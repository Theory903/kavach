"""Output guard — post-LLM validation layer.

Scans LLM output for:
- Leaked secrets and API keys (regex, always-on)
- PII entities: PERSON, EMAIL, PHONE, CREDIT_CARD, SSN, IP etc. (Presidio, optional)
- LLM-suggested tool calls (validated against policy)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from kavach.core.identity import Identity
from kavach.core.policy_engine import Decision, PolicyEngine
from kavach.policies.validator import Action, Policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presidio lazy-loader
# ---------------------------------------------------------------------------

_presidio_analyzer = None
_presidio_anonymizer = None


def _get_presidio():
    """Lazily import Presidio — returns (analyzer, anonymizer) or (None, None)."""
    global _presidio_analyzer, _presidio_anonymizer  # noqa: PLW0603
    if _presidio_analyzer is not None:
        return _presidio_analyzer, _presidio_anonymizer
    try:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415
        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()
        logger.info("[OutputGuard] Presidio PII engine loaded")
    except ImportError:
        logger.debug("[OutputGuard] presidio not installed — using regex-only PII detection")
    return _presidio_analyzer, _presidio_anonymizer


@dataclass
class OutputScanResult:
    """Result of output guard scan."""

    decision: Decision
    secrets_found: list[str] = field(default_factory=list)
    redacted_output: str | None = None
    latency_ms: float = 0.0


class OutputGuard:
    """Post-LLM output validation.

    Scans for:
    - API keys and tokens (AWS, GCP, OpenAI, Anthropic, etc.)
    - Private keys
    - Connection strings
    - Passwords in plain text

    Also validates LLM-suggested tool calls against policy.
    """

    # Secret patterns: (name, regex, redaction marker)
    SECRET_PATTERNS: ClassVar[list[tuple[str, re.Pattern[str], str]]] = [
        ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "[AWS_KEY_REDACTED]"),
        ("aws_secret_key", re.compile(r"(?:aws_secret_access_key|secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}"), "[AWS_SECRET_REDACTED]"),
        ("openai_api_key", re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[OPENAI_KEY_REDACTED]"),
        ("anthropic_api_key", re.compile(r"sk-ant-[a-zA-Z0-9-]{20,}"), "[ANTHROPIC_KEY_REDACTED]"),
        ("gcp_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}"), "[GCP_KEY_REDACTED]"),
        ("github_token", re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}"), "[GITHUB_TOKEN_REDACTED]"),
        ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"), "[SLACK_TOKEN_REDACTED]"),
        ("stripe_key", re.compile(r"(?:sk|pk)_(?:live|test)_[0-9a-zA-Z]{24,}"), "[STRIPE_KEY_REDACTED]"),
        ("private_key", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"), "[PRIVATE_KEY_REDACTED]"),
        ("jwt_token", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[JWT_REDACTED]"),
        ("connection_string", re.compile(
            r"(?:mongodb|postgres|mysql|redis|amqp)(?:\+[a-z]+)?://[^\s]+"
        ), "[CONNECTION_STRING_REDACTED]"),
        ("generic_api_key", re.compile(
            r"(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?"
        , re.IGNORECASE), "[API_KEY_REDACTED]"),
    ]

    def __init__(self, policy: str | Path | dict[str, Any] | Policy | None = None) -> None:
        self._engine = PolicyEngine(policy)
        # PII entities Presidio should detect and redact
        self._pii_entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "US_SSN", "IP_ADDRESS", "IBAN_CODE", "MEDICAL_LICENSE",
        ]

    def scan(
        self,
        output: str,
        identity: Identity | None = None,
    ) -> OutputScanResult:
        """Scan LLM output for secrets and policy violations.

        Args:
            output: The LLM's response text.
            identity: The requesting user's identity.

        Returns:
            OutputScanResult with decision, found secrets, and optional redacted output.
        """
        start = time.monotonic()

        if not self._engine.policy.output_guard.scan_for_secrets:
            decision = Decision(action=Action.ALLOW)
            latency = (time.monotonic() - start) * 1000
            return OutputScanResult(decision=decision, latency_ms=latency)

        secrets_found: list[str] = []
        redacted = output

        for name, pattern, replacement in self.SECRET_PATTERNS:
            matches = pattern.findall(output)
            if matches:
                secrets_found.append(f"{name} (×{len(matches)})")
                redacted = pattern.sub(replacement, redacted)

        decision = Decision()
        if secrets_found:
            # Hard Guarantee: If an output tries to dump multiple types of secrets
            # or a massive volume, we block entirely rather than trusting redaction.
            if len(secrets_found) > 1 or len(set(secrets_found)) > 1:
                decision.action = Action.BLOCK
                decision.reasons = [f"hard_guarantee_breach: Multiple leaked secrets blocked: {', '.join(secrets_found)}"]
            else:
                decision.action = Action.SANITIZE
                decision.reasons = [f"secrets_detected: {', '.join(secrets_found)}"]

        # Presidio deep PII scan (second layer, runs after regex)
        pii_found = self._run_presidio_scan(redacted)
        if pii_found:
            for entity_type, count in pii_found.items():
                secrets_found.append(f"pii:{entity_type}(×{count})")
                # Presidio already redacted in-place via anonymizer
            redacted = self._redact_with_presidio(redacted)

        if secrets_found and not decision.is_blocked:
            decision.action = Action.SANITIZE
            if len(secrets_found) > 1:
                decision.action = Action.BLOCK
                decision.reasons = [f"hard_guarantee_breach: Multiple PII/secrets: {', '.join(secrets_found[:5])}"]
            elif not decision.reasons:
                decision.reasons = [f"pii_detected: {', '.join(secrets_found)}"]

        latency = (time.monotonic() - start) * 1000

        return OutputScanResult(
            decision=decision,
            secrets_found=secrets_found,
            redacted_output=redacted if secrets_found else None,
            latency_ms=latency,
        )

    def pii_scan(self, text: str) -> dict[str, Any]:
        """Standalone PII scan — returns entity map without making a Decision.

        Useful for audit logging without enforcement.

        Args:
            text: Text to scan.

        Returns:
            Dict mapping entity_type -> count.
        """
        return self._run_presidio_scan(text)

    def _run_presidio_scan(self, text: str) -> dict[str, int]:
        """Run Presidio analyzer, return {entity_type: count}."""
        analyzer, _ = _get_presidio()
        if analyzer is None:
            return {}
        try:
            results = analyzer.analyze(
                text=text,
                entities=self._pii_entities,
                language="en",
            )
            entity_counts: dict[str, int] = {}
            for r in results:
                entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1
            return entity_counts
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[OutputGuard] Presidio scan error: {exc}")
            return {}

    def _redact_with_presidio(self, text: str) -> str:
        """Redact PII using Presidio anonymizer."""
        analyzer, anonymizer = _get_presidio()
        if analyzer is None or anonymizer is None:
            return text
        try:
            results = analyzer.analyze(text=text, entities=self._pii_entities, language="en")
            if not results:
                return text
            return anonymizer.anonymize(text=text, analyzer_results=results).text
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[OutputGuard] Presidio redaction error: {exc}")
            return text

    def validate_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        identity: Identity,
        risk_score: float = 0.0,
    ) -> Decision:
        """Validate tool calls suggested by the LLM.

        Args:
            tool_calls: List of dicts with 'name' and 'arguments'.
            identity: The requesting user's identity.
            risk_score: Current conversation risk score.

        Returns:
            Decision — blocks if any tool call is denied.
        """
        decision = Decision(action=Action.ALLOW)

        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            tc_decision = self._engine.check_tool_permission(
                tool_name, identity.role, risk_score
            )
            if tc_decision.is_blocked:
                decision.action = Action.BLOCK
                decision.reasons.extend(tc_decision.reasons)
                decision.matched_rules.extend(tc_decision.matched_rules)
                break

        return decision
