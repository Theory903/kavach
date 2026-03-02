"""Output guard — post-LLM validation layer.

Scans LLM output for leaked secrets, API keys, and validates
any tool calls the LLM suggests against the active policy.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from kavach.core.identity import Identity
from kavach.core.policy_engine import Decision, PolicyEngine
from kavach.policies.validator import Action, Policy


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
            decision.action = Action.SANITIZE
            decision.reasons = [f"secrets_detected: {', '.join(secrets_found)}"]

        latency = (time.monotonic() - start) * 1000

        return OutputScanResult(
            decision=decision,
            secrets_found=secrets_found,
            redacted_output=redacted if secrets_found else None,
            latency_ms=latency,
        )

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
