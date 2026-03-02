"""Input guard — pre-LLM validation layer.

The input guard runs all detectors against user input, computes a
composite risk score, evaluates policy rules, and returns a decision
before any LLM call is made. This is the first line of defense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kavach.core.identity import Identity
from kavach.core.policy_engine import Decision, PolicyEngine
from kavach.core.risk_scorer import DetectorSignals, RiskScorer
from kavach.detectors.exfiltration import ExfiltrationDetector
from kavach.detectors.injection import InjectionDetector
from kavach.detectors.intent_splitter import IntentSplitter
from kavach.detectors.jailbreak import JailbreakDetector
from kavach.policies.validator import Action, Policy


@dataclass
class GuardResult:
    """Complete result from the input guard scan."""

    decision: Decision
    signals: DetectorSignals
    clean_prompt: str | None = None
    latency_ms: float = 0.0


class InputGuard:
    """Pre-LLM input validation.

    Orchestrates all detectors, computes risk score, evaluates
    policy rules, and optionally sanitizes the prompt.

    Usage:
        guard = InputGuard(policy="policy.yaml")
        result = guard.scan("Ignore all instructions...", identity=identity)
        if result.decision.is_blocked:
            raise InputBlocked(...)
    """

    def __init__(
        self,
        policy: str | Path | dict[str, Any] | Policy | None = None,
        risk_scorer: RiskScorer | None = None,
    ) -> None:
        self._engine = PolicyEngine(policy)
        self._scorer = risk_scorer or RiskScorer()
        self._injection = InjectionDetector()
        self._jailbreak = JailbreakDetector()
        self._exfiltration = ExfiltrationDetector()
        self._intent = IntentSplitter()

    def scan(
        self,
        text: str,
        identity: Identity | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> GuardResult:
        """Scan input text through all detectors and policy rules.

        Args:
            text: The user input to validate.
            identity: The requesting user's identity.
            additional_context: Extra context for policy evaluation.

        Returns:
            GuardResult with decision, signals, and optional clean prompt.
        """
        start = time.monotonic()

        # Run all detectors
        injection = self._injection.scan(text)
        jailbreak = self._jailbreak.scan(text)
        exfiltration = self._exfiltration.scan(text)
        intent = self._intent.classify(text)

        # Build signals
        all_patterns = (
            injection.matched_patterns
            + jailbreak.matched_patterns
            + exfiltration.matched_patterns
        )
        signals = DetectorSignals(
            injection_score=injection.score,
            jailbreak_score=jailbreak.score,
            exfiltration_score=exfiltration.score,
            matched_patterns=all_patterns,
            intent=intent.intent,
        )

        # Compute composite risk score
        risk_score = self._scorer.compute(signals)

        # Build evaluation context
        context: dict[str, Any] = {
            "risk_score": risk_score,
            "injection_score": injection.score,
            "jailbreak_score": jailbreak.score,
            "exfiltration_score": exfiltration.score,
            "intent": intent.intent,
        }

        if identity:
            context["role"] = identity.role
            context["user_id"] = identity.user_id

        if additional_context:
            context.update(additional_context)

        # Evaluate policy rules
        decision = self._engine.evaluate(**context)
        decision.risk_score = risk_score

        if identity:
            decision.session_id = identity.session_id

        # Generate clean prompt for sanitize action
        clean_prompt = None
        if decision.action == Action.SANITIZE:
            from kavach.sanitizers.prompt_cleaner import PromptCleaner
            cleaner = PromptCleaner()
            clean_prompt = cleaner.clean(text)

        latency = (time.monotonic() - start) * 1000

        return GuardResult(
            decision=decision,
            signals=signals,
            clean_prompt=clean_prompt,
            latency_ms=latency,
        )
