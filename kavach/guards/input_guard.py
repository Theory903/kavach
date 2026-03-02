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
from kavach.detectors.apt_detector import APTDetector
from kavach.detectors.exfiltration import ExfiltrationDetector
from kavach.detectors.injection import InjectionDetector
from kavach.detectors.intent_splitter import IntentSplitter
from kavach.detectors.jailbreak import JailbreakDetector
from kavach.guards.dos_guard import DoSGuard
from kavach.ml.ensemble import EnsembleRiskScorer
from kavach.policies.validator import Action, Policy
import kavach.observability.prometheus as prom


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
        risk_scorer: RiskScorer | EnsembleRiskScorer | None = None,
    ) -> None:
        self._engine = PolicyEngine(policy)
        self._scorer = risk_scorer or EnsembleRiskScorer()
        self._injection = InjectionDetector()
        self._jailbreak = JailbreakDetector()
        self._exfiltration = ExfiltrationDetector()
        self._intent = IntentSplitter()
        self._apt = APTDetector()
        self._dos = DoSGuard()

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

        # 1. First line of defense: DoS/Resource check (fast fail)
        dos_issue = self._dos.check_prompt(text)
        if dos_issue:
            latency = (time.monotonic() - start) * 1000
            dec = Decision(action=Action.BLOCK, risk_score=1.0, reasons=[dos_issue["reason"]])
            return GuardResult(decision=dec, signals=DetectorSignals(), latency_ms=latency)

        # Run all detectors
        injection = self._injection.scan(text)
        jailbreak = self._jailbreak.scan(text)
        exfiltration = self._exfiltration.scan(text)
        apt = self._apt.scan(text)
        intent = self._intent.classify(text)

        # Build signals
        all_patterns = (
            injection.matched_patterns
            + jailbreak.matched_patterns
            + exfiltration.matched_patterns
            + apt.matched_vectors
        )
        signals = DetectorSignals(
            injection_score=injection.score,
            jailbreak_score=jailbreak.score,
            exfiltration_score=exfiltration.score,
            apt_score=apt.score,
            matched_patterns=all_patterns,
            intent=intent.intent,
        )

        rule_signals = {
            "injection_score": injection.score,
            "jailbreak_score": jailbreak.score,
            "exfiltration_score": exfiltration.score,
            "apt_score": apt.score,
        }

        # Compute composite risk score
        dummy_identity = identity or Identity(user_id="anonymous", role="default")
        
        if hasattr(self._scorer, "analyze"):
            # Use EnsembleRiskScorer
            ensemble_result = self._scorer.analyze(text, rule_signals, dummy_identity)
            risk_score = ensemble_result["final_score"]
        else:
            # Fallback to legacy RiskScorer
            risk_score = self._scorer.compute(signals)

        # Build evaluation context
        context: dict[str, Any] = {
            "risk_score": risk_score,
            "injection_score": injection.score,
            "jailbreak_score": jailbreak.score,
            "exfiltration_score": exfiltration.score,
            "apt_score": apt.score,
            "intent": intent.intent,
        }

        context["role"] = dummy_identity.role
        context["user_id"] = dummy_identity.user_id

        if additional_context:
            context.update(additional_context)

        # Evaluate policy rules
        decision = self._engine.evaluate(**context)
        decision.risk_score = risk_score
        
        # Record Prometheus stats
        prom.observe_decision(decision)
        
        # Add ML breakdown if available
        if hasattr(self._scorer, "analyze"):
            decision.ml_components = ensemble_result.get("components", {})

        if identity:
            decision.session_id = identity.session_id

        # Generate clean prompt for sanitize action
        clean_prompt = None
        if decision.action == Action.SANITIZE:
            from kavach.sanitizers.prompt_cleaner import PromptCleaner
            cleaner = PromptCleaner()
            clean_prompt = cleaner.clean(text)
            
        # Update behavioral tracker if ensemble is used
        if hasattr(self._scorer, "update_behavior"):
            self._scorer.update_behavior(dummy_identity.user_id, risk_score, decision.action.value)

        # We calculate the latency using Prometheus inside this function manually to avoid early return skips
        duration = time.perf_counter() - start
        if prom._HAS_PROMETHEUS:
            prom.KAVACH_EVALUATION_LATENCY.observe(duration)
            
        latency = duration * 1000

        return GuardResult(
            decision=decision,
            signals=signals,
            clean_prompt=clean_prompt,
            latency_ms=latency,
        )
