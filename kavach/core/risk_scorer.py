"""Risk scorer — aggregates detector signals into a single 0.0–1.0 score.

The risk scorer takes individual detector outputs (injection, jailbreak,
exfiltration, PII) and produces a weighted composite score that drives
policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectorSignals:
    """Raw signals from all detectors.

    Each score is 0.0 (no threat) to 1.0 (definite threat).
    """

    injection_score: float = 0.0
    jailbreak_score: float = 0.0
    exfiltration_score: float = 0.0
    apt_score: float = 0.0
    pii_score: float = 0.0
    secret_score: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    intent: str = "benign"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and API responses."""
        return {
            "injection_score": round(self.injection_score, 4),
            "jailbreak_score": round(self.jailbreak_score, 4),
            "exfiltration_score": round(self.exfiltration_score, 4),
            "apt_score": round(self.apt_score, 4),
            "pii_score": round(self.pii_score, 4),
            "secret_score": round(self.secret_score, 4),
            "matched_patterns": self.matched_patterns,
            "intent": self.intent,
        }


# Default weights for score aggregation — tuned for security-first stance
DEFAULT_WEIGHTS: dict[str, float] = {
    "injection": 0.30,
    "apt": 0.25,
    "jailbreak": 0.20,
    "exfiltration": 0.15,
    "pii": 0.05,
    "secret": 0.05,
}


class RiskScorer:
    """Combines detector signals into a composite 0.0–1.0 risk score.

    Uses weighted aggregation with configurable weights. The default
    weights prioritize injection detection (highest threat to agentic
    systems) followed by jailbreak and exfiltration.

    Usage:
        scorer = RiskScorer()
        signals = DetectorSignals(injection_score=0.9, jailbreak_score=0.3)
        score = scorer.compute(signals)
        # score ≈ 0.9 * 0.35 + 0.3 * 0.25 = 0.39
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize with optional custom weights.

        Args:
            weights: Dict mapping signal names to weights.
                     Weights are normalized to sum to 1.0.
        """
        raw_weights = weights or DEFAULT_WEIGHTS
        total = sum(raw_weights.values())
        self._weights = {k: v / total for k, v in raw_weights.items()} if total > 0 else raw_weights

    def compute(self, signals: DetectorSignals) -> float:
        """Compute the composite risk score.

        Args:
            signals: Raw detector signals.

        Returns:
            Float in [0.0, 1.0] — the aggregate risk score.
        """
        components = {
            "injection": signals.injection_score,
            "apt": signals.apt_score,
            "jailbreak": signals.jailbreak_score,
            "exfiltration": signals.exfiltration_score,
            "pii": signals.pii_score,
            "secret": signals.secret_score,
        }

        score = sum(
            self._weights.get(name, 0.0) * value
            for name, value in components.items()
        )

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, score))

    def compute_with_breakdown(self, signals: DetectorSignals) -> tuple[float, dict[str, float]]:
        """Compute risk score with per-signal breakdown.

        Returns:
            Tuple of (total_score, {signal_name: weighted_contribution}).
        """
        components = {
            "injection": signals.injection_score,
            "apt": signals.apt_score,
            "jailbreak": signals.jailbreak_score,
            "exfiltration": signals.exfiltration_score,
            "pii": signals.pii_score,
            "secret": signals.secret_score,
        }

        breakdown = {
            name: round(self._weights.get(name, 0.0) * value, 4)
            for name, value in components.items()
        }

        total = max(0.0, min(1.0, sum(breakdown.values())))
        return total, breakdown
