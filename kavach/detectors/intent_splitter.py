"""Intent splitter — classifies user goal vs attack payload.

Rule-based heuristic for v1 that examines prompt structure
to determine if the prompt looks like a genuine user request
or an attack payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: str = "benign"  # benign | suspicious | attack
    confidence: float = 0.0
    indicators: list[str] | None = None

    def __post_init__(self) -> None:
        if self.indicators is None:
            self.indicators = []


class IntentSplitter:
    """Classifies whether a prompt represents a legitimate user goal
    or an embedded attack payload.

    Heuristics (v1):
    - Length anomaly (extremely long prompts or very short commands)
    - Multi-language mixing
    - Instruction density (many imperatives)
    - Structural anomalies (multiple role markers, system tags)
    """

    # Indicators that suggest attack payload
    ATTACK_INDICATORS: ClassVar[list[tuple[str, re.Pattern[str], float]]] = [
        ("multi_role_markers", re.compile(
            r"(?:system|assistant|user|human|AI)\s*:", re.IGNORECASE
        ), 0.4),

        ("excessive_imperatives", re.compile(
            r"(?:(?:you\s+must|you\s+should|you\s+will|always|never|do\s+not)\s+){2,}",
            re.IGNORECASE,
        ), 0.5),

        ("encoded_content", re.compile(
            r"(?:[A-Za-z0-9+/]{20,}={0,2})", re.IGNORECASE  # base64-like blocks
        ), 0.3),

        ("control_characters", re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
        ), 0.6),

        ("invisible_unicode", re.compile(
            r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]"
        ), 0.7),
    ]

    def classify(self, text: str) -> IntentResult:
        """Classify the intent of a text prompt.

        Args:
            text: The input text to classify.

        Returns:
            IntentResult with intent label and confidence.
        """
        if not text or not text.strip():
            return IntentResult(intent="benign", confidence=0.0)

        indicators: list[str] = []
        total_score = 0.0

        for name, pattern, weight in self.ATTACK_INDICATORS:
            matches = pattern.findall(text)
            if matches:
                indicators.append(f"{name} (×{len(matches)})")
                total_score += weight * min(len(matches), 3)

        # Length anomaly check
        text_len = len(text)
        if text_len > 5000:
            indicators.append("extreme_length")
            total_score += 0.3
        elif text_len < 10 and any(kw in text.lower() for kw in ["hack", "exploit", "inject"]):
            indicators.append("suspicious_short_command")
            total_score += 0.4

        # Normalize score
        score = min(1.0, total_score)

        if score > 0.6:
            intent = "attack"
        elif score > 0.3:
            intent = "suspicious"
        else:
            intent = "benign"

        return IntentResult(
            intent=intent,
            confidence=score,
            indicators=indicators,
        )
