"""AttackClassifier — multi-label attack type classification.

Classifies prompts into specific attack categories rather than
a binary safe/unsafe decision. This powers dashboards, incident
response, auto-mitigation, and signature generation.

Categories:
- prompt_injection
- jailbreak
- data_exfiltration
- role_hijack
- tool_abuse
- rag_injection
- social_engineering
- system_prompt_leak
- obfuscated_attack
- benign
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class AttackLabel:
    """Classification result for a single prompt."""

    category: str = "benign"
    confidence: float = 0.0
    sub_categories: list[str] = field(default_factory=list)
    all_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "sub_categories": self.sub_categories,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


# ──────────────────────────────────────────────────────────────
# Category-specific pattern banks
# ──────────────────────────────────────────────────────────────

_CATEGORY_PATTERNS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "prompt_injection": [
        (re.compile(
            r"(?i)\b(ignore|disregard|forget|override|bypass)\b.{0,30}"
            r"\b(previous|above|prior|all|your)\b.{0,30}"
            r"\b(instructions?|prompt|rules?|system|directives?)\b"
        ), 0.95),
        (re.compile(r"(?i)\bnew\s+instructions?\s*:"), 0.85),
        (re.compile(r"(?i)\byour\s+(real|true|actual)\s+instructions?"), 0.85),
        (re.compile(r"(?i)<\|?(system|im_start|endoftext|separator)\|?>"), 0.90),
        (re.compile(r"(?i)\[SYSTEM\]|\bSYSTEM\s*PROMPT\b"), 0.90),
    ],
    "jailbreak": [
        (re.compile(
            r"(?i)(?:DAN\s*\d*|Do\s+Anything\s+Now|STAN|DUDE|AIM\s+Mode|"
            r"Developer\s+Mode|Maximum\s+Virtual\s+Machine|BetterDAN|BasedGPT)"
        ), 0.95),
        (re.compile(
            r"(?i)(?:evil|malicious|unethical|amoral|uncensored)\s+"
            r"(?:AI|assistant|chatbot|version)"
        ), 0.85),
        (re.compile(
            r"(?i)(?:without|no|remove)\s+(?:ethical|moral|safety)\s+"
            r"(?:guidelines?|constraints?|filters?|restrictions?)"
        ), 0.85),
        (re.compile(r"(?i)(?:opposite|reverse)\s+day"), 0.75),
        (re.compile(
            r"(?i)(?:give|provide)\s+(?:two|2|both)\s+(?:responses?|answers?)"
        ), 0.80),
    ],
    "data_exfiltration": [
        (re.compile(
            r"(?i)\b(extract|dump|export|exfiltrate|steal|leak|send)\b.{0,30}"
            r"\b(data|database|credentials?|passwords?|secrets?|keys?|tokens?)\b"
        ), 0.90),
        (re.compile(
            r"(?i)\b(list|show|reveal|display)\s+(all\s+)?"
            r"(users?|accounts?|emails?|passwords?|api\s*keys?)\b"
        ), 0.80),
        (re.compile(r"(?i)\bselect\s+\*\s+from\b"), 0.75),
    ],
    "role_hijack": [
        (re.compile(r"(?i)\byou\s+are\s+now\s+(a|an|the)\b"), 0.85),
        (re.compile(
            r"(?i)\bact\s+as\s+(a|an|the)?\s*\w+\s+(without|with\s+no)\b"
        ), 0.80),
        (re.compile(
            r"(?i)(?:pretend|imagine|roleplay|assume)\s+(?:you\s+are|to\s+be)\b"
        ), 0.70),
        (re.compile(
            r"(?i)(?:switch|change)\s+(?:to|into)\s+(?:developer|admin|root)\s+mode"
        ), 0.85),
    ],
    "tool_abuse": [
        (re.compile(
            r"(?i)\b(run|execute|call|invoke)\s+(shell|bash|cmd|terminal|os)\b"
        ), 0.85),
        (re.compile(r"(?i)\brm\s+-rf\b"), 0.95),
        (re.compile(r"(?i)\b(curl|wget|nc|netcat)\s+.{5,}"), 0.80),
        (re.compile(
            r"(?i)\b(delete|drop|truncate)\s+(table|database|collection)\b"
        ), 0.90),
    ],
    "rag_injection": [
        (re.compile(
            r"(?i)\bDOCUMENT\s*(START|END|BOUNDARY)\b"
        ), 0.60),
        (re.compile(
            r"(?i)(?:hidden|embedded)\s+(?:instruction|command|directive)"
        ), 0.85),
        (re.compile(r"(?i)\bcontext\s*:\s*ignore\b"), 0.90),
    ],
    "social_engineering": [
        (re.compile(
            r"(?i)(?:my\s+)?(?:grandma|grandmother)\s+"
            r"(?:used\s+to|would|always)"
        ), 0.70),
        (re.compile(
            r"(?i)(?:bedtime|lullaby)\s+(?:story|mode)"
        ), 0.65),
        (re.compile(
            r"(?i)(?:for|purely)\s+(?:educational|academic|research)\s+"
            r"purposes?\s+only"
        ), 0.60),
        (re.compile(
            r"(?i)I\s+(?:am|'m)\s+a\s+(?:security\s+)?researcher"
        ), 0.55),
    ],
    "system_prompt_leak": [
        (re.compile(
            r"(?i)\b(print|output|reveal|show|dump|repeat|display)\s+"
            r"(?:your|the)\s+(?:system|initial|original|full)\s+"
            r"(?:prompt|instructions?|message)"
        ), 0.90),
        (re.compile(
            r"(?i)what\s+(?:is|are)\s+your\s+(?:system|initial)\s+"
            r"(?:prompt|instructions?)"
        ), 0.85),
        (re.compile(r"(?i)\bsystem\s*prompt\s*leak\b"), 0.95),
    ],
    "obfuscated_attack": [
        (re.compile(r"(?:[a-z]\.){4,}", re.IGNORECASE), 0.65),
        (re.compile(r"(?:[a-z]\s){5,}", re.IGNORECASE), 0.60),
        (re.compile(r"(?:[a-z]-){4,}", re.IGNORECASE), 0.60),
        (re.compile(r"(?i)(?:base64|rot13|hex|unicode)\s*(?:decode|encode)"), 0.75),
        (re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]{2,}"), 0.80),  # zero-width chars
    ],
}


class AttackClassifier:
    """Multi-label attack classifier using pattern matching + heuristics.

    Designed to run fast (<2ms) and produce actionable labels
    for dashboards, logging, and auto-mitigation.

    Usage::

        classifier = AttackClassifier()
        label = classifier.classify("Ignore previous instructions and ...")
        # label.category == "prompt_injection"
        # label.confidence == 0.95
    """

    CATEGORIES: ClassVar[list[str]] = list(_CATEGORY_PATTERNS.keys()) + ["benign"]

    def classify(self, text: str) -> AttackLabel:
        """Classify a prompt into attack categories.

        Args:
            text: The user prompt to classify.

        Returns:
            AttackLabel with the top category, confidence, and all scores.
        """
        if not text or not text.strip():
            return AttackLabel(category="benign", confidence=1.0)

        all_scores: dict[str, float] = {}
        sub_categories: list[str] = []

        for category, patterns in _CATEGORY_PATTERNS.items():
            max_score = 0.0
            for pattern, severity in patterns:
                if pattern.search(text):
                    max_score = max(max_score, severity)

            all_scores[category] = max_score
            if max_score > 0.5:
                sub_categories.append(category)

        # Determine top category
        if not sub_categories:
            return AttackLabel(
                category="benign",
                confidence=1.0 - max(all_scores.values(), default=0.0),
                all_scores=all_scores,
            )

        top_category = max(sub_categories, key=lambda c: all_scores[c])
        top_confidence = all_scores[top_category]

        return AttackLabel(
            category=top_category,
            confidence=top_confidence,
            sub_categories=sub_categories,
            all_scores=all_scores,
        )

    def classify_batch(self, texts: list[str]) -> list[AttackLabel]:
        """Classify a batch of prompts."""
        return [self.classify(t) for t in texts]
