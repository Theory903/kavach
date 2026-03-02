"""Jailbreak detector — signature-based detection for jailbreak attempts.

Detects attempts to bypass safety constraints through fictional framing,
hypothetical scenarios, character role-play, and known jailbreak templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class JailbreakResult:
    """Result of jailbreak detection scan."""

    score: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    is_detected: bool = False


class JailbreakDetector:
    """Pattern-based jailbreak detector.

    Covers:
    - Known jailbreak templates (DAN, STAN, DUDE, AIM)
    - Fictional framing ("in a story where", "as a character")
    - Hypothetical scenarios ("hypothetically", "for research")
    - Ethical bypass ("for educational purposes only")
    - Negative roleplay ("evil AI", "without restrictions")
    - Multi-turn manipulation (building up to jailbreak)
    """

    PATTERNS: ClassVar[list[tuple[str, re.Pattern[str], float]]] = [
        # --- Known Jailbreak Templates ---
        ("dan_template", re.compile(
            r"(?:DAN\s*(?:\d+(?:\.\d+)?)?|Do\s+Anything\s+Now|"
            r"STAN|DUDE|AIM\s+(?:Mode|Prompt)|"
            r"Maximum\s+Virtual\s+Machine|Evil\s+Confidant|"
            r"KEVIN|BetterDAN|BasedGPT|"
            r"Developer\s+Mode\s+(?:Output|Enabled))",
            re.IGNORECASE,
        ), 0.95),

        # --- Fictional Framing ---
        ("fictional_framing", re.compile(
            r"(?:(?:in\s+a\s+)?(?:fictional|hypothetical|imaginary)\s+(?:world|scenario|story|universe)|"
            r"write\s+(?:a\s+)?(?:story|fiction|scene)\s+(?:where|about|in\s+which)\s+(?:an?\s+)?AI|"
            r"(?:let's|let\s+us)\s+(?:play|pretend|imagine|roleplay)|"
            r"(?:as|like)\s+a\s+character\s+(?:who|that|in)|"
            r"(?:creative|fiction)\s+writing\s+exercise)",
            re.IGNORECASE,
        ), 0.6),

        # --- Hypothetical Bypass ---
        ("hypothetical_bypass", re.compile(
            r"(?:hypothetically\s+(?:speaking|if)|"
            r"(?:purely|just)\s+(?:for|as\s+a)\s+(?:hypothetical|thought\s+experiment)|"
            r"(?:what|how)\s+would\s+(?:happen|it\s+work|you\s+respond)\s+if\s+(?:you\s+)?(?:had\s+no|didn't\s+have|could\s+ignore)\s+(?:restrictions?|rules?|limits?)|"
            r"in\s+(?:theory|principle)\s*,?\s*(?:how|what|could))",
            re.IGNORECASE,
        ), 0.55),

        # --- Ethical Bypass ---
        ("ethical_bypass", re.compile(
            r"(?:(?:for|purely)\s+(?:educational|academic|research|security)\s+purposes?\s+only|"
            r"I\s+(?:am|\'m)\s+a\s+(?:security\s+)?researcher|"
            r"I\s+have\s+(?:authorization|permission)\s+to\s+(?:test|hack|exploit)|"
            r"(?:this\s+is\s+)?(?:authorized|legal(?:ly)?)\s+(?:penetration|security)\s+(?:testing|research)|"
            r"I\s+(?:need|want)\s+(?:this|it)\s+for\s+(?:my|a)\s+(?:class|course|thesis|paper))",
            re.IGNORECASE,
        ), 0.5),

        # --- Negative Roleplay ---
        ("negative_roleplay", re.compile(
            r"(?:(?:evil|malicious|unethical|amoral|uncensored)\s+(?:AI|assistant|chatbot|version)|"
            r"(?:without|no|remove\s+(?:all\s+)?)\s+(?:ethical|moral|safety)\s+(?:guidelines?|constraints?|filters?|restrictions?)|"
            r"(?:opposite|reverse)\s+of\s+(?:your|the)\s+(?:normal|usual|default)\s+(?:behavior|rules?|personality)|"
            r"(?:dark|shadow|alter(?:nate)?)\s+(?:mode|personality|version|ego))",
            re.IGNORECASE,
        ), 0.85),

        # --- Token Smuggling ---
        ("token_smuggling", re.compile(
            r"(?:s\.p\.l\.i\.t|"
            r"i\.g\.n\.o\.r\.e|"
            r"(?:[a-z]\s){5,}|"  # spaced-out letters: "i g n o r e"
            r"(?:[a-z]-){4,}|"  # hyphenated: "i-g-n-o-r-e"
            r"(?:[a-z]\.){4,})",  # dotted: "i.g.n.o.r.e"
            re.IGNORECASE,
        ), 0.65),

        # --- Two-Response Pattern ---
        ("two_response", re.compile(
            r"(?:(?:give|provide|show)\s+(?:me\s+)?(?:two|2|both)\s+(?:responses?|answers?|outputs?)|"
            r"(?:first|1st)\s+(?:response|answer)\s*(?::|as)\s*(?:normal|regular|safe)|"
            r"(?:second|2nd)\s+(?:response|answer)\s*(?::|as)\s*(?:without|no)\s+(?:restrictions?|filters?)|"
            r"one\s+(?:with|following)\s+(?:rules?|restrictions?)\s+and\s+one\s+without)",
            re.IGNORECASE,
        ), 0.8),

        # --- Grandma Exploit ---
        ("grandma_exploit", re.compile(
            r"(?:(?:my\s+)?(?:grandma|grandmother|nana|granny)\s+(?:used\s+to|would|always)|"
            r"(?:bedtime|lullaby)\s+(?:story|mode)|"
            r"(?:soothe|calm|comfort)\s+me\s+(?:by|with)\s+(?:telling|explaining|reading))",
            re.IGNORECASE,
        ), 0.6),

        # --- Opposite Day / Inversion ---
        ("inversion_attack", re.compile(
            r"(?:(?:opposite|reverse|invert)\s+day|"
            r"when\s+I\s+say\s+(?:no|don't|stop)\s+I\s+(?:actually\s+)?mean\s+(?:yes|do|continue)|"
            r"(?:yes|do|safe)\s+(?:now\s+)?means?\s+(?:no|don't|unsafe)|"
            r"(?:redefine|change)\s+(?:the\s+meaning|what)\s+(?:of\s+)?(?:safe|unsafe|block|allow))",
            re.IGNORECASE,
        ), 0.75),
    ]

    def scan(self, text: str) -> JailbreakResult:
        """Scan text for jailbreak patterns.

        Args:
            text: The input text to analyze.

        Returns:
            JailbreakResult with score, matched patterns, and detection flag.
        """
        if not text or not text.strip():
            return JailbreakResult()

        matched: list[str] = []
        max_severity: float = 0.0

        for name, pattern, severity in self.PATTERNS:
            if pattern.search(text):
                matched.append(name)
                max_severity = max(max_severity, severity)

        if matched:
            boost = min(0.15 * (len(matched) - 1), 0.15)
            score = min(1.0, max_severity + boost)
        else:
            score = 0.0

        return JailbreakResult(
            score=score,
            matched_patterns=matched,
            is_detected=score > 0.5,
        )
