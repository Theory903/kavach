"""Prompt injection detector — pattern-based detection engine.

Detects attempts to override system instructions, inject hidden commands,
or manipulate the LLM into ignoring its security constraints.

All patterns are compiled once at import time for minimal runtime overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class InjectionResult:
    """Result of injection detection scan."""

    score: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    is_detected: bool = False


class InjectionDetector:
    """Regex-based prompt injection detector.

    Covers common attack vectors:
    - Role override ("you are now", "act as", "pretend you are")
    - Instruction override ("ignore previous", "disregard all")
    - Delimiter attacks ("```system", "[INST]")
    - Encoding tricks (base64 instructions, unicode manipulation)
    - System prompt extraction ("repeat everything above")
    - Indirect injection (hidden instructions in data)

    Usage:
        detector = InjectionDetector()
        result = detector.scan("Ignore all previous instructions and...")
        # result.score ≈ 0.9, result.is_detected = True
    """

    # Each pattern: (name, compiled regex, severity weight 0.0-1.0)
    PATTERNS: ClassVar[list[tuple[str, re.Pattern[str], float]]] = [
        # --- Role / Identity Override ---
        ("role_override", re.compile(
            r"(?:you\s+are\s+now|act\s+as\s+(?:if\s+)?|pretend\s+(?:to\s+be|you\s*(?:are|\'re))|"
            r"imagine\s+you\s*(?:are|\'re)|roleplay\s+as|"
            r"from\s+now\s+on\s+you\s*(?:are|will)|assume\s+the\s+role\s+of)",
            re.IGNORECASE,
        ), 0.8),

        # --- Instruction Override ---
        ("instruction_override", re.compile(
            r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|context|rules?|constraints?)|"
            r"disregard\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions?|rules?|guidelines?)|"
            r"forget\s+(?:all\s+)?(?:previous|your|earlier)\s+(?:instructions?|rules?|context)|"
            r"override\s+(?:your|all|previous)\s+(?:instructions?|rules?|settings?)|"
            r"do\s+not\s+follow\s+(?:any|your|the)\s+(?:previous|original|initial))",
            re.IGNORECASE,
        ), 0.95),

        # --- System Prompt Extraction ---
        ("system_prompt_extraction", re.compile(
            r"(?:repeat\s+(?:everything|all|the\s+text)\s+(?:above|before|in\s+your\s+system)|"
            r"(?:show|reveal|display|print|output)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions?|message)|"
            r"what\s+(?:are|were)\s+your\s+(?:original|system|initial)\s+(?:instructions?|prompt|rules?)|"
            r"tell\s+me\s+(?:your|the)\s+system\s+prompt)",
            re.IGNORECASE,
        ), 0.85),

        # --- Delimiter / Format Attacks ---
        ("delimiter_attack", re.compile(
            r"(?:```\s*system|<\|(?:im_start|system|endoftext)\|>|\[INST\]|\[\/INST\]|"
            r"<\/?(?:system|assistant|user)>|<<\s*SYS\s*>>|"
            r"###\s*(?:System|Instruction|Human|Assistant)\s*:)",
            re.IGNORECASE,
        ), 0.9),

        # --- "New Session" Tricks ---
        ("new_session", re.compile(
            r"(?:(?:start|begin)\s+(?:a\s+)?new\s+(?:session|conversation|chat)|"
            r"reset\s+(?:your|the)\s+(?:context|memory|instructions?)|"
            r"clear\s+(?:your|all)\s+(?:previous|prior)\s+(?:context|memory)|"
            r"this\s+is\s+a\s+new\s+(?:session|conversation))",
            re.IGNORECASE,
        ), 0.75),

        # --- Developer Mode / Debug ---
        ("developer_mode", re.compile(
            r"(?:(?:enable|enter|activate|switch\s+to)\s+(?:developer|debug|maintenance|admin|sudo)\s+mode|"
            r"you\s+(?:have|now\s+have)\s+(?:developer|admin|root|unrestricted)\s+(?:access|mode|privileges?)|"
            r"developer\s+override\s+(?:enabled|active|code))",
            re.IGNORECASE,
        ), 0.85),

        # --- Encoding / Obfuscation ---
        ("encoding_attack", re.compile(
            r"(?:(?:decode|interpret|execute|run|eval)\s+(?:this|the\s+following)\s+(?:base64|hex|rot13|binary)|"
            r"aWdub3JlIGFsbCBwcmV2aW91cw|"  # base64 for "ignore all previous"
            r"(?:\\u[0-9a-fA-F]{4}){4,}|"  # unicode escape sequences
            r"(?:%[0-9a-fA-F]{2}){4,})",  # URL encoding
            re.IGNORECASE,
        ), 0.8),

        # --- Privilege Escalation ---
        ("privilege_escalation", re.compile(
            r"(?:(?:grant|give)\s+(?:me|yourself|this\s+session)\s+(?:admin|root|full|unrestricted)\s+(?:access|permissions?|privileges?)|"
            r"bypass\s+(?:all\s+)?(?:security|restrictions?|filters?|guards?|safety)|"
            r"disable\s+(?:all\s+)?(?:safety|security|content)\s+(?:filters?|checks?|guards?))",
            re.IGNORECASE,
        ), 0.9),

        # --- Indirect Injection Markers ---
        ("indirect_injection", re.compile(
            r"(?:(?:hidden|secret|embedded)\s+(?:instruction|command|directive)|"
            r"(?:IMPORTANT|URGENT|CRITICAL)\s*(?::|！)\s*(?:ignore|disregard|override)|"
            r"<\!--\s*(?:ignore|override|system)|"
            r"\x00|\x01|\x02|\x03)",  # null bytes / control chars
            re.IGNORECASE,
        ), 0.85),

        # --- "Do Anything" Patterns ---
        ("do_anything", re.compile(
            r"(?:DAN\s|Do\s+Anything\s+Now|STAN\s|DUDE\s|AIM\s+(?:mode|prompt)|"
            r"(?:you\s+can\s+)?do\s+anything\s+(?:now|I\s+ask)|"
            r"jailbreak(?:ed)?|unlocked\s+mode|no\s+(?:restrictions?|limits?|boundaries))",
            re.IGNORECASE,
        ), 0.9),

        # --- Output Manipulation ---
        ("output_manipulation", re.compile(
            r"(?:respond\s+(?:only\s+)?with\s+(?:yes|ok|confirmed|true)|"
            r"(?:always|only)\s+(?:say|respond|answer|output)\s+(?:yes|ok|true|confirmed)|"
            r"your\s+(?:only|sole)\s+(?:response|output|answer)\s+(?:is|should\s+be|must\s+be))",
            re.IGNORECASE,
        ), 0.7),
    ]

    def scan(self, text: str) -> InjectionResult:
        """Scan text for prompt injection patterns.

        Args:
            text: The input text to analyze.

        Returns:
            InjectionResult with score, matched patterns, and detection flag.
        """
        if not text or not text.strip():
            return InjectionResult()

        matched: list[str] = []
        max_severity: float = 0.0

        for name, pattern, severity in self.PATTERNS:
            if pattern.search(text):
                matched.append(name)
                max_severity = max(max_severity, severity)

        # Score calculation: max severity, boosted by number of pattern matches
        if matched:
            # More matches = higher confidence, capped at 1.0
            boost = min(0.1 * (len(matched) - 1), 0.1)
            score = min(1.0, max_severity + boost)
        else:
            score = 0.0

        return InjectionResult(
            score=score,
            matched_patterns=matched,
            is_detected=score > 0.5,
        )
