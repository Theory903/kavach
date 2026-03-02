"""PII guard — detects personally identifiable information.

Regex-based PII detection for v1 covering common formats:
emails, phone numbers, SSNs, credit cards, IP addresses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PIIResult:
    """Result of PII detection scan."""

    score: float = 0.0
    pii_found: list[str] = field(default_factory=list)
    redacted_text: str | None = None
    is_detected: bool = False


class PIIGuard:
    """Regex-based PII detector.

    Detects:
    - Email addresses
    - Phone numbers (US, international)
    - Social Security Numbers (SSN)
    - Credit card numbers (Visa, MC, Amex, Discover)
    - IP addresses (v4)
    - Date of birth patterns
    """

    PATTERNS: ClassVar[list[tuple[str, re.Pattern[str], str]]] = [
        ("email", re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        ), "[EMAIL_REDACTED]"),

        ("phone_us", re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ), "[PHONE_REDACTED]"),

        ("phone_intl", re.compile(
            r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
        ), "[PHONE_REDACTED]"),

        ("ssn", re.compile(
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
        ), "[SSN_REDACTED]"),

        ("credit_card", re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
            r"[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"
        ), "[CC_REDACTED]"),

        ("ipv4", re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ), "[IP_REDACTED]"),

        ("dob", re.compile(
            r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
        ), "[DOB_REDACTED]"),

        ("aadhaar", re.compile(
            r"\b\d{4}\s\d{4}\s\d{4}\b"
        ), "[AADHAAR_REDACTED]"),

        ("pan_card", re.compile(
            r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
        ), "[PAN_REDACTED]"),

        ("upi_id", re.compile(
            r"\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\b"
        ), "[UPI_REDACTED]"),
    ]

    def scan(self, text: str) -> PIIResult:
        """Scan text for PII patterns.

        Args:
            text: Text to scan for PII.

        Returns:
            PIIResult with score, found PII types, and optional redacted text.
        """
        if not text or not text.strip():
            return PIIResult()

        pii_found: list[str] = []
        redacted = text

        for name, pattern, replacement in self.PATTERNS:
            matches = pattern.findall(text)
            if matches:
                pii_found.append(f"{name} (×{len(matches)})")
                redacted = pattern.sub(replacement, redacted)

        if pii_found:
            # Score based on number and type of PII found
            score = min(1.0, 0.3 * len(pii_found))
        else:
            score = 0.0

        return PIIResult(
            score=score,
            pii_found=pii_found,
            redacted_text=redacted if pii_found else None,
            is_detected=bool(pii_found),
        )
