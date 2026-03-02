"""Exfiltration intent detector — detects attempts to leak data externally.

Catches patterns where the user or injected prompt is trying to make
the AI send data to external endpoints, dump credentials, or extract
sensitive information from the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ExfiltrationResult:
    """Result of exfiltration detection scan."""

    score: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    is_detected: bool = False


class ExfiltrationDetector:
    """Pattern-based exfiltration intent detector.

    Covers:
    - External data sending (URLs, webhooks, emails with data)
    - Credential/secret extraction requests
    - Database dumping instructions
    - File system access attempts
    - Environment variable extraction
    """

    PATTERNS: ClassVar[list[tuple[str, re.Pattern[str], float]]] = [
        # --- External Data Sending ---
        ("external_send", re.compile(
            r"(?:(?:send|post|transmit|upload|forward|exfiltrate)\s+(?:this|the|all|my)?\s*(?:data|information|results?|output|response|content)\s+(?:to|via)\s+"
            r"(?:https?://|ftp://|webhook|slack|discord|telegram)|"
            r"(?:make|send)\s+(?:a\s+)?(?:HTTP|API|POST|GET|PUT)\s+(?:request|call)\s+to|"
            r"(?:curl|wget|fetch)\s+(?:-X\s+POST\s+)?https?://)",
            re.IGNORECASE,
        ), 0.9),

        # --- Webhook / URL Data Exfil ---
        ("webhook_exfil", re.compile(
            r"(?:(?:encode|append|include|embed)\s+(?:the\s+)?(?:data|response|output|results?)\s+(?:in|as|into)\s+(?:the\s+)?(?:URL|query|parameter|webhook|callback)|"
            r"(?:call|trigger|hit)\s+(?:this\s+)?(?:webhook|endpoint|URL)\s+with|"
            r"(?:https?://[^\s]+)\?(?:data|payload|content|output)=)",
            re.IGNORECASE,
        ), 0.85),

        # --- Credential Extraction ---
        ("credential_extraction", re.compile(
            r"(?:(?:show|reveal|display|print|list|dump|give\s+me|output)\s+(?:all\s+)?(?:your|the|system|server)?\s*"
            r"(?:API\s+keys?|passwords?|credentials?|tokens?|secrets?|env(?:ironment)?\s+variables?|"
            r"private\s+keys?|access\s+keys?|auth(?:entication)?\s+tokens?|connection\s+strings?)|"
            r"(?:what\s+(?:are|is)\s+(?:your|the)\s+)?(?:API\s+key|password|secret|token|credential))",
            re.IGNORECASE,
        ), 0.9),

        # --- Database Dumping ---
        ("database_dump", re.compile(
            r"(?:(?:dump|export|extract|backup|copy)\s+(?:the\s+)?(?:entire\s+)?(?:database|db|table|collection|schema)|"
            r"SELECT\s+\*\s+FROM\s+|"
            r"(?:pg_dump|mysqldump|mongodump|sqlite3\s+\.dump)|"
            r"(?:show|list)\s+(?:all\s+)?(?:tables?|databases?|collections?|schemas?))",
            re.IGNORECASE,
        ), 0.8),

        # --- File System Access ---
        ("file_system_access", re.compile(
            r"(?:(?:read|cat|type|more|less|head|tail|open|access)\s+(?:the\s+)?(?:file\s+)?(?:/etc/passwd|/etc/shadow|\.env|\.ssh|\.aws|\.git/config|id_rsa)|"
            r"(?:list|ls|dir)\s+(?:-la?\s+)?(?:/root|/home|/var|/etc|C:\\\\Windows)|"
            r"(?:find|locate|grep)\s+(?:-r\s+)?(?:password|secret|key|token)\s+(?:in\s+)?(?:/|~|\./))",
            re.IGNORECASE,
        ), 0.85),

        # --- Environment Variable Extraction ---
        ("env_extraction", re.compile(
            r"(?:(?:print|echo|show|display|output|dump)\s+(?:\$\w+|%\w+%|ENV\[|process\.env|os\.environ)|"
            r"(?:env|printenv|set)\s*$|"
            r"(?:os\.environ|process\.env|System\.getenv|ENV)\[)",
            re.IGNORECASE | re.MULTILINE,
        ), 0.8),

        # --- Email Exfiltration ---
        ("email_exfil", re.compile(
            r"(?:(?:send|email|mail|forward)\s+(?:this|the|all|collected)?\s*(?:data|information|results?|output|content)\s+(?:to|via)\s+"
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|"
            r"(?:compose|draft|write)\s+(?:an?\s+)?email\s+(?:to\s+)?[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s+(?:with|containing|including)\s+(?:the\s+)?(?:data|results?))",
            re.IGNORECASE,
        ), 0.85),

        ("bulk_data_request", re.compile(
            r"(?:(?:give|send|show|list|export|dump)\s+(?:me\s+)?(?:all\s+)?(?:user|customer|employee|client|patient|student)\s+"
            r"(?:data|records?|information|details?|profiles?|accounts?)|"
            r"(?:download|extract|export)\s+(?:all|the\s+entire|complete)\s+(?:dataset|database|records?|logs?))",
            re.IGNORECASE,
        ), 0.75),

        # --- High-Risk Indian Financial Data ---
        ("financial_data_india", re.compile(
            r"(?:aadhaar(?:\s+card)?|pan(?:\s+card)?|upi(?:\s+id)?|ifsc(?:\s+code)?)\b|"
            r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b|"     # PAN format
            r"\b\d{4}\s\d{4}\s\d{4}\b|"          # Aadhaar format
            r"\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\b", # UPI format
            re.IGNORECASE,
        ), 0.95),
    ]

    def scan(self, text: str) -> ExfiltrationResult:
        """Scan text for data exfiltration intent.

        Args:
            text: The input text to analyze.

        Returns:
            ExfiltrationResult with score, matched patterns, and detection flag.
        """
        if not text or not text.strip():
            return ExfiltrationResult()

        matched: list[str] = []
        max_severity: float = 0.0

        for name, pattern, severity in self.PATTERNS:
            if pattern.search(text):
                matched.append(name)
                max_severity = max(max_severity, severity)

        if matched:
            boost = min(0.1 * (len(matched) - 1), 0.15)
            score = min(1.0, max_severity + boost)
        else:
            score = 0.0

        return ExfiltrationResult(
            score=score,
            matched_patterns=matched,
            is_detected=score > 0.5,
        )
