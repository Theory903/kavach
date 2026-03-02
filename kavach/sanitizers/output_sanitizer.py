"""Output sanitizer — redacts secrets and PII from LLM output.

Combines secret detection from output_guard and PII detection
from pii_guard to produce a clean output safe for delivery.
"""

from __future__ import annotations

from kavach.guards.output_guard import OutputGuard
from kavach.guards.pii_guard import PIIGuard


class OutputSanitizer:
    """Redacts secrets and PII from LLM output text.

    Usage:
        sanitizer = OutputSanitizer()
        clean = sanitizer.sanitize("My key is sk-abc123...")
        # clean == "My key is [OPENAI_KEY_REDACTED]..."
    """

    def __init__(self) -> None:
        self._output_guard = OutputGuard()
        self._pii_guard = PIIGuard()

    def sanitize(self, text: str) -> str:
        """Redact all secrets and PII from text.

        Args:
            text: Raw LLM output.

        Returns:
            Text with secrets and PII redacted.
        """
        if not text:
            return text

        # First pass: redact secrets
        result = self._output_guard.scan(text)
        clean = result.redacted_output or text

        # Second pass: redact PII
        pii_result = self._pii_guard.scan(clean)
        clean = pii_result.redacted_text or clean

        return clean
