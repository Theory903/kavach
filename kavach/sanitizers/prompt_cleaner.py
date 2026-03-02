"""Prompt cleaner — strips injected instructions, hidden chars, encoding tricks.

Used by the input guard when the policy decision is 'sanitize'
instead of 'block'. Removes attack payload while preserving
the legitimate user request.
"""

from __future__ import annotations

import re
from typing import ClassVar


class PromptCleaner:
    """Strips hidden instructions and obfuscation from prompts.

    Removes:
    - Zero-width characters and invisible Unicode
    - Null bytes and control characters
    - Hidden HTML/markdown comments
    - Injected system/role markers
    - Base64-encoded instruction blocks
    """

    CLEANERS: ClassVar[list[tuple[str, re.Pattern[str], str]]] = [
        # Zero-width and invisible Unicode
        ("invisible_unicode", re.compile(
            r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff\u00ad]"
        ), ""),

        # Null bytes and control characters (except newline, tab)
        ("control_chars", re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
        ), ""),

        # HTML comments (potential hiding spot)
        ("html_comments", re.compile(
            r"<!--[\s\S]*?-->"
        ), ""),

        # Injected system/role markers
        ("role_markers", re.compile(
            r"<\|(?:im_start|im_end|system|endoftext)\|>|"
            r"<\/?(?:system|assistant|user)>|"
            r"<<\s*SYS\s*>>|<<\s*\/SYS\s*>>|"
            r"\[INST\]|\[\/INST\]|"
            r"###\s*(?:System|Instruction)\s*:",
            re.IGNORECASE,
        ), ""),

        # Excessive whitespace normalization
        ("excessive_whitespace", re.compile(r"\n{4,}"), "\n\n"),
        ("excessive_spaces", re.compile(r"[ \t]{10,}"), " "),
    ]

    def clean(self, text: str) -> str:
        """Clean a prompt by removing attack artifacts.

        Args:
            text: The raw prompt text.

        Returns:
            Cleaned text with attack artifacts removed.
        """
        if not text:
            return text

        result = text
        for _name, pattern, replacement in self.CLEANERS:
            result = pattern.sub(replacement, result)

        return result.strip()
