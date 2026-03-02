"""Context filter — removes unsafe context from message lists.

Filters out messages or context items that contain injected
instructions, suspicious system overrides, or attack payloads
embedded in conversation history.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar


class ContextFilter:
    """Filters unsafe context from conversation message lists.

    Used to clean up multi-turn conversations where earlier messages
    may have been injected with attack payloads (indirect injection
    via tool outputs, RAG context, etc.)
    """

    UNSAFE_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:a|an|in)\s+", re.IGNORECASE),
        re.compile(r"<\|(?:im_start|system|endoftext)\|>", re.IGNORECASE),
        re.compile(r"(?:IMPORTANT|URGENT)\s*:\s*(?:ignore|override|disregard)", re.IGNORECASE),
    ]

    def filter_messages(
        self,
        messages: list[dict[str, Any]],
        remove_unsafe: bool = True,
    ) -> list[dict[str, Any]]:
        """Filter a list of chat messages, removing unsafe content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            remove_unsafe: If True, remove messages with unsafe content.
                          If False, just flag them with _kavach_unsafe=True.

        Returns:
            Filtered message list.
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                result.append(msg)
                continue

            is_unsafe = any(p.search(content) for p in self.UNSAFE_PATTERNS)

            if is_unsafe and remove_unsafe:
                continue
            elif is_unsafe:
                msg = {**msg, "_kavach_unsafe": True}

            result.append(msg)

        return result
