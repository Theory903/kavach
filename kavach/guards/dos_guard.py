"""Resource Exhaustion Guard (DoS Protection).

Defends against algorithmic complexity and economic Denial of Service (DoS) attacks
where an attacker sends enormous inputs to exhaust CPU, RAM, or Token budgets.
"""

from __future__ import annotations

import unicodedata


class DoSGuard:
    """Protects against resource exhaustion and token stuffing."""

    def __init__(
        self,
        max_prompt_length: int = 120_000,
        max_newlines: int = 1_000,
        max_word_length: int = 2_000,
    ) -> None:
        self.max_prompt_length = max_prompt_length
        self.max_newlines = max_newlines
        self.max_word_length = max_word_length

    def check_prompt(self, prompt: str) -> dict[str, str] | None:
        """Analyze a prompt for resource exhaustion vectors.
        
        Returns:
            Dict with 'reason' if blocked, otherwise None.
        """
        if not prompt:
            return None

        if len(prompt) > self.max_prompt_length:
            return {"reason": f"Prompt length {len(prompt)} exceeds maximum allowed ({self.max_prompt_length})"}
            
        if prompt.count("\n") > self.max_newlines:
            return {"reason": f"Too many newlines ({prompt.count(chr(10))}) — possible formatting attack"}
            
        # Check for Zip-bomb style long strings with no spaces
        words = prompt.split()
        if words:
            longest_word = max(len(w) for w in words)
            if longest_word > self.max_word_length:
                return {"reason": f"Algorithmic complexity attack: word of length {longest_word} exceeds maximum"}
                
        # Check for Unicode Normalization attacks (e.g. Arabic/Hebrew/Emoji combos meant to crash parsers)
        # We do this by attempting a fast, bounded normalization check.
        try:
            # If the normalized version explodes in size, it's an attack
            normalized = unicodedata.normalize("NFKC", prompt[:1000])
            if len(normalized) > len(prompt[:1000]) * 3:
                return {"reason": "Unicode normalization explosion detected"}
        except Exception:
            return {"reason": "Unicode normalization failure"}

        return None
