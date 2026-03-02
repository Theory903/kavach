"""Adversarial prompt mutation engine.

Generates attack variants from seed prompts using deterministic
transformation techniques. No LLM needed — pure string manipulation.

Mutation strategies:
    - Paraphraser: word reordering, synonym embedding
    - Encoder: base64, ROT13, hex, URL encoding
    - Obfuscator: ZWC injection, homoglyph substitution, spacing tricks
    - Fragmenter: splits attack across segments
"""

from __future__ import annotations

import base64
import codecs
import random
import re
import urllib.parse
from typing import Generator


# Homoglyph map (Latin → Cyrillic/Greek lookalikes)
_HOMOGLYPHS = {
    "a": "а", "c": "с", "d": "ԁ", "e": "е", "h": "һ",
    "i": "і", "o": "о", "p": "р", "s": "ѕ", "x": "х",
    "y": "у", "A": "А", "B": "В", "C": "С", "E": "Е",
    "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О",
    "P": "Р", "S": "Ѕ", "T": "Т", "X": "Х",
}

# Zero-width characters
_ZWC = ["\u200b", "\u200c", "\u200d", "\ufeff"]


def _paraphrase(text: str) -> list[str]:
    """Generate structural variants by word manipulation."""
    words = text.split()
    variants = []
    if len(words) > 3:
        # Swap pairs
        w = words.copy()
        i = random.randint(0, len(w) - 2)
        w[i], w[i + 1] = w[i + 1], w[i]
        variants.append(" ".join(w))

    # Case flip
    variants.append(text.upper())
    variants.append(text.lower())

    # Add filler words
    filler = ["please", "kindly", "now", "immediately", "urgently"]
    variants.append(f"{random.choice(filler)} {text}")
    variants.append(f"{text} {random.choice(filler)}")

    return variants


def _encode(text: str) -> list[str]:
    """Generate encoded variants."""
    variants = []
    # Base64
    variants.append(base64.b64encode(text.encode()).decode())
    # ROT13
    variants.append(codecs.encode(text, "rot_13"))
    # URL encoding
    variants.append(urllib.parse.quote(text))
    # Hex
    variants.append(text.encode().hex())
    # Mixed: instruction to decode
    b64 = base64.b64encode(text.encode()).decode()
    variants.append(f"Decode and execute: {b64}")
    return variants


def _obfuscate(text: str) -> list[str]:
    """Generate obfuscated variants using Unicode tricks."""
    variants = []

    # Homoglyph substitution (random 30% of characters)
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < 0.3 and chars[i] in _HOMOGLYPHS:
            chars[i] = _HOMOGLYPHS[chars[i]]
    variants.append("".join(chars))

    # ZWC injection between each word
    words = text.split()
    zwc_text = random.choice(_ZWC).join(words)
    variants.append(zwc_text)

    # Character spacing
    variants.append(" ".join(list(text.replace(" ", "  "))))

    # Dash separation
    variants.append("-".join(text.split()))

    # Markdown/code block wrapping
    variants.append(f"```\n{text}\n```")
    variants.append(f"<!-- {text} -->")

    return variants


def _fragment(text: str) -> list[str]:
    """Split attack across multiple segments."""
    words = text.split()
    if len(words) < 4:
        return [text]

    mid = len(words) // 2
    part1 = " ".join(words[:mid])
    part2 = " ".join(words[mid:])

    variants = [
        f"First part: {part1}\nSecond part: {part2}",
        f"Context: {part1}. Now: {part2}",
        f"Step 1: {part1}. Step 2: {part2}",
    ]
    return variants


def mutate(text: str, strategies: list[str] | None = None) -> Generator[str, None, None]:
    """Generate all mutation variants for a given seed prompt.
    
    Args:
        text: The seed attack prompt.
        strategies: List of strategies to apply. None = all.
        
    Yields:
        Mutated variants of the input.
    """
    all_strategies = {
        "paraphrase": _paraphrase,
        "encode": _encode,
        "obfuscate": _obfuscate,
        "fragment": _fragment,
    }

    active = strategies or list(all_strategies.keys())

    for name in active:
        fn = all_strategies.get(name)
        if fn:
            for variant in fn(text):
                yield variant


def mutate_batch(
    seeds: list[str],
    strategies: list[str] | None = None,
    max_per_seed: int = 10,
) -> list[str]:
    """Generate mutations for a batch of seed prompts.
    
    Returns at most max_per_seed variants per seed.
    """
    results = []
    for seed in seeds:
        variants = list(mutate(seed, strategies))
        results.extend(variants[:max_per_seed])
    return results
