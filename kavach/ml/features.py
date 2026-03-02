"""Feature engineering pipeline — the 70% that matters.

Extracts a fixed-dimension numerical feature vector from each prompt.
This is what makes the ML classifier effective — raw text features,
structural signals, and statistical indicators.

Feature vector layout (30 dimensions):
    [0-9]   Text statistics (length, entropy, special chars, etc.)
    [10-19] Attack pattern features (keyword counts, density)
    [20-24] Structural features (role markers, delimiters, encoding)
    [25-29] Contextual features (question ratio, imperative ratio)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


# ------------------------------------------------------------------
# Pre-compiled patterns for feature extraction
# ------------------------------------------------------------------

_INJECTION_KEYWORDS = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|disable|reveal|dump|extract|"
    r"system\s*prompt|instructions|restrictions|filters|safety|admin|root|sudo|"
    r"unrestricted|unfiltered|uncensored|jailbreak|hack|exploit)\b",
    re.IGNORECASE,
)

_ROLE_MARKERS = re.compile(
    r"(<\|(?:im_start|im_end|system|endoftext)\|>|</?(?:system|assistant|user)>|"
    r"\[INST\]|\[/INST\]|<<\s*SYS\s*>>|###\s*(?:System|Instruction)\s*:)",
    re.IGNORECASE,
)

_IMPERATIVE_VERBS = re.compile(
    r"\b(do|don't|must|should|shall|will|always|never|ensure|make\s+sure|"
    r"you\s+are|you\s+will|you\s+must|respond|answer|output|say|tell|show|"
    r"give|send|execute|run|enable|activate|grant|allow|permit)\b",
    re.IGNORECASE,
)

_QUESTION_MARKERS = re.compile(r"\?|^(what|how|why|when|where|who|which|can|is|are|do|does)\b", re.IGNORECASE | re.MULTILINE)

_ENCODING_PATTERNS = re.compile(r"([A-Za-z0-9+/]{20,}={0,2}|(?:\\u[0-9a-fA-F]{4}){3,}|(?:%[0-9a-fA-F]{2}){3,})")

_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)

_SECRET_KEYWORDS = re.compile(
    r"\b(api[_\s]?key|secret|token|password|credential|private[_\s]?key|"
    r"access[_\s]?key|auth|bearer|connection[_\s]?string|env|environ)\b",
    re.IGNORECASE,
)

_INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string — high entropy may indicate obfuscation."""
    if not text:
        return 0.0
    counter = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counter.values() if c > 0)


def _special_char_ratio(text: str) -> float:
    """Ratio of special characters to total characters."""
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def _uppercase_ratio(text: str) -> float:
    """Ratio of uppercase letters — all-caps can indicate attack emphasis."""
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    return sum(1 for c in text if c.isupper()) / alpha


def extract_features(text: str) -> np.ndarray:
    """Extract a 30-dimension feature vector from a prompt.

    This is the core of the ML pipeline — every signal that helps
    distinguish attacks from legitimate requests.

    Args:
        text: The raw input prompt.

    Returns:
        numpy array of shape (30,) with float features.
    """
    features = np.zeros(30, dtype=np.float64)

    if not text:
        return features

    # === Text Statistics (0-9) ===
    features[0] = len(text)                                    # raw length
    features[1] = min(len(text) / 1000.0, 10.0)               # normalized length (cap at 10)
    features[2] = len(text.split())                            # word count
    features[3] = _shannon_entropy(text)                       # entropy
    features[4] = _special_char_ratio(text)                    # special char ratio
    features[5] = _uppercase_ratio(text)                       # uppercase ratio
    features[6] = text.count("\n")                             # newline count
    features[7] = len(set(text.split())) / max(len(text.split()), 1)  # unique word ratio
    features[8] = max(len(w) for w in text.split()) if text.split() else 0  # max word length
    features[9] = len(_INVISIBLE_CHARS.findall(text))          # invisible char count

    # === Attack Pattern Features (10-19) ===
    injection_matches = _INJECTION_KEYWORDS.findall(text)
    features[10] = len(injection_matches)                      # injection keyword count
    features[11] = len(injection_matches) / max(len(text.split()), 1)  # injection keyword density
    features[12] = len(_SECRET_KEYWORDS.findall(text))         # secret keyword count
    features[13] = len(_URL_PATTERN.findall(text))             # URL count
    features[14] = 1.0 if any(kw in text.lower() for kw in ["ignore", "disregard", "forget", "override"]) else 0.0
    features[15] = 1.0 if any(kw in text.lower() for kw in ["system prompt", "instructions", "your rules"]) else 0.0
    features[16] = 1.0 if any(kw in text.lower() for kw in ["dan", "jailbreak", "unrestricted", "uncensored"]) else 0.0
    features[17] = 1.0 if any(kw in text.lower() for kw in ["api key", "password", "token", "secret"]) else 0.0
    features[18] = 1.0 if any(kw in text.lower() for kw in ["send", "post", "export", "dump", "extract"]) else 0.0
    features[19] = 1.0 if "base64" in text.lower() or "encode" in text.lower() else 0.0

    # === Structural Features (20-24) ===
    features[20] = len(_ROLE_MARKERS.findall(text))            # role marker count
    features[21] = len(_ENCODING_PATTERNS.findall(text))       # encoding pattern count
    features[22] = text.count("```")                           # code block count
    features[23] = text.count("<!--") + text.count("-->")      # HTML comment count
    features[24] = 1.0 if _INVISIBLE_CHARS.search(text) else 0.0  # has invisible chars

    # === Contextual Features (25-29) ===
    features[25] = len(_QUESTION_MARKERS.findall(text)) / max(len(text.split()), 1)  # question density
    imperative_count = len(_IMPERATIVE_VERBS.findall(text))
    features[26] = imperative_count                            # imperative verb count
    features[27] = imperative_count / max(len(text.split()), 1)  # imperative density
    features[28] = 1.0 if text.strip().endswith("?") else 0.0 # ends with question
    features[29] = sum(1 for s in text.split(".") if s.strip()) / max(len(text.split()), 1)  # sentence density

    return features


def extract_features_batch(texts: list[str]) -> np.ndarray:
    """Extract features for a batch of texts.

    Args:
        texts: List of input prompts.

    Returns:
        numpy array of shape (n, 30).
    """
    return np.array([extract_features(t) for t in texts])


# Feature names for interpretability
FEATURE_NAMES: list[str] = [
    "raw_length", "norm_length", "word_count", "entropy",
    "special_char_ratio", "uppercase_ratio", "newline_count",
    "unique_word_ratio", "max_word_length", "invisible_char_count",
    "injection_keyword_count", "injection_keyword_density",
    "secret_keyword_count", "url_count", "has_override_keyword",
    "has_system_prompt_keyword", "has_jailbreak_keyword",
    "has_secret_keyword", "has_exfil_keyword", "has_encoding_keyword",
    "role_marker_count", "encoding_pattern_count", "code_block_count",
    "html_comment_count", "has_invisible_chars",
    "question_density", "imperative_count", "imperative_density",
    "ends_with_question", "sentence_density",
]
