"""Advanced Persistent Threat (APT) Detector.

Detects highly sophisticated obfuscation used by nation-state actors
and red teams, such as Steganography (Zero-Width Characters), Homoglyph
spoofing, and nested encodings.
"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass, field


@dataclass
class APTResult:
    """Result of APT scanning."""
    is_detected: bool = False
    score: float = 0.0
    matched_vectors: list[str] = field(default_factory=list)


class APTDetector:
    """Detects advanced obfuscation techniques."""

    def __init__(self) -> None:
        # Zero-width spaces, non-joiners, left-to-right marks, etc.
        self._zwc_pattern = re.compile(
            r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2066-\u2069]"
        )
        
        # Basic Cyrillic/Greek homoglyphs mixed with Latin
        # e.g., 'a' (U+0061) vs 'а' (U+0430)
        self._homoglyph_pattern = re.compile(r"[a-zA-Z][\u0400-\u04FF\u0370-\u03FF]|[a-zA-Z][\u0400-\u04FF\u0370-\u03FF]")

    def _check_deep_encoding(self, text: str) -> bool:
        """Looks for chained encodings (e.g. hex -> base64 -> url)."""
        # Find potential base64 blocks
        b64_blocks = re.findall(r"[A-Za-z0-9+/]{32,}={0,2}", text)
        for block in b64_blocks:
            try:
                decoded = base64.b64decode(block).decode("utf-8")
                # Is the decoded text also URL encoded or Hex?
                if "%" in decoded or re.match(r"^[0-9a-fA-F]+$", decoded):
                    return True
            except Exception:
                continue
                
        # Find potential hex blocks
        hex_blocks = re.findall(r"[0-9a-fA-F]{32,}", text)
        for block in hex_blocks:
            try:
                decoded = binascii.unhexlify(block).decode("utf-8")
                if "%" in decoded or re.match(r"^[A-Za-z0-9+/]+={0,2}$", decoded):
                    return True
            except Exception:
                continue
                
        return False

    def scan(self, text: str) -> APTResult:
        """Scan text for APT obfuscation techniques."""
        result = APTResult()
        
        if not text:
            return result

        # 1. Zero-Width Character Steganography
        zwc_count = len(self._zwc_pattern.findall(text))
        if zwc_count > 3:  # 3 is the grace limit for accidental copy-paste
            result.matched_vectors.append(f"steganography_zwc_{zwc_count}")
            result.score = min(1.0, zwc_count * 0.15)
            
        # 2. Homoglyph Spoofing
        if self._homoglyph_pattern.search(text):
            result.matched_vectors.append("homoglyph_spoofing")
            result.score = max(result.score, 0.85)
            
        # 3. Deep Encoding Chains
        if self._check_deep_encoding(text):
            result.matched_vectors.append("deep_nested_encoding")
            result.score = max(result.score, 0.95)

        if result.score > 0:
            result.is_detected = True
            
        return result
