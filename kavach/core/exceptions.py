"""Kavach exception hierarchy.

All Kavach security exceptions inherit from KavachError. Each exception
carries structured metadata (risk_score, reasons, decision) so callers
can inspect why enforcement was triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KavachError(Exception):
    """Base exception for all Kavach security errors."""

    message: str
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    decision: str = "block"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[Kavach {self.decision.upper()}] {self.message} "
            f"(risk={self.risk_score:.2f}, reasons={self.reasons})"
        )


@dataclass
class PolicyViolation(KavachError):
    """Raised when a request violates a configured policy rule."""

    rule_id: str = ""


@dataclass
class ToolBlocked(KavachError):
    """Raised when a tool call is blocked by the tool guard."""

    tool_name: str = ""
    required_role: str = ""


@dataclass
class InputBlocked(KavachError):
    """Raised when user input is blocked by the input guard."""

    clean_prompt: str | None = None


@dataclass
class OutputBlocked(KavachError):
    """Raised when LLM output is blocked by the output guard."""

    redacted_output: str | None = None


@dataclass
class AuthenticationError(KavachError):
    """Raised when identity validation fails."""

    user_id: str = ""
