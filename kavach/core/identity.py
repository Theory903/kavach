"""Identity and RBAC context for Kavach.

Provides request-scoped identity binding so every guard and detector
knows who is making the request and what role they hold.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Identity:
    """Immutable identity for a request.

    Attributes:
        user_id: Unique identifier for the user.
        role: Role name used for RBAC policy evaluation.
        session_id: Auto-generated session identifier for audit trail.
        metadata: Arbitrary key-value pairs for custom policy conditions.
    """

    user_id: str
    role: str
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if identity holds a specific role."""
        return self.role == role

    def to_dict(self) -> dict[str, Any]:
        """Serialize identity for logging (no secrets)."""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "session_id": self.session_id,
        }


# Context variable for request-scoped identity propagation
_current_identity: ContextVar[Identity | None] = ContextVar("kavach_identity", default=None)


class IdentityContext:
    """Context manager for binding an Identity to the current execution scope.

    Usage:
        identity = Identity(user_id="u1", role="analyst")
        with IdentityContext(identity):
            # all guards/detectors see this identity
            ...
    """

    def __init__(self, identity: Identity) -> None:
        self._identity = identity
        self._token: Any = None

    def __enter__(self) -> Identity:
        self._token = _current_identity.set(self._identity)
        return self._identity

    def __exit__(self, *args: Any) -> None:
        _current_identity.reset(self._token)

    @staticmethod
    def current() -> Identity | None:
        """Get the identity bound to the current execution scope."""
        return _current_identity.get()

    @staticmethod
    def require() -> Identity:
        """Get the current identity or raise AuthenticationError."""
        identity = _current_identity.get()
        if identity is None:
            from kavach.core.exceptions import AuthenticationError
            raise AuthenticationError(
                message="No identity bound to current context",
                decision="block",
                reasons=["missing_identity"],
            )
        return identity
