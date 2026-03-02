"""SecureClaude — drop-in replacement for the Anthropic client.

Wraps anthropic.Anthropic with Kavach guards, providing
the same messages.create() interface with zero learning curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kavach.core.exceptions import InputBlocked
from kavach.core.gateway import KavachGateway
from kavach.core.identity import Identity, IdentityContext


class _SecureMessages:
    """Wraps anthropic.messages with Kavach guards."""

    def __init__(
        self,
        anthropic_client: Any,
        gateway: KavachGateway,
        identity: Identity,
    ) -> None:
        self._client = anthropic_client
        self._gateway = gateway
        self._identity = identity

    def create(self, **kwargs: Any) -> Any:
        """Guarded replacement for anthropic.messages.create()."""
        messages = kwargs.get("messages", [])

        user_messages = [m for m in messages if m.get("role") == "user"]
        last_user_msg = ""
        if user_messages:
            content = user_messages[-1].get("content", "")
            if isinstance(content, str):
                last_user_msg = content
            elif isinstance(content, list):
                # Handle content blocks
                last_user_msg = " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )

        with IdentityContext(self._identity):
            result = self._gateway.analyze(
                prompt=last_user_msg,
                user_id=self._identity.user_id,
                role=self._identity.role,
                session_id=self._identity.session_id,
            )

            if result["decision"] == "block":
                raise InputBlocked(
                    message="Input blocked by Kavach policy",
                    risk_score=result.get("risk_score", 0.0),
                    reasons=result.get("reasons", []),
                    clean_prompt=result.get("clean_prompt"),
                )

            if result.get("clean_prompt"):
                messages = list(messages)
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": result["clean_prompt"]}
                        break
                kwargs["messages"] = messages

            response = self._client.messages.create(**kwargs)
            return response


class SecureClaude:
    """Drop-in replacement for anthropic.Anthropic with Kavach security."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        policy: str | Path | dict[str, Any] | None = None,
        user_id: str = "anonymous",
        role: str = "default",
        **anthropic_kwargs: Any,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install kavach[anthropic]"
            )

        if anthropic_api_key:
            anthropic_kwargs["api_key"] = anthropic_api_key

        self._anthropic = Anthropic(**anthropic_kwargs)
        self._gateway = KavachGateway(policy=policy)
        self._identity = Identity(user_id=user_id, role=role)

        self.messages = _SecureMessages(self._anthropic, self._gateway, self._identity)

    @property
    def gateway(self) -> KavachGateway:
        return self._gateway
