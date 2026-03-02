"""SecureOpenAI — drop-in replacement for the OpenAI client.

Wraps openai.OpenAI with Kavach input/output guards, providing
the exact same interface with zero learning curve.

Usage:
    from kavach.integrations.openai import SecureOpenAI

    client = SecureOpenAI(
        openai_api_key="sk-...",
        policy="policy.yaml",
        user_id="u1",
        role="analyst"
    )

    # Exact same interface as openai.OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kavach.core.exceptions import InputBlocked
from kavach.core.gateway import KavachGateway
from kavach.core.identity import Identity, IdentityContext


class _SecureChatCompletions:
    """Wraps openai.chat.completions with Kavach guards."""

    def __init__(
        self,
        openai_client: Any,
        gateway: KavachGateway,
        identity: Identity,
    ) -> None:
        self._client = openai_client
        self._gateway = gateway
        self._identity = identity

    def create(self, **kwargs: Any) -> Any:
        """Guarded replacement for openai.chat.completions.create().

        Runs input guard on the last user message, calls OpenAI
        if allowed, then runs output guard on the response.
        """
        messages = kwargs.get("messages", [])

        # Extract the last user message for scanning
        user_messages = [m for m in messages if m.get("role") == "user"]
        last_user_msg = user_messages[-1]["content"] if user_messages else ""

        with IdentityContext(self._identity):
            # Analyze input
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

            # If sanitized, use clean prompt
            if result.get("clean_prompt"):
                messages = list(messages)
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": result["clean_prompt"]}
                        break
                kwargs["messages"] = messages

            # Call OpenAI
            response = self._client.chat.completions.create(**kwargs)

            # Output guard on response
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
                if content:
                    output_result = self._gateway.output_guard.scan(
                        content, identity=self._identity
                    )
                    # Note: we don't modify the response object in v1,
                    # but we log any secrets found
                    if output_result.secrets_found:
                        self._gateway._logger.log_decision(
                            {"decision": "sanitize", "reasons": output_result.secrets_found},
                            identity=self._identity.to_dict(),
                        )

            return response


class _SecureChat:
    """Wraps openai.chat namespace."""

    def __init__(self, completions: _SecureChatCompletions) -> None:
        self.completions = completions


class SecureOpenAI:
    """Drop-in replacement for openai.OpenAI with Kavach security.

    Provides the same interface as the OpenAI client but wraps
    every call with input/output guards and policy enforcement.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        policy: str | Path | dict[str, Any] | None = None,
        user_id: str = "anonymous",
        role: str = "default",
        **openai_kwargs: Any,
    ) -> None:
        """Initialize SecureOpenAI.

        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            policy: Kavach policy source.
            user_id: User identifier for RBAC.
            role: User's role.
            **openai_kwargs: Additional kwargs passed to openai.OpenAI().
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install kavach[openai]"
            )

        if openai_api_key:
            openai_kwargs["api_key"] = openai_api_key

        self._openai = OpenAI(**openai_kwargs)
        self._gateway = KavachGateway(policy=policy)
        self._identity = Identity(user_id=user_id, role=role)

        self.chat = _SecureChat(
            _SecureChatCompletions(self._openai, self._gateway, self._identity)
        )

    @property
    def gateway(self) -> KavachGateway:
        """Access the underlying Kavach gateway."""
        return self._gateway
