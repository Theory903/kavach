"""SecureChain — LangChain middleware with Kavach guards.

Wraps any LangChain chain/runnable with input/output guards.
Two-line integration:

    chain = SecureChain(base_chain=your_chain, policy="policy.yaml")
    chain.invoke({"input": user_input, "user_id": "u1", "role": "analyst"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kavach.core.exceptions import InputBlocked
from kavach.core.gateway import KavachGateway
from kavach.core.identity import Identity, IdentityContext


class SecureChain:
    """Wraps a LangChain chain with Kavach security.

    Works with any LangChain Runnable that has an invoke() method.
    """

    def __init__(
        self,
        base_chain: Any,
        policy: str | Path | dict[str, Any] | None = None,
    ) -> None:
        """Initialize SecureChain.

        Args:
            base_chain: The LangChain chain/runnable to wrap.
            policy: Kavach policy source.
        """
        self._chain = base_chain
        self._gateway = KavachGateway(policy=policy)

    def invoke(self, input_data: dict[str, Any], **kwargs: Any) -> Any:
        """Guarded chain invocation.

        Args:
            input_data: Dict with 'input' key (the prompt) and
                       optional 'user_id' and 'role' keys.
            **kwargs: Additional kwargs passed to the chain.

        Returns:
            Chain output if allowed.

        Raises:
            InputBlocked: If input is blocked by policy.
        """
        prompt = input_data.get("input", "")
        user_id = input_data.pop("user_id", "anonymous")
        role = input_data.pop("role", "default")

        identity = Identity(user_id=user_id, role=role)

        with IdentityContext(identity):
            result = self._gateway.analyze(
                prompt=str(prompt),
                user_id=user_id,
                role=role,
                session_id=identity.session_id,
            )

            if result["decision"] == "block":
                raise InputBlocked(
                    message="Input blocked by Kavach policy",
                    risk_score=result.get("risk_score", 0.0),
                    reasons=result.get("reasons", []),
                    clean_prompt=result.get("clean_prompt"),
                )

            # Use clean prompt if sanitized
            if result.get("clean_prompt"):
                input_data = {**input_data, "input": result["clean_prompt"]}

            return self._chain.invoke(input_data, **kwargs)

    @property
    def gateway(self) -> KavachGateway:
        return self._gateway
