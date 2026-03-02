"""Tool guard — enforcement layer for every tool/function call.

The tool guard is the CORE enforcement primitive of Kavach. It wraps
every tool call with permission checks, risk evaluation, and policy
enforcement before allowing execution.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from kavach.core.exceptions import ToolBlocked
from kavach.core.identity import Identity, IdentityContext
from kavach.core.policy_engine import Decision, PolicyEngine
from kavach.policies.validator import Action, Policy

F = TypeVar("F", bound=Callable[..., Any])


class ToolGuard:
    """Guards every tool call with RBAC and risk-based enforcement.

    Usage:
        guard = ToolGuard(policy="policy.yaml")

        @guard.protect(role_required="admin", risk_threshold=0.6)
        def send_email(to, subject, body):
            ...

        # Or check manually:
        decision = guard.check("send_email", role="analyst", risk_score=0.3)
    """

    def __init__(self, policy: str | Path | dict[str, Any] | Policy | None = None) -> None:
        """Initialize with a policy source.

        Args:
            policy: Path to YAML, dict, Policy object, or None for default.
        """
        self._engine = PolicyEngine(policy)

    @property
    def engine(self) -> PolicyEngine:
        """Access the underlying policy engine."""
        return self._engine

    def check(
        self,
        tool_name: str,
        role: str,
        risk_score: float = 0.0,
    ) -> Decision:
        """Check if a tool call is permitted.

        Args:
            tool_name: Name of the tool being invoked.
            role: The role requesting the tool.
            risk_score: Current risk assessment score.

        Returns:
            Decision with allow/block/require_approval action.
        """
        return self._engine.check_tool_permission(tool_name, role, risk_score)

    def enforce(
        self,
        tool_name: str,
        role: str,
        risk_score: float = 0.0,
    ) -> Decision:
        """Check and enforce — raises ToolBlocked if denied.

        Args:
            tool_name: Name of the tool being invoked.
            role: The role requesting the tool.
            risk_score: Current risk assessment score.

        Returns:
            Decision if allowed.

        Raises:
            ToolBlocked: If the tool call is denied.
        """
        decision = self.check(tool_name, role, risk_score)

        if decision.is_blocked:
            raise ToolBlocked(
                message=f"Tool '{tool_name}' blocked for role '{role}'",
                risk_score=decision.risk_score,
                reasons=decision.reasons,
                decision="block",
                tool_name=tool_name,
                required_role=role,
            )

        return decision

    def protect(
        self,
        role_required: str | None = None,
        risk_threshold: float | None = None,
        tool_name: str | None = None,
    ) -> Callable[[F], F]:
        """Decorator that guards a function with policy enforcement.

        Args:
            role_required: Minimum role required. If None, uses identity from context.
            risk_threshold: Override the policy's risk threshold for this tool.
            tool_name: Override the function name as the tool name.

        Returns:
            Decorator that wraps the function with enforcement.

        Usage:
            @guard.protect(role_required="admin")
            def delete_records(query):
                ...

            @guard.protect(risk_threshold=0.3)
            def search_database(query):
                ...
        """

        def decorator(func: F) -> F:
            resolved_name = tool_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()

                # Resolve identity
                identity = IdentityContext.current()
                role = role_required or (identity.role if identity else "unknown")
                risk = kwargs.pop("_kavach_risk_score", 0.0)

                if risk_threshold is not None:
                    risk = max(risk, 0.0)

                # Check permission
                decision = self.check(resolved_name, role, risk)

                # Apply risk threshold override
                if risk_threshold is not None and risk > risk_threshold:
                    decision.action = Action.BLOCK
                    decision.reasons.append(
                        f"risk {risk:.2f} exceeds tool threshold {risk_threshold}"
                    )

                if decision.is_blocked:
                    raise ToolBlocked(
                        message=f"Tool '{resolved_name}' blocked for role '{role}'",
                        risk_score=risk,
                        reasons=decision.reasons,
                        decision="block",
                        tool_name=resolved_name,
                        required_role=role,
                    )

                # Execute the actual function
                return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def guard_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        identity: Identity,
        risk_score: float = 0.0,
    ) -> Decision:
        """Guard a dynamic tool call (e.g., from LLM tool_use response).

        This is used when the LLM suggests a tool call and we need
        to validate it before execution.

        Args:
            tool_name: Tool the LLM wants to call.
            tool_args: Arguments the LLM provided.
            identity: The identity context.
            risk_score: Current risk score for this conversation.

        Returns:
            Decision allowing or blocking the tool call.
        """
        decision = self.check(tool_name, identity.role, risk_score)

        # Additional check: block unknown tools if configured
        if self._engine.policy.output_guard.block_unknown_tools:
            role_policy = self._engine.get_role_policy(identity.role)
            if not role_policy.is_tool_allowed(tool_name):
                decision.action = Action.BLOCK
                decision.reasons.append(f"unknown tool '{tool_name}' blocked by policy")

        return decision
