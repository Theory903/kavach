"""Policy engine — deterministic rule evaluation.

The policy engine is the brain of Kavach's enforcement system. It evaluates
rules against request context (role, risk scores, intent) and returns
a deterministic decision: allow, block, sanitize, or require_approval.

No randomness, no LLM calls — pure rule evaluation for predictable security.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kavach.policies.loader import load_policy, load_default_policy
from kavach.policies.validator import Action, Policy, RolePolicy, Rule


@dataclass
class Decision:
    """The result of policy evaluation.

    Every Kavach call returns a Decision so callers always know
    exactly what happened and why.
    """

    action: Action = Action.ALLOW
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    clean_prompt: str | None = None
    latency_ms: float = 0.0
    session_id: str = ""
    trace_id: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.action == Action.BLOCK

    @property
    def is_allowed(self) -> bool:
        return self.action == Action.ALLOW

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses and logging."""
        return {
            "decision": self.action.value,
            "risk_score": round(self.risk_score, 4),
            "reasons": self.reasons,
            "matched_rules": self.matched_rules,
            "clean_prompt": self.clean_prompt,
            "latency_ms": round(self.latency_ms, 2),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
        }


# ------------------------------------------------------------------
# Condition evaluator — safe mini-expression engine
# ------------------------------------------------------------------

# Supported operators for condition expressions
_OPERATORS: dict[str, Any] = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# Pattern: "variable operator value"
_SIMPLE_CONDITION = re.compile(
    r"^\s*(\w+)\s*(>=|<=|!=|==|>|<)\s*['\"]?([^'\"]+?)['\"]?\s*$"
)

# Pattern: "variable BETWEEN low AND high"
_BETWEEN_CONDITION = re.compile(
    r"^\s*(\w+)\s+BETWEEN\s+([0-9.]+)\s+AND\s+([0-9.]+)\s*$",
    re.IGNORECASE,
)


def _parse_value(raw: str) -> float | str:
    """Parse a condition value as float or string."""
    try:
        return float(raw)
    except ValueError:
        return raw.strip()


def _evaluate_simple(condition: str, context: dict[str, Any]) -> bool:
    """Evaluate a simple comparison: 'variable op value'."""
    match = _SIMPLE_CONDITION.match(condition)
    if not match:
        return False

    var_name, op_str, raw_value = match.groups()
    ctx_value = context.get(var_name)
    if ctx_value is None:
        return False

    target_value = _parse_value(raw_value)

    # Type coercion: compare same types
    if isinstance(target_value, float) and isinstance(ctx_value, (int, float)):
        return _OPERATORS[op_str](float(ctx_value), target_value)
    return _OPERATORS[op_str](str(ctx_value), str(target_value))


def _evaluate_between(condition: str, context: dict[str, Any]) -> bool:
    """Evaluate: 'variable BETWEEN low AND high'."""
    match = _BETWEEN_CONDITION.match(condition)
    if not match:
        return False

    var_name, low, high = match.groups()
    ctx_value = context.get(var_name)
    if ctx_value is None:
        return False

    return float(low) <= float(ctx_value) <= float(high)


def evaluate_condition(condition: str, context: dict[str, Any]) -> bool:
    """Evaluate a policy condition against a context dictionary.

    Supports:
        - Simple comparisons: "injection_score > 0.8"
        - BETWEEN: "risk_score BETWEEN 0.4 AND 0.7"
        - AND conjunctions: "role == 'analyst' AND exfiltration_score > 0.5"

    Args:
        condition: The condition expression string.
        context: Variable bindings (role, risk_score, injection_score, etc.)

    Returns:
        True if the condition matches.
    """
    # Try BETWEEN first — it contains " AND " literally, so must be
    # checked before splitting on AND conjunctions
    between_match = _BETWEEN_CONDITION.match(condition.strip())
    if between_match:
        return _evaluate_between(condition, context)

    # Handle AND conjunctions (split only if not a BETWEEN expression)
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(evaluate_condition(part.strip(), context) for part in parts)

    # Handle OR conjunctions
    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(evaluate_condition(part.strip(), context) for part in parts)

    # Try simple comparison
    return _evaluate_simple(condition, context)


# ------------------------------------------------------------------
# Policy Engine
# ------------------------------------------------------------------

class PolicyEngine:
    """Core policy engine that evaluates rules against request context.

    The engine is initialized with a policy (from file, dict, or default)
    and provides methods to check tool permissions, evaluate rules, and
    produce security decisions.

    Usage:
        engine = PolicyEngine("policy.yaml")
        decision = engine.evaluate(role="analyst", risk_score=0.85)
        if decision.is_blocked:
            raise InputBlocked(...)
    """

    def __init__(self, policy: str | Path | dict[str, Any] | Policy | None = None) -> None:
        """Initialize with a policy source.

        Args:
            policy: Path to YAML file, dict, Policy object, or None for default.
        """
        if policy is None:
            self._policy = load_default_policy()
        elif isinstance(policy, Policy):
            self._policy = policy
        else:
            self._policy = load_policy(policy)

    @property
    def policy(self) -> Policy:
        """Access the loaded policy."""
        return self._policy

    def get_role_policy(self, role: str) -> RolePolicy:
        """Get the policy for a specific role."""
        return self._policy.get_role_or_default(role)

    def is_tool_allowed(self, tool_name: str, role: str) -> bool:
        """Check if a tool is allowed for a given role.

        Args:
            tool_name: Name of the tool to check.
            role: The role requesting the tool.

        Returns:
            True if the tool is permitted.
        """
        role_policy = self.get_role_policy(role)
        return role_policy.is_tool_allowed(tool_name)

    def evaluate(self, **context: Any) -> Decision:
        """Evaluate all policy rules against the given context.

        Context should include keys like:
            - role: str
            - risk_score: float
            - injection_score: float
            - jailbreak_score: float
            - exfiltration_score: float
            - intent: str

        Returns:
            Decision with the most restrictive matching action.
        """
        decision = Decision(
            risk_score=context.get("risk_score", 0.0),
        )

        # Check role-based risk threshold
        role = context.get("role", "")
        role_blocked = False
        if role:
            role_policy = self.get_role_policy(role)
            risk = context.get("risk_score", 0.0)

            if isinstance(risk, (int, float)) and risk > role_policy.max_risk_score:
                decision.action = Action.BLOCK
                decision.reasons.append(
                    f"risk_score {risk:.2f} exceeds max {role_policy.max_risk_score} for role '{role}'"
                )
                role_blocked = True

        # Evaluate rules in priority order (already sorted)
        for rule in self._policy.rules:
            if evaluate_condition(rule.condition, context):
                decision.matched_rules.append(rule.id)
                if rule.reason:
                    decision.reasons.append(rule.reason)

                # Apply the most restrictive action seen so far
                if _action_severity(rule.action) > _action_severity(decision.action):
                    decision.action = rule.action

        return decision

    def check_tool_permission(self, tool_name: str, role: str, risk_score: float = 0.0) -> Decision:
        """Check if a tool call should be permitted.

        Combines role-based tool permissions with risk evaluation.

        Args:
            tool_name: The tool being invoked.
            role: The requesting role.
            risk_score: Current risk score for the request.

        Returns:
            Decision allowing or blocking the tool call.
        """
        decision = Decision(risk_score=risk_score)
        role_policy = self.get_role_policy(role)

        # Check tool allowed list
        if not role_policy.is_tool_allowed(tool_name):
            decision.action = Action.BLOCK
            decision.reasons.append(f"tool '{tool_name}' not allowed for role '{role}'")
            return decision

        # Check risk threshold
        if risk_score > role_policy.max_risk_score:
            decision.action = Action.BLOCK
            decision.reasons.append(
                f"risk_score {risk_score:.2f} exceeds max {role_policy.max_risk_score}"
            )
            return decision

        # Check approval threshold
        if risk_score > role_policy.require_approval_above:
            decision.action = Action.REQUIRE_APPROVAL
            decision.reasons.append(
                f"risk_score {risk_score:.2f} requires approval (threshold: {role_policy.require_approval_above})"
            )

        return decision


def _action_severity(action: Action) -> int:
    """Severity ranking for choosing the most restrictive action."""
    return {
        Action.ALLOW: 0,
        Action.SANITIZE: 1,
        Action.REQUIRE_APPROVAL: 2,
        Action.BLOCK: 3,
    }.get(action, 0)
