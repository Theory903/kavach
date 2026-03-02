"""Tests for the Kavach policy engine."""

import pytest

from kavach.core.policy_engine import Decision, PolicyEngine, evaluate_condition
from kavach.policies.validator import Action, Policy, RolePolicy, Rule


# --- Condition Evaluator Tests ---

class TestEvaluateCondition:
    """Tests for the safe expression evaluator."""

    def test_simple_greater_than(self) -> None:
        assert evaluate_condition("injection_score > 0.8", {"injection_score": 0.9})
        assert not evaluate_condition("injection_score > 0.8", {"injection_score": 0.5})

    def test_simple_equals(self) -> None:
        assert evaluate_condition("role == 'analyst'", {"role": "analyst"})
        assert not evaluate_condition("role == 'analyst'", {"role": "admin"})

    def test_between(self) -> None:
        assert evaluate_condition(
            "risk_score BETWEEN 0.4 AND 0.7", {"risk_score": 0.5}
        )
        assert not evaluate_condition(
            "risk_score BETWEEN 0.4 AND 0.7", {"risk_score": 0.9}
        )

    def test_and_conjunction(self) -> None:
        ctx = {"role": "analyst", "exfiltration_score": 0.7}
        assert evaluate_condition(
            "role == 'analyst' AND exfiltration_score > 0.5", ctx
        )
        ctx["role"] = "admin"
        assert not evaluate_condition(
            "role == 'analyst' AND exfiltration_score > 0.5", ctx
        )

    def test_missing_variable_returns_false(self) -> None:
        assert not evaluate_condition("missing_var > 0.5", {})

    def test_not_equals(self) -> None:
        assert evaluate_condition("role != 'admin'", {"role": "analyst"})
        assert not evaluate_condition("role != 'admin'", {"role": "admin"})


# --- Role Policy Tests ---

class TestRolePolicy:
    """Tests for role-based tool permissions."""

    def test_wildcard_allows_all(self) -> None:
        policy = RolePolicy(allowed_tools=["*"])
        assert policy.is_tool_allowed("anything")

    def test_explicit_allow(self) -> None:
        policy = RolePolicy(allowed_tools=["search", "read_file"])
        assert policy.is_tool_allowed("search")
        assert not policy.is_tool_allowed("delete_db")

    def test_blocklist_overrides_wildcard(self) -> None:
        policy = RolePolicy(allowed_tools=["*"], blocked_tools=["delete_production_db"])
        assert policy.is_tool_allowed("search")
        assert not policy.is_tool_allowed("delete_production_db")

    def test_blocklist_overrides_explicit(self) -> None:
        policy = RolePolicy(
            allowed_tools=["search", "send_email"],
            blocked_tools=["send_email"],
        )
        assert policy.is_tool_allowed("search")
        assert not policy.is_tool_allowed("send_email")


# --- Policy Engine Tests ---

class TestPolicyEngine:
    """Tests for the full policy engine."""

    @pytest.fixture()
    def engine(self) -> PolicyEngine:
        """Create an engine with a test policy."""
        policy = Policy(
            roles={
                "analyst": RolePolicy(
                    allowed_tools=["search", "summarize"],
                    blocked_tools=["send_email"],
                    max_risk_score=0.5,
                ),
                "admin": RolePolicy(
                    allowed_tools=["*"],
                    blocked_tools=["delete_production_db"],
                    max_risk_score=0.8,
                ),
            },
            rules=[
                Rule(
                    id="injection_block",
                    condition="injection_score > 0.8",
                    action=Action.BLOCK,
                    reason="Injection detected",
                    priority=100,
                ),
                Rule(
                    id="medium_risk_sanitize",
                    condition="risk_score BETWEEN 0.4 AND 0.7",
                    action=Action.SANITIZE,
                    reason="Medium risk — sanitize",
                    priority=50,
                ),
            ],
        )
        return PolicyEngine(policy)

    def test_allows_clean_request(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(
            role="analyst", risk_score=0.1, injection_score=0.0
        )
        assert decision.is_allowed

    def test_blocks_high_injection(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(
            role="analyst", risk_score=0.9, injection_score=0.9
        )
        assert decision.is_blocked
        # Both role risk check and injection rule should fire
        assert len(decision.reasons) >= 1

    def test_blocks_risk_above_max(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(role="analyst", risk_score=0.75)
        assert decision.is_blocked
        assert any("exceeds max" in r for r in decision.reasons)

    def test_sanitizes_medium_risk(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(
            role="admin", risk_score=0.6, injection_score=0.2
        )
        assert decision.action == Action.SANITIZE
        assert "medium_risk_sanitize" in decision.matched_rules

    def test_tool_allowed_for_role(self, engine: PolicyEngine) -> None:
        assert engine.is_tool_allowed("search", "analyst")
        assert not engine.is_tool_allowed("send_email", "analyst")
        assert engine.is_tool_allowed("send_email", "admin")
        assert not engine.is_tool_allowed("delete_production_db", "admin")

    def test_check_tool_permission(self, engine: PolicyEngine) -> None:
        decision = engine.check_tool_permission("search", "analyst", risk_score=0.1)
        assert decision.is_allowed

        decision = engine.check_tool_permission("send_email", "analyst")
        assert decision.is_blocked

    def test_unknown_role_gets_restrictive_default(self, engine: PolicyEngine) -> None:
        assert not engine.is_tool_allowed("search", "unknown_role")

    def test_loads_default_policy(self) -> None:
        engine = PolicyEngine()
        assert engine.policy is not None
        assert "analyst" in engine.policy.roles


class TestPolicyLoading:
    """Tests for loading policies from YAML."""

    def test_load_from_dict(self) -> None:
        data = {
            "version": "1.0",
            "roles": {
                "viewer": {
                    "allowed_tools": ["search"],
                    "max_risk_score": 0.3,
                }
            },
        }
        engine = PolicyEngine(data)
        assert engine.is_tool_allowed("search", "viewer")
        assert not engine.is_tool_allowed("delete", "viewer")

    def test_load_default(self) -> None:
        engine = PolicyEngine()
        assert engine.policy.version == "1.0"
