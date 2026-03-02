"""Tests for Tool Guard enforcement."""

import pytest

from kavach.core.exceptions import ToolBlocked
from kavach.core.identity import Identity, IdentityContext
from kavach.guards.tool_guard import ToolGuard
from kavach.policies.validator import Action, Policy, RolePolicy


@pytest.fixture()
def guard() -> ToolGuard:
    """Create a ToolGuard with a test policy."""
    policy = Policy(
        roles={
            "analyst": RolePolicy(
                allowed_tools=["search", "summarize", "read_file"],
                blocked_tools=["send_email", "export_data"],
                max_risk_score=0.5,
            ),
            "admin": RolePolicy(
                allowed_tools=["*"],
                blocked_tools=["delete_production_db"],
                max_risk_score=0.8,
            ),
        }
    )
    return ToolGuard(policy)


class TestToolGuardCheck:
    """Tests for ToolGuard.check()."""

    def test_allows_permitted_tool(self, guard: ToolGuard) -> None:
        decision = guard.check("search", "analyst")
        assert decision.is_allowed

    def test_blocks_denied_tool(self, guard: ToolGuard) -> None:
        decision = guard.check("send_email", "analyst")
        assert decision.is_blocked

    def test_admin_can_use_most_tools(self, guard: ToolGuard) -> None:
        decision = guard.check("send_email", "admin")
        assert decision.is_allowed

    def test_admin_blocked_from_dangerous_tool(self, guard: ToolGuard) -> None:
        decision = guard.check("delete_production_db", "admin")
        assert decision.is_blocked

    def test_risk_score_blocks_above_threshold(self, guard: ToolGuard) -> None:
        decision = guard.check("search", "analyst", risk_score=0.7)
        assert decision.is_blocked
        assert any("exceeds max" in r for r in decision.reasons)

    def test_unknown_role_is_restrictive(self, guard: ToolGuard) -> None:
        decision = guard.check("search", "guest")
        assert decision.is_blocked


class TestToolGuardEnforce:
    """Tests for ToolGuard.enforce() — raises exceptions."""

    def test_enforce_raises_on_blocked(self, guard: ToolGuard) -> None:
        with pytest.raises(ToolBlocked) as exc_info:
            guard.enforce("send_email", "analyst")
        assert exc_info.value.tool_name == "send_email"
        assert "not allowed" in exc_info.value.reasons[0]

    def test_enforce_returns_decision_on_allowed(self, guard: ToolGuard) -> None:
        decision = guard.enforce("search", "analyst")
        assert decision.is_allowed


class TestToolGuardDecorator:
    """Tests for @guard.protect() decorator."""

    def test_decorator_allows_permitted_call(self, guard: ToolGuard) -> None:
        @guard.protect(role_required="analyst")
        def search(query: str) -> str:
            return f"results for {query}"

        result = search("test query")
        assert result == "results for test query"

    def test_decorator_blocks_denied_call(self, guard: ToolGuard) -> None:
        @guard.protect(role_required="analyst")
        def send_email(to: str, body: str) -> str:
            return "sent"

        with pytest.raises(ToolBlocked):
            send_email("test@example.com", "hello")

    def test_decorator_uses_context_identity(self, guard: ToolGuard) -> None:
        @guard.protect()
        def search(query: str) -> str:
            return f"results for {query}"

        identity = Identity(user_id="u1", role="analyst")
        with IdentityContext(identity):
            result = search("test")
            assert result == "results for test"

    def test_decorator_with_risk_threshold(self, guard: ToolGuard) -> None:
        @guard.protect(role_required="admin", risk_threshold=0.3)
        def dangerous_op() -> str:
            return "done"

        # Should pass with low risk
        result = dangerous_op()
        assert result == "done"

        # Should block with high risk
        with pytest.raises(ToolBlocked):
            dangerous_op(_kavach_risk_score=0.5)

    def test_decorator_with_custom_tool_name(self, guard: ToolGuard) -> None:
        @guard.protect(role_required="analyst", tool_name="search")
        def my_search_function(q: str) -> str:
            return q

        # "search" is allowed for analyst
        result = my_search_function("hello")
        assert result == "hello"
