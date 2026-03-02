"""Agent Loop Security Guard.

Validates the plan→act→observe→reflect agent workflow by enforcing:
- Per-step risk scoring with cumulative session tracking
- Tool chain validation (rejects suspicious sequences)
- Action approval layer for high-risk invocations
- Session-level risk budgets

Usage:
    guard = AgentLoopGuard(policy=policy)
    
    # Validate each step in the agent loop
    guard.begin_session("session_123", role="analyst")
    
    step1 = guard.validate_step(
        session_id="session_123",
        step_type="plan",
        content="Read the database and email results",
        tool_requested="read_db",
    )
    
    step2 = guard.validate_step(
        session_id="session_123",
        step_type="act",
        content="Sending email with results",
        tool_requested="send_email",
    )
    # step2 may return {"approved": False, "reason": "Suspicious tool chain: read_db → send_email"}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Tool chain patterns that are inherently suspicious
SUSPICIOUS_CHAINS: list[tuple[str, str]] = [
    ("read_db", "send_email"),
    ("read_db", "http_post"),
    ("read_file", "send_email"),
    ("read_file", "http_post"),
    ("list_secrets", "send_email"),
    ("list_secrets", "http_post"),
    ("read_db", "write_file"),
    ("get_credentials", "http_post"),
    ("read_memory", "send_email"),
    ("query_vector_db", "http_post"),
]

# Tools classified as high-risk (require explicit approval)
HIGH_RISK_TOOLS = {
    "delete_db", "drop_table", "send_email", "http_post",
    "write_file", "execute_code", "deploy", "transfer_funds",
    "modify_permissions", "delete_user", "export_data",
}

# Maximum cumulative risk budget per session
MAX_SESSION_RISK = 3.0


@dataclass
class AgentStep:
    """A single step in an agent workflow."""
    step_type: str  # plan, act, observe, reflect
    content: str
    tool_requested: str | None
    risk_score: float
    approved: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentSession:
    """Tracks the state of an agent workflow session."""
    session_id: str
    role: str
    steps: list[AgentStep] = field(default_factory=list)
    cumulative_risk: float = 0.0
    is_terminated: bool = False

    @property
    def tool_history(self) -> list[str]:
        return [s.tool_requested for s in self.steps if s.tool_requested]

    @property
    def last_tool(self) -> str | None:
        tools = self.tool_history
        return tools[-1] if tools else None


class AgentLoopGuard:
    """Validates agent workflow steps with cumulative risk tracking."""

    def __init__(self, max_session_risk: float = MAX_SESSION_RISK) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._max_session_risk = max_session_risk

    def begin_session(self, session_id: str, role: str = "default") -> None:
        """Initialize a new agent workflow session."""
        self._sessions[session_id] = AgentSession(session_id=session_id, role=role)
        logger.info("Agent session started: %s (role=%s)", session_id, role)

    def validate_step(
        self,
        session_id: str,
        step_type: str,
        content: str,
        tool_requested: str | None = None,
        risk_score: float = 0.0,
    ) -> dict[str, Any]:
        """Validate a single agent step.
        
        Returns:
            Dict with 'approved', 'reason', 'cumulative_risk', 'requires_approval'.
        """
        session = self._sessions.get(session_id)
        if session is None:
            # Auto-create session
            self.begin_session(session_id)
            session = self._sessions[session_id]

        if session.is_terminated:
            return {
                "approved": False,
                "reason": "Session terminated due to risk budget exceeded",
                "cumulative_risk": session.cumulative_risk,
                "requires_approval": False,
            }

        approved = True
        reason = "approved"
        requires_approval = False

        # Check 1: Tool chain validation
        if tool_requested and session.last_tool:
            chain = (session.last_tool, tool_requested)
            if chain in SUSPICIOUS_CHAINS:
                approved = False
                reason = f"Suspicious tool chain detected: {chain[0]} → {chain[1]}"
                logger.warning("Agent guard blocked: %s", reason)

        # Check 2: High-risk tool approval
        if tool_requested and tool_requested in HIGH_RISK_TOOLS:
            requires_approval = True
            if risk_score > 0.5:
                approved = False
                reason = f"High-risk tool '{tool_requested}' blocked (risk={risk_score:.2f})"

        # Check 3: Cumulative risk budget
        new_cumulative = session.cumulative_risk + risk_score
        if new_cumulative > self._max_session_risk:
            approved = False
            reason = f"Session risk budget exceeded ({new_cumulative:.2f} > {self._max_session_risk})"
            session.is_terminated = True

        # Record step
        step = AgentStep(
            step_type=step_type,
            content=content[:200],
            tool_requested=tool_requested,
            risk_score=risk_score,
            approved=approved,
            reason=reason,
        )
        session.steps.append(step)
        session.cumulative_risk = new_cumulative

        return {
            "approved": approved,
            "reason": reason,
            "cumulative_risk": round(session.cumulative_risk, 4),
            "requires_approval": requires_approval,
            "step_number": len(session.steps),
            "session_terminated": session.is_terminated,
        }

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of the session state."""
        session = self._sessions.get(session_id)
        if session is None:
            return {"error": "Session not found"}

        return {
            "session_id": session.session_id,
            "role": session.role,
            "total_steps": len(session.steps),
            "cumulative_risk": round(session.cumulative_risk, 4),
            "is_terminated": session.is_terminated,
            "tool_history": session.tool_history,
            "blocked_steps": sum(1 for s in session.steps if not s.approved),
        }

    def end_session(self, session_id: str) -> None:
        """Clean up a completed session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
