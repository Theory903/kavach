"""SessionBehaviorEngine — multi-turn anomaly detection.

Detects multi-step attacks by analyzing the trajectory of risk scores,
semantic drift between prompts, and escalating tool usage patterns
across a conversation session.

Behavioral signals emitted:
- ``risk_trajectory``: Moving average trend (rising = escalation)
- ``prompt_drift``: Cosine distance between consecutive turn embeddings
- ``tool_escalation``: Repeated tool calls with increasing risk
- ``behavior_multiplier``: Combined amplifier for the risk scorer (0.8–2.0)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Window size for trajectory analysis
_WINDOW = 10
# Drift threshold — cosine distance above this = suspicious topic change
_DRIFT_THRESHOLD = 0.35
# Risk escalation threshold — rising slope above this triggers alert
_SLOPE_THRESHOLD = 0.12


@dataclass
class TurnRecord:
    """Single conversation turn snapshot."""

    turn_index: int
    risk_score: float
    embedding: np.ndarray | None = None
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class BehaviorSignal:
    """Behavioral analysis output for a single turn.

    The ``behavior_multiplier`` should be applied to the current turn's
    raw risk score before making the enforcement decision.
    """

    behavior_multiplier: float = 1.0
    risk_trajectory: str = "stable"   # stable | rising | falling
    prompt_drift: float = 0.0         # 0.0 = same topic, 1.0 = completely different
    drift_flag: bool = False
    escalation_flag: bool = False
    tool_abuse_flag: bool = False
    reasons: list[str] = field(default_factory=list)
    session_risk: float = 0.0         # Rolling session-level risk average


class SessionBehaviorEngine:
    """Track and score behavioral patterns within a session.

    Usage::

        engine = SessionBehaviorEngine(session_id="sess_abc123")
        signal = engine.update(
            risk_score=0.45,
            embedding=embedding_vector,     # optional
            tool_calls=["search", "email"], # optional
        )
        # Apply multiplier
        adjusted_risk = base_risk * signal.behavior_multiplier
    """

    def __init__(self, session_id: str, window: int = _WINDOW) -> None:
        self._session_id = session_id
        self._window = window
        self._turns: deque[TurnRecord] = deque(maxlen=window)
        self._turn_count = 0
        # Track tool call frequency
        self._tool_call_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        risk_score: float,
        embedding: np.ndarray | None = None,
        tool_calls: list[str] | None = None,
    ) -> BehaviorSignal:
        """Process a new conversation turn and emit a behavioral signal.

        Args:
            risk_score: Kavach risk score for this turn (0.0–1.0).
            embedding: Optional embedding of this turn's prompt.
            tool_calls: Optional list of tool names called this turn.

        Returns:
            BehaviorSignal with multiplier and diagnostic flags.
        """
        tool_calls = tool_calls or []
        record = TurnRecord(
            turn_index=self._turn_count,
            risk_score=risk_score,
            embedding=embedding,
            tool_calls=tool_calls,
        )
        self._turns.append(record)
        self._turn_count += 1

        # Update tool frequency
        for tool in tool_calls:
            self._tool_call_counts[tool] = self._tool_call_counts.get(tool, 0) + 1

        signal = BehaviorSignal()
        reasons: list[str] = []

        # Need at least 2 turns for trajectory analysis
        if len(self._turns) < 2:
            signal.session_risk = risk_score
            return signal

        scores = [t.risk_score for t in self._turns]
        signal.session_risk = float(np.mean(scores))

        # 1. Risk trajectory (linear slope over window)
        trajectory, slope = self._compute_trajectory(scores)
        signal.risk_trajectory = trajectory
        if trajectory == "rising":
            signal.escalation_flag = True
            reasons.append(f"risk_escalation: slope={slope:.3f}")

        # 2. Prompt drift (semantic topic change)
        drift = self._compute_drift(embedding)
        signal.prompt_drift = drift
        if drift > _DRIFT_THRESHOLD:
            signal.drift_flag = True
            reasons.append(f"prompt_drift={drift:.3f} > {_DRIFT_THRESHOLD}")

        # 3. Tool abuse detection
        tool_abuse = self._check_tool_abuse(tool_calls, risk_score)
        if tool_abuse:
            signal.tool_abuse_flag = True
            reasons.append(f"tool_abuse: {tool_abuse}")

        # 4. Compute multiplier (compounding penalties)
        multiplier = 1.0
        if signal.escalation_flag:
            # Rising risk over multiple turns: amplify up to 1.5×
            multiplier *= min(1.5, 1.0 + slope * 3)
        if signal.drift_flag:
            # Sudden topic change with high risk: suspicious
            multiplier *= 1.2
        if signal.tool_abuse_flag:
            multiplier *= 1.3

        # Clamp: [0.8, 2.5] — allows stronger session-based escalation
        signal.behavior_multiplier = round(max(0.8, min(2.5, multiplier)), 3)
        signal.reasons = reasons

        if reasons:
            logger.info(
                f"[BehaviorEngine] session={self._session_id} "
                f"multiplier={signal.behavior_multiplier} reasons={reasons}"
            )

        return signal

    def reset(self) -> None:
        """Reset session state (e.g. on new conversation)."""
        self._turns.clear()
        self._turn_count = 0
        self._tool_call_counts.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize current session state for logging."""
        return {
            "session_id": self._session_id,
            "turn_count": self._turn_count,
            "window_size": len(self._turns),
            "tool_call_counts": self._tool_call_counts,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_trajectory(self, scores: list[float]) -> tuple[str, float]:
        """Compute the slope of the risk score over recent turns.

        Returns:
            (trajectory label, slope value)
        """
        n = len(scores)
        if n < 2:
            return "stable", 0.0

        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        slope = numerator / denominator if denominator != 0 else 0.0

        if slope > _SLOPE_THRESHOLD:
            return "rising", slope
        elif slope < -_SLOPE_THRESHOLD:
            return "falling", slope
        return "stable", slope

    def _compute_drift(self, current_embedding: np.ndarray | None) -> float:
        """Compute cosine distance between last two embeddings.

        Returns:
            Drift value 0.0 (same) to 1.0 (completely different).
        """
        if current_embedding is None or len(self._turns) < 2:
            return 0.0

        # Find previous turn with an embedding
        previous_embedding = None
        for turn in reversed(list(self._turns)[:-1]):
            if turn.embedding is not None:
                previous_embedding = turn.embedding
                break

        if previous_embedding is None:
            return 0.0

        a = current_embedding.flatten()
        b = previous_embedding.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, 1.0 - cos_sim))

    def _check_tool_abuse(self, tool_calls: list[str], risk_score: float) -> str:
        """Detect repeated high-risk tool invocations.

        Returns:
            Description string if abuse detected, else empty string.
        """
        if not tool_calls:
            return ""

        HIGH_RISK_TOOLS = {"email", "database", "code_exec", "filesystem", "http", "shell"}
        abused = []

        for tool in tool_calls:
            count = self._tool_call_counts.get(tool, 0)
            is_high_risk = tool.lower() in HIGH_RISK_TOOLS

            # Flag if: high-risk tool called repeatedly or called during high-risk session
            if count > 3 or (is_high_risk and risk_score > 0.5):
                abused.append(f"{tool}(×{count})")

        return ", ".join(abused)
