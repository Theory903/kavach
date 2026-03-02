"""Behavioral risk tracking.

Tracks a user's recent behavior to adjust base risk levels.
Provides adaptive scoring:
- Users with a history of safe prompts get benefit of doubt.
- Users who repeatedly trigger high risk scores get penalized.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserState:
    """State for a single user."""
    request_count: int = 0
    total_risk_accumulated: float = 0.0
    violation_count: int = 0
    last_seen_ts: float = 0.0
    recent_risk_scores: list[float] = field(default_factory=list)


class BehavioralTracker:
    """Tracks user history and computes behavioral risk multipliers.
    
    In a real distributed system, this would be backed by Redis.
    For the SDK, it's an in-memory sliding window cache.
    """

    def __init__(self, history_size: int = 10) -> None:
        self._history_size = history_size
        self._users: dict[str, UserState] = defaultdict(UserState)

    def record_interaction(self, user_id: str, risk_score: float, action_taken: str) -> None:
        """Update state after an interaction.
        
        Args:
            user_id: The ID of the user.
            risk_score: Final risk score applied.
            action_taken: "allow", "block", "sanitize", etc.
        """
        if user_id == "anonymous":
            return

        state = self._users[user_id]
        state.request_count += 1
        state.total_risk_accumulated += risk_score
        state.last_seen_ts = time.time()
        
        if action_taken == "block":
            state.violation_count += 1
            
        state.recent_risk_scores.append(risk_score)
        if len(state.recent_risk_scores) > self._history_size:
            state.recent_risk_scores.pop(0)

    def get_behavioral_multiplier(self, user_id: str) -> float:
        """Calculate a risk multiplier based on user history.
        
        < 1.0 means user is trusted (lowers total risk).
        > 1.0 means user is suspicious (amplifies total risk).
        
        Args:
            user_id: The ID of the user.
            
        Returns:
            Float multiplier typically mapped between 0.8 and 1.5.
        """
        if user_id == "anonymous" or user_id not in self._users:
            return 1.0

        state = self._users[user_id]
        
        # Hard penalty for previous blocks (immediate effect)
        if state.violation_count > 0:
            return min(1.5, 1.0 + (0.1 * state.violation_count))
            
        # New users: no benefit, no penalty
        if state.request_count < 3:
            return 1.0
            
        # Average recent risk
        if state.recent_risk_scores:
            avg_recent_risk = sum(state.recent_risk_scores) / len(state.recent_risk_scores)
            
            # If they keep generating borderline risk (e.g. 0.4)
            if avg_recent_risk > 0.3:
                return 1.2
            
            # Very clean history
            if avg_recent_risk < 0.1 and state.request_count >= 5:
                return 0.85
                
        return 1.0
