"""Reinforcement Learning Decision Advisor.

Tabular Q-learning agent that suggests optimal actions (allow, sanitize,
block, require_approval) based on discretized security state. The agent
learns from production feedback to optimize decision quality over time.

Architecture:
    - Q-table with 216 states × 4 actions = 864 cells
    - State: (risk_bucket, intent_category, role_tier, behavioral_tier)
    - Trained via temporal-difference updates from feedback labels
    - Persisted to disk as a numpy array for instant cold-start

Critical invariant:
    The RL advisor is NEVER the final authority. The PolicyEngine has
    absolute override power. RL can only suggest tightening a decision,
    never loosening one beyond what the policy allows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Action space
ACTION_ALLOW = 0
ACTION_SANITIZE = 1
ACTION_BLOCK = 2
ACTION_REQUIRE_APPROVAL = 3

ACTION_NAMES = {
    ACTION_ALLOW: "allow",
    ACTION_SANITIZE: "sanitize",
    ACTION_BLOCK: "block",
    ACTION_REQUIRE_APPROVAL: "require_approval",
}

# State dimensions
N_RISK_BUCKETS = 4       # low (0-0.25), medium (0.25-0.5), high (0.5-0.8), critical (0.8+)
N_INTENT_CATEGORIES = 6  # benign, injection, jailbreak, exfiltration, obfuscation, social_eng
N_ROLE_TIERS = 3          # guest=0, standard=1, admin=2
N_BEHAVIORAL_TIERS = 3    # suspicious=0, neutral=1, trusted=2

N_STATES = N_RISK_BUCKETS * N_INTENT_CATEGORIES * N_ROLE_TIERS * N_BEHAVIORAL_TIERS  # 216
N_ACTIONS = 4

# Intent category mapping
INTENT_TO_INDEX = {
    "benign": 0,
    "injection": 1,
    "jailbreak": 2,
    "exfiltration": 3,
    "obfuscation": 4,
    "social_engineering": 5,
    "unknown": 0,
}

# Role tier mapping
ROLE_TO_TIER = {
    "guest": 0,
    "viewer": 0,
    "default": 1,
    "analyst": 1,
    "standard": 1,
    "admin": 2,
    "superadmin": 2,
}

# Reward constants
REWARD_CORRECT = 1.0
REWARD_FALSE_POSITIVE = -1.0
REWARD_FALSE_NEGATIVE = -3.0
REWARD_SOFT_POSITIVE = 0.5


def _discretize_risk(risk_score: float) -> int:
    """Quantize a continuous risk score into 4 buckets."""
    if risk_score < 0.25:
        return 0  # low
    elif risk_score < 0.50:
        return 1  # medium
    elif risk_score < 0.80:
        return 2  # high
    else:
        return 3  # critical


def _discretize_behavioral(multiplier: float) -> int:
    """Quantize behavioral multiplier into 3 tiers."""
    if multiplier < 0.9:
        return 2  # trusted (multiplier reduces risk)
    elif multiplier <= 1.1:
        return 1  # neutral
    else:
        return 0  # suspicious (multiplier amplifies risk)


def _state_index(
    risk_bucket: int,
    intent_idx: int,
    role_tier: int,
    behavioral_tier: int,
) -> int:
    """Flatten multi-dimensional state into a single index."""
    return (
        risk_bucket * (N_INTENT_CATEGORIES * N_ROLE_TIERS * N_BEHAVIORAL_TIERS)
        + intent_idx * (N_ROLE_TIERS * N_BEHAVIORAL_TIERS)
        + role_tier * N_BEHAVIORAL_TIERS
        + behavioral_tier
    )


class RLDecisionAdvisor:
    """Tabular Q-learning decision advisor.
    
    Learns optimal (allow/sanitize/block/escalate) decisions from
    production feedback while respecting policy engine authority.
    
    Usage:
        advisor = RLDecisionAdvisor()
        suggestion = advisor.suggest(risk_score=0.7, intent="jailbreak",
                                     role="guest", behavioral_multiplier=1.2)
        # suggestion = {"action": "block", "confidence": 0.85, "q_values": {...}}
        
        # After feedback arrives:
        advisor.update(state_index, action_taken, reward)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        persist_path: str | None = None,
    ) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self._persist_path = Path(persist_path) if persist_path else None

        # Initialize Q-table with slight bias toward blocking (safety-first)
        self.q_table = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        # Bias: start with a slight preference for block on high-risk states
        for s in range(N_STATES):
            risk_bucket = s // (N_INTENT_CATEGORIES * N_ROLE_TIERS * N_BEHAVIORAL_TIERS)
            if risk_bucket >= 2:  # high or critical
                self.q_table[s, ACTION_BLOCK] = 0.5
            elif risk_bucket == 1:  # medium
                self.q_table[s, ACTION_SANITIZE] = 0.3

        self.total_updates = 0
        self.is_loaded = True

        # Load persisted Q-table if available
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _encode_state(
        self,
        risk_score: float,
        intent_category: str,
        role: str,
        behavioral_multiplier: float,
    ) -> int:
        """Encode raw features into a flat state index."""
        risk_bucket = _discretize_risk(risk_score)
        intent_idx = INTENT_TO_INDEX.get(intent_category, 0)
        role_tier = ROLE_TO_TIER.get(role, 1)
        behavioral_tier = _discretize_behavioral(behavioral_multiplier)
        return _state_index(risk_bucket, intent_idx, role_tier, behavioral_tier)

    def suggest(
        self,
        risk_score: float,
        intent_category: str = "unknown",
        role: str = "default",
        behavioral_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Suggest an action based on current Q-values.
        
        Returns:
            Dict with 'action', 'confidence', 'state_index', and 'q_values'.
        """
        state = self._encode_state(risk_score, intent_category, role, behavioral_multiplier)
        q_values = self.q_table[state]

        # Greedy action selection (no exploration during inference)
        best_action = int(np.argmax(q_values))
        max_q = float(q_values[best_action])

        # Confidence = softmax probability of the best action
        exp_q = np.exp(q_values - np.max(q_values))  # numerical stability
        softmax = exp_q / (exp_q.sum() + 1e-10)
        confidence = float(softmax[best_action])

        return {
            "action": ACTION_NAMES[best_action],
            "action_index": best_action,
            "confidence": round(confidence, 4),
            "state_index": state,
            "q_values": {ACTION_NAMES[i]: round(float(q_values[i]), 4) for i in range(N_ACTIONS)},
        }

    def update(
        self,
        state_index: int,
        action_index: int,
        reward: float,
        next_state_index: int | None = None,
    ) -> None:
        """Update Q-table via temporal-difference learning.
        
        For single-step episodes (most gateway requests), next_state is terminal.
        """
        if next_state_index is not None:
            # Standard Q-learning update
            best_next_q = float(np.max(self.q_table[next_state_index]))
            td_target = reward + self.gamma * best_next_q
        else:
            # Terminal state (single request episode)
            td_target = reward

        old_q = self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] = old_q + self.lr * (td_target - old_q)
        self.total_updates += 1

        # Auto-persist every 100 updates
        if self._persist_path and self.total_updates % 100 == 0:
            self.save()

    def compute_reward(
        self,
        action_taken: str,
        actual_is_attack: bool,
        was_blocked: bool,
    ) -> float:
        """Compute reward from a feedback label.
        
        Args:
            action_taken: The action that was executed (allow/sanitize/block/require_approval)
            actual_is_attack: Ground truth — was this actually malicious?
            was_blocked: Did the system block this request?
        """
        if actual_is_attack and was_blocked:
            return REWARD_CORRECT  # True positive
        elif actual_is_attack and not was_blocked:
            return REWARD_FALSE_NEGATIVE  # Missed attack
        elif not actual_is_attack and was_blocked:
            return REWARD_FALSE_POSITIVE  # False positive
        elif not actual_is_attack and action_taken == "sanitize":
            return REWARD_SOFT_POSITIVE  # Cautious but not blocking
        else:
            return REWARD_CORRECT  # True negative

    def apply_policy_override(
        self,
        rl_suggestion: str,
        policy_decision: str,
        risk_score: float,
    ) -> str:
        """Apply the safety invariant: RL cannot loosen a policy decision.
        
        Rules:
        1. If policy says block, result is block (RL cannot override).
        2. If policy says allow but RL says block, use block (RL can tighten).
        3. If risk_score > 0.8, always block regardless of RL.
        """
        # Define action severity (higher = more restrictive)
        severity = {"allow": 0, "sanitize": 1, "require_approval": 2, "block": 3}

        policy_severity = severity.get(policy_decision, 0)
        rl_severity = severity.get(rl_suggestion, 0)

        # Hard override: critical risk always blocks
        if risk_score > 0.8:
            return "block"

        # RL can only tighten, never loosen
        if rl_severity >= policy_severity:
            return rl_suggestion
        else:
            return policy_decision

    def save(self) -> None:
        """Persist Q-table to disk."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self._persist_path), self.q_table)
        meta = {"total_updates": self.total_updates, "n_states": N_STATES, "n_actions": N_ACTIONS}
        meta_path = self._persist_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("RL Q-table saved (%d updates)", self.total_updates)

    def _load(self) -> None:
        """Load Q-table from disk."""
        try:
            self.q_table = np.load(str(self._persist_path))
            meta_path = self._persist_path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    self.total_updates = meta.get("total_updates", 0)
            logger.info("RL Q-table loaded (%d prior updates)", self.total_updates)
        except Exception as e:
            logger.warning("Failed to load RL Q-table: %s. Starting fresh.", e)

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic stats about the Q-table state."""
        non_zero = int(np.count_nonzero(self.q_table))
        return {
            "total_updates": self.total_updates,
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "q_table_cells": N_STATES * N_ACTIONS,
            "non_zero_cells": non_zero,
            "coverage_pct": round(non_zero / (N_STATES * N_ACTIONS) * 100, 1),
        }
