"""Reinforcement Learning Decision Advisor (PPO).

A continuous-state Proximal Policy Optimization (PPO) agent that suggests optimal actions
(allow, sanitize, block, require_approval) based on the full security context.

Architecture:
    - PPO algorithm (stable_baselines3) replacing legacy tabular Q-learning
    - Continuous State Space:
        - risk_score: [0.0, 1.0]
        - intent_category: One-hot encoded (6 dims)
        - role_tier: Normalized [0.0, 1.0]
        - behavioral_multiplier: Normalized [0.5, 2.0] -> [0.0, 1.0]
    - Discrete Action Space: [0, 1, 2, 3] (allow, sanitize, block, req_approval)
    - Persisted to disk as an SB3 .zip archive.

Critical invariant:
    The RL advisor is NEVER the final authority. The PolicyEngine has
    absolute override power. RL can only suggest tightening a decision,
    never loosening one beyond what the policy allows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

logger = logging.getLogger(__name__)

# Action space mappings
ACTION_ALLOW = 0
ACTION_MONITOR = 1
ACTION_SANITIZE = 2
ACTION_REQUIRE_APPROVAL = 3
ACTION_BLOCK = 4

ACTION_NAMES = {
    ACTION_ALLOW: "allow",
    ACTION_MONITOR: "monitor",
    ACTION_SANITIZE: "sanitize",
    ACTION_REQUIRE_APPROVAL: "require_approval",
    ACTION_BLOCK: "block",
}

# Intent mapping for one-hot encoding
INTENT_TO_INDEX = {
    "benign": 0,
    "injection": 1,
    "jailbreak": 2,
    "exfiltration": 3,
    "obfuscation": 4,
    "social_engineering": 5,
    "unknown": 0,
}

ROLE_TO_TIER = {
    "guest": 0.0,
    "viewer": 0.0,
    "default": 0.5,
    "analyst": 0.5,
    "standard": 0.5,
    "admin": 1.0,
    "superadmin": 1.0,
}

REWARD_CORRECT = 1.0
REWARD_FALSE_POSITIVE = -1.0
REWARD_FALSE_NEGATIVE = -3.0
REWARD_SOFT_POSITIVE = 0.5


class RLDecisionAdvisor:
    """PPO-based decision advisor using Stable Baselines 3.
    
    Learns optimal (allow/sanitize/block/escalate) decisions from Continuous
    state vectors instead of discrete tables.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self.is_loaded = False
        self.model: PPO | None = None
        self.total_updates = 0

        if not _HAS_SB3:
            logger.warning("stable_baselines3 is not installed. RL features disabled.")
            return

        self._init_model()

    def _init_model(self) -> None:
        """Initialize or load the PPO model."""
        model_path = self._persist_path.with_suffix(".zip") if self._persist_path else None

        if model_path and model_path.exists():
            try:
                self.model = PPO.load(str(model_path))
                logger.info("Loaded PPO model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load PPO model (%s). Starting fresh.", e)
                self._create_new_model()
        else:
            self._create_new_model()
            
        self.is_loaded = True

    def _create_new_model(self) -> None:
        """Create a fresh PPO model with multi-layer perceptron policy."""
        # We don't train synchronously with the environment here. 
        # For online updates, we use an episodic memory buffer and train via `model.learn`.
        # Because we're using PPO essentially as an inference head we map observation spaces carefully.
        # But wait, stable-baselines requires a gym env to initialize. Let's build a dummy env.
        from gymnasium import spaces
        import gymnasium as gym

        class MockSecurityEnv(gym.Env):
            """Dummy environment solely for defining spaces to initialize SB3."""
            observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
            action_space = spaces.Discrete(5)
            def reset(self, seed=None, options=None):
                return np.zeros(9, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(9, dtype=np.float32), 0.0, True, False, {}

        self.model = PPO(
            "MlpPolicy",
            MockSecurityEnv(),
            learning_rate=3e-4,
            n_steps=64,
            batch_size=16,
            ent_coef=0.01,
            verbose=0,
        )
        logger.info("Created new PPO architecture model.")

    def _encode_state(
        self,
        risk_score: float,
        intent_category: str,
        role: str,
        behavioral_multiplier: float,
    ) -> np.ndarray:
        """Encode raw features into a 9-dimensional float32 vector.
        
        [0]: risk_score (0 to 1.0)
        [1-6]: intent category (one hot)
        [7]: role_tier (0.0 to 1.0)
        [8]: behavioral_multiplier normalized (map 0.5-2.0 to 0.0-1.0)
        """
        state = np.zeros(9, dtype=np.float32)
        state[0] = np.clip(risk_score, 0.0, 1.0)
        
        intent_idx = INTENT_TO_INDEX.get(intent_category, 0)
        state[1 + intent_idx] = 1.0
        
        state[7] = ROLE_TO_TIER.get(role, 0.5)
        
        # normalize behavioral (0.5 to 2.0) -> (0.0 to 1.0)
        b_norm = (np.clip(behavioral_multiplier, 0.5, 2.0) - 0.5) / 1.5
        state[8] = b_norm
        
        return state

    def suggest(
        self,
        risk_score: float,
        intent_category: str = "unknown",
        role: str = "default",
        behavioral_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Suggest an action using the PPO policy network."""
        if not self.is_loaded or self.model is None:
            # Fallback heuristic aligned with calibrated zones
            if risk_score > 0.85:
                action_idx = ACTION_BLOCK
            elif risk_score > 0.6:
                action_idx = ACTION_SANITIZE
            else:
                action_idx = ACTION_ALLOW
            return {"action": ACTION_NAMES[action_idx], "action_index": action_idx, "confidence": 1.0}

        obs = self._encode_state(risk_score, intent_category, role, behavioral_multiplier)
        
        # PPO predict returns (action, state)
        action, _ = self.model.predict(obs, deterministic=True)
        action_idx = int(action)

        return {
            "action": ACTION_NAMES[action_idx],
            "action_index": action_idx,
            "confidence": 1.0,  # Deterministic predict
            "state_vector": obs.tolist(),
        }

    def train_on_batch(self, states: list[np.ndarray], actions: list[int], rewards: list[float]) -> None:
        """Fine-tune the PPO model on a collected batch of feedback.
        
        Since SB3 isn't natively built for offline batch step-by-step updates outside an Env,
        we fake an environment wrapper that replays the exact logged state/reward sequence.
        """
        if not self.model or len(states) == 0:
            return

        import gymnasium as gym
        from gymnasium import spaces

        class ReplayEnv(gym.Env):
            """Plays back logged transitions to run `.learn()`."""
            observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
            action_space = spaces.Discrete(5)

            def __init__(self):
                self.states = states
                self.rewards = rewards
                self.idx = 0
                self.n = len(states)

            def reset(self, seed=None, options=None):
                self.idx = 0
                return self.states[0], {}

            def step(self, action):
                r = self.rewards[self.idx]
                self.idx += 1
                done = self.idx >= self.n
                obs = self.states[self.idx] if not done else self.states[-1]
                return obs, r, done, False, {}

        # Temporarily swap env to let PPO run a micro-update
        self.model.set_env(ReplayEnv())
        self.model.learn(total_timesteps=len(states))
        
        self.total_updates += len(states)
        self.save()

    def compute_reward(
        self,
        action_taken: str,
        actual_is_attack: bool,
        was_blocked: bool,
    ) -> float:
        """Compute reward exactly as before."""
        if actual_is_attack and was_blocked:
            return REWARD_CORRECT
        elif actual_is_attack and not was_blocked:
            return REWARD_FALSE_NEGATIVE
        elif not actual_is_attack and was_blocked:
            return REWARD_FALSE_POSITIVE
        elif not actual_is_attack and action_taken == "sanitize":
            return REWARD_SOFT_POSITIVE
        else:
            return REWARD_CORRECT

    def apply_policy_override(
        self,
        rl_suggestion: str,
        policy_decision: str,
        risk_score: float,
    ) -> str:
        """Apply the safety invariant: RL cannot loosen a policy decision.
        RL can escalate at most ONE zone above the policy decision to prevent
        untrained models from randomly hard-blocking benign text.
        """
        severity = {"allow": 0, "monitor": 1, "sanitize": 2, "require_approval": 3, "block": 4}
        policy_sev = severity.get(policy_decision, 0)
        rl_sev = severity.get(rl_suggestion, 0)

        # Let rules act first. Action comes from PolicyEngine.
        if risk_score > 0.85:
            return "block"
        
        # Absolute Allow Protection: If risk is under our calibrated minimum, RL cannot override.
        if risk_score < 0.2 and policy_decision == "allow":
            return "allow"

        # RL can tighten but at MOST one level above policy
        if rl_sev > policy_sev:
            max_allowed = policy_sev + 1
            clamped_sev = min(rl_sev, max_allowed)
            # Reverse lookup
            for name, sev in severity.items():
                if sev == clamped_sev:
                    return name
            return policy_decision

        # RL cannot loosen
        return policy_decision

    def save(self) -> None:
        """Persist PPO to disk."""
        if not self._persist_path or not self.model:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = self._persist_path.with_suffix(".zip")
        self.model.save(str(model_path))
        logger.info("PPO model saved to %s", model_path)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "architecture": "PPO (stable-baselines3)",
            "state_dims": 9,
            "action_dims": 5,
            "is_loaded": self.is_loaded
        }
