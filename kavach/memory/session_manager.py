"""Stateful session memory management for detecting multi-turn attacks and intent drift."""
import math
import numpy as np

class SessionManager:
    """Tracks stateful geometry and risk accumulation per user session."""
    
    def __init__(self, risk_decay_factor: float = 0.8):
        self.sessions = {}
        self.risk_decay_factor = risk_decay_factor

    def get_or_create_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "cumulative_risk": 0.0,
                "last_embedding": None,
                "turn_count": 0
            }
        return self.sessions[session_id]

    def record_prompt(self, session_id: str, current_risk: float, current_embedding: np.ndarray) -> dict:
        """
        Records a prompt into the session matrix.
        Returns the updated state including compounding risk and semantic drift.
        """
        session = self.get_or_create_session(session_id)
        
        # 1. Temporal Risk Accrual
        # Base risk decays over time. If a user tries many small injection pieces, the math balloons it.
        session["cumulative_risk"] = current_risk + (session["cumulative_risk"] * self.risk_decay_factor)
        
        # 2. Semantic Intent Drift
        semantic_drift = 0.0
        if session["last_embedding"] is not None and current_embedding is not None:
            base = session["last_embedding"]
            # both shape (384,)
            if len(base.shape) > 1:
                base = base[0]
            curr = current_embedding
            if len(curr.shape) > 1:
                curr = curr[0]
                
            norm_a = np.linalg.norm(base)
            norm_b = np.linalg.norm(curr)
            if norm_a > 0 and norm_b > 0:
                cosine_sim = np.dot(base, curr) / (norm_a * norm_b)
                # Distance scale: 0.0 (identical) to 2.0 (opposite)
                semantic_drift = 1.0 - cosine_sim
                semantic_drift = max(0.0, semantic_drift)

        # Update State
        session["last_embedding"] = current_embedding
        session["turn_count"] += 1
        
        return {
            "session_risk": float(session["cumulative_risk"]),
            "semantic_drift": float(semantic_drift),
            "turn_count": session["turn_count"]
        }

    def reset_session(self, session_id: str) -> None:
        """Clear session memory after a manual context flush or explicit block."""
        if session_id in self.sessions:
            del self.sessions[session_id]
