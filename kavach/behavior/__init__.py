"""Kavach session behavior engine — multi-turn anomaly detection.

Detects multi-step attacks by tracking risk trajectory, prompt
drift, and behavioral patterns across conversation turns.
"""

from kavach.behavior.session_engine import SessionBehaviorEngine, BehaviorSignal

__all__ = ["SessionBehaviorEngine", "BehaviorSignal"]
