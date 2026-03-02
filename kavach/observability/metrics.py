"""Metrics collector for Kavach operations.

In-process counters for latency, block rate, and risk distribution.
Useful for monitoring performance and policy effectiveness.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of Kavach metrics."""

    total_requests: int = 0
    total_blocked: int = 0
    total_allowed: int = 0
    total_sanitized: int = 0
    avg_latency_ms: float = 0.0
    avg_risk_score: float = 0.0
    block_rate: float = 0.0
    decisions_by_action: dict[str, int] = field(default_factory=dict)


class KavachMetrics:
    """Thread-safe in-process metrics collector.

    Tracks:
    - Total requests by action (allow/block/sanitize)
    - Average latency
    - Risk score distribution
    - Block rate

    These are in-memory counters — for production, export to
    Prometheus/Datadog via OpenTelemetry metrics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decisions: dict[str, int] = defaultdict(int)
        self._total_latency: float = 0.0
        self._total_risk: float = 0.0
        self._count: int = 0

    def record(self, action: str, latency_ms: float, risk_score: float) -> None:
        """Record a decision.

        Args:
            action: The decision action (allow, block, sanitize).
            latency_ms: Processing latency in milliseconds.
            risk_score: The computed risk score.
        """
        with self._lock:
            self._decisions[action] += 1
            self._total_latency += latency_ms
            self._total_risk += risk_score
            self._count += 1

    def snapshot(self) -> MetricsSnapshot:
        """Get a point-in-time snapshot of metrics.

        Returns:
            MetricsSnapshot with current aggregated metrics.
        """
        with self._lock:
            total = self._count
            blocked = self._decisions.get("block", 0)
            allowed = self._decisions.get("allow", 0)
            sanitized = self._decisions.get("sanitize", 0)

            return MetricsSnapshot(
                total_requests=total,
                total_blocked=blocked,
                total_allowed=allowed,
                total_sanitized=sanitized,
                avg_latency_ms=round(self._total_latency / total, 2) if total > 0 else 0.0,
                avg_risk_score=round(self._total_risk / total, 4) if total > 0 else 0.0,
                block_rate=round(blocked / total, 4) if total > 0 else 0.0,
                decisions_by_action=dict(self._decisions),
            )

    def reset(self) -> None:
        """Reset all metrics counters."""
        with self._lock:
            self._decisions.clear()
            self._total_latency = 0.0
            self._total_risk = 0.0
            self._count = 0
