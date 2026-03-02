"""Prometheus metrics registry for enterprise observability.

Tracks core RED metrics (Rate, Errors, Duration) for SIEM
and alerting capabilities.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _HAS_PROMETHEUS = True
    
    # Define Enterprise Metrics
    KAVACH_DECISIONS_TOTAL = Counter(
        "kavach_decisions_total",
        "Total number of security decisions made",
        ["action", "highest_risk_factor"]
    )
    
    KAVACH_EVALUATION_LATENCY = Histogram(
        "kavach_evaluation_latency_seconds",
        "Latency of the complete Kavach evaluation pipeline",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    KAVACH_ACTIVE_THREATS = Gauge(
        "kavach_active_threats",
        "Point-in-time active identified threats undergoing evaluation"
    )
    
    KAVACH_ML_INFERENCE_TIME = Histogram(
        "kavach_ml_inference_time_seconds",
        "Latency dedicated specifically to ML feature extraction and ONNX inferencing",
        buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1]
    )
    
except ImportError:
    _HAS_PROMETHEUS = False


logger = logging.getLogger(__name__)


def get_metrics_response() -> tuple[bytes, str]:
    """Get the latest prometheus metrics format.
    
    Returns:
        Tuple of (metrics_bytes, content_type)
    """
    if _HAS_PROMETHEUS:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"", "text/plain"


def observe_decision(decision: Any) -> None:
    """Record a Kavach Decision object into Prometheus metrics.
    
    Args:
        decision: The kavach.core.policy_engine.Decision object
    """
    if not _HAS_PROMETHEUS:
        return
        
    action = decision.action.value
    factors = decision.matched_rules or decision.reasons
    
    # Pick the primary factor if available, else 'benign'
    primary_factor = factors[0] if factors else ("benign" if action == "allow" else "unknown")
    
    KAVACH_DECISIONS_TOTAL.labels(
        action=action,
        highest_risk_factor=primary_factor
    ).inc()


class latency_timer:
    """Context manager to time latency and record to a specific metric."""
    
    def __init__(self, metric: Any) -> None:
        self.metric = metric
        self.start_time = 0.0
        
    def __enter__(self) -> latency_timer:
        if _HAS_PROMETHEUS:
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if _HAS_PROMETHEUS and self.start_time > 0:
            duration = time.perf_counter() - self.start_time
            self.metric.observe(duration)
