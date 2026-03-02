"""OpenTelemetry tracing support for Kavach.

Provides trace context propagation and span creation for
distributed tracing of security decisions.

Optional dependency — gracefully degrades if opentelemetry
is not installed.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Generator


# Try to import OpenTelemetry, fall back to no-op
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class KavachTracer:
    """OpenTelemetry tracer for Kavach operations.

    If OpenTelemetry is installed, creates real spans.
    Otherwise, generates trace IDs but doesn't emit spans.

    Usage:
        tracer = KavachTracer()
        with tracer.span("input_guard.scan") as span:
            span.set_attribute("risk_score", 0.85)
            ...
    """

    def __init__(self, service_name: str = "kavach") -> None:
        self._service_name = service_name
        if _HAS_OTEL:
            self._tracer = trace.get_tracer(service_name)
        else:
            self._tracer = None

    def generate_trace_id(self) -> str:
        """Generate a trace ID (works with or without OTel)."""
        if _HAS_OTEL:
            ctx = trace.get_current_span().get_span_context()
            if ctx.trace_id:
                return format(ctx.trace_id, "032x")
        return f"kavach_{uuid.uuid4().hex[:16]}"

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
        """Create a trace span (real or no-op).

        Args:
            name: Span name (e.g., "input_guard.scan").
            attributes: Key-value attributes to attach.

        Yields:
            The span object (real OTel Span or NoOpSpan).
        """
        if self._tracer is not None:
            with self._tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                yield span
        else:
            yield _NoOpSpan()


class _NoOpSpan:
    """No-op span for when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass
