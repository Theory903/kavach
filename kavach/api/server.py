"""Kavach REST API server — FastAPI application.

Provides REST endpoints for non-Python stacks to use Kavach:
- POST /v1/analyze — analyze a prompt
- POST /v1/secure-run — full guarded execution
- POST /v1/sanitize — just clean a prompt
- GET /v1/policy/{role} — get policy for a role
- GET /v1/metrics — get runtime metrics
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from kavach.core.gateway import KavachGateway

limiter = Limiter(key_func=get_remote_address)


# --- Request / Response Models ---

class AnalyzeRequest(BaseModel):
    """Request body for /v1/analyze and /v1/secure-run."""

    prompt: str = Field(..., max_length=32000, description="The prompt to analyze")
    user_id: str = Field(default="anonymous", description="User identifier")
    role: str = Field(default="default", description="User role for RBAC")
    session_id: str | None = Field(default=None, description="Optional session ID")


class SanitizeRequest(BaseModel):
    """Request body for /v1/sanitize."""

    prompt: str = Field(..., max_length=32000, description="The prompt to sanitize")


class FeedbackRequest(BaseModel):
    """Request body for production feedback loop."""

    trace_id: str = Field(..., description="The original execution trace ID")
    prompt: str = Field(..., max_length=32000)
    decision: str = Field(..., description="Action that was taken (allow/sanitize/block)")
    actual_label: str = Field(..., description="Ground truth category (benign, injection, etc.)")
    was_correct: bool = Field(..., description="Was the system's decision correct?")
    reviewer: str = Field(default="system")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    """Standard response for all analysis endpoints."""

    decision: str
    risk_score: float
    reasons: list[str]
    matched_rules: list[str] = Field(default_factory=list)
    clean_prompt: str | None = None
    latency_ms: float = 0.0
    session_id: str = ""
    trace_id: str = ""


# --- Application Factory ---

def create_app(policy: str | None = None) -> FastAPI:
    """Create the Kavach FastAPI application.

    Args:
        policy: Path to policy YAML file, or None for default.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Kavach API",
        description="Security and permission layer for agentic AI systems",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    gateway = KavachGateway(policy=policy)

    # --- Routes ---

    @app.post("/v1/analyze", response_model=DecisionResponse)
    @limiter.limit("100/minute")
    async def analyze_prompt(request: Request, payload: AnalyzeRequest) -> dict:
        """Analyze a prompt without executing — returns security decision."""
        result = gateway.analyze(
            prompt=payload.prompt,
            user_id=payload.user_id,
            role=payload.role,
            session_id=payload.session_id,
        )
        return result

    @app.post("/v1/secure-run", response_model=DecisionResponse)
    @limiter.limit("100/minute")
    async def secure_run(request: Request, payload: AnalyzeRequest) -> dict:
        """Full guarded execution pipeline.

        Note: In API mode without an LLM callback, this behaves
        like /analyze but with the full pipeline metadata.
        """
        result = gateway.secure_call(
            prompt=payload.prompt,
            user_id=payload.user_id,
            role=payload.role,
            session_id=payload.session_id,
        )
        return result

    @app.post("/v1/sanitize")
    @limiter.limit("200/minute")
    async def sanitize_prompt(request: Request, payload: SanitizeRequest) -> dict:
        """Just sanitize a prompt — no policy evaluation."""
        clean = gateway.sanitize_prompt(payload.prompt)
        return {
            "original_length": len(payload.prompt),
            "clean_length": len(clean),
            "clean_prompt": clean,
        }

    @app.get("/v1/policy/{role}")
    async def get_policy(role: str) -> dict:
        """Get the policy configuration for a specific role."""
        role_policy = gateway._engine.get_role_policy(role)
        return {
            "role": role,
            "policy": role_policy.model_dump(),
        }

    @app.post("/v1/feedback")
    @limiter.limit("200/minute")
    async def ingest_feedback(request: Request, payload: FeedbackRequest) -> dict:
        """Ingest production feedback to update the RL advisor and training dataset."""
        from kavach.ml.feedback import FeedbackStore
        store = FeedbackStore()
        store.record(
            trace_id=payload.trace_id,
            prompt=payload.prompt,
            decision=payload.decision,
            actual_label=payload.actual_label,
            was_correct=payload.was_correct,
            reviewer=payload.reviewer,
            metadata=payload.metadata,
        )
        # Online continuous update of RL advisor if applicable
        # (This is just a light pass; full retraining is done offline via CLI)
        if hasattr(gateway, "_rl_advisor"):
            reward = gateway._rl_advisor.compute_reward(
                action_taken=payload.decision,
                actual_is_attack=(payload.actual_label != "benign"),
                was_blocked=(payload.decision in ["block", "sanitize"])
            )
            # Find terminal state to update (in full prod, state_index stored in trace metadata)
            # For this MVP, we only record it to the jsonl datastore
        return {"status": "ok", "message": "Feedback recorded."}

    @app.get("/v1/metrics")
    async def get_metrics() -> dict:
        """Get current runtime metrics."""
        snapshot = gateway.metrics.snapshot()
        return {
            "total_requests": snapshot.total_requests,
            "total_blocked": snapshot.total_blocked,
            "total_allowed": snapshot.total_allowed,
            "total_sanitized": snapshot.total_sanitized,
            "avg_latency_ms": snapshot.avg_latency_ms,
            "avg_risk_score": snapshot.avg_risk_score,
            "block_rate": snapshot.block_rate,
            "decisions_by_action": snapshot.decisions_by_action,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Any:
        """Standard Prometheus metrics scraping endpoint."""
        from fastapi import Response
        import kavach.observability.prometheus as prom
        
        data, content_type = prom.get_metrics_response()
        return Response(content=data, media_type=content_type)

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "service": "kavach", "version": "0.1.0"}

    return app


# Default app instance (used by uvicorn kavach.api.server:app)
app = create_app()
