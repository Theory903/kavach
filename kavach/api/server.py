"""Kavach REST API server — FastAPI application.

Provides REST endpoints for non-Python stacks to use Kavach:
- POST /v1/analyze — analyze a prompt
- POST /v1/secure-run — full guarded execution
- POST /v1/sanitize — just clean a prompt
- GET /v1/policy/{role} — get policy for a role
- GET /v1/metrics — get runtime metrics
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kavach.core.gateway import KavachGateway


# --- Request / Response Models ---

class AnalyzeRequest(BaseModel):
    """Request body for /v1/analyze and /v1/secure-run."""

    prompt: str = Field(..., description="The prompt to analyze")
    user_id: str = Field(default="anonymous", description="User identifier")
    role: str = Field(default="default", description="User role for RBAC")
    session_id: str | None = Field(default=None, description="Optional session ID")


class SanitizeRequest(BaseModel):
    """Request body for /v1/sanitize."""

    prompt: str = Field(..., description="The prompt to sanitize")


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
    async def analyze_prompt(request: AnalyzeRequest) -> dict:
        """Analyze a prompt without executing — returns security decision."""
        result = gateway.analyze(
            prompt=request.prompt,
            user_id=request.user_id,
            role=request.role,
            session_id=request.session_id,
        )
        return result

    @app.post("/v1/secure-run", response_model=DecisionResponse)
    async def secure_run(request: AnalyzeRequest) -> dict:
        """Full guarded execution pipeline.

        Note: In API mode without an LLM callback, this behaves
        like /analyze but with the full pipeline metadata.
        """
        result = gateway.secure_call(
            prompt=request.prompt,
            user_id=request.user_id,
            role=request.role,
            session_id=request.session_id,
        )
        return result

    @app.post("/v1/sanitize")
    async def sanitize_prompt(request: SanitizeRequest) -> dict:
        """Just sanitize a prompt — no policy evaluation."""
        clean = gateway.sanitize_prompt(request.prompt)
        return {
            "original_length": len(request.prompt),
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

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "service": "kavach", "version": "0.1.0"}

    return app


# Default app instance (used by uvicorn kavach.api.server:app)
app = create_app()
