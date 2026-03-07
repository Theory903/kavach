# ─────────────────────────────────────────────────────────────────────
# Kavach — Multi-stage production Dockerfile
# Stage 1: build + install dependencies
# Stage 2: lean runtime image (non-root, no extras)
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (cache-friendly)
COPY pyproject.toml README.md ./
COPY kavach/ ./kavach/

# Install into a separate prefix so we can copy cleanly
RUN pip install --upgrade pip \
    && pip install --prefix=/install .[ml] \
    && pip install --prefix=/install uvicorn[standard]


# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Kavach"
LABEL org.opencontainers.image.description="Universal AI Security Gateway"
LABEL org.opencontainers.image.version="0.2.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Non-root user for security
RUN addgroup --system kavach && adduser --system --ingroup kavach kavach

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=kavach:kavach kavach/ ./kavach/
COPY --chown=kavach:kavach data/ ./data/
COPY --chown=kavach:kavach policies/ ./policies/ 2>/dev/null || true

# Pre-create data dirs with correct ownership
RUN mkdir -p data/trained_models data/attack_vectors data/rl_ppo \
    && chown -R kavach:kavach data/

USER kavach

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

EXPOSE 8000

# Start server
CMD ["uvicorn", "kavach.api.server:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--access-log"]
