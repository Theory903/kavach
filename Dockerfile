# syntax=docker/dockerfile:1.4
FROM python:3.11-slim-bookworm

# Add non-root user
RUN groupadd -r kavach && useradd -r -g kavach kavach

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies based on the `ml` group + FastAPI
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[ml]"

# Copy source code (this would normally be a wheel install)
COPY kavach/ ./kavach/
COPY policy.yaml ./policy.yaml

# Download the ONNX model layer securely during build phase so no internet required later
RUN python -c "from kavach.ml.embeddings import EmbeddingRiskScorer; EmbeddingRiskScorer().load_and_encode_corpus()"

# Ensure user owns dir
RUN chown -R kavach:kavach /app

USER kavach

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Proxy/Sidecar entrypoint
CMD ["uvicorn", "kavach.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
