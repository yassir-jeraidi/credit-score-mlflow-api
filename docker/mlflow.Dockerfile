# =============================================================================
# MLflow Tracking Server - Multi-Stage Build
# =============================================================================

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install MLflow and dependencies
RUN pip install --upgrade pip && \
    pip install 'mlflow<2.10' psycopg2-binary

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim AS production

# Set labels for image metadata
LABEL maintainer="Credit Score MLOps Team" \
      description="MLflow Tracking Server - Optimized Multi-Stage Build" \
      version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_HOST=0.0.0.0 \
    MLFLOW_PORT=5001

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 mlflow && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mlflow

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /mlflow

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow

# Switch to non-root user
USER mlflow

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:${MLFLOW_PORT}/api/2.0/mlflow/experiments/search?max_results=1 || exit 1

# Default command
CMD ["sh", "-c", "mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts} --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}"]
