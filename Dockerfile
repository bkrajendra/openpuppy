# Phase 3: Production Docker image for Intelligent Agent
FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency and project files first for layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy source and config
COPY src ./src
COPY config ./config
RUN uv sync --no-dev

# Data directory for SQLite and ChromaDB
ENV AGENT_MEMORY_DB=/data/agent_memory.db
ENV ALLOWED_FILE_PATH=/data
VOLUME /data

# Optional: expose Prometheus metrics
ENV PROMETHEUS_METRICS_PORT=9090
EXPOSE 9090

# Default: run CLI (override with agent-telegram etc.)
ENTRYPOINT ["uv", "run", "agent-cli"]
CMD []
