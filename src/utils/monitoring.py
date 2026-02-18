"""Prometheus metrics for observability (Phase 3)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

# Tool executions: total by tool name and status
TOOL_EXECUTIONS = Counter(
    "agent_tool_executions_total",
    "Total tool executions",
    ["tool_name", "status"],
)

# LLM request duration in seconds
LLM_LATENCY = Histogram(
    "agent_llm_latency_seconds",
    "LLM request duration in seconds",
    ["provider"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# Agent invocations (one per user turn)
AGENT_INVOCATIONS = Counter(
    "agent_invocations_total",
    "Total agent invocations",
    ["interface"],
)


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus HTTP server for scraping. Call from main when enabled."""
    start_http_server(port)


def record_tool_execution(tool_name: str, success: bool) -> None:
    status = "success" if success else "failure"
    TOOL_EXECUTIONS.labels(tool_name=tool_name, status=status).inc()


def record_llm_latency(provider: str, duration_seconds: float) -> None:
    LLM_LATENCY.labels(provider=provider).observe(duration_seconds)


def record_agent_invocation(interface: str) -> None:
    AGENT_INVOCATIONS.labels(interface=interface).inc()
