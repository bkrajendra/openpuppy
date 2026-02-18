"""Tool result schema and base types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Structured result from any tool execution.

    Attributes:
        success: Whether the tool completed without error.
        data: Result payload (tool-specific).
        error: Error message if success is False.
        metadata: Optional execution metadata.
        execution_time_ms: Duration in milliseconds.
    """

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
