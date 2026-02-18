"""Tool composition: run_tool allows the agent to invoke one tool from another (Phase 4)."""

from __future__ import annotations

import json
import time

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUN_TOOL_SCHEMA = {
    "properties": {
        "tool_name": {"type": "string", "description": "Name of the tool to run (e.g. web_search, calculator)"},
        "tool_arguments": {"type": "object", "description": "Arguments for the tool as a JSON object"},
    },
    "required": ["tool_name", "tool_arguments"],
}


@tool_registry.register(
    name="run_tool",
    description="Call another tool by name with given arguments. Use for multi-step workflows (e.g. search then summarize). Do not call run_tool from within run_tool.",
    category="meta",
    parameters_schema=RUN_TOOL_SCHEMA,
)
async def run_tool(tool_name: str, tool_arguments: dict) -> ToolResult:
    """Execute another tool (one level of composition; depth limit enforced in registry)."""
    start = time.perf_counter()
    try:
        result = await tool_registry.execute_tool(
            tool_name,
            tool_arguments,
            _call_depth=1,
            _max_call_depth=2,
        )
        return result.model_copy(update={"execution_time_ms": (time.perf_counter() - start) * 1000})
    except Exception as e:
        logger.exception("run_tool_failed", tool_name=tool_name, error=str(e))
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )
