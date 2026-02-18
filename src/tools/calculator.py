"""Safe calculator tool (math expressions only)."""

from __future__ import annotations

import time

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SAFE_BUILTINS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}


def _safe_eval(expr: str):
    """Evaluate a math-only expression with no I/O or imports."""
    return eval(expr, {"__builtins__": {}}, _SAFE_BUILTINS)


CALCULATOR_SCHEMA = {
    "properties": {
        "expression": {"type": "string", "description": "A mathematical expression to evaluate, e.g. '2 + 3 * 4', 'sqrt(16)' is not supported; use pow(16, 0.5)"},
    },
    "required": ["expression"],
}


@tool_registry.register(
    name="calculator",
    description="Evaluate a safe mathematical expression. Supports +, -, *, /, //, %, **, abs, round, min, max, sum, pow. No variables or imports.",
    category="code_execution",
    parameters_schema=CALCULATOR_SCHEMA,
)
async def calculator(expression: str) -> ToolResult:
    start = time.perf_counter()
    try:
        result = _safe_eval(expression)
        return ToolResult(success=True, data={"expression": expression, "result": result}, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.warning("calculator_failed", expression=expression, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)
