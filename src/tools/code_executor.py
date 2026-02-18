"""Sandboxed Python code execution using RestrictedPython."""

from __future__ import annotations

import asyncio
from typing import Any
import time
from concurrent.futures import ThreadPoolExecutor

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import full_write_guard, guarded_iter_unpack_sequence
from RestrictedPython.PrintCollector import PrintCollector

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Allowed builtins and modules for sandbox (Phase 1)
ALLOWED_IMPORTS = frozenset({"math", "statistics", "datetime", "json", "re"})
_MAX_EXECUTION_TIME = 10.0
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="code_exec")
    return _executor


CODE_EXECUTOR_SCHEMA = {
    "properties": {
        "code": {"type": "string", "description": "Python code to run (single expression or statements). No file/network access."},
    },
    "required": ["code"],
}


def _run_restricted_code(code: str) -> tuple[bool, str, Any]:
    """Run code in restricted environment; returns (success, output_or_error, result)."""
    restricted_globals: dict[str, Any] = {
        **safe_globals,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_write_": full_write_guard,
        "_print_": PrintCollector,
    }
    restricted_locals: dict[str, Any] = {}
    try:
        byte_code = compile_restricted(code, filename="<inline>", mode="exec")
        if byte_code is None:
            return False, "Compilation failed", None
        exec(byte_code, restricted_globals, restricted_locals)
        out = restricted_locals.get("_print")
        result_text = out() if callable(out) else ""
        return True, (result_text or "").strip(), None
    except SyntaxError as e:
        return False, str(e), None
    except Exception as e:
        return False, str(e), None


@tool_registry.register(
    name="code_executor",
    description="Run restricted Python code in a sandbox. Allowed: math, statistics, datetime, json, re. No file or network access. Use for calculations and data transformation.",
    category="code_execution",
    parameters_schema=CODE_EXECUTOR_SCHEMA,
)
async def code_executor(code: str) -> ToolResult:
    """
    Execute Python code in a sandboxed environment with timeout.

    Example:
        >>> r = await code_executor("sum(range(101))")
    """
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        result_ok, result_text, _ = await asyncio.wait_for(
            loop.run_in_executor(_get_executor(), _run_restricted_code, code),
            timeout=_MAX_EXECUTION_TIME,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if result_ok:
            return ToolResult(
                success=True,
                data={"output": result_text},
                metadata={"execution_time_limit": _MAX_EXECUTION_TIME},
                execution_time_ms=elapsed_ms,
            )
        return ToolResult(
            success=False,
            data=None,
            error=result_text or "Execution failed",
            execution_time_ms=elapsed_ms,
        )
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("code_executor_timeout", timeout=_MAX_EXECUTION_TIME)
        return ToolResult(
            success=False,
            data=None,
            error=f"Execution timed out after {_MAX_EXECUTION_TIME}s",
            execution_time_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("code_executor_error", error=str(e))
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time_ms=elapsed_ms,
        )
