"""Tool registry with decorator-based registration and OpenAI-compatible schemas."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from src.tools.base import ToolResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """Metadata and handler for a single tool."""

    name: str
    description: str
    category: str
    handler: Callable[..., Awaitable[ToolResult]]
    parameters_schema: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Registry for agent tools with timeout and error handling.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register("echo", "Echo back", "test")
        ... async def echo(msg: str) -> ToolResult:
        ...     return ToolResult(success=True, data=msg, execution_time_ms=0)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._custom_names: set[str] = set()  # names added via UI / register_dynamic (can be removed)

    def register(
        self,
        name: str,
        description: str,
        category: str,
        parameters_schema: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]]:
        """Decorator to register an async tool function."""

        def decorator(fn: Callable[..., Awaitable[ToolResult]]) -> Callable[..., Awaitable[ToolResult]]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                category=category,
                handler=fn,
                parameters_schema=parameters_schema or {},
            )
            return fn

        return decorator

    def register_dynamic(
        self,
        name: str,
        description: str,
        category: str,
        parameters_schema: dict[str, Any],
        handler: Callable[..., Awaitable[ToolResult]],
    ) -> None:
        """Register a tool programmatically (e.g. custom tools added via admin UI). Marked as custom so it can be unregistered."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            category=category,
            handler=handler,
            parameters_schema=parameters_schema or {},
        )
        self._custom_names.add(name)

    def unregister(self, name: str) -> bool:
        """Remove a custom tool by name. Returns True if removed, False if not found or not custom."""
        if name not in self._custom_names:
            return False
        self._custom_names.discard(name)
        self._tools.pop(name, None)
        return True

    def is_custom(self, name: str) -> bool:
        return name in self._custom_names

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": {
                        "type": "object",
                        "properties": defn.parameters_schema.get("properties", {}),
                        "required": defn.parameters_schema.get("required", []),
                    },
                },
            }
            for defn in self._tools.values()
        ]

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
        _call_depth: int = 0,
        _max_call_depth: int = 2,
    ) -> ToolResult:
        """
        Execute a tool by name with timeout and structured error handling.
        Phase 4: _call_depth/_max_call_depth for tool composition (tools calling tools).
        """
        if _call_depth > _max_call_depth:
            return ToolResult(
                success=False,
                data=None,
                error="Max tool call depth exceeded",
                execution_time_ms=0.0,
            )
        if name not in self._tools:
            logger.warning("tool_not_found", tool_name=name)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {name}",
                execution_time_ms=0.0,
            )
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._tools[name].handler(**arguments),
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            if result.execution_time_ms == 0.0:
                result = result.model_copy(update={"execution_time_ms": elapsed_ms})
            logger.info(
                "tool_executed",
                tool_name=name,
                success=result.success,
                execution_time_ms=elapsed_ms,
            )
            try:
                from src.utils.monitoring import record_tool_execution
                record_tool_execution(name, result.success)
            except Exception:
                pass
            return result
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("tool_timeout", tool_name=name, timeout=timeout)
            try:
                from src.utils.monitoring import record_tool_execution
                record_tool_execution(name, False)
            except Exception:
                pass
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool timed out after {timeout}s",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("tool_error", tool_name=name, error=str(e))
            try:
                from src.utils.monitoring import record_tool_execution
                record_tool_execution(name, False)
            except Exception:
                pass
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )


# Global registry instance; tools register on import
tool_registry = ToolRegistry()
