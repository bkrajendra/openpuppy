"""Custom tools added via admin UI: persisted to JSON, executed as HTTP calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import requests

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "custom_tools.json"


def _custom_tools_path() -> Path:
    p = _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_custom_tools(path: Path | None = None) -> list[dict[str, Any]]:
    """Load custom tool definitions from JSON file."""
    p = path or _custom_tools_path()
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tools", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    except Exception as e:
        logger.warning("custom_tools_load_failed", path=str(p), error=str(e))
        return []


def save_custom_tools(tools: list[dict[str, Any]], path: Path | None = None) -> None:
    """Persist custom tool definitions to JSON file."""
    p = path or _custom_tools_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"tools": tools}, f, indent=2)


def _make_http_handler(url: str, method: str) -> Any:
    """Return an async handler that calls the given URL with tool arguments as JSON body or query."""

    async def handler(**kwargs: Any) -> ToolResult:
        try:
            meth = method.upper() if method else "GET"
            if meth == "GET":
                resp = await asyncio.to_thread(requests.get, url, params=kwargs, timeout=30)
            else:
                resp = await asyncio.to_thread(requests.request, meth, url, json=kwargs, timeout=30)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return ToolResult(success=True, data=data, execution_time_ms=0.0)
        except requests.RequestException as e:
            return ToolResult(success=False, data=None, error=str(e), execution_time_ms=0.0)
        except Exception as e:
            logger.exception("custom_tool_error", error=str(e))
            return ToolResult(success=False, data=None, error=str(e), execution_time_ms=0.0)

    return handler


def register_custom_tool_def(t: dict[str, Any]) -> None:
    """Register a single custom tool definition with the global registry."""
    name = t.get("name") or ""
    description = t.get("description") or "Custom tool"
    tool_type = t.get("type") or "http"
    params_schema = t.get("parameters_schema") or {"type": "object", "properties": {}, "required": []}
    if "properties" not in params_schema:
        params_schema["properties"] = params_schema.get("properties", {})
    if "required" not in params_schema:
        params_schema["required"] = params_schema.get("required", [])

    if tool_type == "http":
        url = t.get("url") or ""
        method = t.get("method") or "POST"
        if not url or not name:
            logger.warning("custom_tool_skip_invalid", name=name, url=url)
            return
        handler = _make_http_handler(url, method)
        tool_registry.register_dynamic(
            name=name,
            description=description,
            category="custom",
            parameters_schema=params_schema,
            handler=handler,
        )
        logger.info("custom_tool_registered", name=name, url=url)
    else:
        logger.warning("custom_tool_unknown_type", type=tool_type, name=name)


def load_and_register_all_custom_tools(path: Path | None = None) -> None:
    """Load custom tools from JSON and register each with the global registry."""
    for t in load_custom_tools(path):
        try:
            register_custom_tool_def(t)
        except Exception as e:
            logger.warning("custom_tool_register_failed", name=t.get("name"), error=str(e))


def add_custom_tool(definition: dict[str, Any]) -> dict[str, Any]:
    """
    Add a new custom tool: append to JSON and register. definition must have
    name, description, type (e.g. 'http'), and for http: url, method, parameters_schema.
    """
    tools = load_custom_tools()
    name = (definition.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if any(x.get("name") == name for x in tools):
        raise ValueError(f"Tool '{name}' already exists")
    definition = {**definition, "name": name}
    tools.append(definition)
    save_custom_tools(tools)
    register_custom_tool_def(definition)
    return definition


def remove_custom_tool(name: str) -> bool:
    """Remove a custom tool by name from registry and from JSON."""
    removed = tool_registry.unregister(name)
    tools = [t for t in load_custom_tools() if t.get("name") != name]
    save_custom_tools(tools)
    return removed
