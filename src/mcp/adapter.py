"""MCP adapter: expose tool registry in MCP format and handle tool execution requests."""

from __future__ import annotations

from typing import Any

from src.tools.registry import ToolRegistry
from src.tools.base import ToolResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MCPAdapter:
    """
    Adapter to make tools MCP-compatible for exposure to other systems.
    Phase 3 implementation.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.registry = tool_registry

    def export_mcp_manifest(self) -> dict[str, Any]:
        """Export tools in MCP format (version, tools with name, description, inputSchema)."""
        tools = []
        for defn in self.registry._tools.values():
            schema = defn.parameters_schema
            tools.append({
                "name": defn.name,
                "description": defn.description,
                "inputSchema": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            })
        return {"version": "1.0", "tools": tools}

    async def handle_mcp_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle an incoming MCP tool execution request.
        Expects request with: method="tools/call", params={"name": str, "arguments": dict}.
        Returns result suitable for MCP response (content or error).
        """
        method = request.get("method", "")
        params = request.get("params") or {}
        if method != "tools/call":
            return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            return {"error": {"code": -32602, "message": "Missing tool name"}}
        try:
            result = await self.registry.execute_tool(name, arguments)
            if result.success:
                return {"content": [{"type": "text", "text": str(result.data)}]}
            return {"error": {"code": -32000, "message": result.error or "Tool failed"}}
        except Exception as e:
            logger.exception("mcp_tool_error", tool=name, error=str(e))
            return {"error": {"code": -32000, "message": str(e)}}
