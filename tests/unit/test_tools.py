"""Unit tests for tools."""

import asyncio
import pytest
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry, tool_registry
from src.tools.web_search import web_search
from src.tools.code_executor import code_executor


def test_tool_result_schema():
    r = ToolResult(success=True, data={"x": 1}, execution_time_ms=10.0)
    assert r.success is True
    assert r.data == {"x": 1}
    assert r.error is None


@pytest.mark.asyncio
async def test_code_executor_simple():
    r = await code_executor("print(2 + 2)")
    assert r.success is True
    assert "4" in str(r.data.get("output", ""))


@pytest.mark.asyncio
async def test_code_executor_restricted():
    r = await code_executor("import os; os.system('ls')")
    assert r.success is False
    assert r.error


def test_registry_has_phase1_tools():
    schemas = tool_registry.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "web_search" in names
    assert "code_executor" in names
