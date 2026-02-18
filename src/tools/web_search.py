"""Web search tool using DuckDuckGo (no API key)."""

from __future__ import annotations

from duckduckgo_search import DDGS

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

WEB_SEARCH_SCHEMA = {
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Max results to return (1-10)", "default": 5},
    },
    "required": ["query"],
}


@tool_registry.register(
    name="web_search",
    description="Search the web for current information using DuckDuckGo. Use for facts, news, or recent events.",
    category="information_retrieval",
    parameters_schema=WEB_SEARCH_SCHEMA,
)
async def web_search(
    query: str,
    max_results: int = 5,
) -> ToolResult:
    """
    Run a web search and return snippets and links.

    Example:
        >>> r = await web_search("Python asyncio tutorial", max_results=3)
    """
    import time
    start = time.perf_counter()
    try:
        max_results = max(1, min(10, max_results))
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        data = [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in results
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=True,
            data=data,
            metadata={"query": query, "count": len(data)},
            execution_time_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("web_search_failed", query=query, error=str(e))
        return ToolResult(
            success=False,
            data=None,
            error=str(e),
            execution_time_ms=elapsed_ms,
        )
