"""Wikipedia lookup tool (MediaWiki API, no API key)."""

from __future__ import annotations

import time
from urllib.parse import quote

import requests

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

WIKIPEDIA_SCHEMA = {
    "properties": {
        "query": {"type": "string", "description": "Topic or title to look up on Wikipedia"},
        "sentences": {"type": "integer", "description": "Max summary sentences (1-5)", "default": 3},
    },
    "required": ["query"],
}


@tool_registry.register(
    name="wikipedia_lookup",
    description="Look up a topic on Wikipedia and return a short summary. Use for encyclopedic facts.",
    category="information_retrieval",
    parameters_schema=WIKIPEDIA_SCHEMA,
)
async def wikipedia_lookup(query: str, sentences: int = 3) -> ToolResult:
    start = time.perf_counter()
    try:
        sentences = max(1, min(5, sentences))
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return ToolResult(success=True, data={"summary": "No Wikipedia article found.", "title": None}, execution_time_ms=(time.perf_counter() - start) * 1000)
        title = hits[0].get("title", "")
        params2 = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": sentences,
            "format": "json",
        }
        r2 = requests.get(url, params=params2, timeout=10)
        r2.raise_for_status()
        data2 = r2.json()
        pages = data2.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        summary = (page.get("extract") or "").strip() or "No extract available."
        return ToolResult(
            success=True,
            data={"title": title, "summary": summary, "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"},
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        logger.exception("wikipedia_lookup_failed", query=query, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)
