"""Semantic memory tools: store_memory, retrieve_memory (ChromaDB)."""

from __future__ import annotations

import time
from typing import Any

from src.memory.vector_store import VectorStore
from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


STORE_MEMORY_SCHEMA = {
    "properties": {
        "content": {"type": "string", "description": "Fact or information to remember (e.g. user preference, important detail)"},
    },
    "required": ["content"],
}

RETRIEVE_MEMORY_SCHEMA = {
    "properties": {
        "query": {"type": "string", "description": "Search query to find relevant stored memories"},
        "top_k": {"type": "integer", "description": "Max number of memories to return (1-10)", "default": 5},
    },
    "required": ["query"],
}


@tool_registry.register(
    name="store_memory",
    description="Store a fact or piece of information in long-term semantic memory for later retrieval. Use for user preferences, important facts, or context to remember.",
    category="memory",
    parameters_schema=STORE_MEMORY_SCHEMA,
)
async def store_memory(content: str) -> ToolResult:
    start = time.perf_counter()
    try:
        store = _get_store()
        mem_id = store.add(content)
        return ToolResult(success=True, data={"id": mem_id, "stored": content[:200]}, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.exception("store_memory_failed", error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)


@tool_registry.register(
    name="retrieve_memory",
    description="Search long-term semantic memory for relevant past facts or information.",
    category="memory",
    parameters_schema=RETRIEVE_MEMORY_SCHEMA,
)
async def retrieve_memory(query: str, top_k: int = 5) -> ToolResult:
    start = time.perf_counter()
    try:
        top_k = max(1, min(10, top_k))
        store = _get_store()
        results = store.search(query, top_k=top_k)
        data = [{"content": r["document"], "metadata": r.get("metadata")} for r in results if r.get("document")]
        return ToolResult(success=True, data=data, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.exception("retrieve_memory_failed", error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)
