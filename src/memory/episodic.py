"""Episodic memory: recent conversation turns (time-ordered). Semantic = VectorStore (Phase 4)."""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EpisodicMemory:
    """
    Episodic memory = last N conversation turns per conversation (from SQLite).
    Use for "what did we just discuss" or recent context. Semantic memory = VectorStore for long-term facts.
    """

    def __init__(self, memory_manager: Any, conversation_id: str, limit: int = 20) -> None:
        self.manager = memory_manager
        self.conversation_id = conversation_id
        self.limit = limit
        self._cache: list[dict[str, Any]] | None = None

    async def get_recent_turns(self) -> list[dict[str, Any]]:
        """Return recent messages (episodic) for this conversation."""
        self._cache = await self.manager.get_conversation_history(
            self.conversation_id,
            limit=self.limit,
        )
        return self._cache

    async def get_recent_text(self) -> str:
        """Return a single string of recent user/assistant content for context."""
        turns = await self.get_recent_turns()
        parts = []
        for t in turns:
            role = t.get("role", "")
            content = t.get("content") or ""
            if content:
                parts.append(f"{role}: {content[:500]}")
        return "\n".join(parts[-self.limit:])
