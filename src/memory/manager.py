"""SQLite memory manager for conversations and checkpoints."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Async SQLite persistence for conversations, messages, and tool executions.
    """

    def __init__(self, db_path: str | Path = "data/agent_memory.db") -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Create connection and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._init_schema()

    async def _init_schema(self) -> None:
        """Create tables if not exist."""
        schema_dir = Path(__file__).parent
        schema_file = schema_dir / "schemas.sql"
        if schema_file.exists():
            sql = schema_file.read_text(encoding="utf-8")
            await self._conn.executescript(sql)
            await self._conn.commit()
        else:
            await self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT,
                    tool_calls JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT REFERENCES conversations(id),
                    tool_name TEXT NOT NULL,
                    arguments JSON,
                    result JSON,
                    success BOOLEAN,
                    execution_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    state_snapshot JSON,
                    graph_position TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "MemoryManager":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryManager not connected; use await manager.connect() or async with manager")
        return self._conn

    async def create_conversation(self, user_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Create a new conversation and return its id."""
        conn = self._ensure_conn()
        conv_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {})
        await conn.execute(
            "INSERT INTO conversations (id, user_id, metadata) VALUES (?, ?, ?)",
            (conv_id, user_id, meta_json),
        )
        await conn.commit()
        logger.info("conversation_created", conversation_id=conv_id, user_id=user_id)
        return conv_id

    async def save_conversation_turn(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        tool_executions: list[Any],
    ) -> None:
        """Persist the latest messages and tool executions for a conversation."""
        conn = self._ensure_conn()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if isinstance(content, dict):
                content = json.dumps(content)
            tool_calls = msg.get("tool_calls")
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            await conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, tool_calls_json),
            )
        for te in tool_executions:
            if isinstance(te, dict):
                tool_name = te.get("tool_name", "unknown")
                args = te.get("metadata", {})
                result = te.get("data")
                success = te.get("success", False)
                time_ms = te.get("execution_time_ms", 0.0)
            else:
                tool_name = getattr(te, "tool_name", "unknown")
                args = getattr(te, "metadata", {}) or {}
                result = getattr(te, "data", None)
                success = getattr(te, "success", False)
                time_ms = getattr(te, "execution_time_ms", 0.0)
            await conn.execute(
                """INSERT INTO tool_executions (conversation_id, tool_name, arguments, result, success, execution_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, tool_name, json.dumps(args), json.dumps(result), success, time_ms),
            )
        await conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        await conn.commit()

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent messages for a conversation."""
        conn = self._ensure_conn()
        cursor = await conn.execute(
            "SELECT role, content, tool_calls FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        out = []
        for r in reversed(rows):
            tool_calls = json.loads(r["tool_calls"]) if r["tool_calls"] else None
            out.append({"role": r["role"], "content": r["content"], "tool_calls": tool_calls})
        return out

    async def checkpoint_state(self, checkpoint_id: str, conversation_id: str, state: dict[str, Any], graph_position: str = "") -> None:
        """Save graph state snapshot for resumption."""
        conn = self._ensure_conn()
        # State may contain non-JSON-serializable values; serialize what we can
        snapshot = {k: v for k, v in state.items() if k != "messages" or isinstance(v, list)}
        try:
            snapshot_json = json.dumps(snapshot, default=str)
        except Exception:
            snapshot_json = "{}"
        await conn.execute(
            "INSERT OR REPLACE INTO agent_checkpoints (id, conversation_id, state_snapshot, graph_position) VALUES (?, ?, ?, ?)",
            (checkpoint_id, conversation_id, snapshot_json, graph_position),
        )
        await conn.commit()
