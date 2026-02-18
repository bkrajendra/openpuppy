"""Initialize the agent SQLite database and schema."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root so src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.manager import MemoryManager
from src.utils.config import load_config


async def main() -> None:
    config = load_config()
    db_path = config.get("memory", {}).get("database_path", "data/agent_memory.db")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with MemoryManager(db_path=path) as db:
        print("Database initialized at", path.absolute())


if __name__ == "__main__":
    asyncio.run(main())
