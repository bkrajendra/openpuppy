"""Run the agent scheduler (Phase 4). Configure jobs in config/agent_config.yaml under scheduler.jobs."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scheduler.runner import start_scheduler


async def main() -> None:
    start_scheduler()
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
