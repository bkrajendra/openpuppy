"""APScheduler-based runner for recurring agent tasks."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.logging import get_logger

logger = get_logger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_agent_task(prompt: str, job_id: str) -> None:
    """Run the agent with the given prompt (for scheduled jobs)."""
    try:
        from src.agent.executor import run_agent
        from src.utils.llm_factory import get_llm_from_config
        llm = get_llm_from_config()
        state = await run_agent(prompt, llm=llm, thread_id=f"scheduled_{job_id}")
        response = state.get("final_response", "")
        logger.info("scheduled_agent_done", job_id=job_id, response_len=len(response))
    except Exception as e:
        logger.exception("scheduled_agent_failed", job_id=job_id, error=str(e))


def add_agent_job(
    job_id: str,
    prompt: str,
    cron: str,
    **trigger_kw: Any,
) -> None:
    """
    Add a recurring job that runs the agent with the given prompt.
    cron: cron expression, e.g. "0 9 * * *" for 9am daily, or use trigger_kw (minute=0, hour=9).
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    if trigger_kw:
        trigger = CronTrigger(**trigger_kw)
    else:
        trigger = CronTrigger.from_crontab(cron)
    _scheduler.add_job(
        _run_agent_task,
        trigger=trigger,
        args=[prompt, job_id],
        id=job_id,
        replace_existing=True,
    )
    logger.info("scheduled_job_added", job_id=job_id, cron=cron or str(trigger_kw))


def start_scheduler(jobs: list[dict[str, Any]] | None = None) -> AsyncIOScheduler:
    """
    Start the scheduler. If jobs is provided, add them: [{"id": "...", "prompt": "...", "cron": "0 9 * * *"}, ...].
    Jobs can also be loaded from config scheduler.jobs.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    if jobs is None:
        from src.utils.config import load_config
        config = load_config()
        jobs = config.get("scheduler", {}).get("jobs", [])
    for j in jobs:
        add_agent_job(j.get("id", "job"), j.get("prompt", ""), j.get("cron", "0 * * * *"))
    _scheduler.start()
    logger.info("scheduler_started", job_count=len(jobs))
    return _scheduler


def get_scheduler() -> AsyncIOScheduler | None:
    return _scheduler
