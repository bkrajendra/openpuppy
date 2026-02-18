"""Telegram bot interface for the agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from src.agent.executor import create_agent, _initial_state
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.utils.rate_limit import get_telegram_rate_limiter

logger = get_logger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def _allowed_telegram_user_ids() -> Set[int]:
    """Allowed Telegram user IDs (int). From env TELEGRAM_ALLOWED_USER_IDS=123,456 or config (list or comma string)."""
    config = load_config()
    tg = config.get("interfaces", {}).get("telegram", {})
    raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS")
    if raw is None or raw == "":
        raw = tg.get("allowed_user_ids")
    ids: Set[int] = set()
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, int):
                ids.add(x)
            elif isinstance(x, str) and x.isdigit():
                ids.add(int(x))
    elif isinstance(raw, str) and raw:
        for part in raw.replace(" ", "").split(","):
            if part.isdigit():
                ids.add(int(part))
    return ids


def build_agent():
    """Build compiled agent (no checkpointer for stateless Telegram turns)."""
    from langgraph.checkpoint.memory import MemorySaver
    from src.utils.llm_factory import get_llm_from_config
    llm = get_llm_from_config()
    return create_agent(llm=llm, checkpointer=MemorySaver())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text: run agent, reply with final response. Only allowed user IDs are served."""
    if not update.message or not update.message.text:
        return
    user_message = update.message.text.strip()
    effective_user = update.effective_user
    user_id_int = effective_user.id if effective_user else None
    user_id = str(user_id_int) if user_id_int is not None else "unknown"
    logger.info("telegram_message_received", user_id=user_id)

    allowed = _allowed_telegram_user_ids()

    if allowed and user_id_int is not None and user_id_int not in allowed:
        logger.warning("telegram_unauthorized", user_id=user_id_int)
        await update.message.reply_text("Unauthorized access.")
        return

    limiter = get_telegram_rate_limiter()
    if not limiter.allow(user_id):
        logger.warning("telegram_rate_limited", user_id=user_id)
        await update.message.reply_text("Rate limited. Please try again later.")
        return

    logger.info("telegram_message_received", user_id=user_id)
    try:
        from src.utils.monitoring import record_agent_invocation
        record_agent_invocation("telegram")
    except Exception:
        pass
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        agent = build_agent()
        config = {"configurable": {"thread_id": f"tg_{user_id}"}}
        initial = _initial_state(user_message, metadata={"platform": "telegram", "user_id": user_id})
        final_state = await agent.ainvoke(initial, config=config)
        response = final_state.get("final_response", "").strip() or "I couldn't generate a response."
        if len(response) > 4000:
            response = response[:3997] + "..."
        await update.message.reply_text(response)
    except Exception as e:
        logger.exception("telegram_agent_failed", user_id=user_id, error=str(e))
        await update.message.reply_text(f"Sorry, something went wrong: {e}")


def run_telegram_bot() -> None:
    """Start the Telegram bot (polling)."""
    load_dotenv(_project_root / ".env")
    config = load_config()
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    port = os.getenv("PROMETHEUS_METRICS_PORT", "")
    if port.isdigit():
        try:
            from src.utils.monitoring import start_metrics_server
            start_metrics_server(int(port))
            logger.info("prometheus_metrics_started", port=int(port))
        except Exception as e:
            logger.warning("prometheus_metrics_failed", error=str(e))
    token = os.getenv("TELEGRAM_BOT_TOKEN") or config.get("interfaces", {}).get("telegram", {}).get("bot_token", "")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env or config interfaces.telegram.bot_token")
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("telegram_bot_starting")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_telegram_bot()
