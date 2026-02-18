"""CLI interface for the agent with conversation management."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent.executor import run_agent, stream_agent, create_agent
from src.utils.llm_factory import get_llm_from_config
from src.memory.manager import MemoryManager
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


async def run_conversation_loop(
    *,
    use_memory: bool = True,
    db_path: str | None = None,
    stream: bool = False,
) -> None:
    """
    REPL: read user input, run agent, print final response.
    Optionally persist conversations to SQLite.
    """
    config = load_config()
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level=log_level)

    memory: MemoryManager | None = None
    conversation_id: str | None = None
    thread_id = "cli_default"
    if use_memory:
        db_path = db_path or config.get("memory", {}).get("database_path", "data/agent_memory.db")
        memory = MemoryManager(db_path=db_path)
        await memory.connect()
        try:
            conversation_id = await memory.create_conversation(user_id="cli", metadata={"interface": "cli"})
        except Exception as e:
            logger.warning("memory_init_failed", error=str(e))
            memory = None

    print("Intelligent Agent (Phase 2). Commands: /quit, /stream, /no-stream")
    print("Ask a question (e.g. 'What is the weather in Paris?' or 'Compute sum of 1 to 100').\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Bye.")
            break
        if user_input.lower() == "/stream":
            stream = True
            print("Streaming on.")
            continue
        if user_input.lower() == "/no-stream":
            stream = False
            print("Streaming off.")
            continue

        messages_for_state = None
        if memory and conversation_id:
            try:
                history = await memory.get_conversation_history(conversation_id, limit=20)
                if history:
                    messages_for_state = history + [{"role": "user", "content": user_input}]
            except Exception as e:
                logger.warning("memory_load_failed", error=str(e))

        try:
            llm = get_llm_from_config(config)
            if stream:
                print("Agent: ", end="", flush=True)
                final_response = ""
                async for event in stream_agent(
                    user_input,
                    llm=llm,
                    thread_id=thread_id,
                    config={"configurable": {"thread_id": thread_id}},
                ):
                    for node_name, node_state in event.items():
                        if "final_response" in node_state and node_state["final_response"]:
                            final_response = node_state["final_response"]
                if final_response:
                    print(final_response)
                else:
                    print("(No response)")
            else:
                state = await run_agent(
                    user_input,
                    llm=llm,
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    config={"configurable": {"thread_id": thread_id}},
                    messages=messages_for_state,
                )
                if messages_for_state is None:
                    messages_for_state = state.get("messages", [])
                response = state.get("final_response", "")
                print("Agent:", response)

                if memory and conversation_id and state.get("messages"):
                    try:
                        await memory.save_conversation_turn(
                            conversation_id,
                            state["messages"],
                            state.get("tools_invoked", []),
                        )
                    except Exception as e:
                        logger.warning("memory_save_failed", error=str(e))
        except Exception as e:
            logger.exception("agent_run_failed", error=str(e))
            print("Agent error:", e)

    if memory:
        await memory.close()


def run_cli() -> None:
    """Entry point for CLI."""
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    asyncio.run(run_conversation_loop(use_memory=True))


if __name__ == "__main__":
    run_cli()
