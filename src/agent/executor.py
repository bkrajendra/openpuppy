"""Agent executor: wires LLM + tools, compiles graph, exposes invoke/astream."""

from __future__ import annotations

from typing import Any, AsyncIterator

from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import build_graph
from src.agent import nodes
from src.agent.state import AgentState
from src.llm.base import LLMProvider
from src.llm.openai import OpenAIProvider
from src.tools.registry import tool_registry

# Register all tools on import
import src.tools.web_search  # noqa: F401
import src.tools.code_executor  # noqa: F401
import src.tools.file_operations  # noqa: F401
import src.tools.wikipedia  # noqa: F401
import src.tools.calculator  # noqa: F401
import src.tools.weather  # noqa: F401
import src.tools.memory_tools  # noqa: F401
import src.tools.compose  # noqa: F401
from src.tools.plugins import load_plugins_from_config
from src.tools.custom_tools import load_and_register_all_custom_tools

load_plugins_from_config()
load_and_register_all_custom_tools()


def create_agent(
    llm: LLMProvider | None = None,
    checkpointer: MemorySaver | None = None,
    use_supervisor: bool = True,
) -> Any:
    """
    Create a compiled agent graph with injected LLM and tools.
    If use_supervisor=True (default), adds supervisor node to route to research/code/general tool sets.
    """
    if llm is None:
        llm = OpenAIProvider()
    nodes.set_dependencies(llm, tool_registry)
    from src.agent import supervisor as sup
    sup.set_supervisor_dependencies(llm, tool_registry)
    from src.agent.graph import build_graph
    graph = build_graph(use_supervisor=use_supervisor)
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def _initial_state(
    user_input: str,
    conversation_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build initial state for a turn."""
    if not messages:
        messages = [{"role": "user", "content": user_input}]
    return {
        "user_input": user_input,
        "messages": messages,
        "intent": "",
        "tools_invoked": [],
        "iteration_count": 0,
        "max_iterations": 5,
        "final_response": "",
        "metadata": {"conversation_id": conversation_id or "default", **kwargs.get("metadata", {})},
    }


async def run_agent(
    user_input: str,
    *,
    llm: LLMProvider | None = None,
    conversation_id: str | None = None,
    thread_id: str = "default",
    config: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Run the agent for one user message and return the final state.

    Example:
        >>> state = await run_agent("What is 2+2?")
        >>> print(state["final_response"])
    """
    checkpointer = MemorySaver()
    agent = create_agent(llm=llm, checkpointer=checkpointer)
    config = config or {}
    config["configurable"] = config.get("configurable", {})
    config["configurable"]["thread_id"] = thread_id
    initial = _initial_state(user_input, conversation_id=conversation_id, messages=messages)
    final_state = await agent.ainvoke(initial, config=config)
    return final_state


async def stream_agent(
    user_input: str,
    *,
    llm: LLMProvider | None = None,
    thread_id: str = "default",
    config: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream graph state updates until END."""
    checkpointer = MemorySaver()
    agent = create_agent(llm=llm, checkpointer=checkpointer)
    config = config or {"configurable": {"thread_id": thread_id}}
    initial = _initial_state(user_input)
    async for event in agent.astream(initial, config=config):
        yield event
