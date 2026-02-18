"""Agent state schema for LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _append_messages(left: list[dict[str, Any]], right: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Reducer: append right (list or single message) to left."""
    if isinstance(right, list):
        return left + right
    return left + [right]


class AgentState(TypedDict, total=False):
    """
    State for the agent graph.
    - messages: conversation history (reducer appends)
    - user_input: latest user text
    - intent: routing decision
    - tools_invoked: list of tool results this run
    - iteration_count: current tool loop iteration
    - max_iterations: hard cap (default 5)
    - final_response: last response text
    - metadata: arbitrary dict
    """

    messages: Annotated[list[dict[str, Any]], _append_messages]
    user_input: str
    intent: str  # "direct", "tool_use", "multi_step", "clarification", "synthesize"
    tools_invoked: list[dict[str, Any]]  # ToolResult-like dicts
    iteration_count: int
    max_iterations: int
    final_response: str
    metadata: dict[str, Any]
    team: str  # supervisor: "research" | "code" | "general"
