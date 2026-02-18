"""LangGraph workflow: Supervisor -> Router -> Tool Executor (loop) -> Synthesizer -> END."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent import nodes
from src.agent import supervisor as sup


def route_after_router(state: AgentState) -> str:
    """Route from router: direct/clarification -> synthesizer, tool_use -> tool_executor."""
    intent = state.get("intent", "tool_use")
    if intent == "tool_use":
        return "tool_executor"
    return "synthesizer"


def route_after_tool_executor(state: AgentState) -> str:
    """Route from tool_executor: loop back or go to synthesizer."""
    intent = state.get("intent", "")
    if intent == "synthesize":
        return "synthesizer"
    return "tool_executor"


def build_graph(use_supervisor: bool = True) -> StateGraph:
    """Build the agent StateGraph; optional supervisor routes to specialized tool sets."""
    graph = StateGraph(AgentState)

    if use_supervisor:
        graph.add_node("supervisor", sup.supervisor_node)
    graph.add_node("router", nodes.router_node)
    graph.add_node("tool_executor", nodes.tool_executor_node)
    graph.add_node("synthesizer", nodes.synthesizer_node)

    if use_supervisor:
        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "router")
    else:
        graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"tool_executor": "tool_executor", "synthesizer": "synthesizer"},
    )
    graph.add_conditional_edges(
        "tool_executor",
        route_after_tool_executor,
        {"tool_executor": "tool_executor", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph


def create_compiled_graph(checkpointer: MemorySaver | None = None, use_supervisor: bool = True):
    """Build and compile the graph, optionally with checkpointer and supervisor."""
    graph = build_graph(use_supervisor=use_supervisor)
    return graph.compile(checkpointer=checkpointer)
