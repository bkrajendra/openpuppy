"""LangGraph node implementations: router, tool_executor, synthesizer."""

from __future__ import annotations

import json
from typing import Any

from src.agent.state import AgentState
from src.llm.base import LLMProvider
from src.tools.registry import ToolRegistry
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Injected by executor
_llm: LLMProvider | None = None
_tools: ToolRegistry | None = None


def set_dependencies(llm: LLMProvider, tools: ToolRegistry) -> None:
    """Set LLM and tool registry for nodes."""
    global _llm, _tools
    _llm = llm
    _tools = tools


def _get_llm() -> LLMProvider:
    if _llm is None:
        raise RuntimeError("Nodes not initialized: set_dependencies(llm, tools) first")
    return _llm


def _get_tools() -> ToolRegistry:
    if _tools is None:
        raise RuntimeError("Nodes not initialized: set_dependencies(llm, tools) first")
    return _tools


ROUTER_SYSTEM = """You are a router. Given the user message and conversation so far, output exactly one word: the intent.
- "direct": answer from knowledge only, no tools needed (greetings, simple facts, opinions). Do NOT use direct for weather, current conditions, or anything that needs live/real-time data.
- "tool_use": the user needs information from a tool. Always use tool_use for: weather, current conditions, forecasts, web search, calculations, file operations, Wikipedia lookups, or anything requiring up-to-date or external data.
- "clarification": the request is ambiguous or you need more information from the user.

Reply with only one word: direct, tool_use, or clarification."""


def _messages_for_llm_without_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a message list safe for LLM calls with tools=None.
    OpenAI rejects role='tool' unless preceded by assistant with tool_calls.
    We drop standalone 'tool' messages and fold assistant+tool_calls and the following
    tool results into a single assistant message so the LLM sees the actual tool output.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "")
        if role == "tool":
            i += 1
            continue
        if role == "assistant" and m.get("tool_calls"):
            # Collect this assistant message and the following tool result(s)
            content = (m.get("content") or "").strip()
            tool_results: list[str] = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_content = messages[j].get("content") or ""
                if tool_content:
                    tool_results.append(tool_content)
                j += 1
            if tool_results:
                content = content + "\n\n[Tool results]:\n" + "\n".join(tool_results) if content else "[Tool results]:\n" + "\n".join(tool_results)
            else:
                content = content or "(Used tools in this turn.)"
            out.append({"role": "assistant", "content": content})
            i = j
            continue
        out.append({k: v for k, v in m.items() if k in ("role", "content")})
        i += 1
    return out


async def router_node(state: AgentState) -> dict[str, Any]:
    """Analyze user intent and set intent for conditional edge."""
    llm = _get_llm()
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": state.get("user_input", "")}]
    safe = _messages_for_llm_without_tools(messages[-20:])
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        *safe[-6:],
    ]
    response = await llm.generate(router_messages, tools=None)
    intent = (response.content or "tool_use").strip().lower()
    if intent not in ("direct", "tool_use", "clarification"):
        intent = "tool_use"
    logger.info("router", intent=intent)
    return {"intent": intent}


SYNTHESIZER_SYSTEM = """You are a helpful assistant. The conversation includes [Tool results] with real data (e.g. weather, search results). Your job is to turn that data into a clear, direct answer for the user. Use the tool results as the source of truth; do not say you are unable to provide the information when tool results are present. Do not mention "tool", "API", or internal steps. Be concise and natural."""

TOOL_EXECUTOR_SYSTEM = """You have access to tools. Use them when the user asks for: weather or current conditions (use the weather tool with the location they asked about), web search, calculations, file operations, Wikipedia, or storing/retrieving memory. Do not refuse to use tools; call the appropriate tool with the correct arguments."""


async def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Call LLM with tools; execute any tool_calls and append results to messages.
    Enforce max_iterations; if exceeded or no tool_calls, intent set to synthesize.
    """
    llm = _get_llm()
    tools_reg = _get_tools()
    messages = list(state.get("messages", []))
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    if iteration >= max_iter:
        logger.warning("max_iterations_reached", iteration=iteration)
        return {
            "intent": "synthesize",
            "final_response": "I've hit the step limit. Here's what I have so far.",
        }

    if not messages:
        messages = [{"role": "user", "content": state.get("user_input", "")}]

    # Prepend system hint; filter tools by supervisor team if set (Phase 4)
    tool_messages = [{"role": "system", "content": TOOL_EXECUTOR_SYSTEM}, *messages]
    team = state.get("team")
    if team:
        try:
            from src.agent.supervisor import get_tool_schemas_for_team
            tool_schemas = get_tool_schemas_for_team(tools_reg, team)
        except Exception:
            tool_schemas = tools_reg.get_tool_schemas()
    else:
        tool_schemas = tools_reg.get_tool_schemas()
    response = await llm.generate(tool_messages, tools=tool_schemas)

    if not response.tool_calls:
        return {"intent": "synthesize"}

    # Execute all tool calls (concurrent)
    import asyncio
    results = await asyncio.gather(
        *[tools_reg.execute_tool(tc.name, tc.arguments) for tc in response.tool_calls]
    )

    # Append assistant message with tool_calls (OpenAI format). Arguments must be a JSON string for the API.
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    assistant_msg["tool_calls"] = [
        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments}}
        for tc in response.tool_calls
    ]
    new_messages: list[dict[str, Any]] = [assistant_msg]

    # Append tool results (OpenAI format) - content as short text for LLM (prefer summary when present)
    for tc, res in zip(response.tool_calls, results):
        if res.success and res.data is not None:
            if isinstance(res.data, dict) and res.data.get("summary"):
                content = res.data["summary"]
            elif isinstance(res.data, str):
                content = res.data[:2000]
            else:
                content = str(res.data)[:2000]
        else:
            content = f"Error: {res.error}" if res.error else "Tool failed."
        new_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": content,
        })

    tools_invoked = list(state.get("tools_invoked", []))
    for tc, r in zip(response.tool_calls, results):
        dump = r.model_dump() if hasattr(r, "model_dump") else {"success": getattr(r, "success", False), "data": getattr(r, "data", None)}
        dump["tool_name"] = tc.name
        tools_invoked.append(dump)

    return {
        "messages": new_messages,
        "tools_invoked": tools_invoked,
        "iteration_count": iteration + 1,
    }


async def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Produce final_response from conversation and tool results."""
    llm = _get_llm()
    messages = list(state.get("messages", []))
    if not messages:
        messages = [{"role": "user", "content": state.get("user_input", "")}]
    safe = _messages_for_llm_without_tools(messages)
    synth_messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        *safe,
    ]
    response = await llm.generate(synth_messages, tools=None)
    final = (response.content or "").strip()
    logger.info("synthesizer", response_length=len(final))
    return {"final_response": final}
