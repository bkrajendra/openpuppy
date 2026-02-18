"""Supervisor pattern: route to specialized tool sets (research, code, general)."""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentState
from src.llm.base import LLMProvider
from src.tools.registry import ToolRegistry
from src.utils.logging import get_logger

logger = get_logger(__name__)

_llm: LLMProvider | None = None
_tools: ToolRegistry | None = None

# Which tools each "team" can use (subset of registry names)
TEAM_TOOLS: dict[str, list[str]] = {
    "research": ["web_search", "wikipedia_lookup", "weather", "retrieve_memory"],
    "code": ["code_executor", "calculator", "read_file", "write_file", "list_directory", "retrieve_memory"],
    "general": [],  # empty = all tools
}


def set_supervisor_dependencies(llm: LLMProvider, tools: ToolRegistry) -> None:
    global _llm, _tools
    _llm = llm
    _tools = tools


def get_tools_for_team(team: str) -> ToolRegistry | None:
    """Return registry filtered to team's tools; if general, return full registry."""
    if _tools is None:
        return None
    return _tools


def get_tool_schemas_for_team(registry: ToolRegistry, team: str) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas filtered by team."""
    all_schemas = registry.get_tool_schemas()
    allowed = TEAM_TOOLS.get(team)
    if not allowed:
        return all_schemas
    return [s for s in all_schemas if s["function"]["name"] in allowed]


SUPERVISOR_SYSTEM = """You are a supervisor. Given the user message, choose which specialized team should handle it.
- "research": web search, Wikipedia, weather, lookups, factual questions.
- "code": code execution, calculations, file read/write, scripting.
- "general": mixed or unclear; use all tools.

Reply with exactly one word: research, code, or general."""


def _minimal_messages_for_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool messages and flatten assistant+tool_calls to one assistant line (avoid circular import)."""
    out = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "")
        if role == "tool":
            i += 1
            continue
        if role == "assistant" and m.get("tool_calls"):
            content = (m.get("content") or "").strip() or "(Used tools.)"
            out.append({"role": "assistant", "content": content})
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                i += 1
            continue
        out.append({"role": m.get("role"), "content": m.get("content") or ""})
        i += 1
    return out


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Set state.team to research, code, or general for tool filtering."""
    if _llm is None:
        return {"team": "general"}
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": state.get("user_input", "")}]
    safe = _minimal_messages_for_llm(messages[-6:])
    prompt = [{"role": "system", "content": SUPERVISOR_SYSTEM}, *safe]
    response = await _llm.generate(prompt, tools=None)
    team = (response.content or "general").strip().lower()
    if team not in ("research", "code", "general"):
        team = "general"
    logger.info("supervisor", team=team)
    return {"team": team}
