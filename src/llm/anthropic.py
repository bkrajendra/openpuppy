"""Anthropic Claude provider with tool support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse, ToolCall
from src.utils.logging import get_logger

logger = get_logger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages to Anthropic format (user/assistant with content blocks)."""
    result: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            if result and result[-1].get("role") == "user":
                result[-1]["content"] = result[-1]["content"] + "\n\n[System: " + (m.get("content") or "") + "]"
            else:
                result.append({"role": "user", "content": "[System: " + (m.get("content") or "") + "]"})
            continue
        if role == "user":
            result.append({"role": "user", "content": m.get("content") or ""})
            continue
        if role == "assistant":
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []
            if not tool_calls:
                result.append({"role": "assistant", "content": content})
                continue
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                fid = tc.get("id", "")
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                blocks.append({"type": "tool_use", "id": fid, "name": name, "input": args})
            result.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            result.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": m.get("content") or ""}],
            })
    return result


def _anthropic_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function tools to Anthropic tools format."""
    out = []
    for t in openai_tools:
        fn = t.get("function", {}) or {}
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
        })
    return out


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API with tool/function calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ) -> None:
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Claude Messages API; convert OpenAI-format messages and tools."""
        anthropic_messages = _openai_messages_to_anthropic(messages)
        if not anthropic_messages:
            return LLMResponse(content="", tool_calls=[], finish_reason="error")
        system = None
        if anthropic_messages and anthropic_messages[0].get("role") == "user":
            first_content = anthropic_messages[0].get("content", "")
            if isinstance(first_content, str) and first_content.startswith("[System:"):
                system = first_content.strip("[System: ]")
                anthropic_messages = anthropic_messages[1:]
        request: dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = _anthropic_tools(tools)
        timeout = kwargs.get("timeout", self.timeout)
        try:
            response = await self.client.messages.create(**request, timeout=timeout)
        except Exception as e:
            logger.exception("anthropic_generate_failed", error=str(e))
            raise
        tool_calls_list: list[ToolCall] = []
        content_parts: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                content_parts.append(getattr(block, "text", "") or "")
            if getattr(block, "type", None) == "tool_use":
                tool_calls_list.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", None) or {},
                    )
                )
        return LLMResponse(
            content=" ".join(content_parts).strip(),
            tool_calls=tool_calls_list,
            finish_reason=getattr(response, "stop_reason", None),
        )
