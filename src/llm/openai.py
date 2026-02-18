"""OpenAI provider with function calling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.llm.base import LLMProvider, LLMResponse, ToolCall
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every assistant message with tool_calls has function.arguments as a JSON string."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            normalized_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = tc.get("function") or {}
                fn = dict(fn)
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args)
                elif not isinstance(args, str):
                    fn["arguments"] = json.dumps(args) if args is not None else "{}"
                tc["function"] = fn
                normalized_calls.append(tc)
            m["tool_calls"] = normalized_calls
        out.append(m)
    return out

# Load .env from project root so OPENAI_API_KEY is available
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using chat completions with tool support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
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
        """
        Call OpenAI chat completions; if tools are provided, use function calling.
        """
        messages = _normalize_messages_for_openai(messages)
        request: dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if tools:
            request["tools"] = [{"type": "function", "function": t["function"]} for t in tools]
            request["tool_choice"] = "auto"
        timeout = kwargs.get("timeout", self.timeout)
        try:
            response = await self.client.chat.completions.create(**request, timeout=timeout)
            choice = response.choices[0] if response.choices else None
            if not choice:
                return LLMResponse(content="", tool_calls=[], finish_reason="error")
            msg = choice.message
            tool_calls_list: list[ToolCall] = []
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    import json
                    args = tc.function.arguments if hasattr(tc.function, "arguments") else "{}"
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls_list.append(
                        ToolCall(
                            id=getattr(tc, "id", ""),
                            name=tc.function.name,
                            arguments=args,
                        )
                    )
            return LLMResponse(
                content=(msg.content or "").strip(),
                tool_calls=tool_calls_list,
                finish_reason=choice.finish_reason,
            )
        except Exception as e:
            logger.exception("openai_generate_failed", error=str(e))
            raise
