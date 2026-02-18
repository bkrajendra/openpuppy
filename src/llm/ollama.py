"""Ollama provider (local LLM, OpenAI-compatible API)."""

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

_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
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
        """Call Ollama chat completions; supports tools if the model does."""
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
        except Exception as e:
            logger.warning("ollama_generate_failed", error=str(e), base_url=self.base_url)
            raise
        choice = response.choices[0] if response.choices else None
        if not choice:
            return LLMResponse(content="", tool_calls=[], finish_reason="error")
        msg = choice.message
        tool_calls_list: list[ToolCall] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = getattr(tc.function, "arguments", "{}")
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
