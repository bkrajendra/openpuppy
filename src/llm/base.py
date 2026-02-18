"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Single tool call from LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Anthropic, Ollama)."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response; optionally with tool schemas for function calling."""
        pass
