"""LLM provider factory from config with optional fallback."""

from __future__ import annotations

import os
from typing import Any

from src.llm.base import LLMProvider
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _create_provider(provider: str, **kwargs: Any) -> LLMProvider:
    """Create a single LLM provider by name."""
    provider = (provider or "openai").lower()
    if provider == "openai":
        from src.llm.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model=kwargs.get("model", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
            timeout=kwargs.get("timeout", 60.0),
        )
    if provider == "anthropic":
        from src.llm.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
            model=kwargs.get("model", "claude-sonnet-4-20250514"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
            timeout=kwargs.get("timeout", 60.0),
        )
    if provider == "ollama":
        from src.llm.ollama import OllamaProvider
        return OllamaProvider(
            base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=kwargs.get("model", "llama3.2"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
            timeout=kwargs.get("timeout", 120.0),
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


class FallbackLLMProvider(LLMProvider):
    """Wraps primary and fallback; on primary failure, tries fallback once."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(self, messages: list, tools: list | None = None, stream: bool = False, **kwargs: Any):
        try:
            return await self.primary.generate(messages, tools=tools, stream=stream, **kwargs)
        except Exception as e:
            logger.warning("llm_primary_failed", error=str(e))
            if self.fallback:
                logger.info("llm_fallback_try")
                return await self.fallback.generate(messages, tools=tools, stream=stream, **kwargs)
            raise


def get_llm_from_config(config: dict | None = None) -> LLMProvider:
    """
    Build LLM from config (llm.primary, optional llm.fallback).
    Uses OPENAI_API_KEY / ANTHROPIC_API_KEY from env if not in config.
    """
    config = config or load_config()
    llm_cfg = config.get("llm", {})
    primary_cfg = dict(llm_cfg.get("primary") or {"provider": "openai", "model": "gpt-4o-mini"})
    fallback_cfg = llm_cfg.get("fallback")
    provider_name = primary_cfg.pop("provider", "openai")
    primary = _create_provider(provider_name, **primary_cfg)
    fallback = None
    if fallback_cfg:
        fc = dict(fallback_cfg)
        fallback_name = fc.pop("provider", "ollama")
        fallback = _create_provider(fallback_name, **fc)
        return FallbackLLMProvider(primary, fallback)
    return primary
