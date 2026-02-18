"""A/B testing for prompts: assign variant per conversation and record (Phase 4)."""

from __future__ import annotations

import random
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_prompt_variant(
    key: str,
    variants: dict[str, list[str]],
    variant_id: str | None = None,
) -> str:
    """
    Return one prompt variant for the given key. variant_id can force a variant (e.g. "a" or "b");
    otherwise pick randomly and log for analysis.
    """
    options = variants.get(key, [])
    if not options:
        return ""
    if variant_id and variant_id in options or (variant_id and variant_id.isdigit() and 0 <= int(variant_id) < len(options)):
        idx = int(variant_id) if variant_id.isdigit() else (options.index(variant_id) if variant_id in options else 0)
        chosen = options[idx]
    else:
        idx = random.randint(0, len(options) - 1)
        chosen = options[idx]
    logger.info("prompt_ab", key=key, variant_index=idx, variant_preview=chosen[:50] if chosen else "")
    return chosen


def get_router_prompt(config: dict | None = None) -> str:
    """Return router system prompt (optionally from A/B variants in config.prompts.router list)."""
    from src.utils.config import load_config
    from src.agent.nodes import ROUTER_SYSTEM
    config = config or load_config()
    variants = config.get("prompts", {}).get("router", [])
    if not variants:
        return ROUTER_SYSTEM
    return get_prompt_variant("router", {"router": variants}) or ROUTER_SYSTEM
