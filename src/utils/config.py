"""Configuration loading from YAML and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load agent config from YAML file with optional env var overrides.

    Args:
        config_path: Path to agent_config.yaml. Defaults to config/agent_config.yaml.

    Returns:
        Nested config dict.

    Example:
        >>> cfg = load_config()
        >>> cfg["agent"]["max_iterations"]
        5
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "agent_config.yaml"
    path = Path(config_path)
    if not path.exists():
        return _default_config()
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Env overrides
    if db_path := os.getenv("AGENT_MEMORY_DB"):
        config.setdefault("memory", {})["database_path"] = db_path
    if api_key := os.getenv("OPENAI_API_KEY"):
        config.setdefault("llm", {}).setdefault("primary", {})["api_key"] = api_key
    if model := os.getenv("OPENAI_MODEL"):
        config.setdefault("llm", {}).setdefault("primary", {})["model"] = model
    return config


def get_config_path(config_path: str | Path | None = None) -> Path:
    """Return the path to the config file used for load/save."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "agent_config.yaml"
    return Path(config_path)


def save_config(config: dict[str, Any], config_path: str | Path | None = None) -> None:
    """Write config dict to YAML file. Used by admin UI to persist edits."""
    path = get_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _default_config() -> dict[str, Any]:
    """Default config when no file is present."""
    return {
        "agent": {"name": "IntelligentAgent", "max_iterations": 5, "timeout_seconds": 120},
        "llm": {
            "primary": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        },
        "tools": {"enabled": ["web_search", "code_executor"]},
        "memory": {"database_path": "./data/agent_memory.db"},
        "interfaces": {"cli": {"enabled": True}},
    }
