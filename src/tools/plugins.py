"""Plugin system: load third-party tool modules that register with tool_registry (Phase 4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_plugin_module(module_path: str | Path) -> None:
    """Load a Python module by path (file or dotted name). Module should call tool_registry.register()."""
    path = Path(module_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)
            logger.info("plugin_loaded", path=str(path))
            return
    # Dotted module name
    try:
        importlib.import_module(module_path)
        logger.info("plugin_loaded", module=module_path)
    except Exception as e:
        logger.warning("plugin_load_failed", module=module_path, error=str(e))


def load_plugins_from_config(config: dict | None = None) -> None:
    """Load all plugin modules listed in config.tools.plugins (list of paths or dotted names)."""
    from src.utils.config import load_config
    config = config or load_config()
    plugins = config.get("tools", {}).get("plugins", [])
    for p in plugins:
        load_plugin_module(p)
