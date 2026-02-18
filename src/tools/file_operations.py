"""File operations: read_file, write_file, list_directory (sandboxed to allowed path)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from src.tools.base import ToolResult
from src.tools.registry import tool_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default allowed base path: project data/ or current dir; override via env ALLOWED_FILE_PATH
_ALLOWED_BASE = Path(os.getenv("ALLOWED_FILE_PATH", "data")).resolve()


def _resolve_allowed(path: str | Path) -> Path:
    """Resolve path and ensure it is under the allowed base."""
    p = Path(path).resolve()
    try:
        p.relative_to(_ALLOWED_BASE)
        return p
    except ValueError:
        return _ALLOWED_BASE / path.lstrip("/\\")


READ_FILE_SCHEMA = {
    "properties": {
        "path": {"type": "string", "description": "Relative path to file (under allowed directory)"},
        "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
    },
    "required": ["path"],
}


@tool_registry.register(
    name="read_file",
    description="Read contents of a text file. Path is relative to the allowed workspace (e.g. data/).",
    category="file_operations",
    parameters_schema=READ_FILE_SCHEMA,
)
async def read_file(path: str, encoding: str = "utf-8") -> ToolResult:
    start = time.perf_counter()
    try:
        full = _resolve_allowed(path)
        if not full.exists():
            return ToolResult(success=False, data=None, error=f"File not found: {path}", execution_time_ms=(time.perf_counter() - start) * 1000)
        if not full.is_file():
            return ToolResult(success=False, data=None, error=f"Not a file: {path}", execution_time_ms=(time.perf_counter() - start) * 1000)
        text = full.read_text(encoding=encoding)
        return ToolResult(success=True, data={"path": path, "content": text[:50000], "length": len(text)}, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.exception("read_file_failed", path=path, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)


WRITE_FILE_SCHEMA = {
    "properties": {
        "path": {"type": "string", "description": "Relative path to file (under allowed directory)"},
        "content": {"type": "string", "description": "Content to write"},
        "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
    },
    "required": ["path", "content"],
}


@tool_registry.register(
    name="write_file",
    description="Write text content to a file. Path is relative to the allowed workspace. Creates parent dirs if needed.",
    category="file_operations",
    parameters_schema=WRITE_FILE_SCHEMA,
)
async def write_file(path: str, content: str, encoding: str = "utf-8") -> ToolResult:
    start = time.perf_counter()
    try:
        full = _resolve_allowed(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding=encoding)
        return ToolResult(success=True, data={"path": path, "bytes_written": len(content.encode(encoding))}, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.exception("write_file_failed", path=path, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)


LIST_DIR_SCHEMA = {
    "properties": {
        "path": {"type": "string", "description": "Relative directory path (default: .)"},
    },
    "required": [],
}


@tool_registry.register(
    name="list_directory",
    description="List files and subdirectories in a directory. Path is relative to the allowed workspace.",
    category="file_operations",
    parameters_schema=LIST_DIR_SCHEMA,
)
async def list_directory(path: str = ".") -> ToolResult:
    start = time.perf_counter()
    try:
        full = _resolve_allowed(path)
        if not full.exists():
            return ToolResult(success=False, data=None, error=f"Directory not found: {path}", execution_time_ms=(time.perf_counter() - start) * 1000)
        if not full.is_dir():
            return ToolResult(success=False, data=None, error=f"Not a directory: {path}", execution_time_ms=(time.perf_counter() - start) * 1000)
        entries = [{"name": p.name, "type": "dir" if p.is_dir() else "file"} for p in sorted(full.iterdir())]
        return ToolResult(success=True, data={"path": path, "entries": entries[:200]}, execution_time_ms=(time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.exception("list_directory_failed", path=path, error=str(e))
        return ToolResult(success=False, data=None, error=str(e), execution_time_ms=(time.perf_counter() - start) * 1000)
