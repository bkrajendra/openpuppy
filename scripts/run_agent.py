"""Run the agent CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interfaces.cli import run_cli

if __name__ == "__main__":
    run_cli()
