"""Backup and restore agent data (SQLite DB + vector store directory). Phase 3."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def _default_paths():
    root = Path(__file__).resolve().parent.parent
    return {
        "db": root / "data" / "agent_memory.db",
        "vector": root / "data" / "vector_store",
    }


def backup(output_dir: Path, db_path: Path | None = None, vector_path: Path | None = None) -> None:
    """Copy SQLite DB and vector_store to output_dir with timestamp."""
    paths = _default_paths()
    db = db_path or paths["db"]
    vec = vector_path or paths["vector"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if db.exists():
        shutil.copy2(db, output_dir / "agent_memory.db")
        print(f"Backed up DB to {output_dir / 'agent_memory.db'}")
    if vec.exists():
        shutil.copytree(vec, output_dir / "vector_store", dirs_exist_ok=True)
        print(f"Backed up vector_store to {output_dir / 'vector_store'}")
    if not db.exists() and not vec.exists():
        print("No data found to backup.")


def restore(input_dir: Path, db_path: Path | None = None, vector_path: Path | None = None) -> None:
    """Restore SQLite DB and vector_store from input_dir."""
    paths = _default_paths()
    db_dest = db_path or paths["db"]
    vec_dest = vector_path or paths["vector"]
    input_dir = Path(input_dir)
    db_src = input_dir / "agent_memory.db"
    vec_src = input_dir / "vector_store"
    if db_src.exists():
        db_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_src, db_dest)
        print(f"Restored DB to {db_dest}")
    if vec_src.exists():
        vec_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vec_src, vec_dest, dirs_exist_ok=True)
        print(f"Restored vector_store to {vec_dest}")
    if not db_src.exists() and not vec_src.exists():
        print("No backup found in input dir.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup or restore agent data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backup", help="Backup DB and vector store")
    sub.add_parser("restore", help="Restore from backup dir")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/backups"),
                        help="Backup output dir (default: data/backups)")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/backups"),
                        help="Restore input dir (default: data/backups)")
    args = parser.parse_args()
    if args.command == "backup":
        backup(args.output)
    else:
        restore(args.input)


if __name__ == "__main__":
    main()
