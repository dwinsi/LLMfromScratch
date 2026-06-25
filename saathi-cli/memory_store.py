"""
memory_store.py
===============
Persistent memory for saathi-cli.

Two layers:
  Global   ~/.saathi/memory.json    user preferences, coding style, cross-project facts
  Project  .saathi/memory.json      stack, architecture, key files for this codebase

Both are plain JSON dicts of {key: value} pairs.
The agent writes to them via save_memory / recall_memory tools.
Both are injected into the system prompt at session start so the agent
already knows saved facts without spending a tool call.

Project memory takes precedence over global memory for the same key.
"""

import json
import os
from pathlib import Path


GLOBAL_MEMORY_PATH  = Path.home() / ".saathi" / "memory.json"


def _project_memory_path(cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    return base / ".saathi" / "memory.json"


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class MemoryStore:
    """
    Manages global and project-level memory.

    Usage:
        store = MemoryStore(project_dir="./my-project")
        store.save("global", "preferred_language", "Python")
        store.save("project", "entry_point", "cli.py")
        print(store.recall_all())   # merged view, project overrides global
    """

    def __init__(self, project_dir: str | None = None):
        self.global_path  = GLOBAL_MEMORY_PATH
        self.project_path = _project_memory_path(project_dir)

    def save(self, scope: str, key: str, value: str) -> str:
        """
        Save a fact. scope must be 'global' or 'project'.
        Returns a confirmation string.
        """
        scope = scope.lower().strip()
        if scope not in ("global", "project"):
            return "Error: scope must be 'global' or 'project'."

        path = self.global_path if scope == "global" else self.project_path
        data = _load(path)
        data[key] = value
        _save(path, data)
        return f"Saved to {scope} memory: {key} = {value}"

    def delete(self, scope: str, key: str) -> str:
        """Remove a key from global or project memory."""
        scope = scope.lower().strip()
        if scope not in ("global", "project"):
            return "Error: scope must be 'global' or 'project'."

        path = self.global_path if scope == "global" else self.project_path
        data = _load(path)
        if key not in data:
            return f"Key '{key}' not found in {scope} memory."
        del data[key]
        _save(path, data)
        return f"Deleted '{key}' from {scope} memory."

    def recall_all(self) -> dict:
        """Return merged memory — project values override global for the same key."""
        merged = _load(self.global_path)
        merged.update(_load(self.project_path))
        return merged

    def recall_global(self) -> dict:
        return _load(self.global_path)

    def recall_project(self) -> dict:
        return _load(self.project_path)

    def format_for_prompt(self) -> str:
        """
        Format saved memories as a compact block for injection into the system prompt.
        Returns an empty string if nothing is saved.
        """
        global_mem  = _load(self.global_path)
        project_mem = _load(self.project_path)

        if not global_mem and not project_mem:
            return ""

        lines = ["## Remembered facts\n"]

        if global_mem:
            lines.append("### Global (applies to all projects)")
            for k, v in global_mem.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if project_mem:
            lines.append("### Project-specific")
            for k, v in project_mem.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        return "\n".join(lines)
