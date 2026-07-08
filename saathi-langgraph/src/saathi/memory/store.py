"""Persistent two-scope (global + project) memory store."""

import json
from pathlib import Path


class MemoryStore:
    """
    Two-tier JSON memory: global (~/.saathi/memory.json) and project (.saathi/memory.json).
    Project keys override global keys when both are present.
    """

    def __init__(self) -> None:
        self._global_path = Path.home() / ".saathi" / "memory.json"
        self._project_path = Path(".saathi") / "memory.json"

    def _load(self, path: Path) -> dict[str, str]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self, path: Path, data: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def all(self) -> dict[str, dict[str, str]]:
        return {
            "global": self._load(self._global_path),
            "project": self._load(self._project_path),
        }

    def get(self, scope: str, key: str) -> str | None:
        path = self._global_path if scope == "global" else self._project_path
        return self._load(path).get(key)

    def save(self, scope: str, key: str, value: str) -> None:
        path = self._global_path if scope == "global" else self._project_path
        data = self._load(path)
        data[key] = value
        self._save(path, data)

    def delete(self, scope: str, key: str) -> bool:
        path = self._global_path if scope == "global" else self._project_path
        data = self._load(path)
        if key in data:
            del data[key]
            self._save(path, data)
            return True
        return False

    def clear(self, scope: str) -> None:
        path = self._global_path if scope == "global" else self._project_path
        self._save(path, {})

    def format_for_prompt(self) -> str:
        merged: dict[str, str] = {}
        merged.update(self._load(self._global_path))
        merged.update(self._load(self._project_path))
        if not merged:
            return ""
        lines = [f"  {k}: {v}" for k, v in merged.items()]
        return "\n".join(lines)
