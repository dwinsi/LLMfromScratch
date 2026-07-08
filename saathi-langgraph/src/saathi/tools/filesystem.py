"""File system tools: read, write, patch, list."""

import subprocess
from pathlib import Path

from langchain_core.tools import tool

_MAX_READ_BYTES = 200_000
_turn_snapshots: dict[str, str] = {}


def get_turn_snapshots() -> dict[str, str]:
    return dict(_turn_snapshots)


def clear_turn_snapshots() -> None:
    _turn_snapshots.clear()


@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Use this before editing any file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        size = p.stat().st_size
        if size > _MAX_READ_BYTES:
            return f"Error: file too large ({size} bytes). Read specific sections instead."
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content. Parent directories are created automatically."""
    try:
        p = Path(path)
        if p.exists():
            _turn_snapshots[str(p.resolve())] = p.read_text(encoding="utf-8", errors="replace")
        else:
            _turn_snapshots[str(p.resolve())] = ""
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except OSError as e:
        return f"Error writing {path}: {e}"


@tool
def patch_file(path: str, diff: str) -> str:
    """Apply a unified diff patch to an existing file. Prefer this over write_file for targeted edits."""
    import tempfile

    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"

    original = p.read_text(encoding="utf-8", errors="replace")
    _turn_snapshots[str(p.resolve())] = original

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(diff)
        patch_path = tf.name

    try:
        result = subprocess.run(
            ["patch", str(p), patch_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Patch failed:\n{result.stderr}"
        return f"Patched {path} successfully"
    except FileNotFoundError:
        return "Error: 'patch' command not found. Install it or use write_file instead."
    except subprocess.TimeoutExpired:
        return "Error: patch timed out"
    finally:
        Path(patch_path).unlink(missing_ok=True)


@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory, showing file sizes."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path not found: {path}"
        if not p.is_dir():
            return f"Error: {path} is not a directory"

        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines: list[str] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                lines.append(f"[dir]  {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"[file] {entry.name}  ({size:,} bytes)")
        return "\n".join(lines) if lines else "(empty directory)"
    except OSError as e:
        return f"Error listing {path}: {e}"
