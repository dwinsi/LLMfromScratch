"""Discovery and loading of SAATHI.md project instruction files."""

import contextlib
from pathlib import Path

_FILENAME = "SAATHI.md"


def find_project_instructions(start: Path | None = None) -> str:
    """
    Walk from `start` (default: cwd) up to the home directory, collecting every
    SAATHI.md found. Files closer to the project root are appended last so their
    guidance takes precedence when the model reads top-to-bottom.
    Returns the concatenated content, or "" if none found.
    """
    start = (start or Path.cwd()).resolve()
    home = Path.home().resolve()

    found: list[tuple[Path, str]] = []
    current = start
    while True:
        candidate = current / _FILENAME
        if candidate.is_file():
            with contextlib.suppress(OSError):
                text = candidate.read_text(encoding="utf-8", errors="replace")
                found.append((candidate, text))
        if current == home or current.parent == current:
            break
        current = current.parent

    if not found:
        return ""

    # Nearest-to-cwd wins → place it last. `found` is ordered cwd→root, so reverse.
    found.reverse()
    blocks = [f"# from {path}\n\n{content}".strip() for path, content in found]
    return "\n\n---\n\n".join(blocks)


def instructions_source(start: Path | None = None) -> Path | None:
    """Return the nearest SAATHI.md path, or None — used for the startup notice."""
    start = (start or Path.cwd()).resolve()
    home = Path.home().resolve()
    current = start
    while True:
        candidate = current / _FILENAME
        if candidate.is_file():
            return candidate
        if current == home or current.parent == current:
            break
        current = current.parent
    return None
