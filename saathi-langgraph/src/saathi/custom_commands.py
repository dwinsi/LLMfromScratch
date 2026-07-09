"""User-defined slash commands loaded from ``.saathi/commands/*.md``.

Each markdown file ``foo.md`` registers the command ``/foo``; the file's content
is a prompt template. When invoked, if the template contains the token ``$ARGS``
any text the user typed after the command name is substituted there; otherwise
that text is appended on a new line. Mirrors Claude Code's project commands.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

_COMMANDS_DIR = Path(".saathi") / "commands"
_ARGS_TOKEN = "$ARGS"


def load_custom_commands(directory: Path | None = None) -> dict[str, str]:
    """Return a mapping of command name -> prompt template.

    Names are the file stems, lower-cased. Missing directory returns ``{}``.
    """
    directory = directory or _COMMANDS_DIR
    commands: dict[str, str] = {}
    if not directory.is_dir():
        return commands
    for path in sorted(directory.glob("*.md")):
        with contextlib.suppress(OSError):
            commands[path.stem.lower()] = path.read_text(encoding="utf-8")
    return commands


def render_command(template: str, args: list[str]) -> str:
    """Fill a command template with the user's extra arguments."""
    extra = " ".join(args).strip()
    if _ARGS_TOKEN in template:
        return template.replace(_ARGS_TOKEN, extra)
    if extra:
        return f"{template.rstrip()}\n\n{extra}"
    return template
