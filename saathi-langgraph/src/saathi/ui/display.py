"""Rich-based terminal display helpers."""

import contextlib
import sys

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Ensure our Unicode UI (box drawing, ✓/✗, spinner glyphs, ↳) survives on
# Windows consoles / pipes that default to a legacy codepage like cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        with contextlib.suppress(ValueError, OSError):
            _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()

BANNER = """\
 ____              _   _     _
/ ___|  __ _  __ _| |_| |__ (_)
\\___ \\ / _` |/ _` | __| '_ \\| |
 ___) | (_| | (_| | |_| | | | |
|____/ \\__,_|\\__,_|\\__|_| |_|_|

[dim]your local coding companion[/dim]  •  LangGraph + Ollama
"""

HELP_TEXT = r"""
## Commands

| Input | Action |
|-------|--------|
| `<task>` | Run a task through the agent |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |
| `/init` | Scan the repo and generate a SAATHI.md |
| `/revise-saathi-md` | Update SAATHI.md with this session's learnings |
| `/commit` | Review changes and create a git commit |
| `/doctor` | Health check: Ollama, model, memory, git |
| `/commands` | List custom commands from `.saathi/commands/` |
| `/context <path> ...` | Scope agent to files/folders |
| `/context` | Clear scope |
| `/compact` | Summarize history to free tokens |
| `/rollback [n]` | Undo last n turns (LangGraph checkpoints) |
| `/checkpoints` | List all session checkpoints |
| `/diff` | Show all file changes this session |
| `/export` | Save conversation to Markdown |
| `/copy` | Copy last response to clipboard |
| `/paste` | Multi-line input mode |
| `/model <id>` | Switch Ollama model |
| `/mode explain\|refactor\|debug` | Set agent behaviour preset |
| `/mode` | Show current mode |
| `/memory list` | Show all remembered facts |
| `/memory save <scope> <key> <value>` | Persist a fact |
| `/memory delete <scope> <key>` | Delete a fact |
| `/session save <name>` | Save session |
| `/session load <name>` | Restore a session |
| `/session list` | List saved sessions |
"""


def print_banner(model_id: str) -> None:
    console.print(BANNER, style="cyan bold")
    console.print(f"  Model: [green]{model_id}[/green]  •  Type [bold]help[/bold] for commands\n")


def print_help() -> None:
    console.print(Markdown(HELP_TEXT))


def render_tool_call(tool_name: str, args: dict) -> None:
    args_text = "  ".join(
        f"[cyan]{k}[/cyan]=[yellow]{str(v)[:80]}[/yellow]" for k, v in args.items()
    )
    console.print(f"  [dim]→ {tool_name}[/dim]  {args_text}", highlight=False)


def render_tool_result(tool_name: str, result: str) -> None:
    preview = result[:300] + ("…" if len(result) > 300 else "")
    console.print(
        Panel(preview, title=f"[dim]{tool_name}[/dim]", border_style="dim", box=box.SIMPLE),
        overflow="fold",
    )


def render_checkpoint_table(checkpoints: list[dict]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Thread / Checkpoint ID", style="cyan")
    table.add_column("Messages", justify="right")
    for i, cp in enumerate(checkpoints, 1):
        table.add_row(
            str(i),
            cp.get("checkpoint_id", "—")[:32],
            str(cp.get("message_count", "?")),
        )
    console.print(table)
