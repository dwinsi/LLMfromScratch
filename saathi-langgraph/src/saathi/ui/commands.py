"""Slash command handlers for the interactive session."""

from __future__ import annotations

import difflib
from datetime import datetime
from pathlib import Path

import pyperclip
from langchain_core.messages import BaseMessage

from saathi.memory.store import MemoryStore
from saathi.session.manager import SessionManager, SessionState
from saathi.ui.display import console, render_checkpoint_table


async def handle_context(
    args: list[str],
    state: SessionState,
    messages: list[BaseMessage],
) -> tuple[list[str], list[BaseMessage]]:
    if not args:
        state.context_paths = []
        console.print("[dim]Context scope cleared.[/dim]")
    else:
        state.context_paths = args
        console.print(f"[dim]Scoped to:[/dim] {', '.join(args)}")
        messages = []
    return state.context_paths, messages


def handle_mode(args: list[str], state: SessionState) -> None:
    valid = {"explain", "refactor", "debug"}
    if not args or args[0] == "off":
        state.mode = "default"
        console.print("[dim]Mode cleared — using default behaviour.[/dim]")
    elif args[0] in valid:
        state.mode = args[0]
        console.print(f"[green]Mode:[/green] {state.mode}")
    else:
        console.print(f"[red]Unknown mode.[/red] Valid: {', '.join(sorted(valid))}")


def handle_memory(args: list[str], memory_store: MemoryStore) -> None:
    if not args or args[0] == "list":
        data = memory_store.all()
        for scope, facts in data.items():
            console.print(f"\n[bold]{scope}[/bold]")
            if facts:
                for k, v in facts.items():
                    console.print(f"  [cyan]{k}[/cyan]: {v}")
            else:
                console.print("  [dim](empty)[/dim]")
    elif args[0] == "save" and len(args) >= 4:
        scope, key, value = args[1], args[2], " ".join(args[3:])
        memory_store.save(scope, key, value)
        console.print(f"[green]Saved[/green] [{scope}] {key}")
    elif args[0] == "delete" and len(args) == 3:
        scope, key = args[1], args[2]
        ok = memory_store.delete(scope, key)
        console.print(f"[green]Deleted[/green] {key}" if ok else f"[red]Key not found:[/red] {key}")
    elif args[0] == "clear" and len(args) == 2:
        memory_store.clear(args[1])
        console.print(f"[dim]Cleared {args[1]} memory.[/dim]")
    else:
        console.print(
            "[red]Usage:[/red] /memory list | save <scope> <key> <value> | delete <scope> <key> | clear <scope>"
        )


def handle_session(
    args: list[str],
    state: SessionState,
    messages: list[BaseMessage],
    session_mgr: SessionManager,
) -> tuple[SessionState, list[BaseMessage]] | None:
    if not args:
        console.print("[red]Usage:[/red] /session save|load|list [name]")
        return None

    cmd = args[0]
    if cmd == "list":
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        for name, saved_at in sessions:
            console.print(f"  [cyan]{name}[/cyan]  [dim]{saved_at}[/dim]")
        return None

    if cmd == "save" and len(args) >= 2:
        name = args[1]
        session_mgr.save(name, state, messages)
        console.print(f"[green]Session saved:[/green] {name}")
        return None

    if cmd == "load" and len(args) >= 2:
        name = args[1]
        try:
            loaded_state, loaded_messages = session_mgr.load(name)
            console.print(
                f"[green]Session loaded:[/green] {name} ({len(loaded_messages)} messages)"
            )
            return loaded_state, loaded_messages
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
        return None

    console.print("[red]Unknown session command.[/red]")
    return None


async def handle_rollback(args: list[str], graph, config: dict) -> bool:
    """Roll back n turns using LangGraph's checkpoint history."""
    n = int(args[0]) if args and args[0].isdigit() else 1
    history = [s async for s in graph.aget_state_history(config)]
    # history[0] is current; skip n checkpoints back
    if len(history) <= n:
        console.print("[red]Not enough history to roll back that far.[/red]")
        return False

    target = history[n]
    await graph.aupdate_state(config, target.values, as_node="agent")
    console.print(f"[green]Rolled back {n} turn(s).[/green]")
    return True


async def handle_checkpoints(graph, config: dict) -> None:
    history = [s async for s in graph.aget_state_history(config)]
    rows = []
    for cp in history:
        rows.append(
            {
                "checkpoint_id": cp.config.get("configurable", {}).get("checkpoint_id", "?"),
                "message_count": len(cp.values.get("messages", [])),
            }
        )
    render_checkpoint_table(rows)


def handle_diff(session_start_snapshots: dict[str, str]) -> None:
    changed = False
    for path_str, original in session_start_snapshots.items():
        current_path = Path(path_str)
        if not current_path.exists():
            console.print(f"[red]Deleted:[/red] {path_str}")
            changed = True
            continue
        current = current_path.read_text(encoding="utf-8", errors="replace")
        if current != original:
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{path_str}",
                tofile=f"b/{path_str}",
            )
            diff_text = "".join(diff)
            if diff_text:
                console.print(f"\n[bold]{path_str}[/bold]")
                console.print(diff_text)
                changed = True
    if not changed:
        console.print("[dim]No file changes this session.[/dim]")


def handle_export(messages: list[BaseMessage]) -> None:
    lines = [f"# Saathi Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        lines.append(f"\n**{role}**\n\n{msg.content}\n")
    fname = f"saathi-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    Path(fname).write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Exported:[/green] {fname}")


def handle_copy(last_response: str) -> None:
    if not last_response:
        console.print("[dim]Nothing to copy.[/dim]")
        return
    try:
        pyperclip.copy(last_response)
        console.print("[green]Copied to clipboard.[/green]")
    except Exception as e:
        console.print(f"[red]Copy failed:[/red] {e}")


def handle_paste() -> str:
    console.print("[dim]Paste mode — enter text, then press Ctrl+C to finish:[/dim]")
    lines: list[str] = []
    try:
        while True:
            line = input()
            lines.append(line)
    except (KeyboardInterrupt, EOFError):
        pass
    return "\n".join(lines)
