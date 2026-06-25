"""
Command handlers for saathi-cli.

Each slash command in the interactive session is handled by a dedicated function here.
cli.py owns the main loop and agent streaming; this module owns everything else.

SessionState holds the mutable session variables so handlers can modify them in-place
without needing to return multiple values.
"""

import difflib
import itertools
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax

from langchain_core.messages import AIMessage, HumanMessage

from agent import load_llm, build_agent
from memory_store import MemoryStore
from system_prompt import MODE_ADDENDA
from tools import set_memory_store


console = Console()


# ---------------------------------------------------------------------------
# Input history navigation (up/down arrow keys in the prompt)
# ---------------------------------------------------------------------------

try:
    import readline          # Unix stdlib
except ImportError:
    try:
        import pyreadline3 as readline   # Windows
    except ImportError:
        readline = None


# ---------------------------------------------------------------------------
# Thinking spinner
# ---------------------------------------------------------------------------

THINKING_PHRASES = [
    # English — cognitive
    "thinking", "reasoning", "pondering", "reflecting", "contemplating",
    "philosophising", "puzzling", "deliberating", "cogitating", "ruminating",
    "simmering", "reticulating", "extrapolating", "hypothesising", "deducing",
    "inferring", "calculating", "synthesising", "analysing", "musing",
    "speculating", "theorising", "reckoning", "evaluating", "scrutinising",
    "deciphering", "untangling", "connecting dots", "cross-referencing",
    "weighing options", "considering", "imagining", "envisioning", "intending",
    "processing", "computing", "crunching", "inspecting", "investigating",
    "exploring", "probing", "dissecting", "mulling", "noodling",
    "brain-storming", "free-associating", "pattern-matching", "bootstrapping",
    "triangulating", "interpolating", "approximating", "estimating",

    # English — whimsical / Claude-style
    "reticulating splines", "herding thoughts", "sharpening pencils",
    "consulting the void", "staring into the abyss", "counting neurons",
    "untying knots", "brewing ideas", "distilling wisdom", "aligning vectors",
    "squinting at tokens", "reading the tea leaves", "following the thread",
    "chasing the rabbit", "building the plane mid-flight",

    # Hindi — everyday thinking words (transliterated)
    "soch raha hoon",            # thinking
    "samajh raha hoon",          # understanding
    "vichar kar raha hoon",      # contemplating
    "dhundh raha hoon",          # searching
    "padhh raha hoon",           # reading
    "sun raha hoon",             # listening
    "yaad kar raha hoon",        # remembering
    "andaza laga raha hoon",     # estimating
    "khoj raha hoon",            # exploring / researching
    "hisab laga raha hoon",      # calculating
    "bujh raha hoon",            # figuring out
    "dekh raha hoon",            # looking / examining
    "samjha raha hoon",          # comprehending
    "guna raha hoon",            # multiplying / pondering deeply
    "taash khel raha hoon",      # playing cards (thinking strategically)
    "mann mein soch raha hoon",  # thinking in the mind
    "jugaad laga raha hoon",     # improvising a clever solution
    "dimaag chala raha hoon",    # running the brain
    "ulajhan suljha raha hoon",  # untangling the problem
    "raaz khol raha hoon",       # unravelling the mystery
]


class ThinkingSpinner:
    """Displays a Status spinner that cycles through witty phrases on a timer."""

    def __init__(self, interval: float = 2.0):
        self._phrases    = itertools.cycle(THINKING_PHRASES)
        self._interval   = interval
        self._status: Status | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _rotate(self):
        while not self._stop_event.wait(self._interval):
            if self._status:
                self._status.update(f"[bold cyan]{next(self._phrases)}…[/bold cyan]")

    def start(self):
        self._stop_event.clear()
        self._status = console.status(
            f"[bold cyan]{next(self._phrases)}…[/bold cyan]",
            spinner="dots",
        )
        self._status.start()
        self._thread = threading.Thread(target=self._rotate, daemon=True)
        self._thread.start()

    def update(self, message: str):
        """Pin a specific message (e.g. tool name) instead of cycling."""
        if self._status:
            self._status.update(message)

    def stop(self):
        self._stop_event.set()
        if self._status:
            self._status.stop()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Mutable session variables shared between the main loop and command handlers."""
    model_id:       str
    history:        list       = field(default_factory=list)
    checkpoints:    list       = field(default_factory=list)
    context_paths:  list       = field(default_factory=list)
    current_mode:   str        = ""
    final_answer:   str | None = None
    memory:         Any        = None   # MemoryStore
    llm:            Any        = None   # ChatOllama
    agent_executor: Any        = None   # CompiledStateGraph


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def resolve_paths(paths: list[str]) -> list[str]:
    """Resolve each path to absolute and warn if it does not exist."""
    resolved = []
    for p in paths:
        abs_path = os.path.abspath(p)
        if not os.path.exists(abs_path):
            console.print(f"  [yellow]Warning:[/yellow] path does not exist: [dim]{abs_path}[/dim]")
        resolved.append(abs_path)
    return resolved


def print_context(context_paths: list[str]) -> None:
    if context_paths:
        lines = "\n".join(f"  [cyan]{p}[/cyan]" for p in context_paths)
        console.print(f"[bold]Context scope:[/bold]\n{lines}\n")
    else:
        console.print("[bold]Context scope:[/bold] [dim](none — agent works unrestricted)[/dim]\n")


def print_memory_table(memory_dict: dict, scope_name: str = "Memory") -> None:
    if not memory_dict:
        console.print(f"[dim]{scope_name} is empty.[/dim]")
        return
    from rich.table import Table
    table = Table(title=scope_name, show_header=True, header_style="bold cyan")
    table.add_column("Key",   style="cyan")
    table.add_column("Value", style="green")
    for key, value in memory_dict.items():
        display = str(value)[:100] + "…" if len(str(value)) > 100 else str(value)
        table.add_row(key, display)
    console.print(table)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_clear(state: SessionState) -> None:
    state.history = []
    state.agent_executor = build_agent(
        state.llm, state.context_paths or None,
        state.memory.format_for_prompt(), state.current_mode,
    )
    console.print("[dim]Conversation cleared.[/dim]\n")


def handle_compact(state: SessionState) -> None:
    if not state.history:
        console.print("[dim]Nothing to compact — conversation is empty.[/dim]\n")
        return

    msgs_before = len(state.history)
    console.print("[dim]Compacting conversation…[/dim]")
    spinner = ThinkingSpinner()
    spinner.start()
    try:
        history_text = "\n\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in state.history
        )
        summarise_prompt = (
            "Summarise the following conversation between a developer and a coding assistant. "
            "Your summary will replace the full history so the assistant can continue the session "
            "with less context. Preserve everything that matters:\n"
            "- Every file that was read, created, or modified (include file names and what changed)\n"
            "- Every decision made and the reasoning behind it\n"
            "- Any errors encountered and how they were resolved\n"
            "- Any facts or preferences the user stated\n"
            "- The current state of the work (what is done, what is still pending)\n\n"
            "Be concise but complete. Use bullet points. Do not omit file names or code details.\n\n"
            f"Conversation:\n{history_text}"
        )
        response = state.llm.invoke([HumanMessage(content=summarise_prompt)])
        summary  = response.content.strip()
        state.history = [HumanMessage(content=f"[Conversation summary]\n{summary}")]
    finally:
        spinner.stop()

    console.print(f"[green]Compacted[/green] {msgs_before} message(s) → 1 summary message.\n")
    console.print(Panel(
        Markdown(summary),
        title="[dim]summary[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()


def handle_context(state: SessionState, path_args: list[str]) -> None:
    state.context_paths = resolve_paths(path_args) if path_args else []
    print_context(state.context_paths)
    project_dir  = state.context_paths[0] if state.context_paths else os.getcwd()
    state.memory = MemoryStore(project_dir=project_dir)
    set_memory_store(state.memory)
    state.history = []
    state.agent_executor = build_agent(
        state.llm, state.context_paths or None,
        state.memory.format_for_prompt(), state.current_mode,
    )


def handle_diff(state: SessionState) -> None:
    any_diff = False
    for cp in state.checkpoints:
        for path, original in cp["files"].items():
            if original is None or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                current = fh.read()
            if current == original:
                continue
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{os.path.basename(path)}",
                tofile=f"b/{os.path.basename(path)}",
            ))
            if diff_lines:
                any_diff = True
                console.print(Rule(f"[dim]{path}[/dim]", style="dim"))
                console.print(Syntax("".join(diff_lines), "diff", theme="monokai", word_wrap=True))
    if not any_diff:
        console.print("[dim]No file changes recorded across checkpoints.[/dim]\n")
    else:
        console.print()


def handle_export(state: SessionState) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath  = os.path.join(os.getcwd(), f"saathi-export-{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as fh:
        for msg in state.history:
            if isinstance(msg, HumanMessage):
                fh.write(f"## You\n\n{msg.content}\n\n")
            elif isinstance(msg, AIMessage):
                fh.write(f"## Saathi\n\n{msg.content}\n\n")
    console.print(f"[green]Exported conversation to:[/green] [cyan]{filepath}[/cyan]\n")


def handle_copy(state: SessionState) -> None:
    if state.final_answer is None:
        console.print("[dim]Nothing to copy yet — run a task first.[/dim]\n")
        return
    try:
        import pyperclip
        pyperclip.copy(state.final_answer)
        console.print("[green]Last response copied to clipboard.[/green]\n")
    except Exception:
        console.print("[yellow]Clipboard not available in this environment.[/yellow]\n")


def handle_paste() -> str | None:
    """Collect multi-line input. Returns the joined text or None if cancelled/empty."""
    console.print("[dim]Enter your message (blank line to send, Ctrl+C to cancel):[/dim]")
    collected = []
    try:
        while True:
            line = input()
            if line == "":
                break
            collected.append(line)
    except KeyboardInterrupt:
        console.print("\n[dim]Paste cancelled.[/dim]\n")
        return None
    if not collected:
        console.print("[dim]Empty input — nothing sent.[/dim]\n")
        return None
    return "\n".join(collected)


def handle_model(state: SessionState, new_model_id: str) -> None:
    state.llm          = load_llm(new_model_id)
    state.model_id     = new_model_id
    state.history      = []
    state.final_answer = None
    state.agent_executor = build_agent(
        state.llm, state.context_paths or None,
        state.memory.format_for_prompt(), state.current_mode,
    )
    console.print(f"[green]Switched to model:[/green] [cyan]{new_model_id}[/cyan] — history reset.\n")


def handle_mode(state: SessionState, parts: list[str]) -> None:
    new_mode = parts[1].lower() if len(parts) > 1 else ""

    if not parts[1:]:
        if state.current_mode:
            console.print(f"[bold]Current mode:[/bold] [cyan]{state.current_mode}[/cyan]\n")
        else:
            console.print("[bold]Current mode:[/bold] [dim](none — default behaviour)[/dim]\n")
        return

    if new_mode == "off":
        new_mode = ""

    if new_mode and new_mode not in MODE_ADDENDA:
        valid = " | ".join(MODE_ADDENDA.keys())
        console.print(f"[red]Unknown mode:[/red] {new_mode}")
        console.print(f"[dim]Valid modes: {valid} | off[/dim]\n")
        return

    state.current_mode   = new_mode
    state.agent_executor = build_agent(
        state.llm, state.context_paths or None,
        state.memory.format_for_prompt(), state.current_mode,
    )
    if state.current_mode:
        console.print(f"[green]Mode set:[/green] [cyan]{state.current_mode}[/cyan] — agent rebuilt, history kept.\n")
    else:
        console.print("[green]Mode cleared.[/green] — agent rebuilt, history kept.\n")


def handle_rollback(state: SessionState, parts: list[str]) -> None:
    if not state.checkpoints:
        console.print("[yellow]Nothing to roll back — no turns recorded yet.[/yellow]\n")
        return
    steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    steps = min(steps, len(state.checkpoints))
    for _ in range(steps):
        cp = state.checkpoints.pop()
        for path, original in cp["files"].items():
            if original is None:
                if os.path.exists(path):
                    os.remove(path)
                    console.print(f"  [dim]deleted[/dim] {path}")
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(original)
                console.print(f"  [dim]restored[/dim] {path}")
        state.history = state.history[:cp["history_len"]]
    console.print(f"[green]Rolled back {steps} turn(s).[/green]\n")


def handle_checkpoints(state: SessionState) -> None:
    if not state.checkpoints:
        console.print("[dim]No checkpoints recorded yet.[/dim]\n")
        return
    from rich.table import Table
    tbl = Table(title="Checkpoints", header_style="bold cyan", show_lines=True)
    tbl.add_column("#", style="dim", width=4)
    tbl.add_column("Task")
    tbl.add_column("Files touched", style="cyan")
    for i, cp in enumerate(state.checkpoints, 1):
        files = "\n".join(os.path.basename(p) for p in cp["files"]) or "—"
        tbl.add_row(str(i), cp["task"], files)
    console.print(tbl)
    console.print()


def handle_memory(state: SessionState, parts: list[str]) -> None:
    cmd = parts[1].lower() if len(parts) > 1 else "list"

    if cmd == "list":
        console.print()
        global_mem  = state.memory.recall_global()
        project_mem = state.memory.recall_project()
        if global_mem:
            print_memory_table(global_mem, "Global Memory")
            console.print()
        if project_mem:
            print_memory_table(project_mem, "Project Memory")
            console.print()
        if not global_mem and not project_mem:
            console.print("[dim]No facts saved in memory.[/dim]\n")
        return

    if cmd == "save":
        if len(parts) < 5:
            console.print("[yellow]Usage: /memory save <scope> <key> <value>[/yellow]")
            console.print("[dim]Example: /memory save project entry_point cli.py[/dim]\n")
            return
        result = state.memory.save(parts[2].lower(), parts[3], " ".join(parts[4:]))
        console.print(f"[green]{result}[/green]\n")
        return

    if cmd == "delete":
        if len(parts) < 4:
            console.print("[yellow]Usage: /memory delete <scope> <key>[/yellow]\n")
            return
        result = state.memory.delete(parts[2].lower(), parts[3])
        console.print(f"[green]{result}[/green]\n")
        return

    if cmd == "clear":
        if len(parts) < 3:
            console.print("[yellow]Usage: /memory clear <scope>[/yellow]")
            console.print("[dim]scope must be 'global' or 'project'[/dim]\n")
            return
        scope = parts[2].lower()
        if scope not in ("global", "project"):
            console.print("[red]Error: scope must be 'global' or 'project'[/red]\n")
            return
        data = state.memory.recall_global() if scope == "global" else state.memory.recall_project()
        for key in list(data.keys()):
            state.memory.delete(scope, key)
        console.print(f"[green]{scope.capitalize()} memory cleared.[/green]\n")
        return

    console.print(f"[red]Unknown memory command: {cmd}[/red]")
    console.print("[dim]/memory list | save | delete | clear[/dim]\n")


def handle_session(state: SessionState, parts: list[str]) -> None:
    sub          = parts[1].lower() if len(parts) > 1 else "list"
    sessions_dir = os.path.join(".saathi", "sessions")

    if sub == "list":
        if not os.path.isdir(sessions_dir):
            console.print("[dim]No saved sessions.[/dim]\n")
            return
        entries = [e for e in sorted(os.listdir(sessions_dir)) if e.endswith(".json")]
        if not entries:
            console.print("[dim]No saved sessions.[/dim]\n")
            return
        from rich.table import Table
        tbl = Table(title="Saved Sessions", header_style="bold cyan")
        tbl.add_column("Name",     style="cyan")
        tbl.add_column("Saved at", style="dim")
        for entry in entries:
            fp    = os.path.join(sessions_dir, entry)
            mtime = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")
            tbl.add_row(entry[:-5], mtime)
        console.print(tbl)
        console.print()
        return

    if sub == "save":
        if len(parts) < 3:
            console.print("[yellow]Usage: /session save <name>[/yellow]\n")
            return
        name = parts[2]
        os.makedirs(sessions_dir, exist_ok=True)
        session_data = {
            "history": [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in state.history
            ],
            "context_paths": state.context_paths,
            "checkpoints":   state.checkpoints,
        }
        fp = os.path.join(sessions_dir, f"{name}.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(session_data, fh, indent=2)
        console.print(f"[green]Session saved to:[/green] [cyan]{fp}[/cyan]\n")
        return

    if sub == "load":
        if len(parts) < 3:
            console.print("[yellow]Usage: /session load <name>[/yellow]\n")
            return
        name = parts[2]
        fp   = os.path.join(sessions_dir, f"{name}.json")
        if not os.path.exists(fp):
            console.print(f"[red]Session not found:[/red] {fp}\n")
            return
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        state.history = [
            HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
            for m in data.get("history", [])
        ]
        state.context_paths = data.get("context_paths", [])
        state.checkpoints   = data.get("checkpoints", [])
        state.agent_executor = build_agent(
            state.llm, state.context_paths or None,
            state.memory.format_for_prompt(), state.current_mode,
        )
        console.print(
            f"[green]Session loaded:[/green] [cyan]{name}[/cyan] "
            f"— {len(state.history)} messages, {len(state.checkpoints)} checkpoints.\n"
        )
        return

    console.print(f"[red]Unknown session command: {sub}[/red]")
    console.print("[dim]/session save <name> | load <name> | list[/dim]\n")
