"""
Interactive CLI for the coding agent.

Run this for an ongoing terminal session where you can give
the agent multiple tasks one after another.

Usage:
  python cli.py
  python cli.py --model gemma4:27b
  python cli.py --context ./src ./utils/helpers.py

Commands during a session:
  Type any task and press Enter
  Type 'quit' or 'exit' to end the session
  Type 'clear' to reset the conversation
  Type '/context <path> ...' to set or update the working scope
  Type '/context' with no paths to clear the scope
  Type 'help' to see available commands
"""

import argparse
import difflib
import itertools
import json
import os
import threading
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage

from agent import load_llm, build_agent, compact_history, OLLAMA_MODEL
from system_prompt import MODE_ADDENDA
from memory_store import MemoryStore
from tools import set_memory_store, reset_turn_snapshot, get_turn_snapshot


console = Console()

# Try to enable input history navigation (up/down arrow keys).
try:
    import readline  # Unix stdlib
except ImportError:
    try:
        import pyreadline3 as readline  # Windows
    except ImportError:
        readline = None

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
    "soch raha hoon",       # thinking
    "samajh raha hoon",     # understanding
    "vichar kar raha hoon", # contemplating
    "dhundh raha hoon",     # searching
    "padhh raha hoon",      # reading
    "sun raha hoon",        # listening
    "yaad kar raha hoon",   # remembering
    "andaza laga raha hoon",# estimating
    "khoj raha hoon",       # exploring / researching
    "hisab laga raha hoon", # calculating
    "bujh raha hoon",       # figuring out
    "dekh raha hoon",       # looking / examining
    "samjha raha hoon",     # comprehending
    "guna raha hoon",       # multiplying / pondering deeply
    "taash khel raha hoon", # playing cards (thinking strategically)
    "mann mein soch raha hoon", # thinking in the mind
    "jugaad laga raha hoon",    # improvising a clever solution
    "dimaag chala raha hoon",   # running the brain
    "ulajhan suljha raha hoon", # untangling the problem
    "raaz khol raha hoon",      # unravelling the mystery
]


class ThinkingSpinner:
    """Displays a Status spinner that cycles through witty phrases on a timer."""

    def __init__(self, interval: float = 2.0):
        self._phrases = itertools.cycle(THINKING_PHRASES)
        self._interval = interval
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


def print_banner():
    title = Text("saathi", style="bold cyan") + Text(" — your coding companion")
    subtitle = Text("Powered by Gemma 4 via Ollama", style="dim")
    console.print(Panel(title + "\n" + subtitle, border_style="cyan", padding=(0, 2)))
    console.print("[dim]Type a task and press Enter. Type [bold]help[/bold] for commands.[/dim]\n")


def print_help():
    help_md = """
## Available commands

| Command | Description |
| --- | --- |
| `<any text>` | Run a task through the coding agent |
| `/context <path> ...` | Scope agent to specific files or folders |
| `/context` | Clear scope — agent works unrestricted |
| `clear` | Reset conversation history (keeps scope and memory) |
| `/compact` | Summarise conversation history with the LLM — frees tokens while keeping context |
| `/rollback` | Undo the last turn — restores files and removes it from history |
| `/rollback <n>` | Undo the last n turns |
| `/checkpoints` | List all recorded turns and which files each one touched |
| `/diff` | Show unified diff of every file changed across all checkpoints |
| `/export` | Dump conversation history to a timestamped markdown file |
| `/copy` | Copy the last agent response to clipboard |
| `/paste` | Enter multi-line input mode (blank line to send, Ctrl+C to cancel) |
| `/model <model-id>` | Switch to a different Ollama model mid-session |
| `/mode explain` | Tune agent for clear explanations — prefers reads, never modifies |
| `/mode refactor` | Tune agent for code quality — explains every change, runs tests |
| `/mode debug` | Tune agent for root-cause debugging — reproduces before fixing |
| `/mode` | Show current mode (or clear it with `/mode off`) |
| `/session save <name>` | Save current session (history, context, checkpoints) |
| `/session load <name>` | Restore a saved session |
| `/session list` | List all saved sessions |
| `/memory list` | Show all saved facts (global and project) |
| `/memory save <scope> <key> <value>` | Manually save a fact to memory |
| `/memory delete <scope> <key>` | Delete a single fact from memory |
| `/memory clear <scope>` | Wipe all facts from global or project memory |
| `help` | Show this message |
| `quit` / `exit` | End the session |

## Memory scopes

- `global`: applies across all projects, stored in `~/.saathi/memory.json`
- `project`: specific to current folder, stored in `.saathi/memory.json`

## Example memory commands

```text
/memory list
/memory save project entry_point cli.py
/memory save global preferred_language Python
/memory delete project entry_point
/memory clear project
```

## Startup flags

- `--context <path> ...` — Set initial file/folder scope
- `--model <model-id>` — Use a different Ollama model

## Example tasks

- List all Python files in the current directory
- Read `agent.py` and explain what it does
- Create a new file `hello.py` that prints Hello World
- Search for the word `import` in `tools.py`
"""
    console.print(Markdown(help_md))


def print_memory_table(memory_dict: dict, scope_name: str = "Memory"):
    """Display memory as a rich table."""
    if not memory_dict:
        console.print(f"[dim]{scope_name} is empty.[/dim]")
        return

    from rich.table import Table
    table = Table(title=scope_name, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for key, value in memory_dict.items():
        # Truncate very long values
        display_value = str(value)[:100] + "…" if len(str(value)) > 100 else str(value)
        table.add_row(key, display_value)

    console.print(table)


def resolve_paths(paths: list[str]) -> list[str]:
    """Resolve each path to absolute and warn if it doesn't exist."""
    resolved = []
    for p in paths:
        abs_path = os.path.abspath(p)
        if not os.path.exists(abs_path):
            console.print(f"  [yellow]Warning:[/yellow] path does not exist: [dim]{abs_path}[/dim]")
        resolved.append(abs_path)
    return resolved


def print_context(context_paths: list[str]):
    if context_paths:
        lines = "\n".join(f"  [cyan]{p}[/cyan]" for p in context_paths)
        console.print(f"[bold]Context scope:[/bold]\n{lines}\n")
    else:
        console.print("[bold]Context scope:[/bold] [dim](none — agent works unrestricted)[/dim]\n")


def run_interactive_session(model_id: str, context_paths: list[str] | None = None):
    """
    Run an interactive terminal session.
    Connects to Ollama once at startup, then accepts tasks until the user quits.
    context_paths: optional list of resolved file/folder paths to scope the agent to.
    """
    print_banner()

    context_paths = context_paths or []
    print_context(context_paths)

    # Set up memory — project memory lives next to the code being worked on
    project_dir   = context_paths[0] if context_paths else os.getcwd()
    memory        = MemoryStore(project_dir=project_dir)
    set_memory_store(memory)   # inject into tools so save_memory / recall_memory work

    memory_block  = memory.format_for_prompt()
    if memory_block:
        console.print("[dim]Memory loaded — injecting saved facts into context.[/dim]\n")

    llm            = load_llm(model_id)
    current_mode   = ""
    agent_executor = build_agent(llm, context_paths or None, memory_block, current_mode)
    history: list        = []   # grows each turn; compacted before every call
    checkpoints: list    = []   # each entry: {"files": snapshot, "history_len": n, "task": str}

    final_answer = None  # last agent response; used by /copy

    console.print("[bold green]Agent ready.[/bold green] What would you like to do?\n")

    while True:
        try:
            mode_tag   = f" [dim]({current_mode})[/dim]" if current_mode else ""
            user_input = Prompt.ask(f"[bold cyan]You[/bold cyan]{mode_tag}").strip()

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit'):
                console.print("[dim]Ending session.[/dim]")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if user_input.lower() == 'clear':
                history = []
                agent_executor = build_agent(llm, context_paths or None, memory.format_for_prompt(), current_mode)
                console.print("[dim]Conversation cleared.[/dim]\n")
                continue

            if user_input.lower() == '/compact':
                if not history:
                    console.print("[dim]Nothing to compact — conversation is empty.[/dim]\n")
                    continue
                msgs_before = len(history)
                console.print("[dim]Compacting conversation…[/dim]")
                compact_spinner = ThinkingSpinner()
                compact_spinner.start()
                try:
                    history_text = "\n\n".join(
                        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                        for m in history
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
                    response      = llm.invoke([HumanMessage(content=summarise_prompt)])
                    summary       = response.content.strip()
                    history       = [HumanMessage(content=f"[Conversation summary]\n{summary}")]
                finally:
                    compact_spinner.stop()
                msgs_after = len(history)
                console.print(
                    f"[green]Compacted[/green] {msgs_before} message(s) → {msgs_after} summary message.\n"
                )
                console.print(Panel(
                    Markdown(summary),
                    title="[dim]summary[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                ))
                console.print()
                continue

            if user_input.lower().startswith('/context'):
                parts = user_input.split()[1:]
                context_paths = resolve_paths(parts) if parts else []
                print_context(context_paths)
                # Re-point project memory to the new directory
                project_dir = context_paths[0] if context_paths else os.getcwd()
                memory      = MemoryStore(project_dir=project_dir)
                set_memory_store(memory)
                history = []
                agent_executor = build_agent(llm, context_paths or None, memory.format_for_prompt(), current_mode)
                continue

            if user_input.lower().startswith('/memory'):
                parts = user_input.split()
                cmd = parts[1].lower() if len(parts) > 1 else "list"

                if cmd == "list":
                    console.print()
                    global_mem = memory.recall_global()
                    project_mem = memory.recall_project()
                    if global_mem:
                        print_memory_table(global_mem, "Global Memory")
                        console.print()
                    if project_mem:
                        print_memory_table(project_mem, "Project Memory")
                        console.print()
                    if not global_mem and not project_mem:
                        console.print("[dim]No facts saved in memory.[/dim]\n")
                    continue

                if cmd == "save":
                    if len(parts) < 5:
                        console.print("[yellow]Usage: /memory save <scope> <key> <value>[/yellow]")
                        console.print("[dim]Example: /memory save project entry_point cli.py[/dim]\n")
                        continue
                    scope = parts[2].lower()
                    key = parts[3]
                    value = " ".join(parts[4:])
                    result = memory.save(scope, key, value)
                    console.print(f"[green]{result}[/green]\n")
                    continue

                if cmd == "delete":
                    if len(parts) < 4:
                        console.print("[yellow]Usage: /memory delete <scope> <key>[/yellow]\n")
                        continue
                    scope = parts[2].lower()
                    key = parts[3]
                    result = memory.delete(scope, key)
                    console.print(f"[green]{result}[/green]\n")
                    continue

                if cmd == "clear":
                    if len(parts) < 3:
                        console.print("[yellow]Usage: /memory clear <scope>[/yellow]")
                        console.print("[dim]scope must be 'global' or 'project'[/dim]\n")
                        continue
                    scope = parts[2].lower()
                    if scope not in ("global", "project"):
                        console.print("[red]Error: scope must be 'global' or 'project'[/red]\n")
                        continue
                    # Clear by deleting all keys
                    data = memory.recall_global() if scope == "global" else memory.recall_project()
                    for key in list(data.keys()):
                        memory.delete(scope, key)
                    console.print(f"[green]{scope.capitalize()} memory cleared.[/green]\n")
                    continue

                console.print(f"[red]Unknown memory command: {cmd}[/red]")
                console.print("[dim]/memory list | save | delete | clear[/dim]\n")
                continue

            if user_input.lower().startswith('/rollback'):
                parts = user_input.split()
                steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                steps = min(steps, len(checkpoints))
                if not checkpoints:
                    console.print("[yellow]Nothing to roll back — no turns recorded yet.[/yellow]\n")
                    continue
                for _ in range(steps):
                    cp = checkpoints.pop()
                    # Restore files touched in that turn
                    for path, original in cp["files"].items():
                        if original is None:
                            if os.path.exists(path):
                                os.remove(path)
                                console.print(f"  [dim]deleted[/dim] {path}")
                        else:
                            with open(path, 'w', encoding='utf-8') as fh:
                                fh.write(original)
                            console.print(f"  [dim]restored[/dim] {path}")
                    # Trim history back to before that turn
                    history = history[:cp["history_len"]]
                console.print(f"[green]Rolled back {steps} turn(s).[/green]\n")
                continue

            if user_input.lower() == '/checkpoints':
                if not checkpoints:
                    console.print("[dim]No checkpoints recorded yet.[/dim]\n")
                else:
                    from rich.table import Table
                    tbl = Table(title="Checkpoints", header_style="bold cyan", show_lines=True)
                    tbl.add_column("#", style="dim", width=4)
                    tbl.add_column("Task")
                    tbl.add_column("Files touched", style="cyan")
                    for i, cp in enumerate(checkpoints, 1):
                        files = "\n".join(os.path.basename(p) for p in cp["files"]) or "—"
                        tbl.add_row(str(i), cp["task"], files)
                    console.print(tbl)
                    console.print()
                continue

            if user_input.lower() == '/diff':
                any_diff = False
                for cp in checkpoints:
                    for path, original in cp["files"].items():
                        if original is None:
                            continue
                        if not os.path.exists(path):
                            continue
                        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
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
                continue

            if user_input.lower() == '/export':
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename  = f"saathi-export-{timestamp}.md"
                filepath  = os.path.join(os.getcwd(), filename)
                with open(filepath, 'w', encoding='utf-8') as fh:
                    for msg in history:
                        if isinstance(msg, HumanMessage):
                            fh.write(f"## You\n\n{msg.content}\n\n")
                        elif isinstance(msg, AIMessage):
                            fh.write(f"## Saathi\n\n{msg.content}\n\n")
                console.print(f"[green]Exported conversation to:[/green] [cyan]{filepath}[/cyan]\n")
                continue

            if user_input.lower() == '/copy':
                if final_answer is None:
                    console.print("[dim]Nothing to copy yet — run a task first.[/dim]\n")
                else:
                    try:
                        import pyperclip
                        pyperclip.copy(final_answer)
                        console.print("[green]Last response copied to clipboard.[/green]\n")
                    except Exception:
                        console.print("[yellow]Clipboard not available in this environment.[/yellow]\n")
                continue

            if user_input.lower() == '/paste':
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
                    continue
                if not collected:
                    console.print("[dim]Empty input — nothing sent.[/dim]\n")
                    continue
                user_input = "\n".join(collected)

            if user_input.lower().startswith('/model'):
                parts = user_input.split()
                if len(parts) < 2:
                    console.print("[yellow]Usage: /model <model-id>[/yellow]\n")
                    continue
                new_model_id = parts[1]
                llm            = load_llm(new_model_id)
                agent_executor = build_agent(llm, context_paths or None, memory.format_for_prompt(), current_mode)
                history        = []
                final_answer   = None
                console.print(f"[green]Switched to model:[/green] [cyan]{new_model_id}[/cyan] — history reset.\n")
                continue

            if user_input.lower().startswith('/mode'):
                parts    = user_input.split()
                new_mode = parts[1].lower() if len(parts) > 1 else ""

                if new_mode == "off":
                    new_mode = ""

                if new_mode and new_mode not in MODE_ADDENDA:
                    valid = " | ".join(MODE_ADDENDA.keys())
                    console.print(f"[red]Unknown mode:[/red] {new_mode}")
                    console.print(f"[dim]Valid modes: {valid} | off[/dim]\n")
                    continue

                if not parts[1:]:
                    if current_mode:
                        console.print(f"[bold]Current mode:[/bold] [cyan]{current_mode}[/cyan]\n")
                    else:
                        console.print("[bold]Current mode:[/bold] [dim](none — default behaviour)[/dim]\n")
                    continue

                current_mode   = new_mode
                agent_executor = build_agent(llm, context_paths or None, memory.format_for_prompt(), current_mode)
                if current_mode:
                    console.print(f"[green]Mode set:[/green] [cyan]{current_mode}[/cyan] — agent rebuilt, history kept.\n")
                else:
                    console.print("[green]Mode cleared.[/green] — agent rebuilt, history kept.\n")
                continue

            if user_input.lower().startswith('/session'):
                parts = user_input.split()
                sub   = parts[1].lower() if len(parts) > 1 else "list"
                sessions_dir = os.path.join(".saathi", "sessions")

                if sub == "list":
                    if not os.path.isdir(sessions_dir):
                        console.print("[dim]No saved sessions.[/dim]\n")
                    else:
                        entries = sorted(os.listdir(sessions_dir))
                        if not entries:
                            console.print("[dim]No saved sessions.[/dim]\n")
                        else:
                            from rich.table import Table
                            tbl = Table(title="Saved Sessions", header_style="bold cyan")
                            tbl.add_column("Name", style="cyan")
                            tbl.add_column("Saved at", style="dim")
                            for entry in entries:
                                if entry.endswith(".json"):
                                    fp   = os.path.join(sessions_dir, entry)
                                    mtime = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")
                                    tbl.add_row(entry[:-5], mtime)
                            console.print(tbl)
                            console.print()
                    continue

                if sub == "save":
                    if len(parts) < 3:
                        console.print("[yellow]Usage: /session save <name>[/yellow]\n")
                        continue
                    name = parts[2]
                    os.makedirs(sessions_dir, exist_ok=True)
                    session_data = {
                        "history": [
                            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                            for m in history
                        ],
                        "context_paths": context_paths,
                        "checkpoints": checkpoints,
                    }
                    fp = os.path.join(sessions_dir, f"{name}.json")
                    with open(fp, 'w', encoding='utf-8') as fh:
                        json.dump(session_data, fh, indent=2)
                    console.print(f"[green]Session saved to:[/green] [cyan]{fp}[/cyan]\n")
                    continue

                if sub == "load":
                    if len(parts) < 3:
                        console.print("[yellow]Usage: /session load <name>[/yellow]\n")
                        continue
                    name = parts[2]
                    fp   = os.path.join(sessions_dir, f"{name}.json")
                    if not os.path.exists(fp):
                        console.print(f"[red]Session not found:[/red] {fp}\n")
                        continue
                    with open(fp, 'r', encoding='utf-8') as fh:
                        session_data = json.load(fh)
                    history = [
                        HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
                        for m in session_data.get("history", [])
                    ]
                    context_paths = session_data.get("context_paths", [])
                    checkpoints   = session_data.get("checkpoints", [])
                    agent_executor = build_agent(llm, context_paths or None, memory.format_for_prompt(), current_mode)
                    console.print(f"[green]Session loaded:[/green] [cyan]{name}[/cyan] — {len(history)} messages, {len(checkpoints)} checkpoints.\n")
                    continue

                console.print(f"[red]Unknown session command: {sub}[/red]")
                console.print("[dim]/session save <name> | load <name> | list[/dim]\n")
                continue

            # Build the message list for this turn: compacted history + new message
            reset_turn_snapshot()
            history_len_before = len(history)
            history.append(HumanMessage(content=user_input))
            messages_to_send = compact_history(history)

            # Run the task
            console.print()
            spinner = ThinkingSpinner()
            spinner.start()

            final_answer    = None
            live: Live | None = None
            try:
                for chunk in agent_executor.stream(
                    {"messages": messages_to_send},
                    stream_mode="updates",
                ):
                    for node, val in chunk.items():
                        for msg in val.get("messages", []):
                            tool_calls = getattr(msg, "tool_calls", [])

                            if node == "model" and tool_calls:
                                for tc in tool_calls:
                                    args_text = ", ".join(
                                        f"{k}={repr(v)}" for k, v in tc["args"].items()
                                    )
                                    spinner.update(
                                        f"[bold yellow]⚙ {tc['name']}[/bold yellow]"
                                        f"[dim]({args_text})[/dim]"
                                    )

                            elif node == "tools":
                                content = str(msg.content).strip()
                                preview = content[:300] + "…" if len(content) > 300 else content
                                if live:
                                    live.stop()
                                    live = None
                                spinner.stop()
                                console.print(
                                    Panel(
                                        Syntax(preview, "text", theme="monokai", word_wrap=True),
                                        title="[dim]observation[/dim]",
                                        border_style="dim",
                                        padding=(0, 1),
                                    )
                                )
                                spinner = ThinkingSpinner()
                                spinner.start()

                            elif node == "model" and not tool_calls and msg.content:
                                if live is None:
                                    spinner.stop()
                                    console.print(Rule("[bold cyan]saathi[/bold cyan]", style="cyan"))
                                    live = Live(
                                        Markdown(""),
                                        console=console,
                                        refresh_per_second=15,
                                        vertical_overflow="visible",
                                    )
                                    live.start()
                                final_answer = (final_answer or "") + msg.content
                                live.update(Markdown(final_answer))
            finally:
                if live:
                    live.stop()
                spinner.stop()

            if not final_answer:
                console.print(Rule("[bold cyan]saathi[/bold cyan]", style="cyan"))
                console.print(Markdown("No output returned."))

            console.print()

            # Save checkpoint for this turn so it can be rolled back
            checkpoints.append({
                "task":        user_input,
                "history_len": history_len_before,
                "files":       get_turn_snapshot(),
            })

            # Append the response to history so future turns have context
            if final_answer:
                history.append(AIMessage(content=final_answer))

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type [bold]quit[/bold] to exit cleanly.[/dim]")
            continue

        except Exception as error:
            console.print(f"\n[bold red]Error:[/bold red] {error}")
            console.print("[dim]The agent encountered an error. Try rephrasing your task.[/dim]\n")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="saathi-cli: your coding companion")
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--context",
        type=str,
        nargs="+",
        metavar="PATH",
        help="Files or folders to scope the agent to (can be multiple)",
    )
    args = parser.parse_args()

    initial_context = resolve_paths(args.context) if args.context else []
    run_interactive_session(args.model, initial_context)
