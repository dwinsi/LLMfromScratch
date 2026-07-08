"""Main CLI entry point using Typer."""

from __future__ import annotations

import asyncio
import itertools
import threading
import time

import typer
from langchain_core.messages import HumanMessage
from rich.rule import Rule

from saathi.agent import build_graph, close_graph
from saathi.config import settings
from saathi.diagnostics import run_doctor
from saathi.hooks.runner import HookRunner
from saathi.memory.store import MemoryStore
from saathi.project_context import instructions_source
from saathi.session.manager import SessionManager, SessionState
from saathi.tools import ALL_TOOLS
from saathi.tools.filesystem import clear_turn_snapshots, get_turn_snapshots
from saathi.ui.commands import (
    handle_checkpoints,
    handle_context,
    handle_copy,
    handle_diff,
    handle_export,
    handle_memory,
    handle_mode,
    handle_paste,
    handle_rollback,
    handle_session,
)
from saathi.ui.display import (
    console,
    print_banner,
    print_help,
    render_tool_call,
    render_tool_result,
)

app = typer.Typer(
    name="saathi", help="Local coding agent powered by LangGraph + Ollama", add_completion=False
)

_INIT_PROMPT = (
    "Explore this repository and create a SAATHI.md file at the project root. "
    "Use list_directory and read_file to understand the project. The file should "
    "document: the project's purpose, the tech stack, the main entry point, key "
    "files and their roles, how to build/run/test, and any conventions or gotchas. "
    "Write it with write_file. Keep it concise and factual — do not invent details."
)

_COMMIT_PROMPT = (
    "Review the current git changes and create a commit. First call git_status and "
    "git_diff (and git_diff_staged) to see what changed. Then write a clear, "
    "conventional commit message summarizing the changes and call "
    "git_commit(message=..., add_all=True). Report the result."
)

_SPINNER_PHRASES = itertools.cycle(
    [
        "thinking…",
        "consulting the void…",
        "reading the runes…",
        "pondering…",
        "chewing on it…",
        "soch raha hoon…",
        "connecting the dots…",
        "loading wisdom…",
        "almost there…",
        "jugaad laga raha hoon…",
        "running the numbers…",
        "brewing ideas…",
    ]
)


class ThinkingSpinner:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._label = next(_SPINNER_PHRASES)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        console.print()

    def _run(self) -> None:
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self._stop.is_set():
            char = spinner_chars[i % len(spinner_chars)]
            console.print(f"\r  [cyan]{char}[/cyan]  [dim]{self._label}[/dim]   ", end="")
            time.sleep(0.1)
            i += 1
            if i % 20 == 0:
                self._label = next(_SPINNER_PHRASES)


async def _run_turn(
    graph,
    config: dict,
    task: str,
    state: SessionState,
    messages: list,
) -> tuple[str, list]:
    """Stream one agent turn; return (final_answer, updated_messages)."""

    messages.append(HumanMessage(content=task))
    input_state = {
        "messages": messages,
        "context_paths": state.context_paths,
        "mode": state.mode,
        "session_id": state.session_id,
    }

    spinner = ThinkingSpinner()
    spinner.start()

    final_answer = ""
    updated_messages = list(messages)
    start = time.monotonic()
    in_tokens = 0
    out_tokens = 0

    try:
        async for event in graph.astream_events(input_state, config, version="v2"):
            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if not final_answer:
                        spinner.stop()
                        console.print(Rule(style="dim cyan"))
                    final_answer += chunk.content
                    console.print(chunk.content, end="", highlight=False)

            elif kind == "on_chat_model_end":
                usage = _extract_usage(event["data"].get("output"))
                if usage:
                    in_tokens += usage[0]
                    out_tokens += usage[1]

            elif kind == "on_tool_start":
                spinner.update(f"→ {name}")
                render_tool_call(name, event["data"].get("input", {}))

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                render_tool_result(name, str(output))
                spinner.update(next(_SPINNER_PHRASES))

            elif kind == "on_chain_end" and name == "LangGraph":
                output = event["data"].get("output", {})
                if "messages" in output:
                    updated_messages = output["messages"]

    except KeyboardInterrupt:
        spinner.stop()
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as exc:
        spinner.stop()
        console.print(f"\n[red]Error:[/red] {exc}")
        if settings.debug:
            import traceback

            traceback.print_exc()
    else:
        if final_answer:
            console.print()
        spinner.stop() if not final_answer else None

    elapsed = time.monotonic() - start
    if final_answer or in_tokens or out_tokens:
        console.print(
            f"[dim]↳ {in_tokens:,} in · {out_tokens:,} out · {elapsed:.1f}s[/dim]",
            highlight=False,
        )

    return final_answer, updated_messages


def _extract_usage(output) -> tuple[int, int] | None:
    """Pull (input_tokens, output_tokens) from an AIMessage's usage metadata."""
    if output is None:
        return None
    meta = getattr(output, "usage_metadata", None)
    if meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    # Fallback: ChatOllama sometimes reports under response_metadata
    rmeta = getattr(output, "response_metadata", None)
    if rmeta:
        pc = rmeta.get("prompt_eval_count")
        ec = rmeta.get("eval_count")
        if pc is not None or ec is not None:
            return int(pc or 0), int(ec or 0)
    return None


@app.command()
def main(
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model ID"),
    context: list[str] | None = typer.Option(None, "--context", "-c", help="Scope to paths"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Start an interactive Saathi session."""

    if debug:
        settings.debug = True  # type: ignore[misc]

    model_id = model or settings.ollama_model
    asyncio.run(_interactive_session(model_id, list(context or [])))


async def _interactive_session(model_id: str, context_paths: list[str]) -> None:
    print_banner(model_id)

    memory_store = MemoryStore()
    session_mgr = SessionManager()
    hook_runner = HookRunner()
    state = SessionState(
        model_id=model_id,
        context_paths=context_paths,
    )

    graph = await build_graph(ALL_TOOLS, memory_store, model_id, hook_runner=hook_runner)
    config = {"configurable": {"thread_id": state.session_id}}

    src = instructions_source()
    if src:
        console.print(f"[dim]✓ loaded {src}[/dim]")
    if not hook_runner.config.is_empty:
        console.print("[dim]✓ hooks active (.saathi/hooks.json)[/dim]")

    messages: list = []
    session_start_snapshots: dict[str, str] = {}

    async def execute_task(task: str) -> None:
        """Run one agent turn for a task and record its result + post_turn hooks."""
        nonlocal messages
        clear_turn_snapshots()
        snapshots_before = get_turn_snapshots()

        final_answer, messages = await _run_turn(graph, config, task, state, messages)

        if final_answer:
            state.last_response = final_answer
            # capture any files touched this turn for /diff
            session_start_snapshots.update(snapshots_before)

        # fire post_turn hooks (e.g. run tests, lint) after every completed turn
        if not hook_runner.config.is_empty:
            for result in await hook_runner.run("post_turn"):
                if not result.ok and result.output.strip():
                    console.print(f"[yellow]post_turn hook:[/yellow] {result.output.strip()[:500]}")

    while True:
        try:
            raw = console.input("\n[bold cyan]you[/bold cyan] ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        # ── built-in keywords ───────────────────────────────────────────────
        if raw.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if raw.lower() == "clear":
            messages = []
            console.print("[dim]History cleared.[/dim]")
            continue

        if raw.lower() == "help":
            print_help()
            continue

        # ── slash commands ──────────────────────────────────────────────────
        if raw.startswith("/"):
            parts = raw[1:].split()
            cmd, args = parts[0].lower() if parts else "", parts[1:]

            if cmd == "context":
                state.context_paths, messages = await handle_context(args, state, messages)
                # rebuild graph with new context after scope change
                continue

            if cmd == "mode":
                handle_mode(args, state)
                continue

            if cmd == "memory":
                handle_memory(args, memory_store)
                continue

            if cmd == "session":
                result = handle_session(args, state, messages, session_mgr)
                if result:
                    state, messages = result
                continue

            if cmd == "rollback":
                await handle_rollback(args, graph, config)
                continue

            if cmd == "checkpoints":
                await handle_checkpoints(graph, config)
                continue

            if cmd == "diff":
                handle_diff(session_start_snapshots)
                continue

            if cmd == "export":
                handle_export(messages)
                continue

            if cmd == "copy":
                handle_copy(state.last_response)
                continue

            if cmd == "paste":
                pasted = handle_paste()
                if not pasted.strip():
                    continue
                await execute_task(pasted)
                continue

            if cmd == "model" and args:
                state.model_id = args[0]
                model_id = args[0]
                messages = []
                await close_graph(graph)
                graph = await build_graph(
                    ALL_TOOLS, memory_store, model_id, hook_runner=hook_runner
                )
                config = {"configurable": {"thread_id": state.session_id}}
                console.print(f"[green]Model:[/green] {model_id}  [dim](history cleared)[/dim]")
                continue

            if cmd == "compact":
                console.print(
                    "[dim]Compact: use LangGraph checkpointing — /rollback or /checkpoints instead.[/dim]"
                )
                continue

            if cmd == "doctor":
                run_doctor()
                continue

            if cmd == "init":
                clear_turn_snapshots()
                _, messages = await _run_turn(graph, config, _INIT_PROMPT, state, messages)
                continue

            if cmd == "commit":
                clear_turn_snapshots()
                _, messages = await _run_turn(graph, config, _COMMIT_PROMPT, state, messages)
                continue

            console.print(
                f"[red]Unknown command:[/red] /{cmd}  —  type [bold]help[/bold] for reference"
            )
            continue

        # ── agent turn ──────────────────────────────────────────────────────
        await execute_task(raw)

    # close the checkpointer's SQLite connection on the way out
    await close_graph(graph)
