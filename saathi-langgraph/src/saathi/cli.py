"""Main CLI entry point using Typer."""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
import threading
import time
import uuid

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
    print_task: str | None = typer.Option(
        None,
        "--print",
        "-p",
        help="Run a single task non-interactively, print the result, and exit",
    ),
    output_format: str = typer.Option(
        "text", "--output-format", help="Output for --print: 'text' or 'json'"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model ID"),
    context: list[str] | None = typer.Option(None, "--context", "-c", help="Scope to paths"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Start an interactive Saathi session, or run one task with --print."""

    if debug:
        settings.debug = True

    model_id = model or settings.ollama_model

    if print_task is not None:
        code = asyncio.run(_print_mode(model_id, list(context or []), print_task, output_format))
        raise typer.Exit(code)

    asyncio.run(_interactive_session(model_id, list(context or [])))


async def _print_mode(
    model_id: str,
    context_paths: list[str],
    task: str,
    output_format: str,
) -> int:
    """Run a single task with no interactive UI; emit text or JSON to stdout.

    Returns a process exit code (0 ok, 1 runtime error, 2 usage error). All
    diagnostics go to stderr so stdout stays clean for piping.
    """
    if output_format not in ("text", "json"):
        print(
            f"error: --output-format must be 'text' or 'json', got {output_format!r}",
            file=sys.stderr,
        )
        return 2

    memory_store = MemoryStore()
    hook_runner = HookRunner()
    graph = await build_graph(ALL_TOOLS, memory_store, model_id, hook_runner=hook_runner)
    session_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "messages": [HumanMessage(content=task)],
        "context_paths": context_paths,
        "mode": "default",
        "session_id": session_id,
    }

    try:
        result = await graph.ainvoke(input_state, config)
    except Exception as exc:  # noqa: BLE001 — surface any failure as a clean error
        if settings.debug:
            import traceback

            traceback.print_exc()
        _emit_error(str(exc), output_format)
        return 1
    finally:
        await close_graph(graph)

    messages: list[BaseMessage] = result.get("messages", [])
    response = _final_text(messages)

    if output_format == "json":
        payload = {
            "model": model_id,
            "task": task,
            "response": response,
            "tool_calls": _collect_tool_calls(messages),
            "usage": _collect_usage(messages),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(response)
    return 0


def _emit_error(message: str, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)


def _final_text(messages: list[BaseMessage]) -> str:
    """The last assistant message with textual content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return ""


def _collect_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    calls: list[dict] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append({"name": tc["name"], "args": tc.get("args", {})})
    return calls


def _collect_usage(messages: list[BaseMessage]) -> dict[str, int]:
    in_tokens = out_tokens = 0
    for msg in messages:
        usage = _extract_usage(msg)
        if usage:
            in_tokens += usage[0]
            out_tokens += usage[1]
    return {"input_tokens": in_tokens, "output_tokens": out_tokens}


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

        final_answer, messages = await _run_turn(graph, config, task, state, messages)

        # Record the pre-edit content of files this turn touched, keeping the
        # earliest snapshot per file so /diff shows changes since the session
        # started (setdefault never overwrites an earlier original).
        for path, original in get_turn_snapshots().items():
            session_start_snapshots.setdefault(path, original)

        if final_answer:
            state.last_response = final_answer

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
                    "[dim]Compact: use /rollback or /checkpoints (LangGraph checkpointing).[/dim]"
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
