# Chapter 17 — CLI Design: Typer, REPL Loops, and Scripting Mode

> "The Unix philosophy: write programs that do one thing and do it well, write programs that work together, write programs that handle text streams."
>
> — Doug McIlroy

---

## 17.1 Typer — Declarative CLIs from Type Hints

Building command-line interfaces in Python has evolved dramatically. The early days had `sys.argv` parsing by hand. Then `optparse`, then `argparse`. For years `argparse` was the standard: verbose, explicit, functional. Then came `click` — a decorator-based framework that made CLIs composable and testable. And then came **Typer**.

Typer is built on top of click and takes its declarative philosophy one step further: instead of decorators with explicit `type=` parameters, Typer derives CLI argument types and help text directly from Python type hints. You write normal Python function signatures; Typer turns them into a fully featured CLI.

### The Basic Pattern

```python
import typer

app = typer.Typer()

@app.command()
def greet(name: str, count: int = 3):
    for _ in range(count):
        print(f"Hello, {name}!")

if __name__ == "__main__":
    app()
```

Running `python greet.py --help` produces:

```text
Usage: greet.py [OPTIONS] NAME

Arguments:
  NAME  [required]

Options:
  --count INTEGER  [default: 3]
  --help           Show this message and exit.
```

The `name: str` becomes a required positional argument. The `count: int = 3` becomes an optional `--count` flag with a default of `3`. Type hints drive everything.

### Saathi's Typer Application

Saathi's CLI lives in `src/saathi/cli.py`. The top-level application object:

```python
# src/saathi/cli.py
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from saathi.config import settings as default_settings
from saathi.graph import build_graph
from saathi.display import print_response, print_tool_call, print_error
from saathi.usage import extract_usage

app = typer.Typer(
    name="saathi",
    help="An agentic coding assistant powered by Ollama and LangGraph.",
    add_completion=False,   # disable shell completion install prompts
    no_args_is_help=False,  # we want an interactive session with no args
    rich_markup_mode="rich",  # enable Rich markup in help text
)

console = Console(stderr=True)  # diagnostics to stderr
out = Console()                  # output to stdout
```

A few things to note:

- `add_completion=False` disables Typer's default behaviour of showing shell tab-completion installation prompts. saathi users generally do not need this.
- `no_args_is_help=False` means running `saathi` with no arguments starts an interactive session rather than showing help. This is the right default for a REPL.
- `rich_markup_mode="rich"` enables Rich markup (e.g., `[bold]text[/bold]`) in docstrings and help text.
- Two consoles: `console` writes to `stderr` (for diagnostics), `out` writes to `stdout` (for actual output). This is important for the `--print` scripting mode (§17.6).

### The Main Command Signature

```python
@app.command()
def main(
    task: Optional[str] = typer.Argument(
        default=None,
        help="Task to execute in --print mode. If omitted, starts interactive session.",
    ),
    model: Optional[str] = typer.Option(
        default=None,
        "--model", "-m",
        help=(
            "Ollama model to use. Overrides SAATHI_MODEL env var. "
            "Example: --model llama3.2:3b"
        ),
    ),
    context: Optional[int] = typer.Option(
        default=None,
        "--context", "-c",
        help=(
            "Context window size in tokens. Overrides SAATHI_CONTEXT_WINDOW. "
            "Must match the selected model."
        ),
    ),
    print_mode: bool = typer.Option(
        False,
        "--print", "-p",
        help=(
            "Non-interactive mode: run TASK and print the result to stdout. "
            "Useful for shell pipelines and CI scripts."
        ),
    ),
    output_format: str = typer.Option(
        default="text",
        "--output-format",
        help="Output format for --print mode. Options: text, json.",
        callback=validate_output_format,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging to stderr.",
    ),
) -> None:
    """Saathi — an agentic coding assistant.

    Run with no arguments to start an interactive REPL session.
    Run with --print TASK to use as a shell filter or CI step.

    [bold]Examples:[/bold]

        [green]# Interactive session[/green]
        saathi

        [green]# Non-interactive: run a task and print the result[/green]
        saathi --print "Explain the main() function in cli.py"

        [green]# Use a specific model[/green]
        saathi --model llama3.2:3b

        [green]# JSON output for shell parsing[/green]
        saathi --print "What files exist?" --output-format json | jq .response
    """
    # Resolve settings, applying any CLI overrides.
    current_settings = _resolve_settings(model=model, context=context, debug=debug)

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if print_mode:
        if task is None:
            console.print(
                "[red]Error:[/red] --print requires a TASK argument. "
                "Example: saathi --print 'Explain the code'",
                style="red",
            )
            raise typer.Exit(code=2)
        asyncio.run(_print_mode(task=task, fmt=output_format, cfg=current_settings))
    else:
        asyncio.run(_interactive_session(initial_task=task, cfg=current_settings))
```

Let us unpack the option declarations:

**`typer.Argument` vs `typer.Option`**

- `Argument`: positional parameter. `saathi "do something"` — the string is a positional argument.
- `Option`: named flag. `saathi --model llama3.2:3b`.

**Long and short forms**: `"--model", "-m"` registers both `--model` and `-m` as aliases for the same option. Typer passes both as positional arguments to `typer.Option()`.

**`callback=validate_output_format, is_eager=True`**: The callback runs before the command body. `is_eager=True` means it runs even if an error would otherwise short-circuit execution. This lets us validate `--output-format` before building the graph (§17.7).

**`Optional[str]`**: Using `Optional[str]` with `default=None` means the option is not required. Typer distinguishes between "not provided" (`None`) and "provided as empty string" (`""`).

### `_resolve_settings` — Merging CLI Overrides with Config

```python
def _resolve_settings(
    model: Optional[str],
    context: Optional[int],
    debug: bool,
) -> "Settings":
    """Merge CLI flags into the global settings object.

    CLI flags take the highest precedence (above env vars and .env file).
    Fields not overridden by CLI flags retain their env/default values.
    """
    from saathi.config import Settings

    overrides: dict = {}
    if model is not None:
        overrides["model"] = model
    if context is not None:
        overrides["context_window"] = context
    if debug:
        overrides["debug"] = True
        overrides["log_level"] = "DEBUG"

    if overrides:
        return default_settings.model_copy(update=overrides)
    return default_settings
```

`model_copy(update=...)` creates a new `Settings` instance with specific fields overridden, leaving everything else unchanged. This is the correct pattern for "apply overrides" without mutating the global singleton.

---

## 17.2 `asyncio.run` — Bridging Sync Typer and Async LangGraph

Typer calls your command function synchronously. LangGraph's `ainvoke` and `astream` are `async`. The bridge is `asyncio.run()`.

### The Basic Pattern 1

```python
@app.command()
def main(...) -> None:
    asyncio.run(_interactive_session(...))

async def _interactive_session(...) -> None:
    graph = build_graph(...)
    async for chunk in graph.astream(state, config):
        ...
```

`asyncio.run()` creates a new event loop, runs the coroutine to completion, and closes the loop. This is the standard pattern for an async entry point from a sync context.

### Why Async?

LangGraph's graph execution is inherently concurrent. When the agent decides to call three tools simultaneously, LangGraph dispatches all three calls concurrently on the event loop. Synchronous execution would serialize these calls, making the agent 3× slower for parallelizable workloads.

Additionally, streaming requires async: you cannot `yield` from a synchronous function in a way that lets you render each token as it arrives. `async for chunk in graph.astream(...)` gives you each token or event as it is ready, allowing live rendering.

### Windows and `asyncio.WindowsProactorEventLoopPolicy`

On Windows, the default event loop policy changed in Python 3.8. For subprocess-based tools (which saathi's tools may use), you need the `ProactorEventLoop`:

```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

This is set at the top of `cli.py`, before any `asyncio.run()` calls. Without it, subprocess tools that use pipes can raise `NotImplementedError` on Windows.

---

## 17.3 The REPL Loop — `_interactive_session`

The heart of saathi is `_interactive_session`, an async function that implements the Read-Eval-Print Loop (REPL). Here is the full structure:

```python
async def _interactive_session(
    initial_task: Optional[str],
    cfg: "Settings",
) -> None:
    """Run an interactive saathi session.

    Initialises the LangGraph graph once, then loops:
      1. Read user input (or use initial_task for the first turn)
      2. Dispatch slash commands or route to the graph
      3. Display the result
      4. Repeat

    The loop exits on /exit, Ctrl-C, or Ctrl-D (EOF).
    """
    from saathi.graph import build_graph
    from saathi.memory import load_memory
    from saathi.display import print_welcome, print_token_footer

    # ── Startup ──────────────────────────────────────────────────────── #

    print_welcome(cfg)

    graph = build_graph(cfg)
    state: dict = {"messages": [], "model": cfg.model}

    # Load persistent memory into initial state.
    memory_content = await load_memory(cfg.memory_dir)
    if memory_content:
        state["memory"] = memory_content

    # ── Main loop ────────────────────────────────────────────────────── #

    first_turn = True

    while True:
        try:
            # Get input: use initial_task for the first turn if provided,
            # otherwise prompt the user interactively.
            if first_turn and initial_task is not None:
                user_input = initial_task
                first_turn = False
            else:
                first_turn = False
                try:
                    user_input = await _async_prompt()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Goodbye.[/dim]")
                    break

            user_input = user_input.strip()
            if not user_input:
                continue

            # ── Slash command dispatch ────────────────────────────── #

            if user_input.startswith("/"):
                result = await _dispatch_slash_command(
                    user_input, state=state, cfg=cfg, graph=graph
                )
                if result == "EXIT":
                    break
                if result == "CLEAR":
                    state["messages"] = []
                    console.print("[dim]History cleared.[/dim]")
                continue

            # ── Agent turn ───────────────────────────────────────── #

            await execute_task(
                task=user_input,
                state=state,
                graph=graph,
                cfg=cfg,
            )

        except KeyboardInterrupt:
            console.print("\n[dim](Interrupted. Press Ctrl-D or type /exit to quit.)[/dim]")
            continue


async def execute_task(
    task: str,
    state: dict,
    graph,
    cfg: "Settings",
) -> None:
    """Run a single agent turn: add user message, stream graph, display result.

    This is a nested helper defined inside _interactive_session in the actual
    implementation so it captures `state` by reference (mutations are visible
    to the outer loop). Extracted here for clarity.

    Args:
        task: The user's input string.
        state: The current agent state dict (mutated in place with new messages).
        graph: The compiled LangGraph graph.
        cfg: Current settings.
    """
    from langchain_core.messages import HumanMessage
    from saathi.display import print_response, print_tool_call, print_token_footer
    from saathi.usage import extract_usage

    # Append the human message to state.
    state["messages"].append(HumanMessage(content=task))

    start_time = time.monotonic()
    final_message = None
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    # Stream the graph. Each chunk is a partial state update.
    async for chunk in graph.astream(state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "agent":
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print_tool_call(tc)
                    elif hasattr(msg, "content") and msg.content:
                        if cfg.stream:
                            out.print(msg.content, end="", highlight=False)
                        final_message = msg
            elif node_name == "tools":
                messages = node_output.get("messages", [])
                for msg in messages:
                    pass  # tool results are not printed directly

    if not cfg.stream and final_message is not None:
        out.print(final_message.content, highlight=False)

    # Update state with the latest messages from the graph output.
    # (LangGraph returns the full updated state after streaming.)
    result_state = await graph.ainvoke(
        state,
        # Note: in real implementation, we track state updates during streaming
        # rather than doing a second ainvoke. This is simplified for clarity.
    )
    state.update(result_state)

    # Token usage footer.
    elapsed = time.monotonic() - start_time
    if final_message is not None:
        usage = extract_usage(final_message)
        if usage:
            print_token_footer(usage, elapsed)
```

A few important design decisions in this loop:

**State mutation**: `state` is a dict that accumulates messages across turns. Each `HumanMessage` and the graph's response `AIMessage` are appended. LangGraph's reducer (`add_messages`) handles the bookkeeping.

**Streaming**: With `stream_mode="updates"`, LangGraph yields partial state updates as each node completes. We render the agent's text content as it arrives.

**Separation of concerns**: The loop itself is lean. Complex logic (memory loading, tool display, usage extraction) is delegated to helper modules.

---

## 17.4 Slash Command Dispatch

Slash commands are saathi's power-user interface. They give direct access to internal operations without going through the LLM. The dispatch works on a simple prefix check.

### The Dispatch Function

```python
async def _dispatch_slash_command(
    user_input: str,
    state: dict,
    cfg: "Settings",
    graph,
) -> Optional[str]:
    """Dispatch a slash command to the appropriate handler.

    Returns:
        "EXIT" if the session should end.
        "CLEAR" if history should be cleared.
        None for all other commands (already handled).
    """
    # Split command from arguments: "/commit feat: add logging" → ("/commit", "feat: add logging")
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # ── Built-in commands ─────────────────────────────────────────── #

    if command in ("/exit", "/quit", "/bye"):
        return "EXIT"

    elif command == "/help" or command == "/commands":
        _show_commands(cfg)

    elif command == "/clear":
        return "CLEAR"

    elif command == "/compact":
        from saathi.memory import compact_messages
        state["messages"] = await compact_messages(state["messages"], cfg)
        n = len(state["messages"])
        console.print(f"[dim]Compacted to {n} messages.[/dim]")

    elif command == "/init":
        from saathi.tools.init import run_init
        await run_init(cfg)

    elif command == "/commit":
        from saathi.tools.commit import run_commit
        await run_commit(args or None, cfg)

    elif command == "/revise-saathi-md":
        from saathi.tools.revise import run_revise_saathi_md
        await run_revise_saathi_md(cfg)

    elif command == "/code-review":
        from saathi.tools.review import run_code_review
        await run_code_review(args or None, cfg, graph, state)

    elif command == "/review":
        from saathi.tools.review import run_pr_review
        await run_pr_review(args, cfg, graph, state)

    elif command == "/doctor":
        from saathi.doctor import run_doctor
        await run_doctor(cfg)

    elif command == "/paste":
        await _handle_paste(state=state, cfg=cfg, graph=graph)

    elif command == "/rollback":
        from saathi.memory import rollback_memory
        await rollback_memory(cfg, steps=int(args) if args.isdigit() else 1)

    elif command == "/checkpoints":
        from saathi.memory import list_checkpoints
        checkpoints = await list_checkpoints(cfg)
        for i, cp in enumerate(checkpoints, 1):
            console.print(f"  {i}. {cp}")

    elif command == "/diff":
        from saathi.tools.git_diff import show_diff
        await show_diff(args or None)

    elif command == "/export":
        from saathi.export import export_session
        path = await export_session(state["messages"], args or None, cfg)
        console.print(f"[green]Exported to:[/green] {path}")

    elif command == "/copy":
        from saathi.clipboard import copy_last_response
        copy_last_response(state["messages"])
        console.print("[green]Copied to clipboard.[/green]")

    elif command == "/model":
        if args:
            cfg_dict = cfg.model_dump()
            cfg_dict["model"] = args.strip()
            from saathi.config import Settings
            new_cfg = Settings(**cfg_dict)
            cfg.__dict__.update(new_cfg.__dict__)
            console.print(f"[green]Model set to:[/green] {args.strip()}")
        else:
            console.print(f"Current model: [bold]{cfg.model}[/bold]")

    elif command == "/mode":
        # /mode fast | /mode careful | /mode creative
        _set_mode(args, cfg)

    elif command == "/memory":
        await _handle_memory_command(args, cfg)

    elif command == "/session":
        await _handle_session_command(args, state, cfg)

    elif command == "/context":
        from saathi.context import show_context_window
        show_context_window(state["messages"], cfg)

    else:
        # Check custom commands before declaring unknown.
        from saathi.custom_commands import load_custom_commands, find_custom_command
        custom_cmds = load_custom_commands(cfg.commands_dir)
        matched = find_custom_command(command, custom_cmds)
        if matched:
            from saathi.custom_commands import render_command
            rendered = render_command(matched, args)
            await execute_task(task=rendered, state=state, cfg=cfg, graph=graph)
        else:
            console.print(
                f"[yellow]Unknown command:[/yellow] {command}. "
                "Type [bold]/commands[/bold] for a list.",
                style="yellow",
            )

    return None
```

### The Command Table

Here is saathi's full slash command set:

| Command | Arguments | Description |
| --------- | ----------- | ------------- |
| `/exit`, `/quit`, `/bye` | — | End the session. |
| `/help`, `/commands` | — | Show all available commands. |
| `/clear` | — | Clear the message history. |
| `/compact` | — | Summarise and compress message history. |
| `/init` | — | Initialise `.saathi/` directory structure. |
| `/commit` | `[message]` | Stage changes and commit (asks LLM for message if omitted). |
| `/revise-saathi-md` | — | Ask the LLM to update `SAATHI.md` based on recent changes. |
| `/code-review` | `[ref]` | Review the current diff (or diff against `ref`). |
| `/review` | `<PR-URL\|branch>` | Review a GitHub PR or branch. |
| `/doctor` | — | Run health checks (Ollama, model, paths, tools). |
| `/paste` | — | Enter multi-line paste mode. |
| `/rollback` | `[N]` | Roll back memory N checkpoints (default 1). |
| `/checkpoints` | — | List available memory checkpoints. |
| `/diff` | `[ref]` | Show git diff (optionally against `ref`). |
| `/export` | `[path]` | Export conversation to a Markdown file. |
| `/copy` | — | Copy the last assistant response to clipboard. |
| `/model` | `[name]` | Show or change the current model. |
| `/mode` | `fast\|careful\|creative` | Switch temperature preset. |
| `/memory` | `show\|edit\|reset` | Manage the persistent memory file. |
| `/session` | `save\|load\|list` | Save, load, or list sessions. |
| `/context` | — | Show context window usage (tokens used vs. budget). |

Custom commands (from `.saathi/commands/*.md`) appear dynamically and fall through to the `else` branch.

---

## 17.5 `/paste` — Multi-line Input

### The Problem

The standard `input()` call reads until a newline. Pasting a multi-line code snippet into a terminal sends each line separately, so only the first line is read as the prompt. This is frustrating when you want to paste a function definition and ask "what does this do?"

Earlier versions of saathi hit this bug differently: `/paste` was not implemented, so the command fell through to the "unknown command" branch and the user got an error instead of multi-line input. The fix added explicit `/paste` handling.

### The Implementation

```python
async def _handle_paste(state: dict, cfg: "Settings", graph) -> None:
    """/paste — collect multi-line input until an empty line, then run as task.

    Usage:
        /paste
        [paste your text here]
        [paste more lines]
                        ← press Enter on an empty line to submit
    """
    console.print(
        "[dim]Paste mode: enter your text, then press Enter on an empty line to submit.[/dim]"
    )
    lines: list[str] = []

    while True:
        try:
            # We use sys.stdin.readline() directly here rather than input()
            # to handle the case where stdin is redirected (e.g., in tests).
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line or line == "\n":
                break
            lines.append(line.rstrip("\n"))
        except (EOFError, KeyboardInterrupt):
            break

    if not lines:
        console.print("[dim]No input received.[/dim]")
        return

    pasted_text = "\n".join(lines)
    console.print(f"[dim]Running with {len(lines)} line(s) of input...[/dim]")

    await execute_task(task=pasted_text, state=state, cfg=cfg, graph=graph)
```

Key design choices:

- Uses `run_in_executor(None, sys.stdin.readline)` to read stdin without blocking the event loop. A plain `input()` call would block the async loop.
- Terminates on an empty line (just Enter), which is the universal multi-line input convention in Unix-style tools (e.g., `git commit` body input).
- Falls through to `execute_task` — the pasted content becomes a task for the LLM, just like typed input.

---

## 17.6 `--print` Mode — Non-Interactive One-Shot

The `--print` flag turns saathi into a shell filter: it reads a task from the command line, runs it through the agent once, writes the result to stdout, and exits. This is the "scripting mode" described in Chapter 1.

```python
async def _print_mode(
    task: str,
    fmt: str,
    cfg: "Settings",
) -> None:
    """Non-interactive mode: run a single task and print the result.

    Writes the agent's final response to stdout.
    Writes all diagnostics (model, timing, errors) to stderr.
    Exits with:
        0 — success
        1 — runtime error (LLM call failed, tool error)
        2 — usage error (bad arguments — handled before this function)

    Args:
        task: The task string to execute.
        fmt: Output format, one of "text" or "json".
        cfg: Resolved settings.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from saathi.graph import build_graph
    from saathi.memory import load_memory
    from saathi.usage import extract_usage
    import json as json_module

    console.print(
        f"[dim]Model: {cfg.model} | Task: {task[:60]}{'...' if len(task) > 60 else ''}[/dim]"
    )

    try:
        graph = build_graph(cfg)
        state = {"messages": [HumanMessage(content=task)], "model": cfg.model}

        # Load memory context.
        memory = await load_memory(cfg.memory_dir)
        if memory:
            state["memory"] = memory

        start_time = time.monotonic()

        # Run the graph to completion (no streaming in --print mode).
        result = await graph.ainvoke(state)

        elapsed = time.monotonic() - start_time

        # Find the final AIMessage.
        final_message: Optional[AIMessage] = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final_message = msg
                break

        if final_message is None:
            console.print("[red]Error: no response from agent.[/red]")
            raise typer.Exit(code=1)

        # Collect tool calls for JSON output.
        tool_calls_summary = []
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in (msg.tool_calls or []):
                    tool_calls_summary.append({
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                    })

        # Extract usage metadata.
        usage = extract_usage(final_message)

        # ── Output ──────────────────────────────────────────────── #

        if fmt == "json":
            payload = {
                "model": cfg.model,
                "task": task,
                "response": final_message.content,
                "tool_calls": tool_calls_summary,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0) if usage else 0,
                    "output_tokens": usage.get("output_tokens", 0) if usage else 0,
                },
                "elapsed_seconds": round(elapsed, 3),
            }
            out.print(json_module.dumps(payload, indent=2), highlight=False)
        else:
            out.print(final_message.content, highlight=False)

        # Diagnostics to stderr.
        if usage:
            console.print(
                f"[dim]↳ {usage.get('input_tokens', 0):,} in · "
                f"{usage.get('output_tokens', 0):,} out · "
                f"{elapsed:.1f}s[/dim]"
            )

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if cfg.debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=1)
```

### Why stdout vs stderr Separation Matters

In the scripting mode, stdout must contain only the agent's output. If you write diagnostics to stdout, downstream shell tools (`grep`, `jq`, `awk`) break.

Consider:

```bash
# This must work:
saathi --print "List all Python files" | grep "\.py$"

# This must work:
saathi --print "Summarize the changes" --output-format json | jq .response
```

If `console.print(f"Model: {cfg.model}")` wrote to stdout, `jq` would fail with a JSON parse error. By routing all diagnostics through `Console(stderr=True)`, stdout is clean.

### Exit Codes

| Code | Meaning | When |
| ------ | --------- | ------ |
| `0` | Success | Agent responded successfully. |
| `1` | Runtime error | LLM call failed, tool raised exception, no response returned. |
| `2` | Usage error | Bad arguments (invalid `--output-format`, missing `--print` task). |

Exit code 2 is important: it signals a **usage error** (the user invoked the command incorrectly) rather than a **runtime error** (something went wrong during execution). This mirrors the convention in standard Unix tools (`curl`, `git`, etc.) and allows callers to distinguish between "I called this wrong" and "something failed at runtime".

### Shell Pipeline Examples

```bash
# Ask saathi to explain a file, capture output to a variable.
EXPLANATION=$(saathi --print "Explain src/saathi/graph.py" 2>/dev/null)

# Run a review and write to a file.
saathi --print "Review the current diff" > review.md 2>&1

# JSON output parsed with jq.
RESPONSE=$(saathi --print "List all classes" --output-format json | jq -r .response)

# Use in a Makefile target.
.PHONY: describe
describe:
    saathi --print "Describe the purpose of this project in one paragraph"

# Pipe another command's output into saathi via /paste equivalent.
git diff HEAD~1 | saathi --print "$(cat -)"
```

---

## 17.7 `--output-format` Validation — Early Exit with Code 2

The `--output-format` flag only makes sense in `--print` mode. But we want to validate it early—before the graph is built, before Ollama is contacted—so that a typo in `--output-format json5` fails immediately with a helpful error message and exit code 2, not after a 2-second startup delay.

### The Callback

```python
def validate_output_format(value: str) -> str:
    """Validate --output-format before building the graph.

    This runs as a typer callback with is_eager=True, meaning it fires
    immediately when the option is parsed, before the command body runs.

    Args:
        value: The raw string value of --output-format.

    Returns:
        The validated value (lowercased).

    Raises:
        typer.BadParameter: If value is not one of the accepted formats.
    """
    value = value.lower().strip()
    valid = {"text", "json"}
    if value not in valid:
        raise typer.BadParameter(
            f"'{value}' is not a valid output format. "
            f"Choose from: {', '.join(sorted(valid))}",
            param_hint="--output-format",
        )
    return value
```

`typer.BadParameter` causes Typer to print an error and exit with code 2. The `param_hint` parameter names the offending flag in the error message:

```error
Error: Invalid value for '--output-format': 'json5' is not a valid output format. Choose from: json, text
```

### Why `is_eager=True`?

Without `is_eager=True`, Typer processes callbacks in the order options appear on the command line, after all options are collected. `is_eager=True` runs the callback immediately when the option is encountered during parsing, before other options are processed.

For our validation callback, this is correct: we want to validate `--output-format` before any work starts.

### Testing the Validation

```python
# tests/test_print_mode.py

from typer.testing import CliRunner
from saathi.cli import app

runner = CliRunner()


def test_invalid_output_format_exits_2():
    """An invalid --output-format should exit with code 2."""
    result = runner.invoke(app, ["--print", "hello", "--output-format", "invalid"])
    assert result.exit_code == 2
    assert "invalid" in result.output.lower()
    assert "output-format" in result.output.lower()


def test_valid_output_format_text():
    """--output-format text should be accepted."""
    # This test would need Ollama running to complete; use mock in CI.
    result = runner.invoke(app, ["--output-format", "text", "--help"])
    assert result.exit_code == 0


def test_print_without_task_exits_2():
    """--print with no TASK argument should exit with code 2."""
    result = runner.invoke(app, ["--print"])
    # Typer raises Missing argument 'TASK' — exit code 2.
    assert result.exit_code == 2


def test_output_format_is_case_insensitive():
    """--output-format JSON and --output-format json should both be accepted."""
    result = runner.invoke(app, ["--output-format", "JSON", "--help"])
    assert result.exit_code == 0
```

`typer.testing.CliRunner` is the standard way to test Typer apps. It invokes the app as if from the command line and captures output, without actually running a subprocess.

---

## 17.8 Token Usage Footer

After every agent turn, saathi prints a footer showing token consumption and elapsed time:

```text
↳ 1,240 in · 312 out · 1.8s
```

This is displayed in the terminal with dim styling (grey), so it is visible but does not compete with the agent's response. It is written to stderr (the diagnostic console), not stdout, so it does not appear in `--print` mode output.

### `extract_usage` in `usage.py`

```python
# src/saathi/usage.py
"""Utilities for extracting token usage metadata from LangChain messages."""

from __future__ import annotations
from typing import Optional


def extract_usage(message) -> Optional[dict]:
    """Extract token usage from a LangChain AIMessage.

    LangChain stores usage in message.usage_metadata (preferred) or
    message.response_metadata['usage'] (legacy). We check both.

    Args:
        message: A LangChain AIMessage or similar message object.

    Returns:
        A dict with keys 'input_tokens' and 'output_tokens', or None
        if no usage data is available.

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> msg = AIMessage(
        ...     content="Hello",
        ...     usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ... )
        >>> extract_usage(msg)
        {'input_tokens': 100, 'output_tokens': 50}
    """
    # LangChain v0.2+ standardised usage_metadata.
    if hasattr(message, "usage_metadata") and message.usage_metadata:
        meta = message.usage_metadata
        return {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
        }

    # Fallback: response_metadata (varies by provider).
    if hasattr(message, "response_metadata") and message.response_metadata:
        rmeta = message.response_metadata
        # Ollama format.
        if "prompt_eval_count" in rmeta:
            return {
                "input_tokens": rmeta.get("prompt_eval_count", 0),
                "output_tokens": rmeta.get("eval_count", 0),
            }
        # OpenAI format (if ever used).
        usage = rmeta.get("usage", rmeta.get("token_usage", {}))
        if usage:
            return {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            }

    return None


def format_usage_footer(usage: dict, elapsed: float) -> str:
    """Format a token usage dict and elapsed time as a human-readable footer.

    Args:
        usage: Dict with 'input_tokens' and 'output_tokens'.
        elapsed: Elapsed time in seconds.

    Returns:
        A formatted string like '↳ 1,240 in · 312 out · 1.8s'
    """
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return f"↳ {inp:,} in · {out:,} out · {elapsed:.1f}s"
```

### Why Track Token Usage?

Even with Ollama (local, $0 cost), tracking token usage is valuable:

1. **Context window awareness.** If you are approaching the 8096-token limit, you see it in the footer numbers before the model starts truncating your history.
2. **Performance debugging.** If a response took 12 seconds, the footer tells you whether that was a large prompt (many input tokens) or a long response (many output tokens).
3. **Model comparison.** When trying different models, the footer lets you compare their verbosity (output tokens) and context sensitivity (input tokens) across turns.
4. **Future cost awareness.** If you ever switch to a cloud LLM, you already have the instrumentation in place to see exactly what each turn costs.

---

## 17.9 Rich Console — Beyond `print()`

Saathi uses [Rich](https://rich.readthedocs.io/) for terminal output. Rich is a Python library for rich text and beautiful formatting in the terminal. It supports colours, bold/italic/dim text, markdown rendering, tables, progress bars, syntax highlighting, panels, and more.

### Why Rich Over `print()`?

```python
# Without Rich:
print(f"Error: {message}")  # plain text, no colour

# With Rich:
from rich.console import Console
console = Console(stderr=True)
console.print(f"[red]Error:[/red] {message}")  # red "Error:", normal message
```

Rich uses a markup language inspired by BBCode. Tags like `[red]`, `[bold]`, `[dim]`, `[green]` apply colours and styles. `[/red]` closes the tag.

### saathi's Console Objects

```python
# In cli.py:
console = Console(stderr=True)  # diagnostics to stderr (coloured)
out = Console()                  # output to stdout (no colours in --print mode)
```

The `Console(stderr=True)` console writes to `sys.stderr`. This is for:

- Error messages
- Warnings
- Loading spinners
- The token usage footer
- Debug information

The plain `Console()` writes to `sys.stdout`. This is for:

- The agent's actual response
- The output of `--print` mode

### Markdown Rendering

Rich can render Markdown in the terminal:

```python
from rich.markdown import Markdown

md = Markdown(agent_response)
out.print(md)
```

This renders headings with bold underlines, code blocks with syntax highlighting, lists with bullet characters, and so on. For an AI agent that frequently produces Markdown responses, this is a significant UX improvement.

However, there is a caveat: Markdown rendering in the terminal only looks good in full-colour terminals. When stdout is redirected to a file or a pipe, ANSI escape codes are stripped by Rich automatically (it detects non-TTY contexts and disables formatting). This is handled correctly by Rich's `Console` class.

### Panels, Tables, and Trees

For structured output (e.g., `/doctor` results, `/context` window breakdown), Rich provides:

```python
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Panel — boxed content.
console.print(Panel("Ollama is reachable ✓", title="Health Check", style="green"))

# Table — columnar data.
table = Table(title="Settings")
table.add_column("Setting", style="cyan")
table.add_column("Value", style="green")
table.add_row("model", settings.model)
table.add_row("temperature", str(settings.temperature))
console.print(table)

# Tree — hierarchical data (e.g., tool call structure).
tree = Tree("Agent Turn")
tree.add("LLM call (512 tokens)")
tools_branch = tree.add("Tool calls")
tools_branch.add("read_file: main.py")
tools_branch.add("run_shell: pytest")
console.print(tree)
```

---

## 17.10 `display.py` — Unicode Safety on Windows

Windows is the bête noire of CLI developers. The default console encoding on Windows is often `cp1252` (Windows Latin-1) or `cp850` (Western European), not UTF-8. Characters outside these code pages—Unicode box-drawing characters (┌, ─, └), arrows (↳), checkmarks (✓), and non-ASCII symbols—cause `UnicodeEncodeError` when written to stdout or stderr.

### The Error

```text
UnicodeEncodeError: 'charmap' codec can't encode character '✓' in position 0:
character maps to <undefined>
```

This error, `✓` is `✓`, fires when saathi tries to print `✓ Ollama reachable` to a Windows terminal that is not configured for UTF-8.

### The Fix in `display.py`

```python
# src/saathi/display.py
"""Display helpers for the saathi CLI.

Handles Unicode safety on Windows, Rich console initialisation,
and formatted output functions.
"""

from __future__ import annotations

import sys
from typing import Optional

from rich.console import Console


def _reconfigure_streams() -> None:
    """Reconfigure stdout and stderr for UTF-8 on Windows.

    On Windows, sys.stdout and sys.stderr use the system code page by default
    (typically cp1252 or cp850). This causes UnicodeEncodeError for characters
    outside the code page, such as box-drawing characters and Unicode symbols.

    This function reconfigures both streams to use UTF-8 with 'replace' error
    handling (unknown characters are replaced with '?' rather than raising).
    It is a no-op on Linux and macOS, which default to UTF-8.
    """
    if sys.platform != "win32":
        return

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # reconfigure() may fail on some Windows configurations
                # (e.g., when output is redirected to a non-text stream).
                # Silently ignore — we tried.
                pass


# Run once at import time.
_reconfigure_streams()


# Safe console instances that use UTF-8.
console = Console(stderr=True, highlight=False)
out = Console(highlight=False)


def print_welcome(cfg) -> None:
    """Print the saathi welcome banner."""
    console.print(
        f"[bold cyan]saathi[/bold cyan] [dim]v0.1.0 | model: {cfg.model}[/dim]"
    )
    console.print(
        "[dim]Type your task, or /commands for available slash commands. "
        "Ctrl-D or /exit to quit.[/dim]"
    )


def print_tool_call(tool_call: dict) -> None:
    """Print a tool call notification with consistent formatting.

    Args:
        tool_call: A dict with 'name' and optional 'args' keys.
    """
    name = tool_call.get("name", "unknown_tool")
    args = tool_call.get("args", {})

    # Summarise args to a short string.
    if args:
        args_str = ", ".join(
            f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3]
        )
        if len(args) > 3:
            args_str += f", +{len(args) - 3} more"
    else:
        args_str = ""

    console.print(
        f"  [dim cyan]tool:[/dim cyan] [bold]{name}[/bold]"
        + (f"({args_str})" if args_str else "()")
    )


def print_token_footer(usage: dict, elapsed: float) -> None:
    """Print the token usage footer to stderr.

    Args:
        usage: Dict with 'input_tokens' and 'output_tokens'.
        elapsed: Elapsed time in seconds.
    """
    from saathi.usage import format_usage_footer
    footer = format_usage_footer(usage, elapsed)
    console.print(f"[dim]{footer}[/dim]")


def print_error(message: str, *, hint: Optional[str] = None) -> None:
    """Print an error message with optional hint."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        console.print(f"[dim]Hint: {hint}[/dim]")


def print_response(content: str) -> None:
    """Print the agent's response to stdout."""
    from rich.markdown import Markdown
    md = Markdown(content)
    out.print(md)
```

The key line is:

```python
stream.reconfigure(encoding="utf-8", errors="replace")
```

- `encoding="utf-8"` tells the stream to encode output as UTF-8.
- `errors="replace"` means that if a character still cannot be encoded (rare with UTF-8, but possible), replace it with `?` rather than raising an exception.

This is called at module import time via `_reconfigure_streams()`. The moment `display.py` is imported (which happens at CLI startup), the streams are reconfigured. No code elsewhere needs to worry about Windows encoding.

---

## 17.11 Custom Commands — `.saathi/commands/*.md`

Saathi supports user-defined slash commands. These are Markdown files stored in `.saathi/commands/`. The filename (without `.md`) becomes the command name.

### File Format

```markdown
<!-- .saathi/commands/explain-func.md -->
---
description: Explain a Python function in plain English.
args: "FUNCTION_NAME"
---

Explain the Python function `$ARGS` in this project.
Find the function definition, understand what it does, and explain it in
plain English as if to a junior developer. Include:
- What the function's purpose is
- What its parameters are
- What it returns
- Any notable edge cases or caveats

Keep the explanation under 200 words.
```

Usage:

```text
/explain-func build_graph
```

The string `$ARGS` is replaced with the argument text (`build_graph`), and the resulting text is submitted as a task to the LLM.

### `custom_commands.py`

```python
# src/saathi/custom_commands.py
"""Custom slash command loading and rendering.

Custom commands live in .saathi/commands/*.md. Each file's stem becomes
a command name (e.g., explain-func.md → /explain-func).

File format:
    Optional YAML frontmatter (between --- markers) with:
        description: Short description shown in /commands output.
        args: Description of expected arguments (displayed in help).
    Body: The prompt template. Use $ARGS for argument substitution.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def load_custom_commands(commands_dir: Path) -> dict[str, dict]:
    """Load all custom commands from a directory.

    Args:
        commands_dir: Path to the directory containing *.md command files.

    Returns:
        A dict mapping command names (with leading slash) to their metadata:
            {
                "/explain-func": {
                    "name": "explain-func",
                    "description": "...",
                    "args": "...",
                    "template": "...",
                    "path": Path("..."),
                }
            }
    """
    commands: dict[str, dict] = {}

    if not commands_dir.exists():
        return commands

    for path in sorted(commands_dir.glob("*.md")):
        cmd = _parse_command_file(path)
        if cmd:
            commands[f"/{cmd['name']}"] = cmd

    return commands


def _parse_command_file(path: Path) -> Optional[dict]:
    """Parse a single command file.

    Args:
        path: Path to the .md file.

    Returns:
        A dict with command metadata, or None if the file is invalid.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    name = path.stem.lower()

    # Parse optional YAML frontmatter.
    description = f"Custom command: {name}"
    args_hint = ""
    template = content

    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
    )
    if frontmatter_match:
        fm_text, template = frontmatter_match.group(1), frontmatter_match.group(2)
        for line in fm_text.splitlines():
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"\'')
            elif line.startswith("args:"):
                args_hint = line.split(":", 1)[1].strip().strip('"\'')

    template = template.strip()

    return {
        "name": name,
        "description": description,
        "args": args_hint,
        "template": template,
        "path": path,
    }


def find_custom_command(
    command: str,
    custom_cmds: dict[str, dict],
) -> Optional[dict]:
    """Look up a command by name.

    Args:
        command: The command string including leading slash (e.g., "/explain-func").
        custom_cmds: The dict returned by load_custom_commands().

    Returns:
        The command dict if found, None otherwise.
    """
    return custom_cmds.get(command.lower())


def render_command(cmd: dict, args: str) -> str:
    """Render a command template with argument substitution.

    Replaces all occurrences of $ARGS (case-sensitive) in the template
    with the provided args string.

    Args:
        cmd: A command dict from load_custom_commands().
        args: The argument string from the user (may be empty).

    Returns:
        The rendered prompt string.

    Example:
        >>> cmd = {"template": "Explain the function $ARGS in detail."}
        >>> render_command(cmd, "build_graph")
        'Explain the function build_graph in detail.'
    """
    template = cmd["template"]
    return template.replace("$ARGS", args)
```

### Dispatch in the REPL

In `_dispatch_slash_command`, after all built-in commands are checked, custom commands are the fallback:

```python
else:
    # Check custom commands before declaring unknown.
    custom_cmds = load_custom_commands(cfg.commands_dir)
    matched = find_custom_command(command, custom_cmds)
    if matched:
        rendered = render_command(matched, args)
        await execute_task(task=rendered, state=state, cfg=cfg, graph=graph)
    else:
        console.print(
            f"[yellow]Unknown command:[/yellow] {command}. "
            "Type [bold]/commands[/bold] for a list.",
        )
```

This means custom commands can never shadow built-in commands (they are checked last). If a user creates a custom command named `exit.md`, the built-in `/exit` still works.

---

## 17.12 `/doctor` — Health Checks

`/doctor` is the first thing to run when saathi is not working. It performs a series of non-destructive checks and prints the results with clear pass/fail indicators.

```python
# src/saathi/doctor.py
"""Health check command for saathi (/doctor)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from saathi.config import Settings

console = Console(stderr=True)


async def run_doctor(cfg: Settings) -> None:
    """Run all health checks and display results.

    Never raises. Each check is wrapped in try/except so that a failing
    check does not prevent subsequent checks from running.
    """
    table = Table(title="/doctor — saathi health check", show_header=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    # 1. Ollama reachability.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{cfg.ollama_base_url}/api/tags")
            if r.status_code == 200:
                table.add_row("Ollama reachable", "[green]✓ pass[/green]", cfg.ollama_base_url)
            else:
                table.add_row(
                    "Ollama reachable",
                    "[red]✗ fail[/red]",
                    f"HTTP {r.status_code} from {cfg.ollama_base_url}",
                )
    except Exception as exc:
        table.add_row(
            "Ollama reachable",
            "[red]✗ fail[/red]",
            f"Cannot connect: {exc}. Is `ollama serve` running?",
        )

    # 2. Model available.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{cfg.ollama_base_url}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Check exact match or prefix match (qwen2.5:14b matches qwen2.5:14b).
                model_available = any(
                    m == cfg.model or m.startswith(cfg.model.split(":")[0])
                    for m in models
                )
                if model_available:
                    table.add_row(
                        "Model available",
                        "[green]✓ pass[/green]",
                        f"{cfg.model} found",
                    )
                else:
                    available = ", ".join(models[:5]) or "none"
                    table.add_row(
                        "Model available",
                        "[yellow]⚠ warn[/yellow]",
                        f"{cfg.model} not found. Available: {available}. "
                        f"Run: ollama pull {cfg.model}",
                    )
    except Exception as exc:
        table.add_row("Model available", "[dim]? skip[/dim]", f"Skipped (Ollama check failed): {exc}")

    # 3. Memory directory writable.
    try:
        cfg.memory_dir.mkdir(parents=True, exist_ok=True)
        test_file = cfg.memory_dir / ".doctor_test"
        test_file.write_text("ok")
        test_file.unlink()
        table.add_row(
            "Memory dir writable",
            "[green]✓ pass[/green]",
            str(cfg.memory_dir),
        )
    except Exception as exc:
        table.add_row(
            "Memory dir writable",
            "[red]✗ fail[/red]",
            f"{cfg.memory_dir}: {exc}",
        )

    # 4. Sessions directory writable.
    try:
        cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        table.add_row(
            "Sessions dir writable",
            "[green]✓ pass[/green]",
            str(cfg.sessions_dir),
        )
    except Exception as exc:
        table.add_row(
            "Sessions dir writable",
            "[red]✗ fail[/red]",
            f"{cfg.sessions_dir}: {exc}",
        )

    # 5. git on PATH.
    git_path = shutil.which("git")
    if git_path:
        table.add_row("git on PATH", "[green]✓ pass[/green]", git_path)
    else:
        table.add_row(
            "git on PATH",
            "[yellow]⚠ warn[/yellow]",
            "git not found. /commit and /diff will not work.",
        )

    # 6. patch on PATH (needed for apply_patch tool).
    patch_path = shutil.which("patch")
    if patch_path:
        table.add_row("patch on PATH", "[green]✓ pass[/green]", patch_path)
    else:
        table.add_row(
            "patch on PATH",
            "[yellow]⚠ warn[/yellow]",
            "patch not found. apply_patch tool will not work.",
        )

    # 7. Python version.
    import sys
    py = sys.version.split()[0]
    major, minor, *_ = sys.version_info
    if major >= 3 and minor >= 11:
        table.add_row("Python version", "[green]✓ pass[/green]", py)
    else:
        table.add_row(
            "Python version",
            "[yellow]⚠ warn[/yellow]",
            f"{py} — saathi requires Python 3.11+",
        )

    console.print(table)
```

Design principles of `/doctor`:

1. **Never raises.** Each check is wrapped in `try/except`. A failing check prints a failure row; it does not abort the rest.
2. **Always actionable.** Failure messages include what to do: "Run: `ollama pull qwen2.5:14b`".
3. **Three outcomes.** `✓ pass` (green), `✗ fail` (red), `⚠ warn` (yellow). Warnings are "works but degraded"; failures are "will not work".
4. **Fast.** All checks run in under 5 seconds. The Ollama check has a 5-second timeout; if Ollama is not running, the check fails fast rather than hanging.

---

## 17.13 Session Management — `/session`

Sessions allow saving the full conversation history to disk and reloading it later. This is useful for long-running tasks that span multiple terminal sessions.

```python
async def _handle_session_command(args: str, state: dict, cfg: "Settings") -> None:
    """Handle /session subcommands.

    Subcommands:
        /session save [name]   — save current state to .saathi/sessions/
        /session load <name>   — load a saved session
        /session list          — list all saved sessions
    """
    import json
    from datetime import datetime

    cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
    parts = args.strip().split(maxsplit=1)
    subcommand = parts[0].lower() if parts else "list"
    session_name = parts[1].strip() if len(parts) > 1 else None

    if subcommand == "save":
        # Generate a name if none provided.
        if not session_name:
            session_name = datetime.now().strftime("session-%Y%m%d-%H%M%S")

        # Sanitise name.
        safe_name = re.sub(r"[^\w\-]", "_", session_name)
        path = cfg.sessions_dir / f"{safe_name}.json"

        # Serialise messages.
        messages_data = []
        for msg in state.get("messages", []):
            messages_data.append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None),
            })

        payload = {
            "name": safe_name,
            "saved_at": datetime.now().isoformat(),
            "model": cfg.model,
            "messages": messages_data,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Session saved:[/green] {path}")

    elif subcommand == "load":
        if not session_name:
            console.print("[yellow]Usage:[/yellow] /session load <name>")
            return
        safe_name = re.sub(r"[^\w\-]", "_", session_name)
        path = cfg.sessions_dir / f"{safe_name}.json"
        if not path.exists():
            console.print(f"[red]Session not found:[/red] {safe_name}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruct messages.
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        type_map = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
        }
        messages = []
        for m in payload.get("messages", []):
            cls = type_map.get(m["type"])
            if cls:
                messages.append(cls(content=m.get("content", "")))
        state["messages"] = messages
        console.print(
            f"[green]Session loaded:[/green] {safe_name} "
            f"({len(messages)} messages, saved {payload.get('saved_at', 'unknown')})"
        )

    elif subcommand == "list":
        sessions = sorted(cfg.sessions_dir.glob("*.json"))
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
            return
        table = Table(title="Saved Sessions")
        table.add_column("Name", style="cyan")
        table.add_column("Saved At", style="dim")
        table.add_column("Messages", justify="right")
        for s in sessions:
            try:
                data = json.loads(s.read_text(encoding="utf-8"))
                table.add_row(
                    s.stem,
                    data.get("saved_at", "?"),
                    str(len(data.get("messages", []))),
                )
            except Exception:
                table.add_row(s.stem, "?", "?")
        console.print(table)

    else:
        console.print(
            f"[yellow]Unknown /session subcommand:[/yellow] {subcommand}. "
            "Use save, load, or list."
        )
```

---

## 17.14 `/export` — Saving the Conversation to Markdown

`/export` saves the conversation to a Markdown file for external sharing, archiving, or further editing.

```python
# src/saathi/export.py
"""Export conversation history to Markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage


async def export_session(
    messages: list,
    path_arg: Optional[str],
    cfg,
) -> Path:
    """Export the conversation to a Markdown file.

    Args:
        messages: The message list from agent state.
        path_arg: User-supplied path (or None for auto-generated).
        cfg: Current settings.

    Returns:
        Path to the written file.
    """
    # Resolve output path.
    if path_arg:
        output_path = Path(path_arg).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"saathi-export-{timestamp}.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# saathi Conversation Export",
        f"",
        f"- **Exported**: {datetime.now().isoformat()}",
        f"- **Model**: {cfg.model}",
        f"- **Messages**: {len(messages)}",
        f"",
        "---",
        "",
    ]

    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            lines.append(f"## Turn {i} — Human")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
        elif isinstance(msg, AIMessage):
            lines.append(f"## Turn {i} — Assistant")
            lines.append("")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"*Tool call: `{tc.get('name')}`*")
                    lines.append("")
            if msg.content:
                lines.append(msg.content)
            lines.append("")
        else:
            lines.append(f"## Turn {i} — {msg.__class__.__name__}")
            lines.append("")
            lines.append(str(getattr(msg, "content", "")))
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
```

The export format is simple and readable:

```markdown
# saathi Conversation Export

- **Exported**: 2026-07-09T14:30:00
- **Model**: qwen2.5:14b
- **Messages**: 6

---

## Turn 1 — Human

Explain the build_graph function.

## Turn 2 — Assistant

*Tool call: `read_file`*

The `build_graph` function constructs the LangGraph `StateGraph` that...
```

---

## Summary

Saathi's CLI architecture demonstrates several important patterns for building robust command-line AI tools:

- **Typer** turns type hints into a fully featured CLI with minimal boilerplate. Options, arguments, callbacks, and validation come for free.
- **`asyncio.run`** bridges the sync Typer entry point to the async LangGraph execution.
- **The REPL loop** is simple: read input → dispatch → execute → display → repeat. Complexity lives in the dispatch handlers, not the loop.
- **Slash commands** use a simple prefix dispatch table with custom commands as the fallback. Built-ins cannot be shadowed.
- **`/paste`** solves multi-line input with `asyncio.run_in_executor` for non-blocking stdin reading.
- **`--print` mode** separates stdout (output) from stderr (diagnostics), enabling shell pipeline usage.
- **Exit codes** (0/1/2) follow Unix conventions and enable scripting.
- **Rich Console** provides colours, tables, Markdown rendering, and panels without terminal glitches.
- **Unicode safety on Windows** requires `stream.reconfigure(encoding="utf-8", errors="replace")` at startup.
- **`/doctor`** runs defensive health checks that never raise, always give actionable output, and complete in under 5 seconds.
- **Session save/load** uses JSON files in `.saathi/sessions/`, reconstructing LangChain message objects.
- **`/export`** produces clean, shareable Markdown.

The full CLI source is `src/saathi/cli.py`, with display helpers in `src/saathi/display.py`, custom commands in `src/saathi/custom_commands.py`, and export logic in `src/saathi/export.py`.
