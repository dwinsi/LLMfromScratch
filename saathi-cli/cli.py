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
import os

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent, compact_history, load_llm, OLLAMA_MODEL
from memory_store import MemoryStore
from tools import get_turn_snapshot, reset_turn_snapshot, set_memory_store

from commands import (
    SessionState,
    ThinkingSpinner,
    console,
    handle_checkpoints,
    handle_clear,
    handle_compact,
    handle_context,
    handle_copy,
    handle_diff,
    handle_export,
    handle_memory,
    handle_mode,
    handle_model,
    handle_paste,
    handle_rollback,
    handle_session,
    print_context,
    resolve_paths,
)


def print_banner() -> None:
    title    = Text("saathi", style="bold cyan") + Text(" — your coding companion")
    subtitle = Text("Powered by Gemma 4 via Ollama", style="dim")
    console.print(Panel(title + "\n" + subtitle, border_style="cyan", padding=(0, 2)))
    console.print("[dim]Type a task and press Enter. Type [bold]help[/bold] for commands.[/dim]\n")


def print_help() -> None:
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


def run_interactive_session(model_id: str, context_paths: list[str] | None = None) -> None:
    """
    Run an interactive terminal session.
    Connects to Ollama once at startup, then accepts tasks until the user quits.
    context_paths: optional list of resolved file/folder paths to scope the agent to.
    """
    print_banner()

    context_paths = context_paths or []
    print_context(context_paths)

    project_dir = context_paths[0] if context_paths else os.getcwd()
    memory      = MemoryStore(project_dir=project_dir)
    set_memory_store(memory)

    memory_block = memory.format_for_prompt()
    if memory_block:
        console.print("[dim]Memory loaded — injecting saved facts into context.[/dim]\n")

    llm = load_llm(model_id)

    state = SessionState(
        model_id      = model_id,
        context_paths = context_paths,
        memory        = memory,
        llm           = llm,
        agent_executor = build_agent(llm, context_paths or None, memory_block),
    )

    console.print("[bold green]Agent ready.[/bold green] What would you like to do?\n")

    while True:
        try:
            mode_tag   = f" [dim]({state.current_mode})[/dim]" if state.current_mode else ""
            user_input = Prompt.ask(f"[bold cyan]You[/bold cyan]{mode_tag}").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                console.print("[dim]Ending session.[/dim]")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "clear":
                handle_clear(state)
                continue

            if user_input.lower() == "/compact":
                handle_compact(state)
                continue

            if user_input.lower().startswith("/context"):
                handle_context(state, user_input.split()[1:])
                continue

            if user_input.lower().startswith("/memory"):
                handle_memory(state, user_input.split())
                continue

            if user_input.lower().startswith("/rollback"):
                handle_rollback(state, user_input.split())
                continue

            if user_input.lower() == "/checkpoints":
                handle_checkpoints(state)
                continue

            if user_input.lower() == "/diff":
                handle_diff(state)
                continue

            if user_input.lower() == "/export":
                handle_export(state)
                continue

            if user_input.lower() == "/copy":
                handle_copy(state)
                continue

            if user_input.lower() == "/paste":
                pasted = handle_paste()
                if pasted is None:
                    continue
                user_input = pasted

            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) < 2:
                    console.print("[yellow]Usage: /model <model-id>[/yellow]\n")
                    continue
                handle_model(state, parts[1])
                continue

            if user_input.lower().startswith("/mode"):
                handle_mode(state, user_input.split())
                continue

            if user_input.lower().startswith("/session"):
                handle_session(state, user_input.split())
                continue

            # ----------------------------------------------------------------
            # Agent turn
            # ----------------------------------------------------------------
            reset_turn_snapshot()
            history_len_before = len(state.history)
            state.history.append(HumanMessage(content=user_input))
            messages_to_send = compact_history(state.history)

            console.print()
            spinner          = ThinkingSpinner()
            spinner.start()
            state.final_answer = None
            live: Live | None  = None

            try:
                for chunk in state.agent_executor.stream(
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
                                console.print(Panel(
                                    Syntax(preview, "text", theme="monokai", word_wrap=True),
                                    title="[dim]observation[/dim]",
                                    border_style="dim",
                                    padding=(0, 1),
                                ))
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
                                state.final_answer = (state.final_answer or "") + msg.content
                                live.update(Markdown(state.final_answer))
            finally:
                if live:
                    live.stop()
                spinner.stop()

            if not state.final_answer:
                console.print(Rule("[bold cyan]saathi[/bold cyan]", style="cyan"))
                console.print(Markdown("No output returned."))

            console.print()

            checkpoints_entry = {
                "task":        user_input,
                "history_len": history_len_before,
                "files":       get_turn_snapshot(),
            }
            state.checkpoints.append(checkpoints_entry)

            if state.final_answer:
                state.history.append(AIMessage(content=state.final_answer))

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
