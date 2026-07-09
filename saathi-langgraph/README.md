# Saathi LangGraph

A production-grade local coding agent powered by **LangGraph** and **Ollama** — the same features as saathi-cli, rebuilt on the 2026 stack.

📐 **[ARCHITECTURE.md](ARCHITECTURE.md)** — LangGraph concepts, enterprise patterns, and diagrams · 🧪 **[tests/README.md](tests/README.md)** — how to run the test suite · 🖥️ **[docs/ollama-remote.md](docs/ollama-remote.md)** — run the model on a remote server

## What's new vs. saathi-cli

| Feature | saathi-cli (LangChain) | saathi-langgraph |
| --------- | ---------------------- | ------------------ |
| Agent framework | `create_agent()` | `StateGraph` + `ToolNode` |
| Checkpointing | Manual JSON snapshots | `AsyncSqliteSaver` (built-in) |
| Rollback | Custom file restore | `graph.aget_state_history()` |
| Async | No | Yes — `astream_events` throughout |
| Config | Hardcoded constants | `pydantic-settings` + `.env` |
| Packaging | `requirements.txt` | `pyproject.toml` + `hatch` |
| CLI framework | `argparse` | `typer` |
| State | Global mutable dicts | Typed `AgentState` TypedDict |

## Setup

```bash
# 1. Install uv or hatch
pip install hatch

# 2. Create virtualenv and install
cd saathi-langgraph
hatch env create
hatch shell

# 3. Configure
cp .env.example .env
# Edit .env to taste

# 4. Start Ollama
ollama serve
ollama pull gemma4:12b

# 5. Run
python -m saathi
# or
saathi
```

## Quick start

```bash
# Default model
saathi

# Different model
saathi --model gemma4:27b

# Scope to a directory
saathi --context ./src --context ./tests
```

## Scripting mode (`--print`)

Run a single task non-interactively and exit — for shell pipelines and CI:

```bash
# plain text: just the final answer on stdout
saathi --print "what does src/saathi/cli.py do?"

# machine-readable JSON (response + tool calls + token usage)
saathi -p "summarize the tools" --output-format json | jq .response
```

Diagnostics go to **stderr**, results to **stdout**, so piping stays clean.
Exit codes: `0` success, `1` runtime error (e.g. Ollama down), `2` bad usage.
The JSON payload has `model`, `task`, `response`, `tool_calls`, and `usage`.

## Commands

| Input | Action |
| ------- | -------- |
| `<task>` | Run a task through the agent |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |
| `/init` | Scan the repo and generate a `SAATHI.md` |
| `/revise-saathi-md` | Update `SAATHI.md` with this session's learnings |
| `/commit` | Review changes and create a git commit |
| `/code-review` | Multi-reviewer review of the working diff |
| `/doctor` | Health check: Ollama, model, memory dirs, git/patch |
| `/context <path> ...` | Scope agent to files/folders |
| `/context` | Clear scope |
| `/rollback [n]` | Undo last n turns via LangGraph checkpoints |
| `/checkpoints` | List all checkpoint snapshots |
| `/compact` | Summarize old turns to free context window |
| `/diff` | Show file changes this session |
| `/export` | Save conversation to Markdown |
| `/copy` | Copy last response to clipboard |
| `/paste` | Multi-line input mode |
| `/model <id>` | Switch Ollama model mid-session |
| `/mode explain\|refactor\|debug` | Agent behaviour preset |
| `/memory list\|save\|delete\|clear` | Manage persistent facts |
| `/session save\|load\|list` | Save/restore sessions |

## Tools

| Tool | Description |
| ------ | ------------- |
| `read_file` | Read file contents (up to 200 KB) |
| `write_file` | Create or overwrite a file |
| `patch_file` | Apply a unified diff patch |
| `list_directory` | List files with sizes |
| `run_bash` | Execute shell commands (60s timeout) |
| `search_in_file` | Regex search within one file |
| `search_across_files` | Recursive grep across a directory |
| `search_web` | DuckDuckGo (default) or Brave Search |
| `save_memory` | Persist a fact to global/project memory |
| `recall_memory` | Read all remembered facts |
| `git_status` | Show working-tree status |
| `git_diff` / `git_diff_staged` | Show unstaged / staged changes |
| `git_log` | Recent commits (one-line) |
| `git_commit` | Create a commit (optionally `git add -A` first) |

## Architecture

```folder
saathi/
├── cli.py              Typer app + async REPL loop
├── config.py           Pydantic-settings (reads .env)
├── project_context.py  SAATHI.md discovery + loading
├── diagnostics.py      /doctor health checks
├── agent/
│   ├── state.py        AgentState TypedDict
│   ├── prompts.py      System prompt + mode addenda
│   ├── nodes.py        LangGraph node: calls LLM
│   ├── tool_node.py    Parallel tool node (blocking + hooks)
│   └── graph.py        StateGraph + AsyncSqliteSaver
├── tools/
│   ├── filesystem.py   read/write/patch/list
│   ├── shell.py        run_bash
│   ├── search.py       file + web search
│   ├── git.py          status/diff/log/commit
│   └── memory_tools.py save/recall memory
├── hooks/
│   └── runner.py       pre/post_tool, post_turn, block_paths
├── memory/
│   └── store.py        Two-scope JSON persistence
├── session/
│   └── manager.py      Save/load sessions as JSON
└── ui/
    ├── display.py      Rich console helpers
    └── commands.py     Slash command handlers
```

## Modes

- **explain** — reads only, cites file:line, no modifications
- **refactor** — uses `patch_file` over `write_file`, explains every change
- **debug** — reproduces bug first, smallest possible fix

## Project instructions (SAATHI.md)

Drop a `SAATHI.md` at your project root (mirrors Claude Code's `CLAUDE.md`). On
startup Saathi walks from the cwd up to your home directory, loads every
`SAATHI.md` it finds (nearest wins), and injects them into the system prompt.

- `/init` — let the agent scan the repo and write a `SAATHI.md` for you
- `/revise-saathi-md` — after a working session, fold what was learned back into
  `SAATHI.md` (conventions, gotchas, decisions) so the next session starts smarter
- Edit it by hand afterwards; changes apply on the next session

## Token usage

After every turn Saathi prints a dim footer with token counts and wall-clock time:

```text
↳ 1,240 in · 312 out · 1.8s
```

Counts come from Ollama's `usage_metadata` / `prompt_eval_count` fields.

## Health check

`/doctor` verifies Ollama is reachable, the configured model is pulled, the
global and project memory directories are writable, and that `git`/`patch` are
on PATH — a one-stop troubleshooter.

Run with `--debug` for structured (`structlog`) logs of tool blocks, errors, and
hook runs. Logs are quiet by default and always go to stderr, so `--print`
output stays clean.

Transient Ollama **connection** failures (server still starting up) are retried
with exponential backoff — tune via `SAATHI_OLLAMA_MAX_RETRIES` and
`SAATHI_OLLAMA_RETRY_BASE_DELAY`. Slow responses are not retried.

## Code review

`/code-review` runs several **specialist reviewers** over your uncommitted diff
(`git diff HEAD`) — bugs, error-handling, design, and security — each returning
structured findings with a **confidence score**. Findings below the threshold are
dropped to cut noise, and the rest are shown ranked by severity:

```
Code review — 2 finding(s)

╭─ HIGH 90% calc.py:2 (bugs) ─────────────────────────────╮
│ Division by zero is unhandled when b == 0.              │
│ → Guard against b == 0 or catch ZeroDivisionError.     │
╰─────────────────────────────────────────────────────────╯
```

- The reviewers are dispatched concurrently (`asyncio.gather`); actual parallelism
  depends on your Ollama server (`OLLAMA_NUM_PARALLEL`).
- Tune the noise filter with `SAATHI_REVIEW_MIN_CONFIDENCE` (default 70).

## MCP servers

Connect external [Model Context Protocol](https://modelcontextprotocol.io)
servers and their tools join the agent's toolset automatically. Declare them in
`.saathi/mcp.json` (Claude-Desktop style):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "weather": { "url": "http://localhost:8000/mcp/" }
  }
}
```

- `command`/`args` → stdio transport; `url` → HTTP (`transport` inferred, or set
  it explicitly to `stdio` / `streamable_http` / `sse`).
- On startup you'll see `✓ N MCP tool(s) from M server(s)`.
- Loading is **best-effort**: a broken or unreachable server logs a warning and
  is skipped — it never blocks startup.
- A minimal example server ships in [`examples/mcp_echo_server.py`](examples/mcp_echo_server.py);
  see [`examples/mcp.example.json`](examples/mcp.example.json).

## Custom commands

Drop markdown files in `.saathi/commands/` to define your own slash commands
(mirrors Claude Code's project commands). A file `review-diff.md` registers
`/review-diff`; its content is the prompt that runs. See [`examples/commands/`](examples/commands).

```bash
mkdir -p .saathi/commands
cp examples/commands/*.md .saathi/commands/
saathi        # startup shows: ✓ 2 custom command(s): /explain, /review-diff
```

- If the template contains `$ARGS`, text typed after the command is substituted
  there (e.g. `/explain src/saathi/cli.py`); otherwise it's appended.
- `/commands` lists the loaded custom commands.
- Built-in commands always take precedence over same-named custom ones.

## History compaction

Long sessions eventually fill the context window. Saathi summarizes the older
turns into a single summary message and keeps recent turns verbatim:

- **`/compact`** — do it now
- **Automatic** — before a turn, if the history estimate exceeds 75% of the
  context window (`SAATHI_CONTEXT_WINDOW`), Saathi compacts first

The cut is made at a **user-turn boundary**, so the retained tail is always a
valid sequence (never a dangling tool result). Compaction continues on a fresh
checkpoint thread, so the summarized history becomes the new baseline — the
trade-off is that `/rollback` cannot cross a compaction boundary.

## Parallel tool execution

When the model emits several tool calls in one turn (e.g. reading five files),
Saathi runs them **concurrently**. Each call flows through its own
`block-check → pre_tool hook → execute → post_tool hook` pipeline, and all
pipelines run under an `asyncio` semaphore bounded by `SAATHI_MAX_PARALLEL_TOOLS`
(default 8). A batch finishes in about the time of the slowest call rather than
the sum. Every tool call is still answered with exactly one result — including
blocked calls and calls that raise — so the agent loop never desyncs.

## Hooks

Drop a `.saathi/hooks.json` (copy `hooks.example.json`) to run shell commands
around agent activity — auto-format, lint, tests, and sensitive-file protection.

```json
{
  "pre_tool":   ["echo about to run $SAATHI_TOOL_NAME >&2"],
  "post_tool":  ["ruff format $SAATHI_TOOL_ARG_PATH"],
  "post_turn":  ["pytest -q"],
  "block_paths": ["*.env", "**/secrets/*", "*.pem"]
}
```

- **pre_tool** — runs before each tool call; a **non-zero exit blocks the call**
  and the output is returned to the model as the reason
- **post_tool** — runs after each successful tool call (e.g. format the edited file)
- **post_turn** — runs after every completed turn (e.g. run the test suite)
- **block_paths** — glob patterns; any `write_file`/`patch_file` targeting a match
  is refused before it runs — mirrors Claude Code's sensitive-file protection

Commands receive context via environment variables: `SAATHI_EVENT`,
`SAATHI_TOOL_NAME`, `SAATHI_TOOL_ARGS` (JSON), and `SAATHI_TOOL_ARG_PATH` (the
target path, when the tool has one).

## Memory

- **Global** (`~/.saathi/memory.json`) — user preferences and style
- **Project** (`.saathi/memory.json`) — stack, entry points, architecture facts
- Project keys override global keys for the same name
- Both are injected into the system prompt automatically

## Testing

An offline `pytest` suite lives in [`tests/`](tests/README.md) — 114 tests, no
Ollama or network needed.

```bash
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pytest
```

Covers tool registration, parallel execution + semaphore cap, hook blocking,
`SAATHI.md` loading, memory, graph compilation, and token-usage extraction.
`asyncio_mode = "auto"` means async tests need no decorator; everything writes
to `tmp_path`, so no test touches your real `~/.saathi` or repo.

### Quality gates

The same checks run in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
on every push and PR, across Python 3.12 and 3.13:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy src                # type check
pytest                  # tests
```

## LangGraph checkpointing

Every agent turn is automatically checkpointed in `.saathi/checkpoints.db` (SQLite).

- `/checkpoints` — browse all snapshots with message counts
- `/rollback 2` — rewind to state before the last 2 turns

No custom rollback code needed — this is built into LangGraph's `AsyncSqliteSaver`.
