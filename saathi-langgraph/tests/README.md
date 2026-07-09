# Saathi Test Suite

A `pytest` suite of **104 tests** covering the whole application. It runs
**offline** — no Ollama server, no network, no API keys (one MCP test spawns a
local Python echo server over stdio). Tests write only to a temporary directory,
so running the suite never touches your real `~/.saathi` memory or git repository.

---

## 1. One-time setup

Install the project with its `dev` extras (pytest, pytest-asyncio, ruff, mypy)
into the virtual environment:

```bash
# with uv (fast)
uv pip install -e ".[dev]"

# or with plain pip
pip install -e ".[dev]"
```

You only need to do this once (or after changing dependencies).

---

## 2. Running the tests

All commands are run from the project root (`saathi-langgraph/`).

### Run everything

```bash
pytest
```

Expected output ends with a line like:

```text
104 passed in 24.09s
```

### Run one file

```bash
pytest tests/test_hooks.py
```

### Run one specific test

Use `::` to point at a single function:

```bash
pytest tests/test_parallel.py::test_calls_run_in_parallel
```

### Run tests whose name matches a keyword

`-k` filters by substring across all files:

```bash
pytest -k "parallel or hook"      # anything about parallelism or hooks
pytest -k "block"                 # every blocking-related test
pytest -k "not git"               # skip the git tests
```

### See each test name as it runs

```bash
pytest -v
```

### Show print output / live logs

By default pytest hides stdout from passing tests. To see it:

```bash
pytest -s
```

### Stop at the first failure

```bash
pytest -x
```

### Re-run only what failed last time

```bash
pytest --lf
```

> **Tip:** if the `pytest` command isn't found, call it through the venv
> explicitly: `./.venv/Scripts/python.exe -m pytest` (Windows) or
> `.venv/bin/python -m pytest` (macOS/Linux).

---

## 3. Reading the results

- `.` — a passing test
- `F` — a failed assertion
- `E` — an error (exception during the test)
- `s` — skipped (e.g. git tests when `git` isn't installed)

A failure prints the exact assertion and the values involved, plus the file and
line number, so you can jump straight to the problem.

---

## 4. What each test file checks

### `test_imports.py` (3 tests)

Confirms the package imports cleanly and that **all 15 tools** are registered
exactly once (no missing or duplicate tools). A fast canary — if this fails,
something structural is broken.

### `test_config.py` (3 tests)

The `pydantic-settings` configuration: sensible defaults, that `SAATHI_*`
environment variables override them, and that the history token budget is 75% of
the context window.

### `test_tools.py` (8 tests)

The core file/search/shell tools:

- `write_file` then `read_file` round-trips content
- reading a missing file returns a friendly error (no crash)
- `list_directory` shows files and folders
- `patch_file` reports a missing target gracefully
- `search_in_file` finds regex matches with line numbers
- `search_across_files` greps a directory
- `run_bash` echoes output **and** refuses a dangerous command from the denylist

### `test_snapshots.py` (7 tests)

The file-change snapshots behind `/diff` (regression coverage for the bug that
once made `/diff` always say "no changes"):

- a new file snapshots as empty, an existing file snapshots its original content
- repeated writes in one turn keep the **first** original, not the intermediate
- `clear_turn_snapshots` empties the store
- `/diff` renders modifications, reports deletions, and says "no changes" when
  content is unchanged

### `test_git.py` (3 tests)

Git tools **without ever creating a commit or touching a real repo**:

- `git_status` / `git_log` return a clean "not a git repository" message when run
  in an empty temp directory
- the "git binary missing" path is exercised by faking a `FileNotFoundError`

These auto-**skip** if `git` isn't on your `PATH`.

### `test_parallel.py` (4 tests)

The headline feature — parallel tool execution:

- **speedup**: 5 tools that each sleep 0.3s finish in well under half the 1.5s a
  serial run would take
- **semaphore cap**: with `max_parallel_tools=2`, no more than 2 tools ever run
  at once
- **every tool call is answered**: the count of results always matches the count
  of calls (this is what keeps the agent loop from desyncing)
- an empty tool-call list returns nothing

### `test_hooks.py` (10 tests)

The hooks system:

- `block_paths` refuses `write_file`/`patch_file` to sensitive globs (e.g.
  `*.env`) but leaves reads and normal files alone
- a `pre_tool` hook that exits non-zero **blocks** the call; exit-zero lets it run
- a mixed batch (blocked + ok + unknown tool) still answers every call, and the
  blocked write never creates the file
- `post_tool` shell hooks actually execute, and receive context via environment
  variables (`SAATHI_TOOL_NAME`, etc.)
- the example config parses; a missing or malformed config is treated as empty

### `test_prompts.py` (6 tests)

System-prompt assembly: the base persona is always present, mode addenda
(`explain`/`refactor`/`debug`) are injected, an unknown mode is ignored, and
memory, context scope, and `SAATHI.md` instructions all land in the prompt.

### `test_project_context.py` (3 tests)

`SAATHI.md` discovery: nothing found returns empty; a single file loads; and in a
nested layout the **nearest** file wins (appended last so it takes precedence).

### `test_memory.py` (6 tests)

The two-scope memory store: save/get, delete (returns `True`/`False`), clear, and
that a **project** value overrides a **global** value for the same key. Also that
a corrupt JSON file is tolerated as empty instead of crashing.

### `test_graph.py` (6 tests)

LangGraph wiring **without calling Ollama**: the graph compiles with the expected
`agent` and `tools` nodes, the SQLite checkpoint DB is created, checkpoint history
is queryable, `close_graph()` is safe to call twice, the `agent` node has **only
conditional** outgoing edges (regression guard — an unconditional `agent → END`
would break the ReAct loop), and a fresh `thread_id` **isolates** a compacted
history from the old thread's accumulated messages.

### `test_diagnostics.py` (1 test)

`/doctor` runs to completion without raising even when Ollama is unreachable —
the normal case in CI.

### `test_usage.py` (4 tests)

The token-count extraction behind the per-turn `↳ … in · … out` footer: reads
`usage_metadata`, falls back to Ollama's `prompt_eval_count`/`eval_count`, and
returns `None` when there's nothing to report.

### `test_print_mode.py` (5 tests)

The non-interactive `--print` mode: picking the final assistant message,
collecting tool calls and summing token usage for the JSON payload, and
rejecting an invalid `--output-format` with exit code 2 (before any graph is
built, so it stays offline).

### `test_custom_commands.py` (6 tests)

User-defined `/commands` from `.saathi/commands/*.md`: loading files keyed by
lower-cased stem (ignoring non-`.md`), an empty result for a missing directory,
and `$ARGS` rendering (substitution, append-when-no-token, and unchanged).

### `test_logging.py` (4 tests)

Structured logging: `configure_logging` runs at both levels, `get_logger` emits
structured events, and the tool node logs `tool_blocked` / `tool_unknown` events
(captured with `structlog.testing.capture_logs`).

### `test_retry.py` (6 tests)

Async retry/backoff: success on the first try (no sleep), retry-then-succeed with
exponential delays, exhausting all attempts, non-retryable errors propagating
immediately, `max_delay` capping, and the `on_retry` callback. Uses an injected
recording sleep, so tests never actually wait.

### `test_compaction.py` (7 tests)

History compaction with a fake LLM: token estimation, the `needs_compaction`
threshold, splitting at a **user-turn boundary** (so the retained tail never
starts with an orphaned `ToolMessage`), a no-op when there aren't enough turns,
summarizing older turns into a leading summary message, and a smaller token
estimate afterward.

### `test_mcp.py` (12 tests)

MCP client: config normalization (`mcpServers` wrapper vs flat, stdio/http
transport inference, explicit transport, skipping entries without a
command/url), graceful degradation (empty config, unreachable server → no tools,
no raise), `_result_to_text` coercion (strings vs MCP content blocks), and a
**live round-trip** against the bundled stdio echo server
([`examples/mcp_echo_server.py`](../examples/mcp_echo_server.py)).

---

## 5. How the suite is organized

```folder
tests/
├── README.md              this file
├── __init__.py            makes `tests` a package (enables `tests.helpers`)
├── conftest.py            shared fixtures: sample_file, isolated_memory
├── helpers.py             ai_with_tool_calls(), tool_call() state builders
└── test_*.py              the tests themselves
```

- **Fixtures** (reusable setup) live in `conftest.py` and are injected by name:
  `sample_file` is a temp file with known content; `isolated_memory` is a
  `MemoryStore` redirected to `tmp_path`.
- **Helpers** (plain functions) live in `helpers.py` and are imported with
  `from tests.helpers import ...`.
- **Async tests** need no decorator — `pyproject.toml` sets
  `asyncio_mode = "auto"`, so any `async def test_*` is awaited automatically.

---

## 6. Troubleshooting

| Symptom | Fix |
| --------- | ----- |
| `pytest: command not found` | Run via the venv: `./.venv/Scripts/python.exe -m pytest` |
| `ModuleNotFoundError: saathi` | You didn't install the package: `pip install -e ".[dev]"` |
| git tests show `s` (skipped) | `git` isn't on `PATH` — install it or ignore |
| Everything is slow (~20s+) | Normal: `/doctor` waits on Ollama connection timeouts |
| Unicode errors in output | Already handled — the app forces UTF-8 stdout on Windows |

---

## 7. Adding a new test

1. Create or open a `tests/test_*.py` file.
2. Write a function named `test_something`. For async code, just use
   `async def test_something` — no decorator needed.
3. Use `tmp_path` (a pytest built-in fixture) for any files you create, so the
   test stays isolated.
4. Need a graph state with tool calls? Import the builders:

   ```python
   from tests.helpers import ai_with_tool_calls, tool_call
   ```

5. Run `pytest tests/test_yourfile.py -v` to check it.
