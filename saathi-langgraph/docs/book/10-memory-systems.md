# Chapter 10 — Memory Systems: Persistence Beyond the Context Window

> "The palest ink is better than the best memory."
> — Chinese proverb

---

## Overview

This chapter is about one of the most important unsolved problems in applied AI: **how do you give a language model memory?**

Not in a research sense. In a practical, production sense. You are building an agent that helps a developer work on a codebase. Tomorrow the developer comes back. The agent has forgotten everything. The developer has to re-explain the project, the team conventions, the fact that they use `ruff` for linting and `pytest` for tests and that the `legacy_api/` folder must never be touched. Every single session.

This is not a hypothetical annoyance. It is a fundamental architectural gap. Language models are stateless computation units. They transform input tokens into output tokens. They retain nothing between API calls. The only "memory" an LLM has during a conversation is whatever you put into the context window.

This chapter covers how `saathi` addresses that gap with a layered memory architecture: in-context memory, out-of-context memory via JSON files, project instructions via `SAATHI.md`, session save/load, and the design principles behind each decision. We also look ahead at where this architecture goes at scale — vector stores, semantic retrieval, multi-user isolation — so you can extend it when your requirements grow.

---

## 10.1 The Memory Problem

### 10.1.1 What Statelessness Actually Means

When you call the Anthropic API, you send a list of messages. The model reads them all, generates the next token, and stops. No state is persisted on the server. No information crosses between API calls unless you explicitly include it in the next call's message list.

This is by design. Statelessness makes the API horizontally scalable, deterministic, and easy to reason about. Each call is independent. But for an AI assistant, it creates a fundamental UX problem.

Consider the difference between these two experiences:

**Experience A (no memory):**

```text
Day 1, 9am:
  User: "We use ruff for linting, not flake8. Please remember that."
  Agent: "Got it! I'll use ruff for linting."

Day 2, 9am:
  User: "Can you check this file for lint errors?"
  Agent: "Sure! Let me run flake8..."
  User: "No! We use ruff. I told you this yesterday."
  Agent: "I apologize, I don't have memory of previous conversations."
```

**Experience B (with memory):**

```text
Day 1, 9am:
  User: "We use ruff for linting, not flake8. Please remember that."
  Agent: "Got it! I've saved that to project memory. I'll use ruff going forward."

Day 2, 9am:
  User: "Can you check this file for lint errors?"
  Agent: "Running ruff on the file..."  [because it read the saved fact at startup]
```

Experience B is not magic. It is engineering. The agent, at the end of Day 1, wrote the fact "we use ruff for linting" to a file. At the start of Day 2, it read that file and injected the facts into the system prompt before the conversation began. The model "knows" it because you told it at the start of the session.

### 10.1.2 The Context Window as Memory

Before we talk about external memory, let us be precise about what the context window actually gives us.

A context window is the set of tokens the model can attend to during a forward pass. As of mid-2025, state-of-the-art models have context windows ranging from 128k tokens (Claude Haiku) to 200k tokens (Claude Sonnet, Claude Opus) to 1 million tokens (Gemini). This sounds large. And it is large for many tasks.

But consider:

- A large Python codebase might have 2 million tokens of source code.
- A 6-month conversation history at 500 tokens/turn, 10 turns/day = ~900,000 tokens.
- A large PDF document might be 300,000 tokens.
- Your conversation history, plus your codebase, plus the current task, plus tool call results, plus the system prompt — it all adds up fast.

Even with 200k tokens, you will hit the limit. And hitting the limit is not graceful. You get a hard error, or the oldest messages are silently truncated, and the model loses context.

More subtly: even when everything fits in the context window, the model's ability to attend to information degrades with distance. Information at the very beginning or very end of the context is attended to most reliably. Information buried in the middle of a 150k-token context is less reliably retrieved. This is the "lost in the middle" problem.

So even if your context window is theoretically large enough, you should not rely on it as your only memory mechanism. You need external memory.

### 10.1.3 What Kinds of Things Need to Be Remembered?

Not all information is the same. When designing a memory system, it helps to categorize what you need to remember:

**1. Procedural knowledge (how to do things):**

- "Run tests with `pytest tests/ -v`"
- "Deploy with `./scripts/deploy.sh staging`"
- "Never commit directly to main, always use a feature branch"

**2. Factual knowledge (facts about the project):**

- "This project uses Python 3.11"
- "The main entry point is `src/saathi/main.py`"
- "The `legacy_api/` folder is deprecated and should not be modified"

**3. Preference knowledge (how the user likes things done):**

- "I prefer docstrings in Google style"
- "When suggesting refactors, show me the diff first, don't apply automatically"
- "I like verbose git commit messages with a body"

**4. Episodic knowledge (what happened in past sessions):**

- "Last week we refactored the database layer"
- "We started a migration to async/await in the HTTP client"
- "There's a known bug in the rate limiter that we haven't fixed yet"

Saathi's memory system is designed primarily for categories 1-3. Category 4 (episodic memory) is partially handled by session save/load, but full episodic memory would require semantic search over conversation histories — a more complex system we discuss in section 10.13.

### 10.1.4 The Design Constraints

When designing saathi's memory system, several constraints shaped the choices:

1. **No external dependencies.** Memory should work without Redis, PostgreSQL, a vector database, or any server. It is a CLI tool. It should work offline, in air-gapped environments, on a developer's laptop without a DBA.

2. **Human-readable.** A developer should be able to open the memory file in a text editor and understand it, edit it, delete entries. JSON files satisfy this. SQLite would also work. A proprietary binary format would not.

3. **Fast startup.** The memory load happens at every session start. It must be nearly instantaneous. Reading a small JSON file from disk: ~1ms. Querying a remote vector database: ~100ms. For a personal tool, the JSON file wins.

4. **Two scopes: global and project.** Some knowledge is universal (your preferred docstring style). Some is project-specific (which linter this project uses). The system must support both, with project memory overriding global memory for the same key.

5. **Writable by the agent.** The agent must be able to add and update memories itself, via tools. Otherwise the user has to manually edit JSON files, which defeats the purpose.

With these constraints, the design becomes relatively clear: two JSON files, a simple key-value store, injected into the system prompt at startup.

---

## 10.2 Two Kinds of Memory

### 10.2.1 In-Context Memory

In-context memory is everything in the `messages` list passed to the API. It is ephemeral: it exists only for the duration of the conversation and must be reconstructed from scratch next session.

Saathi populates in-context memory from several sources:

**The system prompt.** The first "message" (technically, the `system` parameter in the Anthropic API) contains:

- The agent's core instructions and personality
- The injected contents of `SAATHI.md` (if present)
- The injected contents of the memory store (all saved facts)
- Tool descriptions

**The conversation history.** Every user message, every assistant message, every tool call result from earlier in the current session. This is the most volatile memory: it grows with every turn and is gone the next session.

**Tool call results.** When the agent calls `read_file`, the file contents appear in the conversation as a tool result message. This is in-context memory: the model "knows" the file contents because they are in the messages list.

In-context memory is immediately available and requires no lookup. But it is bounded by the context window and does not persist across sessions.

### 10.2.2 Out-of-Context Memory

Out-of-context memory lives outside the model's context window. It is stored externally (on disk, in a database, in a vector store) and injected into the context at the right moment.

Saathi uses two forms of out-of-context memory:

**The memory store.** Key-value pairs stored in JSON files. Injected into the system prompt at session start. This is the primary persistent memory mechanism.

**SAATHI.md.** A markdown file in the project directory containing project instructions. Injected into the system prompt at session start. This is a more structured form of out-of-context memory designed for project-level conventions.

The injection happens at the boundary between sessions: when you start saathi, it reads these external stores and populates the system prompt. Within a session, these are in-context. Across sessions, they are out-of-context.

### 10.2.3 The Retrieval Problem

Out-of-context memory introduces a new problem: which memories to inject? If you have 10 saved facts, injecting all 10 is fine. If you have 10,000 saved facts, injecting all 10,000 would consume enormous context and most would be irrelevant to the current task.

Saathi's current approach is simple: inject all memories, always. This works because saathi is a personal tool and the memory store is expected to remain small (tens to hundreds of entries at most). A developer using saathi for a few years might accumulate a few hundred project-specific facts and a few dozen global preferences. At typical token counts for short facts, this is well under 10,000 tokens total — a small fraction of the context window.

For systems where memory can grow large, you need selective retrieval: embed each memory as a vector, embed the current query, retrieve the top-k most semantically similar memories. We cover this in section 10.13.

---

## 10.3 Saathi's Two-Scope MemoryStore

### 10.3.1 Architecture Overview

Saathi's memory system has two scopes:

**Global memory** (`~/.saathi/memory.json`): Facts that apply across all projects. Things like: your preferred coding style, your name, your timezone, tool preferences that are not project-specific.

**Project memory** (`.saathi/memory.json` in the project root): Facts specific to this project. Things like: the linter being used, test commands, architectural decisions, which files to avoid.

When the agent reads memory, it starts with global memory and then overlays project memory. Project keys override global keys of the same name. This allows you to have a global default ("I prefer async/await") that can be overridden per-project ("This project uses callbacks because it targets Node 10").

### 10.3.2 The MemoryStore Class

```python
# src/saathi/memory_store.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class MemoryStore:
    """
    A two-scope key-value memory store backed by JSON files.

    Global scope:  ~/.saathi/memory.json
    Project scope: .saathi/memory.json  (in the current working directory)

    Project keys override global keys when both define the same key.
    """

    GLOBAL_DIR = Path.home() / ".saathi"
    GLOBAL_FILE = GLOBAL_DIR / "memory.json"

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._project_root = project_root or Path.cwd()
        self._project_file = self._project_root / ".saathi" / "memory.json"
        self._global: Dict[str, Any] = self._load(self.GLOBAL_FILE)
        self._project: Dict[str, Any] = self._load(self._project_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, key: str, value: Any, scope: str = "project") -> None:
        """
        Save a key-value pair to the specified scope.

        Args:
            key:   Identifier for the memory entry. Use dot-notation for
                   namespacing, e.g. "linter.command" or "conventions.docstyle".
            value: Any JSON-serializable value. Usually a string.
            scope: Either "project" (default) or "global".

        Raises:
            ValueError: If scope is not "project" or "global".
        """
        if scope not in ("project", "global"):
            raise ValueError(f"Unknown scope {scope!r}. Use 'project' or 'global'.")

        if scope == "project":
            self._project[key] = value
            self._persist(self._project_file, self._project)
        else:
            self._global[key] = value
            self._persist(self.GLOBAL_FILE, self._global)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by key.

        Project scope takes precedence over global scope.
        Returns `default` if the key is not found in either scope.
        """
        if key in self._project:
            return self._project[key]
        if key in self._global:
            return self._global[key]
        return default

    def delete(self, key: str, scope: str = "project") -> bool:
        """
        Delete a key from the specified scope.

        Returns True if the key was found and deleted, False otherwise.
        Does NOT delete from the other scope: deleting from project scope
        will cause the global value (if any) to become visible again.
        """
        if scope == "project":
            if key in self._project:
                del self._project[key]
                self._persist(self._project_file, self._project)
                return True
            return False
        elif scope == "global":
            if key in self._global:
                del self._global[key]
                self._persist(self.GLOBAL_FILE, self._global)
                return True
            return False
        else:
            raise ValueError(f"Unknown scope {scope!r}.")

    def clear(self, scope: str = "project") -> int:
        """
        Remove all entries from the specified scope.

        Returns the number of entries deleted.
        """
        if scope == "project":
            count = len(self._project)
            self._project = {}
            self._persist(self._project_file, self._project)
            return count
        elif scope == "global":
            count = len(self._global)
            self._global = {}
            self._persist(self.GLOBAL_FILE, self._global)
            return count
        else:
            raise ValueError(f"Unknown scope {scope!r}.")

    def get_all(self) -> Dict[str, Any]:
        """
        Return the merged view of all memories.

        Project keys shadow global keys of the same name.
        The returned dict is a copy; mutating it does not affect the store.
        """
        merged = dict(self._global)
        merged.update(self._project)
        return merged

    def get_scope(self, scope: str) -> Dict[str, Any]:
        """
        Return all entries from a single scope (no merging).

        Useful for display commands that show where each fact lives.
        """
        if scope == "project":
            return dict(self._project)
        elif scope == "global":
            return dict(self._global)
        else:
            raise ValueError(f"Unknown scope {scope!r}.")

    def has(self, key: str) -> bool:
        """Return True if key exists in either scope."""
        return key in self._project or key in self._global

    def which_scope(self, key: str) -> Optional[str]:
        """
        Return the effective scope for a key.

        Returns "project" if the key is in project scope (even if also in
        global scope, since project overrides global). Returns "global" if
        only in global scope. Returns None if not found in either scope.
        """
        if key in self._project:
            return "project"
        if key in self._global:
            return "global"
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        """
        Load a JSON memory file.

        If the file does not exist: returns an empty dict.
        If the file exists but is malformed: logs a warning and returns
        an empty dict. We NEVER raise on corrupt files because that would
        prevent saathi from starting at all, which is worse than losing
        memory state.
        """
        if not path.exists():
            return {}
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                # File contains valid JSON but not an object (e.g., a list).
                # Treat as empty and overwrite on next save.
                return {}
            return data
        except json.JSONDecodeError:
            # Corrupt file. Log and continue.
            import warnings
            warnings.warn(
                f"Memory file {path} contains invalid JSON. Treating as empty. "
                f"The file will be overwritten on the next save.",
                stacklevel=2,
            )
            return {}
        except OSError as exc:
            import warnings
            warnings.warn(
                f"Could not read memory file {path}: {exc}. Treating as empty.",
                stacklevel=2,
            )
            return {}

    @staticmethod
    def _persist(path: Path, data: Dict[str, Any]) -> None:
        """
        Write data to a JSON file, creating parent directories if needed.

        Uses an atomic write pattern: write to a .tmp file, then rename.
        This prevents corrupt files if the process is killed mid-write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)  # Atomic on POSIX; best-effort on Windows.
        except OSError as exc:
            # If we can't write memory, log and continue.
            # Don't crash the agent because memory persistence failed.
            import warnings
            warnings.warn(
                f"Could not write memory file {path}: {exc}.",
                stacklevel=2,
            )
```

### 10.3.3 How Project Keys Override Global Keys

The merge logic in `get_all()` is intentionally simple:

```python
merged = dict(self._global)
merged.update(self._project)
```

Start with all global entries. Then update with all project entries. Since `dict.update()` overwrites existing keys, any project key that matches a global key will take precedence.

This gives you a natural override pattern:

```json
// ~/.saathi/memory.json (global)
{
  "linter": "ruff",
  "test_command": "pytest",
  "docstring_style": "google"
}

// project/.saathi/memory.json (project)
{
  "linter": "eslint",
  "test_command": "jest --coverage",
  "entry_point": "src/index.ts"
}
```

After merging, the effective memory is:

```json
{
  "linter": "eslint",         // project overrides global
  "test_command": "jest --coverage",  // project overrides global
  "docstring_style": "google",        // only in global
  "entry_point": "src/index.ts"       // only in project
}
```

The agent sees `linter = eslint`, correctly, for this JavaScript project. But if you switch to a Python project that has no `linter` key in its project memory, the agent will fall back to `linter = ruff` from global memory.

### 10.3.4 The Atomic Write Pattern

The `_persist` method uses an atomic write pattern:

```python
tmp = path.with_suffix(".json.tmp")
text = json.dumps(data, indent=2, ensure_ascii=False)
tmp.write_text(text, encoding="utf-8")
tmp.replace(path)  # rename, not copy
```

Why not just `path.write_text(...)`?

If the process is killed between opening the file for writing and finishing the write, you end up with a truncated or partially written JSON file. On the next load, `json.loads()` will raise `JSONDecodeError`, and your memory will appear empty. Worse, the corrupt file will be silently replaced with an empty dict on the next save, permanently losing all your memories.

The atomic pattern avoids this: the rename (`.replace()`) is atomic on POSIX systems. The old file remains valid until the new file is fully written and renamed. On Windows, `Path.replace()` is not fully atomic (it uses `MoveFileExW` which is not guaranteed atomic under all conditions), but it is still much safer than a direct overwrite.

For production systems handling valuable data, you would use a write-ahead log or a proper database. For a personal developer tool, the `.tmp` rename is sufficient.

---

## 10.4 Memory Injection

### 10.4.1 How Memories Become System Prompt Content

The memory store is useless unless the model can see it. The mechanism that makes memories visible is **injection into the system prompt**.

Saathi's system prompt is assembled fresh at the start of every session. The assembly process in `prompts.py` calls `recall_memory()` which returns all saved facts formatted as a string, then inserts that string into the system prompt template.

```python
# src/saathi/prompts.py

from __future__ import annotations

from typing import Optional
from .memory_store import MemoryStore
from .project_context import load_project_context


SYSTEM_PROMPT_TEMPLATE = """\
You are Saathi, an AI coding assistant. You help developers understand,
modify, and debug code. You have access to tools for reading and writing
files, running bash commands, and searching the web.

{project_context_section}

{memory_section}

## Core Principles

- Be concise but complete. Don't truncate code.
- When you modify a file, always show the diff or the full modified section.
- Prefer small, focused changes over large rewrites.
- Ask for clarification when requirements are ambiguous.
- If you're unsure, say so. Do not hallucinate APIs or file paths.

{tool_descriptions}
"""


def build_system_prompt(
    store: MemoryStore,
    tools: list,
    project_root: Optional[str] = None,
) -> str:
    """
    Assemble the complete system prompt for a new session.

    This is called once at session startup and becomes part of the
    first API call's `system` parameter.
    """
    project_context_section = _build_project_context_section(project_root)
    memory_section = _build_memory_section(store)
    tool_descriptions = _build_tool_descriptions(tools)

    return SYSTEM_PROMPT_TEMPLATE.format(
        project_context_section=project_context_section,
        memory_section=memory_section,
        tool_descriptions=tool_descriptions,
    )


def _build_memory_section(store: MemoryStore) -> str:
    """
    Format all saved memories as a markdown section for the system prompt.

    If there are no memories, returns an empty string (no section header).
    """
    all_memories = store.get_all()

    if not all_memories:
        return ""

    lines = ["## Remembered Facts\n"]
    lines.append(
        "The following facts have been saved across previous sessions. "
        "Use them to inform your responses:\n"
    )

    for key, value in sorted(all_memories.items()):
        if isinstance(value, str):
            lines.append(f"- **{key}**: {value}")
        else:
            # Non-string values: serialize to JSON for display
            import json
            lines.append(f"- **{key}**: {json.dumps(value)}")

    return "\n".join(lines)


def _build_project_context_section(project_root: Optional[str]) -> str:
    """
    Load SAATHI.md content (if present) and format as a system prompt section.
    """
    context = load_project_context(project_root)
    if not context:
        return ""
    return f"## Project Context\n\n{context}"


def _build_tool_descriptions(tools: list) -> str:
    """Format the list of available tools for the system prompt."""
    if not tools:
        return ""
    names = ", ".join(t.name for t in tools)
    return f"## Available Tools\n\nYou have access to: {names}"
```

### 10.4.2 Why the System Prompt and Not a Separate Message?

An alternative design would inject memories as a separate user or assistant message at the start of the conversation, rather than in the system prompt. Why do we use the system prompt?

1. **Reliability.** Instructions in the system prompt are more reliably followed than instructions buried in conversation history. The system prompt is designed for persistent instructions; conversation messages are designed for dynamic content.

2. **Cleanliness.** Injecting memory as a fake user message would pollute the conversation history display. The system prompt is invisible to the user in saathi's UI.

3. **API semantics.** The Anthropic API treats `system` as a special parameter, distinct from the conversation `messages`. This separation is semantically correct: the system prompt is the agent's persistent configuration, not part of the dialogue.

### 10.4.3 What the System Prompt Looks Like

For a project with some memories saved, the assembled system prompt might look like:

```text
You are Saathi, an AI coding assistant. You help developers understand,
modify, and debug code. ...

## Project Context

# MyApp

A FastAPI application serving the company's internal API.

## Stack
- Python 3.11
- FastAPI 0.110
- PostgreSQL 15 via SQLAlchemy async
- Deployed on GCP Cloud Run

## Conventions
- Always use type annotations.
- Tests live in tests/. Run with: pytest tests/ -v
- Never modify the legacy_api/ directory.

## Remembered Facts

The following facts have been saved across previous sessions. Use them to
inform your responses:

- **deploy_command**: ./scripts/deploy.sh staging
- **linter**: ruff
- **linter.command**: ruff check . --fix
- **test_command**: pytest tests/ -v --cov=src

## Core Principles
...
```

Every key fact is visible to the model before the first user message. The model does not need to be asked "what linter do we use?" — it already knows.

---

## 10.5 `save_memory` and `recall_memory` Tools

### 10.5.1 The Agent Can Write Its Own Memories

Saathi exposes two tools that allow the LLM itself to interact with the memory store: `save_memory` and `recall_memory`. This is the crucial design point: the agent is not just a consumer of memories, it is also a producer.

When a user says "remember that we use ruff for linting," the model recognizes this as a command to persist information and calls `save_memory`. When a user asks "what do you remember about this project?", the model calls `recall_memory` to list all saved facts.

### 10.5.2 `save_memory` Tool

```python
# src/saathi/tools/memory_tools.py

from __future__ import annotations

import json
from typing import Any, Optional
from ..memory_store import MemoryStore


async def save_memory(
    key: str,
    value: str,
    scope: str = "project",
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Save a fact to persistent memory.

    The saved fact will be injected into the system prompt at the start of
    future sessions. Use this when the user asks you to remember something,
    or when you discover a project convention that should persist.

    Args:
        key:   A short, descriptive identifier. Use dot-notation for grouping:
               "linter.command", "conventions.docstyle", "deploy.staging_url".
        value: The value to remember. Should be a complete, self-contained fact.
               Example: "ruff check . --fix && ruff format ."
        scope: "project" to save for this project only (default).
               "global" to save across all projects.

    Returns:
        A confirmation message.

    Examples:
        save_memory("linter", "ruff", scope="project")
        save_memory("test_command", "pytest tests/ -v --cov=src", scope="project")
        save_memory("preferred_docstyle", "google", scope="global")
    """
    if store is None:
        store = MemoryStore()

    # Validate scope early for a cleaner error message to the model.
    if scope not in ("project", "global"):
        return (
            f"Error: scope must be 'project' or 'global', got {scope!r}. "
            f"Use 'project' to save for this project, "
            f"'global' to save across all projects."
        )

    store.save(key, value, scope=scope)

    # Confirm with context so the model can relay this to the user.
    scope_display = "project memory" if scope == "project" else "global memory"
    return (
        f"Saved to {scope_display}: {key!r} = {value!r}. "
        f"This will be available in future sessions."
    )
```

The function signature uses a `store` parameter with a default of `None` (resolved to a fresh `MemoryStore()`). This makes the function easy to test: in tests, pass a `MemoryStore` pointing to a temp directory. In production, the agent infrastructure injects the shared store.

### 10.5.3 `recall_memory` Tool

```python
async def recall_memory(
    key: Optional[str] = None,
    scope: Optional[str] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Recall saved memories.

    Use this when the user asks what has been remembered, or when you need
    to check if a specific fact has been saved.

    Args:
        key:   If provided, recall only this specific key.
               If omitted, recall all saved memories.
        scope: If provided, limit recall to this scope ("project" or "global").
               If omitted, shows memories from both scopes (merged view).

    Returns:
        A formatted string listing the recalled memories, or a message
        indicating that no memories were found.

    Examples:
        recall_memory()                  # Show all memories
        recall_memory(key="linter")      # Show one specific key
        recall_memory(scope="global")    # Show only global memories
    """
    if store is None:
        store = MemoryStore()

    # Specific key lookup
    if key is not None:
        value = store.get(key)
        if value is None:
            # Try to be helpful: suggest similar keys
            all_keys = list(store.get_all().keys())
            similar = [k for k in all_keys if key.lower() in k.lower()]
            if similar:
                return (
                    f"No memory found for key {key!r}. "
                    f"Similar keys: {', '.join(similar)}"
                )
            return f"No memory found for key {key!r}. No memories have been saved yet."
        which = store.which_scope(key)
        return f"{key!r} = {value!r}  [scope: {which}]"

    # Scope-specific listing
    if scope is not None:
        if scope not in ("project", "global"):
            return f"Error: scope must be 'project' or 'global', got {scope!r}."
        memories = store.get_scope(scope)
        if not memories:
            return f"No memories saved in {scope} scope."
        lines = [f"## {scope.title()} Memory\n"]
        for k, v in sorted(memories.items()):
            lines.append(f"- {k!r}: {v!r}")
        return "\n".join(lines)

    # All memories, merged
    all_memories = store.get_all()
    if not all_memories:
        return (
            "No memories have been saved yet. "
            "Use save_memory() to save facts for future sessions."
        )

    global_memories = store.get_scope("global")
    project_memories = store.get_scope("project")

    lines = []

    if project_memories:
        lines.append("## Project Memory (this project only)\n")
        for k, v in sorted(project_memories.items()):
            lines.append(f"- {k!r}: {v!r}")
        lines.append("")

    if global_memories:
        lines.append("## Global Memory (all projects)\n")
        for k, v in sorted(global_memories.items()):
            lines.append(f"- {k!r}: {v!r}")
        lines.append("")

    if not project_memories and not global_memories:
        return "No memories saved."

    return "\n".join(lines)
```

### 10.5.4 Tool Registration

These tools are registered with the saathi agent infrastructure like any other tool:

```python
# src/saathi/tools/__init__.py

from functools import partial
from ..memory_store import MemoryStore
from .memory_tools import save_memory, recall_memory


def create_memory_tools(store: MemoryStore) -> list:
    """
    Create memory tools bound to a specific MemoryStore instance.

    We use functools.partial to bind the store so the tool functions
    have the right store without needing to be called with it explicitly.
    """
    return [
        partial(save_memory, store=store),
        partial(recall_memory, store=store),
    ]
```

The agent infrastructure wraps these in a `StructuredTool` (if using LangChain) or the Anthropic tool format. The key point: the `store` parameter is pre-bound via `partial`, so the LLM never has to pass it. The LLM only sees `key`, `value`, and `scope` as parameters for `save_memory`.

### 10.5.5 Natural Language Triggers

The model is instructed to call `save_memory` when it detects certain patterns in user messages:

- "Remember that..."
- "Save that..."
- "Note that..."
- "Always use X for Y"
- "Never do X"
- "From now on, X"

And it calls `recall_memory` when it detects:

- "What do you remember about..."
- "Do you know our..."
- "What was that command for..."
- "List all memories"

These are soft instructions in the system prompt. The model uses its own judgment about when a fact is worth saving. In practice, models are quite good at recognizing when a user is trying to establish a persistent rule versus just making a one-time observation.

---

## 10.6 JSON File Storage

### 10.6.1 Why JSON Files and Not a Database?

The choice of JSON files for memory storage is intentional and worth defending explicitly, because it is a common question.

**Arguments for a "proper" database (SQLite, PostgreSQL):**

- Concurrent access safety (file locking)
- Richer query capabilities (search by value, not just key)
- Transactions
- Indexing

**Arguments for JSON files:**

- Zero external dependencies
- Human-readable and human-editable
- Works in any environment without setup
- Easy to version control (you can commit your memories if you want)
- Easy to back up (it's just a file)
- Easy to reset (delete the file)
- Fast enough for the use case (reads are <1ms for small files)

For a personal CLI tool used by one person at a time, the database advantages are theoretical while the JSON file advantages are practical. Saathi is not a multi-user web application. It is a tool you run in a terminal on your laptop. JSON files are the right choice.

That said, there is one real concern: **concurrent access**. If you had two saathi sessions running simultaneously and both tried to write memory, you could get a race condition. The atomic write pattern (write to .tmp, rename) mitigates data corruption, but it does not prevent one session's writes from overwriting another's.

In practice this is rarely a problem: developers typically run one saathi session at a time. If it ever becomes a problem, the solution is to add file locking using Python's `fcntl.flock()` on POSIX or a lockfile on Windows.

### 10.6.2 The Read/Write Pattern with Corruption Handling

```python
@staticmethod
def _load(path: Path) -> Dict[str, Any]:
    """Load memory from a JSON file. Never raises."""
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        # Corrupt file: warn but don't crash.
        import warnings
        warnings.warn(
            f"Memory file {path} contains invalid JSON. "
            f"Treating as empty. File will be overwritten on next save.",
            stacklevel=2,
        )
        return {}
    except OSError as exc:
        import warnings
        warnings.warn(f"Could not read memory file {path}: {exc}.", stacklevel=2)
        return {}
```

The `_load` method handles three cases:

1. File does not exist: return empty dict (first time running, normal)
2. File exists but has invalid JSON: warn and return empty dict (corruption)
3. File exists but contains non-dict JSON (e.g., `[]`): return empty dict (unexpected format)

The contract is: `_load` never raises. It always returns a dict. This is defensive programming: a corrupt memory file should not prevent saathi from starting.

### 10.6.3 The JSON Format

The memory file is just a flat JSON object:

```json
{
  "linter": "ruff",
  "linter.command": "ruff check . --fix",
  "test_command": "pytest tests/ -v --cov=src",
  "deploy.staging": "./scripts/deploy.sh staging",
  "deploy.production": "./scripts/deploy.sh production",
  "conventions.docstyle": "google",
  "conventions.imports": "isort with profile=black",
  "entry_point": "src/saathi/main.py",
  "architecture.summary": "LangGraph agent with tool nodes for file ops and bash"
}
```

Keys use dot-notation as a convention (not enforced by code) to suggest namespacing. The values are strings in almost all cases. The store supports any JSON-serializable value (lists, nested dicts) but in practice you want facts to be short, self-contained strings that read naturally when injected into the system prompt.

### 10.6.4 Memory File Locations

```folder
~/.saathi/
    memory.json          # global memory (applies to all projects)

project_root/
    .saathi/
        memory.json      # project memory (overrides global for this project)
        hooks.json       # hook configuration (see Chapter 11)
```

The `.saathi/` directory at the project root is saathi's project-level configuration directory. It is analogous to `.git/` for git or `.venv/` for Python virtual environments. You may want to add it to `.gitignore` if your memories contain project-specific secrets, or commit it if the memories are conventions you want to share with your team.

---

## 10.7 SAATHI.md — Project Instructions

### 10.7.1 The Concept

Memory facts are great for short key-value pairs: "linter = ruff", "test_command = pytest". But some project knowledge is richer than a key-value pair. You need to explain the architecture. You need to describe which directories do what. You need to document gotchas that require a paragraph to explain.

For this, saathi uses `SAATHI.md`: a markdown file in the project directory containing free-form project instructions. It is read at startup and injected into the system prompt, just like memory facts.

The concept is borrowed directly from Claude Code's `CLAUDE.md`. If you have used Claude Code, you are already familiar with the pattern. `SAATHI.md` is saathi's equivalent.

### 10.7.2 `project_context.py`

```python
# src/saathi/project_context.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_project_context(project_root: Optional[str] = None) -> str:
    """
    Load SAATHI.md files from the project directory and parent directories.

    Walks from the project root (or cwd) up to the user's home directory,
    collecting any SAATHI.md files found along the way. Files are returned
    in order from most distant (home directory) to nearest (project root),
    so the nearest SAATHI.md takes precedence by appearing last.

    This allows a "global" SAATHI.md at ~/SAATHI.md for cross-project
    conventions, overridden by project-specific SAATHI.md files.

    Args:
        project_root: The directory to treat as the project root.
                      Defaults to os.getcwd().

    Returns:
        The concatenated contents of all found SAATHI.md files,
        separated by horizontal rules. Returns empty string if none found.
    """
    start = Path(project_root or os.getcwd()).resolve()
    home = Path.home().resolve()

    # Walk from project root up to home directory.
    # Collect all SAATHI.md files found along the way.
    found = []
    current = start
    while True:
        candidate = current / "SAATHI.md"
        if candidate.exists() and candidate.is_file():
            found.append(candidate)
        if current == home:
            break
        parent = current.parent
        if parent == current:
            # Reached filesystem root without hitting home. Stop.
            break
        current = parent

    if not found:
        return ""

    # Reverse so home comes first, project comes last.
    # Project instructions override home instructions for the same topic.
    found.reverse()

    sections = []
    for path in found:
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(content)
        except OSError:
            pass  # Unreadable file, skip silently.

    return "\n\n---\n\n".join(sections)
```

### 10.7.3 Walk From cwd to Home

The walk from cwd up to home is the key design choice here. It supports a hierarchy:

```folder
~/SAATHI.md                   # Global conventions (all projects)
~/work/SAATHI.md              # Work-specific conventions (all work projects)
~/work/myapp/SAATHI.md        # Project-specific conventions
~/work/myapp/src/SAATHI.md    # (unusual, but supported)
```

When saathi starts in `~/work/myapp/src/`, it reads:

1. `~/SAATHI.md` (if exists) — global
2. `~/work/SAATHI.md` (if exists) — work-level
3. `~/work/myapp/SAATHI.md` (if exists) — project-level
4. `~/work/myapp/src/SAATHI.md` (if exists) — subproject-level

They are concatenated in this order (global first, most specific last). The most specific file has the last word — its instructions come later in the system prompt and take precedence when there is a conflict.

### 10.7.4 Writing a Good SAATHI.md

A good `SAATHI.md` answers these questions for the agent:

1. **What is this project?** One sentence.
2. **What is the tech stack?** Languages, frameworks, databases, deployment.
3. **How do I run the project?** Development server, tests, build.
4. **What are the coding conventions?** Formatting, linting, style.
5. **What are the gotchas?** Things that are counterintuitive or that have tripped people up.
6. **What should I never do?** Hard constraints.

Here is a template:

```markdown
# ProjectName

One-sentence description of what this project does.

## Tech Stack

- Language: Python 3.11
- Framework: FastAPI 0.110
- Database: PostgreSQL 15 (async via SQLAlchemy 2.0)
- Message queue: Celery + Redis
- Deployment: Docker Compose (dev), GCP Cloud Run (production)

## Getting Started

```bash
# Install dependencies
uv sync

# Start development server
uvicorn src.app:app --reload

# Run tests
pytest tests/ -v

# Lint and format
ruff check . --fix && ruff format .
```

## Architecture

The application follows a service-layer pattern:

- `src/api/` — FastAPI routers
- `src/services/` — Business logic (no database access)
- `src/models/` — SQLAlchemy models
- `src/schemas/` — Pydantic schemas for request/response

The service layer NEVER imports from `src/api/`. The API layer ALWAYS goes
through the service layer.

## Conventions

- All functions must have type annotations.
- Docstrings use Google style.
- Database sessions are injected via FastAPI's dependency injection.
- Never use `SELECT *` — always specify columns.
- All database queries must be in `src/repositories/`, not in services.

## Gotchas

- `legacy_api/` is deprecated. Do NOT modify it. It is kept only for backward
  compatibility and will be removed in v3.0.
- The `User.email` field is encrypted at rest. Use `user.get_email()` to read it,
  never access the raw field directly.
- Running tests requires a test PostgreSQL instance. See `docker-compose.test.yml`.

## Never Do

- Commit directly to `main`. Always use a feature branch.
- Disable type checking with `# type: ignore` without a comment explaining why.
- Use `print()` for logging. Use `structlog` from `src/logging.py`.

This kind of structured project context dramatically improves the agent's usefulness. Without it, the agent has to discover every convention from scratch or ask the user. With it, the agent starts with the knowledge of a developer who has been on the project for weeks.

---

## 10.8 `/init` — Generating SAATHI.md

### 10.8.1 The Problem

Writing a good `SAATHI.md` from scratch takes time. For existing projects with years of history, it can be daunting. Saathi solves this with `/init`: a command that analyzes the project and generates a `SAATHI.md` automatically.

The generation process:

1. Read git log (recent commits, active contributors)
2. List directory structure (top-level dirs, key files)
3. Read package files (`pyproject.toml`, `package.json`, `requirements.txt`, `Cargo.toml`, etc.)
4. Check for common config files (`.ruff.toml`, `.eslintrc`, `pytest.ini`, etc.)
5. Ask the LLM to synthesize a `SAATHI.md` from this information

### 10.8.2 The `/init` Command Implementation

```python
# src/saathi/commands/init_command.py

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


INIT_ANALYSIS_PROMPT = """\
You are analyzing a software project to generate a SAATHI.md file.
SAATHI.md is a project instructions file for an AI assistant. It should
contain everything a new developer would need to know to work on the project:
tech stack, conventions, how to run things, and gotchas.

## Project Analysis

### Git History (last 20 commits)
{git_log}

### Directory Structure
{dir_structure}

### Package Files
{package_files}

### Config Files Found
{config_files}

## Task

Generate a SAATHI.md file for this project. The file should:

1. Start with a one-line description of the project (inferred from git log
   and directory names).
2. List the tech stack (languages, frameworks, libraries — infer from
   package files and imports).
3. Show how to run the project (development, tests, build, lint — infer
   from scripts in package.json, Makefile, pyproject.toml, etc.).
4. Describe the architecture and directory structure at a high level.
5. List any coding conventions you can infer (from config files, git history,
   existing docstrings).
6. List gotchas or special knowledge (anything that looks non-obvious from
   git commit messages).
7. List hard constraints (if any "never do X" patterns are visible).

Write the SAATHI.md in clean markdown. Be specific, not generic. Use actual
command names from the package files, not placeholders like "your test command".
If you are unsure about something, omit it rather than guessing.
"""


async def run_init_command(
    project_root: Optional[Path] = None,
    agent=None,  # The saathi agent, for LLM calls
    overwrite: bool = False,
) -> str:
    """
    Generate or regenerate SAATHI.md for the current project.

    Args:
        project_root: Project root. Defaults to cwd.
        agent:        The saathi agent instance, for making LLM calls.
        overwrite:    If True, overwrite existing SAATHI.md without prompting.

    Returns:
        A message describing what happened.
    """
    root = project_root or Path.cwd()
    saathi_md_path = root / "SAATHI.md"

    if saathi_md_path.exists() and not overwrite:
        return (
            f"SAATHI.md already exists at {saathi_md_path}. "
            f"Use '/init --overwrite' to regenerate it, or "
            f"'/revise-saathi-md' to update it based on the current session."
        )

    # Gather project information
    git_log = _get_git_log(root)
    dir_structure = _get_dir_structure(root)
    package_files = _get_package_files(root)
    config_files = _get_config_files(root)

    prompt = INIT_ANALYSIS_PROMPT.format(
        git_log=git_log,
        dir_structure=dir_structure,
        package_files=package_files,
        config_files=config_files,
    )

    # Ask the LLM to generate SAATHI.md
    response = await agent.generate(prompt)
    saathi_content = _extract_markdown(response)

    # Write the file
    saathi_md_path.write_text(saathi_content, encoding="utf-8")

    return (
        f"Generated SAATHI.md at {saathi_md_path}. "
        f"Review it, then edit as needed. "
        f"The file will be read at the start of future sessions."
    )


def _get_git_log(root: Path) -> str:
    """Get recent git log as a string."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "(no git history)"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "(git not available)"


def _get_dir_structure(root: Path) -> str:
    """Get top-level directory structure."""
    lines = []
    try:
        for item in sorted(root.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                lines.append(f"{item.name}/")
            else:
                lines.append(item.name)
    except OSError:
        pass
    return "\n".join(lines) or "(could not read directory)"


def _get_package_files(root: Path) -> str:
    """Read relevant package and dependency files."""
    candidates = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "build.gradle",
        "pom.xml",
        "Makefile",
    ]
    sections = []
    for name in candidates:
        path = root / name
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                # Truncate large files
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                sections.append(f"### {name}\n```\n{content}\n```")
            except OSError:
                pass
    return "\n\n".join(sections) or "(no package files found)"


def _get_config_files(root: Path) -> str:
    """List config files present in the project."""
    config_patterns = [
        ".ruff.toml", "ruff.toml",
        ".flake8", ".pylintrc",
        ".eslintrc", ".eslintrc.json", ".eslintrc.js",
        "pytest.ini", ".pytest.ini",
        ".pre-commit-config.yaml",
        "docker-compose.yml", "docker-compose.yaml",
        "Dockerfile",
        ".github/",
        "tox.ini",
        "mypy.ini", ".mypy.ini",
    ]
    found = []
    for pattern in config_patterns:
        if (root / pattern).exists():
            found.append(pattern)
    return ", ".join(found) or "(no config files found)"


def _extract_markdown(response: str) -> str:
    """
    Extract markdown content from a model response.

    If the model wrapped the SAATHI.md in a code fence, strip the fence.
    """
    import re
    # Check for ```markdown ... ``` wrapper
    match = re.search(r"```(?:markdown)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
```

### 10.8.3 What the Generated SAATHI.md Looks Like

For the saathi project itself, `/init` would produce something like:

```markdown
# saathi

An AI-powered coding assistant CLI built with LangGraph and the Anthropic API.

## Tech Stack

- Python 3.11+
- LangGraph (agent orchestration)
- Anthropic Claude API
- structlog (structured logging)
- rich (terminal UI)
- pytest (tests)

## Getting Started

```bash
# Install dependencies
uv sync

# Run in development
python -m saathi

# Run tests
pytest tests/ -v

# Lint and format
ruff check . --fix && ruff format .
```

...

```

The generated file is a starting point. Review it, add any facts the LLM missed, and remove anything incorrect. Once it looks good, it will improve every future session.

---

## 10.9 `/revise-saathi-md`

### 10.9.1 Folding Learnings Back In

Over the course of a session, you might discover things about a project that are not yet in `SAATHI.md`. The `/revise-saathi-md` command automates updating it.

The command:
1. Reads the current `SAATHI.md`
2. Reads the current conversation history (summarized)
3. Asks the LLM: "What new facts emerged in this session that should be added to SAATHI.md?"
4. Generates an updated `SAATHI.md`
5. Shows a diff and asks for confirmation before writing

### 10.9.2 The Implementation

```python
# src/saathi/commands/revise_saathi_md.py

from __future__ import annotations

from pathlib import Path
from typing import Optional


REVISE_PROMPT = """\
You are updating a SAATHI.md project instructions file for an AI assistant.

## Current SAATHI.md

{current_saathi_md}

## Session Summary

The following new information emerged during this session:

{session_summary}

## Task

Review the current SAATHI.md and the session summary. Identify any new facts,
conventions, or gotchas that should be added to SAATHI.md. Then produce an
updated version of the full SAATHI.md file.

Rules:
- Do not remove existing correct information.
- Do add new information that would be useful in future sessions.
- Do correct any information that turns out to be wrong.
- Keep the file concise. Do not add speculative or uncertain facts.
- Preserve the existing structure and style.

Output the complete updated SAATHI.md file.
"""


async def run_revise_command(
    project_root: Optional[Path] = None,
    agent=None,
    conversation_history: Optional[list] = None,
) -> str:
    """
    Update SAATHI.md based on learnings from the current session.
    """
    root = project_root or Path.cwd()
    saathi_md_path = root / "SAATHI.md"

    if not saathi_md_path.exists():
        return (
            "No SAATHI.md found in this project. "
            "Run '/init' first to generate one."
        )

    current_content = saathi_md_path.read_text(encoding="utf-8")

    # Summarize the session
    session_summary = _summarize_session(conversation_history or [])

    prompt = REVISE_PROMPT.format(
        current_saathi_md=current_content,
        session_summary=session_summary,
    )

    updated_content = await agent.generate(prompt)
    updated_content = _extract_markdown(updated_content)

    # Show diff and confirm before writing
    diff = _simple_diff(current_content, updated_content)
    if not diff:
        return "No changes needed. SAATHI.md is already up to date."

    return {
        "action": "confirm_write",
        "path": str(saathi_md_path),
        "diff": diff,
        "content": updated_content,
        "message": (
            f"Proposed changes to SAATHI.md:\n\n{diff}\n\n"
            f"Write these changes? (yes/no)"
        ),
    }


def _summarize_session(history: list) -> str:
    """
    Extract key facts from conversation history for the revision prompt.

    We take the last N user+assistant message pairs and summarize them.
    We don't send the full history because it might be very long.
    """
    if not history:
        return "(no conversation history)"

    # Take last 20 messages
    recent = history[-20:]
    parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content[:500]}")  # Truncate long messages

    return "\n\n".join(parts)


def _simple_diff(old: str, new: str) -> str:
    """
    Generate a simple line-level diff between two strings.
    """
    import difflib
    lines_old = old.splitlines(keepends=True)
    lines_new = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_old, lines_new,
        fromfile="SAATHI.md (current)",
        tofile="SAATHI.md (proposed)",
        n=3,
    )
    return "".join(diff)


def _extract_markdown(response: str) -> str:
    import re
    match = re.search(r"```(?:markdown)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
```

The `/revise-saathi-md` command is designed to be run at the end of a productive session, especially after:

- Discovering and documenting an architectural decision
- Finding a gotcha (e.g., "turns out you need to restart the Docker container when changing environment variables")
- Establishing a new convention ("we decided to switch from `black` to `ruff format`")

---

## 10.10 Session Save/Load

### 10.10.1 Sessions vs. Memory

Memory stores facts. Sessions store conversations.

A session is a snapshot of the full conversation history — every message, every tool call, every tool result — at a particular point in time. Sessions allow you to:

- Save work before an experimental change
- Resume a complex debugging session the next day
- Share a session with a colleague ("here's the conversation where we figured out the deadlock")

Sessions are user-visible artifacts with names. Memory is a behind-the-scenes key-value store. They serve different purposes and should not be confused.

### 10.10.2 Session Storage

Sessions are stored in `~/.saathi/sessions/`:

```folder
~/.saathi/
    sessions/
        2024-01-15-database-refactor.json
        2024-01-16-auth-bug.json
        2024-01-17-performance-investigation.json
```

Each session file is a JSON array of message objects:

```json
[
  {
    "role": "user",
    "content": "I need to refactor the database layer to use async SQLAlchemy."
  },
  {
    "role": "assistant",
    "content": "Let me start by reading the current database code...",
    "tool_calls": [...]
  },
  {
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "..."
  }
]
```

### 10.10.3 The Session Manager

```python
# src/saathi/session_manager.py

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class SessionManager:
    """
    Manages session save/load for conversation histories.

    Sessions are stored as JSON files in ~/.saathi/sessions/.
    Session names are slugified and prefixed with the current date.
    """

    SESSION_DIR = Path.home() / ".saathi" / "sessions"

    def __init__(self) -> None:
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        messages: List[dict],
        overwrite: bool = False,
    ) -> Path:
        """
        Save the current conversation to a named session file.

        Args:
            name:      Human-readable name for the session. Will be slugified.
            messages:  The full conversation history.
            overwrite: If True, overwrite existing session. Otherwise raises.

        Returns:
            The path to the written session file.
        """
        slug = self._slugify(name)
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_prefix}-{slug}.json"
        path = self.SESSION_DIR / filename

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Session {filename!r} already exists. "
                f"Use overwrite=True to replace it."
            )

        data = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, name_or_path: str) -> List[dict]:
        """
        Load a session by name or path.

        Searches for a session file whose name contains the given string.
        If multiple matches, returns the most recent.

        Args:
            name_or_path: Session name (partial match OK) or full file path.

        Returns:
            The list of messages from the session.

        Raises:
            FileNotFoundError: If no matching session is found.
        """
        # Try as direct path first
        direct = Path(name_or_path)
        if direct.exists():
            return self._read_session(direct)

        # Search by name
        matches = [
            p for p in self.SESSION_DIR.glob("*.json")
            if name_or_path.lower() in p.stem.lower()
        ]

        if not matches:
            raise FileNotFoundError(
                f"No session found matching {name_or_path!r}. "
                f"Available sessions: {self._list_names()}"
            )

        # Return most recent match (sort by modification time)
        most_recent = max(matches, key=lambda p: p.stat().st_mtime)
        return self._read_session(most_recent)

    def list_sessions(self) -> List[dict]:
        """
        List all available sessions, sorted by most recent first.

        Returns a list of dicts with keys: name, saved_at, message_count, path.
        """
        sessions = []
        for path in sorted(
            self.SESSION_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "name": data.get("name", path.stem),
                    "saved_at": data.get("saved_at", "unknown"),
                    "message_count": data.get("message_count", 0),
                    "path": str(path),
                })
            except (json.JSONDecodeError, OSError):
                pass

        return sessions

    def delete(self, name: str) -> bool:
        """Delete a session by name. Returns True if deleted, False if not found."""
        matches = [
            p for p in self.SESSION_DIR.glob("*.json")
            if name.lower() in p.stem.lower()
        ]
        if not matches:
            return False
        for path in matches:
            path.unlink()
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a name to a filesystem-safe slug."""
        slug = name.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_-]+", "-", slug)
        slug = slug.strip("-")
        return slug or "session"

    @staticmethod
    def _read_session(path: Path) -> List[dict]:
        """Read a session file and return its messages."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except (json.JSONDecodeError, OSError) as exc:
            raise OSError(f"Could not read session {path}: {exc}") from exc

    def _list_names(self) -> str:
        """Return a comma-separated string of session names for error messages."""
        sessions = self.list_sessions()
        if not sessions:
            return "(none)"
        return ", ".join(s["name"] for s in sessions[:5])
```

### 10.10.4 Session Commands in the CLI

```python
# In cli.py, the /session command handling:

async def handle_session_command(parts: list, agent, session_manager: SessionManager):
    """Handle /session save <name>, /session load <name>, /session list."""

    if len(parts) < 2:
        return (
            "Usage:\n"
            "  /session save <name>  — save current conversation\n"
            "  /session load <name>  — load a previous conversation\n"
            "  /session list         — list all saved sessions\n"
        )

    subcommand = parts[1].lower()

    if subcommand == "list":
        sessions = session_manager.list_sessions()
        if not sessions:
            return "No sessions saved yet."
        lines = ["## Saved Sessions\n"]
        for s in sessions:
            lines.append(
                f"- **{s['name']}** ({s['message_count']} messages, "
                f"saved {s['saved_at'][:10]})"
            )
        return "\n".join(lines)

    elif subcommand == "save":
        if len(parts) < 3:
            return "Usage: /session save <name>"
        name = " ".join(parts[2:])
        try:
            path = session_manager.save(name, agent.messages)
            return f"Session saved to {path.name}"
        except FileExistsError as exc:
            return str(exc)

    elif subcommand == "load":
        if len(parts) < 3:
            return "Usage: /session load <name>"
        name = " ".join(parts[2:])
        try:
            messages = session_manager.load(name)
            agent.messages = messages
            return (
                f"Loaded session {name!r} ({len(messages)} messages). "
                f"Conversation history restored."
            )
        except FileNotFoundError as exc:
            return str(exc)

    else:
        return f"Unknown session subcommand {subcommand!r}. Use: save, load, list."
```

---

## 10.11 LangGraph's Built-in Memory

### 10.11.1 What LangGraph Provides

LangGraph has its own memory primitives that are worth understanding, even though saathi uses its own simpler system.

LangGraph offers:

**1. Thread checkpoints.** Each conversation (thread) in LangGraph can have its full state automatically checkpointed to a store. On the next call with the same `thread_id`, the state is restored. This is built-in session save/load, handled automatically by the graph runtime.

**2. `InMemoryStore`.** A cross-thread store that supports namespaced key-value storage and semantic search (if you provide an embedding function). Multiple threads can read and write to it.

```python
from langgraph.store.memory import InMemoryStore

# Create a store with embedding support
store = InMemoryStore(
    index={
        "embed": some_embedding_function,
        "dims": 1536,
    }
)

# Save a memory
store.put(
    namespace=("user_123", "memories"),
    key="linter_preference",
    value={"text": "This project uses ruff for linting"}
)

# Search semantically
results = store.search(
    namespace=("user_123", "memories"),
    query="what linter does this project use?",
    limit=5,
)
```

**3. `PostgresSaver` and `SqliteSaver`.** Production-grade checkpoint stores backed by databases. Your conversation state is persisted durably and can survive process restarts.

### 10.11.2 Why Saathi Uses Its Own System

Given that LangGraph provides memory, why did saathi implement its own?

1. **Simplicity for the use case.** LangGraph's memory is designed for production multi-user systems. For a single-user CLI tool, it is overkill. The `InMemoryStore` lives in process memory and is lost on restart unless paired with a database backend. Saathi's JSON files are simpler and always persist.

2. **Human visibility.** Saathi's memory files are plain JSON. You can open them, read them, edit them. LangGraph's checkpoint stores are not designed for human inspection.

3. **No infrastructure required.** LangGraph's `PostgresSaver` requires a running PostgreSQL instance. `SqliteSaver` requires SQLite. Saathi's JSON files require nothing.

4. **Control over injection.** Saathi injects memories into the system prompt at a specific place, with specific formatting. LangGraph's memory is injected by the graph runtime in whatever format the graph specifies.

### 10.11.3 When to Use LangGraph's Memory

For production systems, you should use LangGraph's memory:

- **Multi-user systems.** LangGraph's namespace support makes it trivial to isolate memories per user. JSON files in a home directory don't work for a hosted service.

- **Scale.** If memory can grow to millions of entries, you need database-backed storage with indexing. JSON files don't scale beyond thousands of entries.

- **Semantic retrieval.** If you need "find the 5 most relevant memories to this query," you need `InMemoryStore` with an embedding function, or an external vector store.

- **Crash recovery.** If your agent is long-running (hours or days), you need durable checkpoints that survive process crashes. LangGraph's `PostgresSaver` handles this.

---

## 10.12 Memory in Multi-User Systems

### 10.12.1 The Problem

Saathi is designed for a single user. Its memory files live in the user's home directory. This design does not scale to a hosted service with multiple users.

If you were to deploy saathi as a web service (a "saathi-as-a-service"), each user would need completely isolated memory. User A's memory of "my linter is ruff" must never contaminate User B's sessions.

### 10.12.2 Namespacing by User ID

The simplest approach: namespace all memory by `user_id`.

```python
class MultiUserMemoryStore:
    """
    A memory store that namespaces all data by user_id.

    File layout:
        storage_root/
            users/
                user_abc123/
                    global/memory.json
                    projects/
                        proj_xyz789/memory.json
                        proj_abc456/memory.json
    """

    def __init__(self, storage_root: Path, user_id: str) -> None:
        self._user_dir = storage_root / "users" / user_id
        self._global_file = self._user_dir / "global" / "memory.json"
        # project memories namespaced by project_id
        self._projects_dir = self._user_dir / "projects"

    def get_project_store(self, project_id: str) -> MemoryStore:
        """Get a project-scoped store for a specific project."""
        project_file = self._projects_dir / project_id / "memory.json"
        return MemoryStore._from_files(
            global_file=self._global_file,
            project_file=project_file,
        )
```

File paths now include the user ID, so no cross-user contamination is possible at the filesystem level.

### 10.12.3 Database-Backed Storage

For a production service with many users, filesystem storage has limitations:

- File listing across all users is slow
- No atomic cross-file transactions
- Backups are complex

A database approach:

```sql
CREATE TABLE memories (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    project_id  VARCHAR(255),        -- NULL means global
    key         VARCHAR(255) NOT NULL,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, project_id, key)
);

CREATE INDEX memories_user_project ON memories (user_id, project_id);
```

The `MemoryStore` interface remains the same (same `save`, `get`, `delete`, `get_all` methods), but the implementation hits the database instead of files.

### 10.12.4 Access Control

In a multi-user system, you must also ensure that users cannot read each other's memories through the API. This means:

- Every memory query must include the `user_id` in the WHERE clause
- There is no way to get memories without specifying a user_id
- Admin endpoints for inspecting a user's memory require explicit admin credentials

The application layer (not the agent) is responsible for enforcing this. The agent should never receive a `user_id` as a parameter it can choose; it should always be injected by the request context.

---

## 10.13 Semantic Memory with Embeddings

### 10.13.1 The Limitation of Key-Value Memory

Key-value memory has a fundamental limitation: you need to know the key to retrieve the value. If you saved "linter = ruff" and later ask "what tool do we use for code style?", a key-value lookup for "code style" will find nothing, even though "linter = ruff" is the answer.

Semantic memory solves this with embeddings: instead of looking up by exact key, you embed the query and find the most semantically similar stored memories.

### 10.13.2 How Embedding-Based Memory Works

```python
# Conceptual design — not in saathi, but the pattern for extending it

from typing import List, Tuple


class SemanticMemoryStore:
    """
    A memory store that supports semantic (similarity) retrieval.

    Memories are stored as (key, value, embedding) triples.
    Retrieval embeds the query and returns top-k most similar memories.
    """

    def __init__(self, embed_fn, vector_store):
        self._embed_fn = embed_fn    # e.g., OpenAI embeddings
        self._vector_store = vector_store  # e.g., Chroma, Qdrant

    def save(self, key: str, value: str, metadata: dict = None) -> None:
        """
        Save a memory with its embedding.
        """
        embedding = self._embed_fn(value)
        self._vector_store.upsert(
            id=key,
            vector=embedding,
            payload={"key": key, "value": value, **(metadata or {})}
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find memories most semantically similar to the query.

        Returns list of (key, value, score) tuples.
        """
        query_embedding = self._embed_fn(query)
        results = self._vector_store.search(
            vector=query_embedding,
            limit=top_k,
        )
        return [(r.payload["key"], r.payload["value"], r.score) for r in results]

    def get_relevant(self, query: str, top_k: int = 5) -> str:
        """
        Return a formatted string of relevant memories for injection.
        """
        memories = self.search(query, top_k=top_k)
        if not memories:
            return ""

        lines = ["## Relevant Memories\n"]
        for key, value, score in memories:
            lines.append(f"- **{key}** (relevance: {score:.2f}): {value}")
        return "\n".join(lines)
```

### 10.13.3 When to Use Semantic Memory

Semantic memory makes sense when:

1. **Memory grows large.** If you have thousands of memories, injecting all of them would be impractical. Semantic retrieval lets you inject only the relevant ones.

2. **Keys are not known in advance.** If memories are saved with natural-language keys ("The team decided in the Q3 planning meeting to migrate to async database access"), you can't look them up by exact key but you can find them with a query like "what did we decide about the database?"

3. **Approximate matching is acceptable.** Semantic similarity is approximate. "linter" and "code style tool" will match, but so might unrelated memories with similar vocabulary.

For saathi's current use case (a personal CLI tool with tens to hundreds of memories), the complexity of embedding-based retrieval is not warranted. The simple key-value approach with full injection works well.

### 10.13.4 Vector Store Options

If you were to add semantic memory to saathi:

| Vector Store | Deployment | Notes |
| --- | --- | --- |
| Chroma | Local, in-process | Easiest to add, no separate service |
| Qdrant | Local or hosted | Good performance, REST API |
| pgvector | PostgreSQL extension | Good if already using Postgres |
| LanceDB | Local, file-based | Fast, no server, good for local tools |
| Pinecone | Hosted only | Managed, scalable, but requires API key |

For a local CLI tool like saathi, Chroma or LanceDB would be the best choices: no external service required, simple Python API.

---

## 10.14 Memory Limits and Eviction

### 10.14.1 What Happens When Memory Grows Large?

Saathi currently has no memory eviction. Memories accumulate until you manually delete them. For a personal tool used by one developer on a handful of projects, this is acceptable. Even after years of use, the memory files will likely contain hundreds of entries at most — well within the context window budget.

But for completeness, let us consider what strategies exist if memory were to grow large.

### 10.14.2 LRU Eviction

Least Recently Used (LRU) eviction removes the memory that has not been accessed for the longest time.

```python
import time
from collections import OrderedDict


class LRUMemoryStore:
    """
    A memory store that evicts the least recently used entries when
    the size limit is exceeded.
    """

    def __init__(self, max_entries: int = 500) -> None:
        self._max = max_entries
        self._data: OrderedDict = OrderedDict()  # key -> (value, last_accessed)

    def save(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (value, time.time())

        if len(self._data) > self._max:
            # Evict the least recently used entry
            oldest_key, _ = self._data.popitem(last=False)
            # Optionally log the eviction
            import warnings
            warnings.warn(f"Memory limit reached. Evicted oldest entry: {oldest_key!r}")

    def get(self, key: str, default=None):
        if key not in self._data:
            return default
        value, _ = self._data[key]
        self._data.move_to_end(key)  # Update access time
        self._data[key] = (value, time.time())
        return value
```

LRU eviction is simple but has a problem for memory stores: "access time" is tricky to track. A memory that was accessed 6 months ago but is still critical (e.g., "never modify legacy_api/") might get evicted by LRU, causing the agent to forget an important constraint.

### 10.14.3 Importance-Based Eviction

A better approach: tag memories with importance levels.

```python
# When saving a memory, the agent also saves an importance score.
# Eviction removes low-importance memories first.

async def save_memory(key: str, value: str, importance: int = 3) -> str:
    """
    importance: 1 (can be forgotten) to 5 (critical, never evict)
    """
    ...
```

The agent is instructed to rate importance:

- 5: Hard constraints ("never modify legacy_api/")
- 4: Project conventions ("use ruff for linting")
- 3: Common knowledge ("tests run with pytest")
- 2: Helpful context ("the staging server is at staging.example.com")
- 1: Temporary or easily re-discoverable ("the last PR was #142")

During eviction, importance-1 and importance-2 memories are removed first.

### 10.14.4 Summarization

The most sophisticated eviction strategy: instead of deleting old memories, summarize them. Merge related facts into fewer, denser entries.

This requires another LLM call:

```python
async def condense_memories(store: MemoryStore, max_entries: int = 100) -> str:
    """
    When memory exceeds max_entries, ask the LLM to condense related facts.
    """
    all_memories = store.get_all()
    if len(all_memories) <= max_entries:
        return f"Memory has {len(all_memories)} entries. No condensation needed."

    prompt = f"""
    The following project memories have grown too large ({len(all_memories)} entries).
    Please condense them by merging related facts and removing redundancies.
    Preserve all critical information. Output as a JSON object with fewer entries.

    Current memories:
    {json.dumps(all_memories, indent=2)}

    Output a JSON object with at most {max_entries} entries.
    """

    condensed_json = await agent.generate(prompt)
    condensed = json.loads(condensed_json)
    store.clear(scope="project")
    for key, value in condensed.items():
        store.save(key, value, scope="project")

    return f"Condensed {len(all_memories)} memories to {len(condensed)} entries."
```

This is the most powerful approach but also the most expensive: it requires an LLM call and risks losing information if the model is careless in summarization. For saathi's use case (personal tool, small memory), it is not needed.

### 10.14.5 Saathi's Approach: No Eviction

Saathi makes a deliberate design choice: no eviction.

Reasons:

1. For a personal tool, memory stays small. A power user might save 200-300 facts. At ~20 tokens per fact, that's 4,000-6,000 tokens. With a 200k token context window, this is 2-3% of the available context.

2. Eviction is lossy and potentially dangerous. Evicting "never modify legacy_api/" because it was saved two years ago would be a serious bug.

3. The user can manually delete memories they no longer need via `/memory delete <key>` or by editing the JSON file directly.

The right time to add eviction is when the memory files actually become a problem. Premature optimization in a personal tool wastes engineering time and introduces complexity without benefit.

---

## Summary

Memory is what separates a useful assistant from an amnesiac one. Saathi's memory system is intentionally simple, built on the following principles:

1. **Two scopes** (global and project) with clear override semantics (project beats global). This gives you cross-project defaults with per-project customization.

2. **JSON file storage** — human-readable, zero dependencies, always persistent. The right choice for a personal CLI tool.

3. **Injection at startup** — memories are injected into the system prompt before the first API call. The model knows everything you have saved before you type your first message.

4. **Agent-writable via tools** — `save_memory` and `recall_memory` let the model manage its own memory. You say "remember that," and it does.

5. **SAATHI.md for richer context** — free-form markdown for project conventions that don't fit neatly in key-value pairs. Loaded from a hierarchy of directories.

6. **Session save/load** for conversation snapshots — distinct from memory facts, these preserve the full conversation history.

7. **No eviction** for the personal use case — embrace simplicity while you can; add sophistication only when data shows you need it.

The architecture is designed to be extended. When you need semantic retrieval, swap the `MemoryStore` for a `SemanticMemoryStore` backed by Chroma or Qdrant. When you need multi-user isolation, add a user_id namespace to all storage paths. When you need scale, put LangGraph's `PostgresSaver` behind the same interface. The interface stays the same; the implementation grows with your requirements.

---

*Next: Chapter 11 — Hooks and Security: what happens when your AI assistant has file write access and shell execution, and how you keep it safe.*
