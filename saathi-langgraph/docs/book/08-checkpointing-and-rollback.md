# Chapter 8 — Checkpointing and Rollback: Persistent Agent State

> **About This Chapter**
>
> This chapter covers one of LangGraph's most powerful features: the ability to persist the complete state of a running agent to durable storage, retrieve it later, and roll back to any prior point in the conversation. We will walk through every layer of saathi's checkpointing system — from the abstract interface LangGraph exposes, to the SQLite-backed implementation saathi uses, to the `/checkpoints` and `/rollback` CLI commands that expose these capabilities to the user.
>
> By the end of this chapter you will understand not just *how* to wire up checkpointing, but *why* each design decision was made and what the trade-offs are.

---

## Table of Contents

1. Why Checkpointing Matters
2. The Checkpointer Interface — `BaseCheckpointSaver`
3. `AsyncSqliteSaver` — SQLite-Backed Async Checkpointing
4. Why NOT `AsyncSqliteSaver.from_conn_string()`
5. The Thread Model
6. Checkpoint Data Structure
7. `aget_state` — Reading Current State
8. `aget_state_history` — Browsing the Full History
9. `aupdate_state` — Injecting State from Outside
10. The `/rollback` Command — Full Implementation Walkthrough
11. The `/checkpoints` Display Command
12. The `close_graph` Pattern — Clean Shutdown
13. `aiosqlite` — Why Async SQLite Matters
14. Checkpoint Storage Location and Configuration
15. Other Checkpointers — When to Upgrade
16. Testing Checkpointing
17. Summary
18. Key Takeaways

---

## 1. Why Checkpointing Matters

### 1.1 The Ephemeral Agent Problem

Every LangGraph agent, at its core, is a state machine. At any given moment the agent holds a snapshot of everything it knows: the full conversation history, the files it has been told to consider, the mode it is operating in, and any metadata accumulated during the session. This state lives in memory — inside a Python dictionary that LangGraph manages.

The problem is simple and familiar to every developer who has worked with long-running processes: memory is ephemeral. When the process ends, the state disappears. For a conversational AI assistant like saathi, this is a significant limitation:

- The user opens saathi, has a productive 30-minute session debugging a machine learning pipeline, then closes their terminal. The next time they open saathi, the entire conversation is gone.
- The user is in the middle of a complex multi-step task. Their laptop runs out of battery. When they resume, they have to reconstruct the context from scratch.
- A bug in saathi causes an unhandled exception. The traceback is printed, the process exits, and all the context built up during the session is lost.

These are not edge cases — they are the normal conditions under which a developer tool operates. A CLI tool that cannot survive a restart is fundamentally less useful than one that can.

### 1.2 What Checkpointing Gives You

Checkpointing solves the ephemeral agent problem by automatically saving the full agent state to durable storage after every node execution. The benefits are layered:

**Persistence across restarts.** When you open saathi again and provide the same thread ID, the agent picks up exactly where you left off. Every message, every file path, every decision is restored. The conversation continues as if nothing happened.

**Rollback to any prior state.** Every checkpoint is retained (by default). You can ask saathi to go back to the state it was in five messages ago, before you went down a particular line of questioning. The `/rollback` command implements this. It is not just an "undo last message" — it is a full time-travel capability that can jump back to any point in the session's history.

**Reproducibility.** If you find that the agent produced a particularly good response at a specific point, you can load that checkpoint and replay from there, potentially with a different prompt, to explore alternative paths. This is invaluable for debugging and for understanding how the agent's reasoning evolves.

**Debugging past state.** When something goes wrong — the agent hallucinated a file path, produced incorrect code, or made a bad decision — you can load the exact state at that moment and inspect it. You can see precisely what messages were in the context, what the agent knew, and what it did not know.

**Audit trails.** Every checkpoint records a timestamp and step number. This gives you a complete audit trail of the agent's activity, which is useful for understanding usage patterns and for diagnosing problems after the fact.

### 1.3 Checkpointing in the Context of saathi

saathi is a single-user CLI tool designed for a developer working on a Python project. Its checkpointing requirements are accordingly straightforward:

- Single user, single machine — no distributed concurrency concerns.
- Sessions may be long (30 minutes to several hours) but are not always resumed.
- The primary checkpointing use case is rollback within a session, with secondary use being resumption after a crash or restart.
- Storage should be transparent and low-maintenance — the user should not have to think about it.

These requirements point to SQLite as the right storage backend: a zero-configuration, single-file database that lives alongside the project. We will examine this choice in detail in Section 3.

### 1.4 A Mental Model: Git for Agent State

If you are familiar with Git, checkpointing maps cleanly onto Git concepts:

| Git Concept | Checkpointing Equivalent |
| --- | --- |
| Commit | Checkpoint |
| Commit hash | `checkpoint_id` |
| Parent commit | `parent_checkpoint_id` |
| Branch | `thread_id` |
| `git log` | `aget_state_history()` |
| `git checkout <hash>` | `aupdate_state(checkpoint_id)` |
| Working tree | Current in-memory agent state |
| Repository | Checkpoint database (`checkpoints.db`) |

This analogy is not perfect — you cannot branch from a checkpoint (yet), for example — but it is a useful mental model for understanding what the checkpointing system is doing and why.

> **Callout: Checkpointing Is Not Caching**
>
> It is tempting to think of checkpointing as a caching mechanism — a way to avoid recomputing expensive LLM calls. This is not its primary purpose. Checkpointing saves state at the *agent* level (messages, metadata, decisions), not at the *LLM call* level (prompt/response pairs). Saathi does not use LLM response caching. Checkpointing is about *state persistence*, not *computation reuse*.

### 1.5 Before and After: A Concrete Comparison

To make the benefit concrete, consider what happens without checkpointing versus with it in a typical workflow.

**Without checkpointing:**

```text
$ saathi
saathi> I'm debugging a training loop in src/train.py. The loss isn't decreasing.
[saathi analyzes the file and gives a detailed response...]
saathi> Can you show me the gradient flow?
[saathi traces the computation graph...]
saathi> ^C
$
$ saathi   # new session — context completely lost
saathi> I'm debugging a training loop in src/train.py...
# User has to re-explain everything
```

**With checkpointing:**

```text
$ saathi
saathi> I'm debugging a training loop in src/train.py. The loss isn't decreasing.
[saathi analyzes the file and gives a detailed response, saves checkpoint]
saathi> Can you show me the gradient flow?
[saathi traces the computation graph, saves checkpoint]
saathi> ^C
$
$ saathi --session my-previous-session-id
[saathi loads checkpoint — conversation history fully restored]
saathi> Now let's look at the learning rate schedule...
# User continues exactly where they left off
```

The difference is not cosmetic. For complex, multi-session debugging work, checkpointing is the difference between a useful tool and a frustrating one.

### 1.6 The Cost of Checkpointing

Nothing is free. Checkpointing adds:

- **Latency**: Each node execution writes a checkpoint before returning. This adds 5–20 milliseconds per node, which is negligible compared to LLM API call latency (typically 500ms–5s).
- **Storage**: Each checkpoint stores the full state. For a session with 20 exchanges, this might be 2–10 MB of SQLite data (see Section 14 for details).
- **Complexity**: The application must manage the checkpoint database connection lifecycle.

For saathi's use case, all three costs are acceptable. The latency is imperceptible. The storage is modest. The complexity is hidden behind a clean API.

---

## 2. The Checkpointer Interface — `BaseCheckpointSaver`

### 2.1 The Abstract Base Class

LangGraph defines a standard interface for checkpoint storage through `BaseCheckpointSaver`. Any object that implements this interface can be used as the checkpointer for a compiled graph. The interface is intentionally minimal — it defines exactly what LangGraph needs and nothing more.

The core abstract methods are:

```python
class BaseCheckpointSaver(ABC):

    @abstractmethod
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint and return the updated config with checkpoint_id."""
        ...

    @abstractmethod
    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> Optional[CheckpointTuple]:
        """Get the latest checkpoint for a config (or a specific checkpoint_id)."""
        ...

    @abstractmethod
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints, optionally filtered."""
        ...
```

For async usage, there are corresponding async variants:

```python
    async def aput(self, ...) -> RunnableConfig: ...
    async def aget_tuple(self, ...) -> Optional[CheckpointTuple]: ...
    async def alist(self, ...) -> AsyncIterator[CheckpointTuple]: ...
```

### 2.2 What LangGraph Does With the Checkpointer

When you compile a graph with a checkpointer:

```python
graph = builder.compile(checkpointer=my_checkpointer)
```

LangGraph wraps every node execution in a checkpoint operation. The sequence for a single `ainvoke` call is:

1. **Load**: LangGraph calls `aget_tuple(config)` to load the most recent checkpoint for the given `thread_id`. If no checkpoint exists, the agent starts with an empty state.
2. **Execute**: LangGraph runs the graph from the loaded state, executing each node in sequence.
3. **Save**: After each node completes, LangGraph calls `aput(config, checkpoint, metadata, new_versions)` to save the updated state.

This means checkpoints are saved at the *node* granularity, not at the call granularity. If a graph has three nodes (say, `check_context`, `run_agent`, `post_process`) and all three execute in a single `ainvoke` call, three checkpoints will be written to the database.

Here is a diagram showing the checkpoint write pattern for a single `ainvoke` call through a three-node graph:

```text
ainvoke(state, config)
        │
        ▼
[Load latest checkpoint for thread_id]
        │
        ▼
┌─────────────────┐
│  check_context  │ node executes
└────────┬────────┘
         │ write checkpoint (step N+1)
         ▼
┌─────────────────┐
│   run_agent     │ node executes
└────────┬────────┘
         │ write checkpoint (step N+2)
         ▼
┌─────────────────┐
│  post_process   │ node executes
└────────┬────────┘
         │ write checkpoint (step N+3)
         ▼
[Return final state to caller]
```

Each of those three checkpoint writes goes to the SQLite database. The next `ainvoke` call loads the checkpoint at step N+3 and continues from there.

### 2.3 You Never Call These Methods Directly

This is an important point that confuses many LangGraph beginners: **you do not call `put`, `get_tuple`, or `list` directly in application code.** LangGraph calls them internally as part of graph execution.

The methods you *do* call — as an application developer — are the graph-level APIs that query and manipulate checkpoint history:

- `graph.aget_state(config)` — get the current state snapshot
- `graph.aget_state_history(config)` — iterate over all checkpoints
- `graph.aupdate_state(config, values)` — inject state changes

These graph-level methods internally call the checkpointer's methods, but they do so through LangGraph's state management layer, which handles serialization, channel versioning, and metadata correctly. You should always use the graph-level APIs rather than calling the checkpointer directly.

### 2.4 The `setup()` Method

Many checkpointer implementations, including `AsyncSqliteSaver`, expose a `setup()` method:

```python
await checkpointer.setup()
```

This method creates the necessary database tables (or other storage structures) if they do not already exist. It is idempotent — calling it on an already-initialized database is safe. You must call it before the first use of the checkpointer, but you do not need to call it every time (it checks whether the tables exist before creating them).

> **Callout: Async vs Sync Checkpointers**
>
> LangGraph supports both synchronous and asynchronous graph execution. Saathi uses async execution throughout (via `ainvoke` and `astream`), so it needs an async checkpointer — one that implements `aput`, `aget_tuple`, and `alist`. The sync methods (`put`, `get_tuple`, `list`) are still required by the interface but are not used in saathi's execution path. `AsyncSqliteSaver` implements all six methods.

### 2.5 The `CheckpointTuple` Data Structure

When the checkpointer returns a checkpoint, it wraps it in a `CheckpointTuple`:

```python
@dataclass
class CheckpointTuple:
    config: RunnableConfig          # includes thread_id and checkpoint_id
    checkpoint: Checkpoint          # the actual state snapshot
    metadata: CheckpointMetadata    # step, source, writes
    parent_config: Optional[RunnableConfig]  # link to prior checkpoint
    pending_writes: Optional[list]  # uncommitted writes (rarely non-empty)
```

The `Checkpoint` itself contains:

- `v`: schema version (currently 1)
- `id`: unique checkpoint identifier (UUID)
- `ts`: timestamp as ISO 8601 string
- `channel_values`: the serialized state dict
- `channel_versions`: version counters for each channel
- `versions_seen`: for tracking what each node has seen

You will rarely interact with `CheckpointTuple` or `Checkpoint` directly. The graph-level APIs transform these into `StateSnapshot` objects that are much easier to work with.

### 2.6 Why This Interface Exists

The `BaseCheckpointSaver` interface is valuable because it decouples the graph logic from the storage implementation. The graph does not know or care whether checkpoints go to SQLite, PostgreSQL, Redis, or an in-memory dictionary. You can swap checkpointers at graph compile time without touching any graph logic.

This design follows the Dependency Inversion Principle: the high-level graph module (which defines agent behavior) does not depend on low-level storage modules (SQLite, PostgreSQL). Both depend on the abstraction (`BaseCheckpointSaver`).

---

## 3. `AsyncSqliteSaver` — SQLite-Backed Async Checkpointing

### 3.1 Why SQLite?

For a single-user CLI tool like saathi, SQLite is an almost ideal checkpoint storage backend:

**Zero configuration.** SQLite requires no server process, no network configuration, no authentication setup. The database is a single file on disk. There is nothing to install beyond the Python `aiosqlite` package.

**ACID guarantees.** SQLite provides full ACID transactions. Checkpoint writes are atomic — either the entire checkpoint is written, or none of it is. There is no risk of writing partial state that leaves the database in an inconsistent condition.

**Durability.** SQLite uses write-ahead logging (WAL mode) by default in newer versions, which means checkpoint writes are durable even if the process crashes mid-write.

**Adequate performance.** For a single-user tool, SQLite's throughput is more than sufficient. A checkpoint write typically takes 5–20 milliseconds. Given that LLM API calls take hundreds of milliseconds to several seconds, checkpointing adds negligible latency.

**Portability.** The checkpoint database is a single file that can be copied, moved, or archived. If you want to share a session state with a colleague, you can copy the `.db` file.

**Inspectability.** You can open the checkpoint database with any SQLite client and inspect its contents directly. This is valuable for debugging.

The main limitation of SQLite is that it supports only a single writer at a time. For a multi-user system, this would be a significant bottleneck. But saathi is explicitly a single-user tool — this limitation simply does not apply.

### 3.2 The `AsyncSqliteSaver` Class

`AsyncSqliteSaver` is provided by the `langgraph-checkpoint-sqlite` package. Its constructor takes an `aiosqlite.Connection` object:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

conn = await aiosqlite.connect("path/to/checkpoints.db")
checkpointer = AsyncSqliteSaver(conn)
await checkpointer.setup()
```

The `setup()` call creates the following tables in the database:

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB NOT NULL,
    metadata BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
```

The `checkpoints` table stores one row per checkpoint. The `checkpoint_writes` table stores individual channel writes (used for sub-graph support and pending writes). The `checkpoint_migrations` table tracks schema versions.

### 3.3 The Initialization Pattern in saathi

saathi uses the following pattern to initialize the graph with a checkpointer:

```python
import aiosqlite
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph

async def create_graph(db_path: Path) -> CompiledGraph:
    """Create and compile the LangGraph graph with SQLite checkpointing."""

    # Build the graph structure
    builder = StateGraph(SaathiState)
    builder.add_node("check_context", check_context_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_edge(START, "check_context")
    builder.add_conditional_edges("check_context", route_after_context)
    builder.add_edge("run_agent", END)

    # Initialize the SQLite connection
    conn = await aiosqlite.connect(str(db_path))

    # Create the checkpointer
    checkpointer = AsyncSqliteSaver(conn)

    # Create the tables if they don't exist
    await checkpointer.setup()

    # Compile the graph with the checkpointer attached
    return builder.compile(checkpointer=checkpointer)
```

The key steps are:

1. Build the graph structure using `StateGraph` and its builder methods.
2. Open an async SQLite connection to the checkpoint database file.
3. Instantiate `AsyncSqliteSaver` with that connection.
4. Call `setup()` to create the database tables.
5. Compile the graph, passing the checkpointer as an argument.

The compiled graph now automatically persists every state transition to the SQLite database.

### 3.4 Connection Lifetime

The `aiosqlite.Connection` object is long-lived — it is created once when the graph is first created and remains open for the lifetime of the application. This is intentional: creating a new SQLite connection for each operation is expensive (though less so than for network databases), and SQLite handles concurrent reads perfectly well on a single connection.

The connection is closed explicitly when the application exits, via the `close_graph()` function described in Section 12.

> **Callout: The DB File Is Created On First Connect**
>
> If the SQLite database file does not exist when `aiosqlite.connect()` is called, SQLite creates it automatically. You do not need to create the file manually or run any setup scripts. The `await checkpointer.setup()` call then creates the tables. This means the first time saathi runs in a new project directory, it silently creates the checkpoint database with no user interaction required.

### 3.5 Package Installation

To use `AsyncSqliteSaver`, install the separate checkpoint package:

```bash
pip install langgraph-checkpoint-sqlite
```

This is separate from the main `langgraph` package. The `langgraph` package contains the core graph machinery and the `InMemorySaver`. The SQLite, PostgreSQL, and other backend checkpointers are in separate packages. This keeps the core package lean and allows you to install only the storage backends you need.

saathi's `pyproject.toml` includes this dependency:

```toml
[project]
dependencies = [
    "langgraph>=0.2.0",
    "langgraph-checkpoint-sqlite>=1.0.0",
    "aiosqlite>=0.19.0",
    # ... other dependencies
]
```

---

## 4. Why NOT `AsyncSqliteSaver.from_conn_string()`

### 4.1 The Convenience Method

`AsyncSqliteSaver` provides a class method `from_conn_string()` that appears to simplify initialization:

```python
# This looks convenient...
checkpointer = AsyncSqliteSaver.from_conn_string("path/to/checkpoints.db")
```

This pattern is prominently featured in early LangGraph documentation and many tutorials online. It seems cleaner than the three-step pattern (connect, instantiate, setup). So why does saathi not use it?

### 4.2 What `from_conn_string()` Actually Returns

The answer lies in what `from_conn_string()` actually returns. Inspecting its implementation reveals:

```python
@classmethod
@contextlib.asynccontextmanager
async def from_conn_string(
    cls, conn_string: str
) -> AsyncIterator["AsyncSqliteSaver"]:
    """Create a new AsyncSqliteSaver from a connection string."""
    async with aiosqlite.connect(conn_string) as conn:
        yield cls(conn)
```

The `@asynccontextmanager` decorator means this method returns an *async context manager*, not an `AsyncSqliteSaver` instance. The correct way to use it is:

```python
async with AsyncSqliteSaver.from_conn_string("path/to/checkpoints.db") as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    # ... use graph inside this block
```

### 4.3 The Bug That Happens Without `async with`

If you write:

```python
# WRONG - common mistake
checkpointer = AsyncSqliteSaver.from_conn_string("path/to/checkpoints.db")
await checkpointer.setup()  # AttributeError!
```

You get an `AttributeError` because `checkpointer` is a `_AsyncGeneratorContextManager` object, not an `AsyncSqliteSaver`. It does not have a `setup()` method. Even more confusingly, if you then try:

```python
graph = builder.compile(checkpointer=checkpointer)
await graph.ainvoke(state, config)  # Fails at runtime!
```

The graph compiles without error (the type annotation accepts any `BaseCheckpointSaver`, and Python's duck typing does not catch this at compile time), but it fails when it tries to actually call `checkpointer.aput()` — because the context manager object does not have that method either.

This is a particularly insidious bug because it only manifests at runtime, under specific conditions, with an error message that does not clearly point to the real cause.

### 4.4 The Structural Problem with `async with`

Even if you correctly use `async with`, the pattern creates a structural problem for saathi's architecture. The `async with` block must encompass *all* uses of the checkpointer — which means the entire application lifetime must be inside the context manager:

```python
async def main():
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        await run_cli_loop(graph)  # The entire app lives in here
    # Connection is closed when the block exits
```

This is architecturally awkward. It forces the entire application to be nested inside the checkpointer initialization, making the code harder to read and test. The explicit connection pattern — connect, instantiate, setup, and later close explicitly — maps more cleanly onto the actual application lifecycle.

### 4.5 The Explicit Pattern Is Better

saathi's pattern is more verbose but unambiguous:

```python
conn = await aiosqlite.connect(str(db_path))
checkpointer = AsyncSqliteSaver(conn)
await checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)
```

And at shutdown:

```python
await conn.close()
```

Every step is explicit. There are no hidden context managers, no risk of using a context manager object where an instance was expected, and the lifecycle of the connection is clear from reading the code. The `close_graph()` function (Section 12) encapsulates the shutdown logic.

> **Callout: Prefer Explicit Lifetimes**
>
> This is a broader principle that applies beyond just `AsyncSqliteSaver`: when working with long-lived resources (database connections, file handles, network connections), prefer explicit open/close over context managers when the resource lifetime spans more than a single function. Context managers are excellent for short-lived, scoped resources. For application-lifetime resources, explicit lifecycle management is clearer.

### 4.6 Documenting This Decision

Because this is a non-obvious decision that future maintainers might want to "fix" (it looks like the explicit pattern is unnecessarily verbose), saathi includes a comment at the initialization site:

```python
# We use the explicit connection pattern rather than
# AsyncSqliteSaver.from_conn_string() because from_conn_string()
# returns an async context manager (_AsyncGeneratorContextManager),
# not an AsyncSqliteSaver instance. Calling it without `async with`
# would give a runtime AttributeError when setup() or aput() is called.
# See: docs/book/08-checkpointing-and-rollback.md Section 4.
conn = await aiosqlite.connect(str(db_path))
checkpointer = AsyncSqliteSaver(conn)
await checkpointer.setup()
```

This comment prevents a well-intentioned "cleanup" from introducing a subtle bug.

---

## 5. The Thread Model

### 5.1 Thread IDs in LangGraph

Every `ainvoke` or `astream` call in LangGraph takes a `config` dictionary as its second argument. The most important field in this config is the `thread_id`:

```python
config = {"configurable": {"thread_id": "my-session-123"}}
result = await graph.ainvoke(user_input, config)
```

The `thread_id` is the primary key for checkpoint retrieval. When LangGraph processes this call:

1. It searches the checkpoint database for the most recent checkpoint with `thread_id = "my-session-123"`.
2. If found, it loads that checkpoint as the starting state.
3. If not found, it starts with the initial state (empty messages, default values).
4. After execution, it saves the new checkpoint with the same `thread_id`.

All checkpoints for a session share the same `thread_id`. The checkpoint history for a session is thus a linear sequence of checkpoints, all tagged with the same `thread_id`, ordered by step number and timestamp.

### 5.2 Thread ID Design in saathi

saathi uses a UUID-based thread ID for each session:

```python
import uuid

def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())
```

The session ID is generated when a new session starts. If the user provides a `--session` argument on the command line, saathi uses that value as the thread ID, allowing them to resume a previous session:

```bash
# Start a new session (new thread_id generated automatically)
saathi

# Resume session abc123
saathi --session abc123

# Start a fresh session with a custom name
saathi --session my-project-debug-session
```

The session ID is stored alongside the checkpoint data, so when the user lists their sessions (via `/sessions`), saathi can display session IDs that can be passed to `--session` for resumption.

### 5.3 Thread Isolation

Different thread IDs are completely isolated from each other. The checkpoint database may contain hundreds of sessions, but when you invoke the graph with `thread_id = "abc"`, LangGraph only sees checkpoints with `thread_id = "abc"`. There is no cross-contamination between sessions.

This isolation is absolute — it is enforced at the database query level, not at the application level. Even if there were a bug in saathi's session management code, LangGraph's checkpointer would not load the wrong session's data, because the thread_id is passed directly to the SQLite query:

```sql
SELECT * FROM checkpoints
WHERE thread_id = ?
ORDER BY checkpoint_id DESC
LIMIT 1
```

### 5.4 The `checkpoint_id` Within a Thread

Within a single thread, each checkpoint has a unique `checkpoint_id` — a UUID that identifies this specific snapshot. The checkpoint also records its `parent_checkpoint_id` — the `checkpoint_id` of the previous checkpoint in the sequence.

This forms a linked list of checkpoints:

```text
[checkpoint_id=abc, parent=None]  -- First checkpoint (empty state)
        |
        v
[checkpoint_id=def, parent=abc]  -- After first message
        |
        v
[checkpoint_id=ghi, parent=def]  -- After first response
        |
        v
[checkpoint_id=jkl, parent=ghi]  -- After second message
        |
        v
[checkpoint_id=mno, parent=jkl]  -- After second response  <- Current
```

The `aget_state_history()` method traverses this chain from the latest checkpoint back to the beginning, yielding each `StateSnapshot` in reverse chronological order.

### 5.5 The `config` Dictionary in Practice

The full `config` dictionary that saathi passes to `ainvoke` looks like this:

```python
config = {
    "configurable": {
        "thread_id": self.session_id,
    }
}
```

When you want to load a *specific* checkpoint (rather than the most recent one), you add a `checkpoint_id`:

```python
config = {
    "configurable": {
        "thread_id": self.session_id,
        "checkpoint_id": "specific-checkpoint-uuid-here",
    }
}
```

LangGraph interprets a config with a `checkpoint_id` as a request to load that specific checkpoint rather than the most recent one. This is how the rollback mechanism works — you specify the `checkpoint_id` of the state you want to restore.

### 5.6 Thread ID Collisions

Since thread IDs are user-supplied strings, there is a theoretical risk of collision: two users running saathi with the same `--session` value would share checkpoint history. For a single-user tool, this is not a concern — there is only one user. For a multi-user deployment, this would require adding a user ID prefix to the thread ID.

UUID-generated session IDs (the default when no `--session` is provided) have a collision probability so low it is effectively zero: there are 2^122 possible UUID v4 values, giving approximately 5.3 × 10^36 unique identifiers. At a rate of one new session per second, you would need to run saathi for about 100 trillion years before expecting a collision.

---

## 6. Checkpoint Data Structure

### 6.1 What Gets Stored

Every checkpoint stores a complete snapshot of the entire agent state. For saathi, the state is defined by `SaathiState`:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class SaathiState(TypedDict):
    messages: Annotated[list, add_messages]  # Full conversation history
    context_paths: list[str]                  # Files/directories in context
    mode: str                                 # Current agent mode
    session_id: str                           # Session identifier
    error: Optional[str]                      # Last error, if any
```

Every field of this `TypedDict` is serialized and stored in the checkpoint. This includes:

- **`messages`**: The complete list of `HumanMessage`, `AIMessage`, `SystemMessage`, and `ToolMessage` objects. This is the most significant contributor to checkpoint size — each message includes its full text content, plus any tool calls and tool results.
- **`context_paths`**: The list of file and directory paths that have been added to the agent's context via `/context add`. These are just strings (paths), so they are small.
- **`mode`**: A short string (`"chat"`, `"code"`, `"debug"`, etc.).
- **`session_id`**: The UUID of the session.
- **`error`**: A short error string or `None`.

### 6.2 The Full Checkpoint Schema

A complete checkpoint stored in the database has the following structure:

```flow
CheckpointTuple
├── config: RunnableConfig
│   └── configurable
│       ├── thread_id: str          # "3f2a1b4c-..."
│       ├── checkpoint_ns: str      # "" (empty for top-level graphs)
│       └── checkpoint_id: str      # "1a2b3c4d-..."
│
├── checkpoint: Checkpoint
│   ├── v: int                      # Schema version, currently 1
│   ├── id: str                     # Same as checkpoint_id above
│   ├── ts: str                     # "2024-01-15T10:30:45.123456+00:00"
│   ├── channel_values: dict        # The actual state
│   │   ├── messages: [...]         # Serialized message objects
│   │   ├── context_paths: [...]    # List of path strings
│   │   ├── mode: "chat"
│   │   ├── session_id: "3f2a1b4c-..."
│   │   └── error: null
│   └── channel_versions: dict      # Version counters per channel
│       ├── __start__: int
│       ├── messages: int
│       ├── context_paths: int
│       └── ...
│
├── metadata: CheckpointMetadata
│   ├── source: str                 # "loop" | "input" | "update"
│   ├── step: int                   # 0, 1, 2, 3, ... (monotonically increasing)
│   ├── writes: dict                # What the last node wrote
│   └── parents: dict               # Parent checkpoint references
│
└── parent_config: Optional[RunnableConfig]
    └── configurable
        ├── thread_id: str          # Same thread_id
        ├── checkpoint_ns: str      # ""
        └── checkpoint_id: str      # The *previous* checkpoint's id
```

### 6.3 Checkpoint Metadata Explained

The `metadata` field provides context about *how* this checkpoint was created:

**`source`** indicates the origin of the checkpoint:

- `"input"` — this checkpoint was created when the graph received new input
- `"loop"` — this checkpoint was created by a node executing within the graph
- `"update"` — this checkpoint was created by `aupdate_state()` (external injection)

**`step`** is a monotonically increasing integer starting at 0. Each checkpoint within a thread has a step number one higher than its parent. This is more reliable than timestamps for ordering checkpoints, because clock skew or clock adjustments could cause timestamps to be non-monotonic.

**`writes`** records what values the last node wrote to the state. For example, if the `run_agent` node wrote a new AI message, `writes` would contain `{"messages": [AIMessage(...)]}`. This is useful for understanding what changed between checkpoints.

**`parents`** is a dictionary mapping checkpoint namespace to parent checkpoint ID. For flat (non-nested) graphs like saathi, this is a simple `{"": "parent-checkpoint-id"}`.

### 6.4 Serialization

LangGraph serializes checkpoint state using its built-in serializer, which handles:

- Python primitives (str, int, float, bool, None) — stored as JSON
- LangChain message objects (`HumanMessage`, `AIMessage`, etc.) — serialized to a standard dict format
- Lists and dicts — recursively serialized
- Custom TypedDict types — serialized field by field

The serialized checkpoint is stored as a BLOB in the SQLite database. The serialization format is internal to LangGraph and may change between versions, which is one reason you should not attempt to read checkpoint BLOBs directly — use the graph APIs instead.

### 6.5 Checkpoint Size

The dominant factor in checkpoint size is the `messages` list. A typical LLM message might be 500–5000 characters of text. After serialization and any LangGraph metadata overhead, a checkpoint for a session with 10 exchanges (20 messages) might be 50–200 KB.

The checkpoint database grows with every message, because every checkpoint contains the *full* messages list (not just the delta). This is intentional — it allows any checkpoint to be loaded independently, without replaying from the beginning. The trade-off is storage size.

For a typical saathi session:

- 10 exchanges: ~100 KB per checkpoint, ~20 checkpoints, ~2 MB total
- 50 exchanges: ~500 KB per checkpoint, ~100 checkpoints, ~50 MB total
- 100 exchanges: ~1 MB per checkpoint, ~200 checkpoints, ~200 MB total

These are rough estimates. The actual size depends heavily on the length of the messages and the number of tool calls. For a developer machine with tens of gigabytes of free storage, these sizes are not concerning. For very long sessions (hundreds of exchanges), pruning old checkpoints becomes worthwhile — this is a future enhancement for saathi.

---

## 7. `aget_state` — Reading Current State

### 7.1 The Method Signature

`aget_state` retrieves the current state of a thread as a `StateSnapshot`:

```python
snapshot: StateSnapshot = await graph.aget_state(config)
```

Where `config` is a `RunnableConfig` with at least a `thread_id` in `configurable`. If `checkpoint_id` is also provided, `aget_state` returns the state at that specific checkpoint rather than the most recent one.

### 7.2 The `StateSnapshot` Object

`StateSnapshot` is a dataclass with the following fields:

```python
@dataclass
class StateSnapshot:
    values: dict[str, Any]      # The state dict (messages, context_paths, etc.)
    next: tuple[str, ...]        # Which nodes would run next from this state
    config: RunnableConfig       # Config with this snapshot's checkpoint_id
    metadata: Optional[CheckpointMetadata]  # Step, source, writes
    created_at: Optional[str]   # ISO 8601 timestamp
    parent_config: Optional[RunnableConfig]  # Link to prior checkpoint
    tasks: tuple                 # Pending tasks (usually empty)
```

The most important fields are:

**`values`**: The actual state dictionary. For saathi, this is:

```python
{
    "messages": [HumanMessage(...), AIMessage(...), ...],
    "context_paths": ["/path/to/file.py", ...],
    "mode": "chat",
    "session_id": "abc123",
    "error": None,
}
```

**`next`**: A tuple of node names that would execute next if the graph were to continue from this state. If the graph has reached its terminal state (after the last node runs and reaches `END`), this is an empty tuple.

**`config`**: The config for this specific snapshot, including its `checkpoint_id`. This is what you pass to `aupdate_state()` to roll back to this state.

**`created_at`**: When this checkpoint was written, as an ISO 8601 timestamp string.

### 7.3 Using `aget_state` in saathi

saathi uses `aget_state` in several places:

**In the main chat loop**, to check whether the previous invocation completed successfully:

```python
snapshot = await self.graph.aget_state(self.config)
if snapshot.values.get("error"):
    self._display_error(snapshot.values["error"])
```

**In the `/context` command**, to read the current context_paths before modifying them:

```python
snapshot = await self.graph.aget_state(self.config)
current_paths = snapshot.values.get("context_paths", [])
```

**In the `/status` command**, to display a summary of the current session state:

```python
async def handle_status_command(self) -> None:
    snapshot = await self.graph.aget_state(self.config)
    values = snapshot.values

    message_count = len(values.get("messages", []))
    context_count = len(values.get("context_paths", []))
    mode = values.get("mode", "unknown")

    print(f"Session: {self.session_id}")
    print(f"Mode: {mode}")
    print(f"Messages: {message_count}")
    print(f"Context paths: {context_count}")
    if snapshot.created_at:
        print(f"Last checkpoint: {snapshot.created_at}")
```

### 7.4 When `aget_state` Returns None

If no checkpoint exists for the given `thread_id`, `aget_state` returns a snapshot with empty state (default values for all fields) and `next` set to the starting node of the graph. This represents the "before the first message" state — the graph is ready to run but has not run yet.

This means you can safely call `aget_state` even on a brand-new session that has never received any input. The snapshot you get back will have `messages = []`, `context_paths = []`, and so on.

### 7.5 The `next` Field

The `next` field is particularly useful for understanding what state the graph is in:

```python
snapshot = await graph.aget_state(config)

if not snapshot.next:
    print("Graph has completed — at END node")
elif "run_agent" in snapshot.next:
    print("Graph is waiting to run the agent node")
elif "check_context" in snapshot.next:
    print("Graph is waiting to run the context check")
```

For saathi's graph, after a normal `ainvoke` completes (reaches `END`), `snapshot.next` will be an empty tuple. If execution was interrupted mid-graph (for example, due to a timeout or exception), `snapshot.next` will indicate the node that was about to run when execution stopped.

---

## 8. `aget_state_history` — Browsing the Full History

### 8.1 The Method Signature

`aget_state_history` is an async generator that yields `StateSnapshot` objects for all checkpoints in a thread, from newest to oldest:

```python
async for snapshot in graph.aget_state_history(config):
    print(snapshot.created_at, len(snapshot.values.get("messages", [])))
```

It accepts optional parameters to limit the number of results:

```python
# Get only the 10 most recent checkpoints
async for snapshot in graph.aget_state_history(config, limit=10):
    ...
```

And to paginate from a specific checkpoint (useful for very long sessions):

```python
# Get checkpoints before a specific one
async for snapshot in graph.aget_state_history(config, before=some_config):
    ...
```

### 8.2 What `aget_state_history` Yields

Each yielded `StateSnapshot` is a complete checkpoint, with all the fields described in Section 7.2. Because the history is yielded newest-first, the first snapshot is the current state, the second is the state before the most recent node execution, and so on.

The `step` counter in the metadata decreases as you iterate:

- First snapshot: `metadata.step = N` (current step, highest number)
- Second snapshot: `metadata.step = N-1`
- ...
- Last snapshot: `metadata.step = 0`

### 8.3 Understanding the Step Granularity

A key detail: `aget_state_history` yields one snapshot per *node execution*, not one per *user message*. If a user sends one message and the graph executes three nodes (`check_context`, `run_agent`, `post_process`), there will be three new checkpoints in the history.

For displaying history to the user, you typically want to filter to only the checkpoints that represent complete AI responses — those created by the `run_agent` node (or wherever the final message is added). saathi does this by looking for checkpoints where the last message in the state is an `AIMessage`:

```python
async def get_turn_checkpoints(self) -> list[StateSnapshot]:
    """Get checkpoints that represent completed conversation turns."""
    turns = []
    async for snapshot in self.graph.aget_state_history(self.config):
        messages = snapshot.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            turns.append(snapshot)
    return turns
```

### 8.4 The `/checkpoints` Command Implementation

saathi's `/checkpoints` command uses `aget_state_history` to display a numbered list of conversation checkpoints:

```python
async def handle_checkpoints_command(self) -> None:
    """Display the checkpoint history for the current session."""
    snapshots = []

    async for snapshot in self.graph.aget_state_history(self.config):
        messages = snapshot.values.get("messages", [])
        # Only include checkpoints that have at least one AI response
        if any(isinstance(m, AIMessage) for m in messages):
            snapshots.append(snapshot)

    if not snapshots:
        print("No checkpoints found for this session.")
        return

    print(f"\nCheckpoint history for session {self.session_id}:")
    print("-" * 60)

    for i, snapshot in enumerate(snapshots):
        messages = snapshot.values.get("messages", [])
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        step = snapshot.metadata.step if snapshot.metadata else "?"
        ts = snapshot.created_at or "unknown"

        # Format: "  [0] Step 12 — 3 exchanges — 2024-01-15 10:30:45"
        exchange_count = min(human_count, ai_count)
        print(f"  [{i}] Step {step:3d} — {exchange_count:2d} exchanges — {ts[:19]}")

    print("-" * 60)
    print(f"Use /rollback <n> to restore checkpoint [n]")
    print(f"  Example: /rollback 2  (restore to 2 checkpoints ago)")
```

This displays output like:

```text
Checkpoint history for session 3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c:
------------------------------------------------------------
  [0] Step  18 — 4 exchanges — 2024-01-15 10:45:23  <- current
  [1] Step  15 — 3 exchanges — 2024-01-15 10:40:11
  [2] Step  12 — 2 exchanges — 2024-01-15 10:35:44
  [3] Step   9 — 1 exchanges — 2024-01-15 10:30:22
------------------------------------------------------------
Use /rollback <n> to restore checkpoint [n]
  Example: /rollback 2  (restore to 2 checkpoints ago)
```

### 8.5 Performance Considerations

`aget_state_history` loads and deserializes every checkpoint in the thread. For a session with hundreds of checkpoints, this can take a noticeable amount of time (hundreds of milliseconds to a few seconds). For the saathi use case, this is acceptable — `/checkpoints` is not called in a hot loop, and the user is waiting for the output anyway.

If performance became a concern (for very long sessions), a few optimizations are possible:

1. Add a `limit` parameter to only load the N most recent checkpoints.
2. Store message counts as separate metadata fields to avoid full deserialization just for counting.
3. Cache the checkpoint list between `/checkpoints` calls, invalidating the cache when a new message is added.

For the current saathi implementation, none of these optimizations are implemented, as they add complexity without clear benefit.

### 8.6 The Async Generator Pattern

`aget_state_history` returns an async generator, not a list. This is an important distinction for memory efficiency: LangGraph does not load all checkpoints into memory at once; it fetches them from the database one at a time as you iterate.

For typical saathi usage (dozens to a few hundred checkpoints), this distinction is not important — the total data fits comfortably in memory. But for sessions with thousands of checkpoints, the async generator pattern prevents out-of-memory errors.

Because `aget_state_history` returns an async generator, you must iterate it with `async for` — not `for`. Attempting to iterate it with a regular `for` loop will raise a `TypeError`.

---

## 9. `aupdate_state` — Injecting State from Outside

### 9.1 What `aupdate_state` Does

`aupdate_state` allows you to modify the graph's state from *outside* the graph — that is, without running any graph nodes. It creates a new checkpoint in the database with the provided state values, and the next `ainvoke` or `astream` call will start from this new checkpoint.

The method signature is:

```python
new_config = await graph.aupdate_state(
    config,           # The thread to update (and optionally a specific checkpoint)
    values,           # The values to update (partial state update)
    as_node=None,     # Optional: which node to pretend this update came from
)
```

It returns a new `config` dict that includes the `checkpoint_id` of the newly created checkpoint.

### 9.2 Partial State Updates

`aupdate_state` performs a *partial* state update — you only need to specify the fields you want to change. Fields you do not specify retain their current values. For example, to change only the mode:

```python
await graph.aupdate_state(config, {"mode": "debug"})
```

For list fields like `messages` that use `add_messages` as their reducer, the update *appends* to the existing list (it does not replace it). For simple scalar fields, the update *replaces* the existing value.

This means that if you want to do a full state replacement for rollback purposes, you must provide *all* the state fields, not just the ones that changed. Providing only `messages` would leave `mode`, `context_paths`, and other fields at their current (post-rollback-target) values, which is usually not what you want.

### 9.3 Full State Replacement for Rollback

For rollback, saathi provides the complete `values` dict from the target checkpoint:

```python
old_values = target_snapshot.values
# old_values contains ALL state fields from the target checkpoint

new_config = await graph.aupdate_state(self.config, old_values)
# All state fields are now set to their values from the target checkpoint
```

This works correctly because `old_values` is the full state dict, and `aupdate_state` with a full dict replaces every field. The only subtlety is the `messages` reducer: because `messages` uses `add_messages`, providing `old_values["messages"]` directly would append those messages to the current messages list, not replace it.

To handle this correctly for rollback, saathi uses the `as_node` parameter and a special state structure:

```python
# For rollback, we need to replace messages entirely, not append.
# We do this by treating the update as coming from the START node,
# which bypasses the add_messages reducer.
new_config = await graph.aupdate_state(
    self.config,
    old_values,
    as_node="__start__",
)
```

The `as_node="__start__"` parameter tells LangGraph to treat this update as the initial state input (which replaces rather than merges). This is the correct approach for full state replacement.

### 9.4 The `as_node` Parameter

The `as_node` parameter tells LangGraph to treat this update as if it came from a specific node. This affects:

1. What gets stored in `metadata.writes` for this checkpoint.
2. Which node LangGraph considers to be "next" after this update.
3. How reducers are applied to the updated values.

Common values for `as_node`:

- `None` (default): LangGraph determines the node based on graph topology.
- `"__start__"`: Treat as initial input — replaces rather than merges.
- A specific node name (e.g., `"run_agent"`): Pretend the update came from that node.

### 9.5 The Return Value

`aupdate_state` returns a new `RunnableConfig` with the `checkpoint_id` of the newly created checkpoint:

```python
new_config = await graph.aupdate_state(config, values)
new_checkpoint_id = new_config["configurable"]["checkpoint_id"]
```

The original `config` object is not modified. To verify that the update took effect, you can call `aget_state(new_config)` and inspect the returned snapshot.

Note that the original `config` (without `checkpoint_id`) still points to "the most recent checkpoint" — after `aupdate_state`, this is the checkpoint just created, so `aget_state(config)` and `aget_state(new_config)` will return the same snapshot.

---

## 10. The `/rollback` Command — Full Implementation Walkthrough

### 10.1 The Command Syntax

The `/rollback` command accepts a numeric argument specifying how many "turns" (complete exchange pairs) to go back:

```text
/rollback 1    # Go back one turn (undo the last exchange)
/rollback 3    # Go back three turns
/rollback      # With no argument, shows the checkpoint list (same as /checkpoints)
```

### 10.2 Defining a "Turn"

For rollback purposes, a "turn" is defined as one complete exchange: one user message plus the AI's response. Going back N turns means restoring the state to before the Nth-most-recent exchange.

This definition is more intuitive for users than "go back N checkpoints" (which would refer to individual node executions) or "go back N steps" (the step counter in checkpoint metadata, which also counts node-level checkpoints).

The mapping from "turns" to checkpoints:

```text
Turn 1:   user message → [node: check_context] → [node: run_agent] → AI response
Turn 2:   user message → [node: check_context] → [node: run_agent] → AI response
Turn 3:   user message → [node: check_context] → [node: run_agent] → AI response
```

Each turn generates approximately 2–4 checkpoints in the database. The turn-level filtering (find checkpoints where the last message is an `AIMessage`) selects one checkpoint per turn — the one representing the completed exchange.

### 10.3 The Full Rollback Implementation

```python
async def handle_rollback_command(self, args: str) -> None:
    """Handle the /rollback command."""
    from langchain_core.messages import AIMessage

    # Parse the argument
    if not args.strip():
        # No argument — show checkpoint list instead
        await self.handle_checkpoints_command()
        return

    try:
        turns_back = int(args.strip())
    except ValueError:
        print(f"Error: /rollback requires a number. Got: {args!r}")
        print("Usage: /rollback <n>  (e.g., /rollback 1)")
        return

    if turns_back < 1:
        print("Error: /rollback requires a positive integer.")
        return

    # Collect turn-level checkpoints (one per AI response)
    turn_checkpoints: list[StateSnapshot] = []
    async for snapshot in self.graph.aget_state_history(self.config):
        messages = snapshot.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            turn_checkpoints.append(snapshot)

    # turn_checkpoints[0] = current state
    # turn_checkpoints[1] = one turn ago
    # turn_checkpoints[N] = N turns ago

    if not turn_checkpoints:
        print("No conversation history found. Nothing to roll back.")
        return

    # Check if the requested rollback is possible
    # We need turns_back to index into the list, but index 0 is current state
    # So "go back 1 turn" means using index 1
    target_index = turns_back
    if target_index >= len(turn_checkpoints):
        available = len(turn_checkpoints) - 1  # Subtract 1 for current state
        print(f"Error: Cannot go back {turns_back} turns.")
        print(f"Maximum rollback for this session: {available} turn(s).")
        return

    target_snapshot = turn_checkpoints[target_index]

    # Confirm the rollback with the user
    target_messages = target_snapshot.values.get("messages", [])
    target_exchanges = sum(
        1 for m in target_messages if isinstance(m, AIMessage)
    )
    target_ts = (target_snapshot.created_at or "unknown")[:19]

    print(f"\nRolling back to state from {target_ts}")
    print(f"  Exchanges at that point: {target_exchanges}")
    print(f"  Current exchanges: {len(turn_checkpoints) - 1}")
    print()

    # Perform the rollback
    await self.rollback_to_snapshot(target_snapshot)

    # Confirm success
    print(f"Rollback complete. Went back {turns_back} turn(s).")
    print("The next message will continue from the restored state.")

async def rollback_to_snapshot(self, snapshot: StateSnapshot) -> None:
    """Restore the graph state to a specific snapshot."""
    old_values = snapshot.values

    # Create a new checkpoint with the old state values.
    # as_node="__start__" ensures messages are replaced, not appended.
    new_config = await self.graph.aupdate_state(
        self.config,
        old_values,
        as_node="__start__",
    )

    checkpoint_short = new_config["configurable"]["checkpoint_id"][:8]
    print(f"New checkpoint created: {checkpoint_short}...")
```

### 10.4 What Happens After Rollback

After a successful rollback:

1. The checkpoint database now has a new checkpoint (at the current step N + 1) with the state values from the historical checkpoint.
2. The `aget_state()` call will return this new checkpoint.
3. The next `ainvoke()` call will start from this new checkpoint.
4. The historical checkpoints are still in the database — they are not deleted. This means you could "rollback the rollback" if you wanted to, though saathi does not expose this capability directly.

The conversation continues from the restored state. To the user, it is as if the last N turns of conversation never happened.

### 10.5 An Example Rollback Scenario

Here is a concrete example to make this tangible:

```text
Session start (thread_id = "abc")

Turn 1:
  User: "What does the train() function do?"
  AI: "The train() function initializes the model, sets up the optimizer..."
  Checkpoint saved (step 3, 2 messages)

Turn 2:
  User: "How can I add dropout?"
  AI: "To add dropout, you would modify the model architecture..."
  Checkpoint saved (step 6, 4 messages)

Turn 3:
  User: "Write me a complete rewrite of the training loop"
  AI: <produces 200 lines of code, some incorrect>
  Checkpoint saved (step 9, 6 messages)

User: /rollback 1
```

After `/rollback 1`:

- The state is restored to the checkpoint at step 6 (4 messages: turns 1 and 2).
- The AI's incorrect rewrite of the training loop is gone from the state.
- The next message will continue from turn 2.

```text
Turn 3 (retry):
  User: "Actually, just show me how to add dropout to the existing loop"
  AI: <produces a focused, correct diff>
  Checkpoint saved (step 11, 6 messages again — but different content)
```

Note that the "incorrect" checkpoint from step 9 still exists in the database. Only the *current* state pointer has changed.

### 10.6 Error Handling in Rollback

The rollback command includes several error cases:

- **No argument**: Shows the checkpoint list (a graceful degradation).
- **Non-numeric argument**: Clear error message with usage instructions.
- **Zero or negative number**: Clear error message.
- **Number larger than available history**: Shows how many turns are available.
- **No history at all**: Explains that there is nothing to roll back.

Each error case gives the user actionable information and does not leave the agent in an inconsistent state. The graph's state is only modified by `aupdate_state`, which is only called after all argument validation has passed.

### 10.7 Future Enhancement: Named Rollback Targets

A natural future enhancement is to allow users to name checkpoints and roll back to them by name:

```text
/checkpoint save before-refactor
... several turns of conversation ...
/rollback before-refactor
```

This would require storing checkpoint names as additional metadata in the database and implementing a name-based lookup in the rollback command. The core mechanism (`aupdate_state` with the target snapshot's values) remains the same; only the lookup logic changes.

---

## 11. The `/checkpoints` Display Command

### 11.1 Design Goals

The `/checkpoints` command serves several purposes:

1. **Discoverability**: Let the user see how much history is available.
2. **Rollback preparation**: Help the user identify the right rollback target before running `/rollback`.
3. **Debugging**: Show timestamps and exchange counts for understanding the session timeline.

### 11.2 Display Format Design

The display format was designed to balance information density with readability:

```text
Checkpoint history for session abc12345-...
------------------------------------------------------------
  [0] Step  18 — 4 exchanges — 2024-01-15 10:45:23  <- current
  [1] Step  15 — 3 exchanges — 2024-01-15 10:40:11
  [2] Step  12 — 2 exchanges — 2024-01-15 10:35:44
  [3] Step   9 — 1 exchange  — 2024-01-15 10:30:22
------------------------------------------------------------
Use /rollback <n> to restore checkpoint [n]
  Example: /rollback 2  (restores to 2 exchanges ago)
```

Key design decisions:

- **Numbered from 0**: Index 0 is always the current state. This makes it easy to relate `/checkpoints` output to `/rollback n` — `/rollback 2` restores to what was checkpoint `[2]` in the list.
- **Exchange count, not message count**: "4 exchanges" (meaning 4 human messages and 4 AI responses) is more intuitive than "8 messages".
- **Step number shown**: For debugging, the step number helps correlate with LangGraph's internal checkpoint indexing.
- **Timestamps truncated to seconds**: Full microsecond precision (`2024-01-15T10:45:23.123456+00:00`) is rarely useful and clutters the display. Truncating to `2024-01-15 10:45:23` is easier to read.
- **"current" marker**: The `<- current` marker on index 0 prevents confusion about which direction the list is ordered.

### 11.3 Complete Implementation

```python
async def handle_checkpoints_command(self) -> None:
    """Display the checkpoint history for the current session."""
    from langchain_core.messages import AIMessage, HumanMessage

    # Collect turn-level checkpoints
    snapshots: list[StateSnapshot] = []
    async for snapshot in self.graph.aget_state_history(self.config):
        messages = snapshot.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            snapshots.append(snapshot)

    if not snapshots:
        print("\nNo conversation history found for this session.")
        print("Start chatting to create checkpoints.")
        return

    # Display header
    session_short = self.session_id[:8]
    print(f"\nCheckpoint history for session {session_short}...")
    print("-" * 60)

    for i, snapshot in enumerate(snapshots):
        messages = snapshot.values.get("messages", [])
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        step = snapshot.metadata.step if snapshot.metadata else 0

        # Format timestamp
        ts = (snapshot.created_at or "unknown time")
        # Convert ISO 8601 to readable format
        if len(ts) >= 19:
            ts_display = ts[:10] + " " + ts[11:19]
        else:
            ts_display = ts

        # Pluralize "exchange"
        exchange_word = "exchange " if ai_count == 1 else "exchanges"

        # Mark current
        marker = "  <- current" if i == 0 else ""

        print(
            f"  [{i}] Step {step:3d} — "
            f"{ai_count:2d} {exchange_word} — "
            f"{ts_display}"
            f"{marker}"
        )

    print("-" * 60)
    print(f"Use /rollback <n> to restore checkpoint [n]")
    if len(snapshots) > 1:
        example_n = min(2, len(snapshots) - 1)
        print(f"  Example: /rollback {example_n}")
```

### 11.4 Handling Edge Cases in Display

**No history**: If the session has not yet had any complete exchanges, the command explains this gracefully rather than displaying an empty table.

**Single exchange**: The "exchange" / "exchanges" pluralization handles the case where there is only one exchange.

**Very long sessions**: For sessions with many exchanges, the list can be long. A future enhancement would add a `--limit` option. For now, the full list is displayed with a suggestion to use `/rollback N` where N is the desired rollback point.

**Malformed timestamps**: The `or "unknown time"` fallback handles the case where `created_at` is `None` (which should not happen in practice, but defensive programming is worthwhile).

### 11.5 Integration with `/rollback`

The `/checkpoints` command is intentionally designed to feed into `/rollback`. The index numbers shown by `/checkpoints` correspond directly to the argument passed to `/rollback`:

```text
/checkpoints shows:
  [0] current state
  [1] one turn ago
  [2] two turns ago

/rollback 2 restores to what /checkpoints shows as [2]
```

This consistency is important for usability — the user should be able to look at the `/checkpoints` output and immediately know what argument to pass to `/rollback` without any mental arithmetic.

---

## 12. The `close_graph` Pattern — Clean Shutdown

### 12.1 Why Clean Shutdown Matters

The `AsyncSqliteSaver` holds an open `aiosqlite.Connection`. When the application exits, this connection should be closed cleanly. If the connection is not closed, several things can happen:

- SQLite's write-ahead log (WAL) file may not be checkpointed, leaving it on disk alongside the `.db` file.
- Python's garbage collector will eventually close the connection, but timing is nondeterministic.
- In tests, not closing connections can cause "database is locked" errors when multiple tests share a database file.

For a CLI tool like saathi, clean shutdown is especially important because the tool is designed to be run frequently — once per terminal session, often multiple times per day. Accumulating unclosed connections (or WAL files) over time would be noisy and potentially problematic.

### 12.2 The `close_graph` Function

saathi's `close_graph` function encapsulates the shutdown logic:

```python
async def close_graph(graph: CompiledGraph) -> None:
    """Close the graph's checkpointer connection cleanly.

    Safe to call multiple times — idempotent.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return

    conn = getattr(checkpointer, "conn", None)
    if conn is None:
        return

    # Check if the connection is already closed
    # aiosqlite sets _connection to None after closing
    if getattr(conn, "_connection", None) is None:
        return  # Already closed, nothing to do

    try:
        await conn.close()
    except Exception as e:
        # Log but don't raise — shutdown should not fail the app
        import logging
        logging.getLogger(__name__).warning(
            f"Error closing checkpoint database: {e}"
        )
```

### 12.3 Why Idempotency Matters

The `close_graph` function checks whether the connection is already closed before attempting to close it. This makes it safe to call multiple times — for example:

- From the main cleanup handler at the end of the session.
- From the signal handler (SIGINT/SIGTERM).
- From a test's `teardown` method.

Without idempotency, calling `close_graph` twice would raise an error on the second call (because `aiosqlite` raises if you close an already-closed connection). With idempotency, the second call is a no-op.

The idempotency check works by examining `conn._connection` — `aiosqlite` sets this to `None` after closing the underlying `sqlite3.Connection`. This is an implementation detail that could change between `aiosqlite` versions, so it is worth noting in the code as a potentially fragile check.

### 12.4 Integration with Application Lifecycle

`close_graph` is called in saathi's main function, in a `finally` block to ensure it always runs regardless of how the application exits:

```python
async def main() -> None:
    graph = None
    try:
        db_path = get_checkpoint_db_path()
        graph = await create_graph(db_path)
        await run_interactive_session(graph)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving state...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    finally:
        if graph is not None:
            await close_graph(graph)
            print("Checkpoint database closed.")
```

The `finally` block ensures that even if the application raises an unexpected exception, the checkpoint database connection is properly closed before the process exits.

### 12.5 Signal Handling

For robustness, saathi also registers a signal handler for `SIGTERM` (the signal sent by `kill` and by process managers):

```python
import asyncio
import signal

def register_shutdown_handlers(graph: CompiledGraph) -> None:
    """Register signal handlers for clean shutdown."""
    loop = asyncio.get_event_loop()

    def handle_sigterm():
        print("\nReceived SIGTERM. Closing checkpoint database...")
        loop.create_task(close_graph(graph))

    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
```

Note that `SIGINT` (Ctrl+C) is handled by the `KeyboardInterrupt` exception in the `try/finally` block above — no separate signal handler is needed for it.

### 12.6 What Happens If You Don't Close Cleanly

If the process exits without calling `close_graph`, SQLite handles it gracefully:

1. SQLite detects that the connection was closed unexpectedly.
2. If WAL mode is enabled, the WAL file remains on disk but is self-consistent.
3. The next time the database is opened, SQLite automatically processes the WAL and brings the database to a consistent state.

So not closing cleanly is not catastrophic — SQLite is designed to survive process crashes. But it is messier (WAL files left on disk, delayed checkpoint processing) and is avoidable with a clean shutdown.

---

## 13. `aiosqlite` — Why Async SQLite Matters

### 13.1 The Problem with Synchronous SQLite

The standard Python `sqlite3` module is synchronous. When you call `sqlite3.connect()`, `cursor.execute()`, or `cursor.fetchall()`, the current thread blocks until the operation completes.

In a synchronous Python program, this is fine. But saathi uses an async event loop (`asyncio`) for all its I/O operations — LLM API calls, file system operations, and now database operations. If you use synchronous `sqlite3` in an async program:

```python
# WRONG — blocks the event loop
import sqlite3
conn = sqlite3.connect("checkpoints.db")
cursor = conn.execute("SELECT * FROM checkpoints WHERE thread_id = ?", [thread_id])
rows = cursor.fetchall()  # Event loop is blocked here!
```

The `cursor.fetchall()` call blocks the *entire event loop* for the duration of the database query. While the query is running, no other async tasks can make progress — not the LLM API response streaming, not the UI update, not the keyboard input handling.

For a database on a local disk, typical SQLite queries complete in 5–50 milliseconds. This is long enough to cause noticeable stuttering if it happens frequently, and it is architecturally incorrect — async programs should not block the event loop.

### 13.2 How `aiosqlite` Solves This

`aiosqlite` is a thin wrapper around the standard `sqlite3` module that runs all database operations in a separate thread pool. From the event loop's perspective, a database query is an asynchronous operation that yields control while waiting for the thread pool worker to complete the query.

The internal implementation:

1. When you call `await conn.execute(...)`, `aiosqlite` packages the query into a task.
2. The task is submitted to a `concurrent.futures.ThreadPoolExecutor`.
3. The `await` returns control to the event loop.
4. When the thread pool worker completes the SQLite query, it signals the event loop.
5. The event loop resumes the coroutine with the query result.

From the application code's perspective, you simply add `await` to database calls:

```python
# CORRECT — async SQLite
import aiosqlite

async def get_checkpoints(db_path: str, thread_id: str) -> list:
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ?",
            [thread_id]
        )
        rows = await cursor.fetchall()
        return rows
```

### 13.3 The Connection Pattern in saathi

saathi holds a single long-lived `aiosqlite.Connection` rather than opening a new connection per query:

```python
# In create_graph():
conn = await aiosqlite.connect(str(db_path))
checkpointer = AsyncSqliteSaver(conn)
```

This is more efficient than using `async with aiosqlite.connect(...)` per operation, because:

1. Connection creation has overhead (opening the file, reading the WAL, initializing the page cache).
2. A single connection allows SQLite to keep frequently accessed pages in its page cache between queries.
3. The connection can be reused across the entire session without re-establishing it.

The `aiosqlite.connect()` coroutine opens the connection and returns an `aiosqlite.Connection` object. Importantly, `aiosqlite.Connection` can also be used as an async context manager, which automatically closes the connection when the `async with` block exits. saathi does not use this pattern (instead using explicit `close_graph`), for the reasons discussed in Section 4.4.

### 13.4 `aiosqlite` Configuration

saathi uses `aiosqlite` with default settings. For reference, the key configuration options available:

```python
conn = await aiosqlite.connect(
    database="checkpoints.db",  # Path to the database file
    timeout=5.0,                 # Seconds to wait for a lock (default: 5)
    isolation_level=None,        # None = autocommit mode
)
```

`isolation_level=None` (autocommit mode) is appropriate for `aiosqlite` because `AsyncSqliteSaver` manages its own transactions. Using the default isolation level (`""`) would cause every statement to start a transaction implicitly, which can conflict with `AsyncSqliteSaver`'s own transaction management.

### 13.5 Performance Characteristics of `aiosqlite`

Because `aiosqlite` uses a thread pool worker, there is a small overhead per operation: the cost of scheduling work to the thread pool and receiving the result. This overhead is typically 0.1–0.5 milliseconds per operation.

For saathi's use case:

- Each `ainvoke` call triggers 2–5 checkpoint writes.
- Each checkpoint write requires 1–3 SQLite operations.
- Total `aiosqlite` overhead per `ainvoke`: approximately 1–7 milliseconds.

This overhead is imperceptible compared to LLM API call latency and is a worthwhile trade-off for keeping the event loop responsive.

---

## 14. Checkpoint Storage Location and Configuration

### 14.1 Default Location

By default, saathi stores its checkpoint database in a `.saathi` directory within the current working directory:

```folder
your-project/
├── .saathi/
│   └── checkpoints.db     <- checkpoint database
├── src/
│   └── ...
└── README.md
```

The `.saathi` directory name was chosen to:

- Be clearly associated with saathi (prevents confusion with other tools' hidden directories).
- Be hidden by default (dotfile convention on Unix-like systems).
- Be project-local (each project has its own independent checkpoint history).

### 14.2 Path Resolution Logic

saathi determines the checkpoint database path at startup:

```python
from pathlib import Path

def get_checkpoint_db_path(
    project_dir: Optional[Path] = None,
    db_name: str = "checkpoints.db",
) -> Path:
    """Get the path to the checkpoint database.

    Uses the project directory (current working directory by default)
    and creates the .saathi subdirectory if it does not exist.
    """
    if project_dir is None:
        project_dir = Path.cwd()

    saathi_dir = project_dir / ".saathi"
    saathi_dir.mkdir(parents=True, exist_ok=True)

    return saathi_dir / db_name
```

The `mkdir(parents=True, exist_ok=True)` call creates the `.saathi` directory and any missing parent directories, and does nothing if the directory already exists. This is idempotent — safe to call every time saathi starts.

### 14.3 Configuring the Database Location

The database location can be overridden via the `--db` command-line argument:

```bash
# Use a custom database path
saathi --db /path/to/my/checkpoints.db

# Use a different project directory
saathi --project /path/to/different/project
```

This is useful for:

- Storing checkpoints in a shared location (e.g., a team's shared drive).
- Using different checkpoint databases for different aspects of a project.
- Testing saathi with a throwaway database.

### 14.4 What Happens If You Delete the Database

Deleting the checkpoint database (the `.db` file) gives you a clean slate:

```bash
rm .saathi/checkpoints.db
```

The next time saathi starts, it creates a new empty database. All prior session history is permanently lost. For most users, this is a straightforward way to start fresh if the checkpoint database grows too large or becomes corrupted.

saathi does not currently provide a `/clear-history` command (which would delete all checkpoints via SQL), but this is a natural future addition.

### 14.5 Size Considerations and Growth

The checkpoint database grows indefinitely under normal use. Each message exchange adds approximately one to three checkpoints, each of which stores the full conversation history at that point.

**Approximate growth rates:**

| Usage Level | Messages/Day | DB Growth/Day |
| --- | --- | --- |
| Light | 20 | ~2 MB |
| Moderate | 100 | ~10 MB |
| Heavy | 500 | ~50 MB |
| Very Heavy | 2000 | ~200 MB |

These estimates assume average message length of ~500 characters. Code-heavy sessions will be larger.

**Storage implications:**

- After 30 days of moderate use: ~300 MB
- After 1 year of moderate use: ~3.6 GB

For most developer machines, this is manageable but grows indefinitely. Future saathi versions should implement checkpoint pruning: a configurable retention policy that removes checkpoints older than N days, or retains only the most recent K checkpoints per thread.

### 14.6 Database Inspection

Because the checkpoint database is a standard SQLite file, you can inspect it with any SQLite tool:

```bash
# Using the sqlite3 CLI
sqlite3 .saathi/checkpoints.db

# Check table sizes
sqlite3 .saathi/checkpoints.db "SELECT count(*) FROM checkpoints;"
sqlite3 .saathi/checkpoints.db "SELECT thread_id, count(*) FROM checkpoints GROUP BY thread_id;"

# Find the total database size
ls -lh .saathi/checkpoints.db
```

Note that the `checkpoint` column in the `checkpoints` table is a BLOB containing serialized Python/LangGraph data. It is not human-readable directly, but you can read the metadata columns (`thread_id`, `checkpoint_id`, `parent_checkpoint_id`, `type`) directly.

### 14.7 `.gitignore` Considerations

The `.saathi` directory should typically be added to `.gitignore`:

```gitignore
# saathi checkpoint database — session-specific, not for version control
.saathi/
```

Checkpoint databases contain personal session history, may be large, and are specific to each developer's machine. They should not be committed to version control. The `.saathi` directory entry in `.gitignore` prevents accidental commits.

### 14.8 Backup and Migration

Because the checkpoint database is a single file, backup is simple:

```bash
# Backup
cp .saathi/checkpoints.db .saathi/checkpoints.db.bak

# Restore from backup
cp .saathi/checkpoints.db.bak .saathi/checkpoints.db
```

Migration between machines is also straightforward — copy the `.db` file to the same relative location on the new machine. Since the thread IDs are stored in the database, the session history is fully portable.

---

## 15. Other Checkpointers — When to Upgrade

### 15.1 `InMemorySaver`

`InMemorySaver` stores checkpoints in a Python dictionary in memory. It is the simplest possible checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**When to use `InMemorySaver`:**

- **Tests**: For unit and integration tests where you want checkpointing behavior but do not want to create files on disk. Tests can run faster and do not need cleanup.
- **Prototyping**: When experimenting with graph structure and not yet ready to set up persistent storage.
- **Ephemeral sessions**: For applications where sessions are intentionally not persisted (e.g., a one-shot query tool).

**When NOT to use `InMemorySaver`:**

- Any application where session persistence is required (saathi's primary use case).
- Long-running sessions where checkpoints grow large (memory is finite).

saathi's test suite uses `InMemorySaver` for most tests, only using `AsyncSqliteSaver` for the specific tests that verify checkpointing behavior:

```python
# In conftest.py or test_graph.py
import pytest
from langgraph.checkpoint.memory import InMemorySaver

@pytest.fixture
def in_memory_graph():
    """Compile the saathi graph with an in-memory checkpointer for testing."""
    builder = build_saathi_graph()
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
```

### 15.2 `AsyncPostgresSaver`

For multi-user or production deployments, `AsyncPostgresSaver` (from `langgraph-checkpoint-postgres`) replaces SQLite with PostgreSQL:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncpg

async def create_production_graph(db_url: str) -> CompiledGraph:
    conn = await asyncpg.connect(db_url)
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)
```

**When to upgrade to PostgreSQL:**

- **Multiple concurrent users**: PostgreSQL handles concurrent writes from many clients. SQLite does not.
- **Separate storage from compute**: If you are running saathi (or a saathi-derived service) in containers or serverless functions, storing the checkpoint database in a separate PostgreSQL instance ensures data persists across container restarts.
- **Large-scale deployments**: PostgreSQL is more efficient for very large checkpoint databases.
- **Existing PostgreSQL infrastructure**: If your project already uses PostgreSQL, using it for checkpoints avoids introducing a second database technology.

**When to stay with SQLite:**

- Single-user CLI tool (saathi's current use case).
- No existing PostgreSQL infrastructure.
- Development and testing.
- Environments where zero-configuration is valuable.

### 15.3 Redis-Based Checkpointing

There is no official LangGraph checkpointer for Redis, but one can be implemented against the `BaseCheckpointSaver` interface. Redis would be appropriate for:

- **High-throughput, short-lived sessions**: Redis's in-memory storage makes it extremely fast.
- **Automatic TTL**: Redis keys can be set to expire automatically, solving the "database grows indefinitely" problem.
- **Existing Redis infrastructure**: If your application already uses Redis for caching.

Redis is not persistent by default (though RDB snapshots and AOF logging provide durability options), so it is not appropriate for applications where checkpoint durability is critical.

### 15.4 Custom Checkpointers

For specialized requirements, you can implement `BaseCheckpointSaver` directly. Use cases:

- Storing checkpoints in a cloud object store (S3, GCS, Azure Blob).
- Encrypting checkpoints before storage.
- Routing checkpoints to different backends based on content (e.g., large checkpoints to object store, small ones to Redis).
- Adding metrics and observability to checkpoint operations.

A custom checkpointer must implement at minimum: `put`, `get_tuple`, `list` (sync versions), and ideally also the async versions `aput`, `aget_tuple`, `alist`.

### 15.5 The Checkpointer Decision Matrix

| Requirement | InMemorySaver | AsyncSqliteSaver | AsyncPostgresSaver |
| --- | --- | --- | --- |
| Zero configuration | Yes | Yes | No |
| Persistence across restarts | No | Yes | Yes |
| Multi-user support | No | No | Yes |
| Concurrent writes | No | No (single file) | Yes |
| Suitable for tests | Yes | Yes (slow) | Rarely |
| Production-ready | No | Single-user only | Yes |
| Storage limit | RAM | Disk | Disk/managed |
| saathi default | No | **Yes** | No |

---

## 16. Testing Checkpointing

### 16.1 Testing Philosophy

Testing checkpointing code requires a different mindset than testing pure business logic. Checkpointing involves I/O (disk writes), timing (timestamps), and stateful operations (sequential checkpoint writes and reads). Good tests for checkpointing should:

1. Verify that checkpoints are written correctly.
2. Verify that state can be loaded from checkpoints.
3. Verify that the rollback mechanism works end-to-end.
4. Verify that `close_graph` is safe and idempotent.
5. Verify that different thread IDs are truly isolated.

### 16.2 Test Setup: The `tmp_db_path` Fixture

For tests that need an actual SQLite database, saathi uses a pytest fixture that creates a temporary database in a temporary directory:

```python
import pytest
import asyncio
from pathlib import Path
import tempfile

@pytest.fixture
async def tmp_db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite database path for tests."""
    return tmp_path / "test_checkpoints.db"

@pytest.fixture
async def sqlite_graph(tmp_db_path: Path):
    """Create a compiled saathi graph with a real SQLite checkpointer."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from saathi.graph import build_saathi_graph

    builder = build_saathi_graph()
    conn = await aiosqlite.connect(str(tmp_db_path))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    yield graph

    # Cleanup
    from saathi.graph import close_graph
    await close_graph(graph)
```

Using pytest's `tmp_path` fixture ensures that each test gets a fresh, isolated temporary directory. The database file is automatically cleaned up when the test session ends.

### 16.3 Test: Graph Compilation Creates the DB

```python
@pytest.mark.asyncio
async def test_graph_compilation_creates_database(tmp_db_path: Path):
    """Compiling the graph creates the SQLite database file."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from saathi.graph import build_saathi_graph, close_graph

    # Database does not exist yet
    assert not tmp_db_path.exists()

    # Compile the graph
    builder = build_saathi_graph()
    conn = await aiosqlite.connect(str(tmp_db_path))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    # Database now exists
    assert tmp_db_path.exists()

    # Tables were created
    async with aiosqlite.connect(str(tmp_db_path)) as inspect_conn:
        cursor = await inspect_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}

    assert "checkpoints" in tables
    assert "checkpoint_writes" in tables

    await close_graph(graph)
```

### 16.4 Test: Checkpoint History Is Queryable

```python
@pytest.mark.asyncio
async def test_checkpoint_history_is_queryable(sqlite_graph):
    """After sending messages, checkpoint history is queryable."""
    import uuid
    from langchain_core.messages import AIMessage

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Send two messages
    await sqlite_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config
    )
    await sqlite_graph.ainvoke(
        {"messages": [{"role": "user", "content": "How are you?"}]},
        config
    )

    # Collect history
    snapshots = []
    async for snapshot in sqlite_graph.aget_state_history(config):
        snapshots.append(snapshot)

    # Should have multiple checkpoints
    assert len(snapshots) >= 2

    # Most recent checkpoint should have 4 messages (2 human + 2 AI)
    latest = snapshots[0]
    messages = latest.values.get("messages", [])
    human_msgs = [m for m in messages if hasattr(m, "type") and m.type == "human"]
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    assert len(human_msgs) == 2
    assert len(ai_msgs) == 2

    # Checkpoints should be ordered newest-first (step decreasing)
    steps = [s.metadata.step for s in snapshots if s.metadata]
    assert steps == sorted(steps, reverse=True)
```

### 16.5 Test: `close_graph` Is Safe to Call Twice

```python
@pytest.mark.asyncio
async def test_close_graph_is_idempotent(tmp_db_path: Path):
    """close_graph can be called multiple times without error."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from saathi.graph import build_saathi_graph, close_graph

    builder = build_saathi_graph()
    conn = await aiosqlite.connect(str(tmp_db_path))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    # First close — should succeed
    await close_graph(graph)

    # Second close — should be a no-op, not raise an exception
    await close_graph(graph)

    # Third close — still safe
    await close_graph(graph)
```

### 16.6 Test: Fresh Thread ID Isolates History

```python
@pytest.mark.asyncio
async def test_thread_id_isolation(sqlite_graph):
    """Different thread IDs have completely independent checkpoint histories."""
    import uuid

    thread_a = str(uuid.uuid4())
    thread_b = str(uuid.uuid4())
    config_a = {"configurable": {"thread_id": thread_a}}
    config_b = {"configurable": {"thread_id": thread_b}}

    # Send a message on thread A
    await sqlite_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Thread A message"}]},
        config_a
    )

    # Thread B should have no history
    snapshots_b = []
    async for snapshot in sqlite_graph.aget_state_history(config_b):
        snapshots_b.append(snapshot)

    # Thread B either has no checkpoints or has an empty initial state
    if snapshots_b:
        # If there is a snapshot, it should have no messages from thread A
        messages_b = snapshots_b[0].values.get("messages", [])
        for msg in messages_b:
            content = getattr(msg, "content", "")
            assert "Thread A message" not in content

    # Thread A should have exactly the message we sent
    snapshots_a = []
    async for snapshot in sqlite_graph.aget_state_history(config_a):
        snapshots_a.append(snapshot)

    assert len(snapshots_a) >= 1
    messages_a = snapshots_a[0].values.get("messages", [])
    contents = [getattr(m, "content", "") for m in messages_a]
    assert any("Thread A message" in c for c in contents)
```

### 16.7 Test: Rollback Restores Prior State

```python
@pytest.mark.asyncio
async def test_rollback_restores_prior_state(sqlite_graph):
    """Rolling back to a prior checkpoint restores the correct state."""
    import uuid
    from langchain_core.messages import AIMessage

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Send first message
    await sqlite_graph.ainvoke(
        {"messages": [{"role": "user", "content": "First message"}]},
        config
    )

    # Get state after first message
    state_after_first = await sqlite_graph.aget_state(config)
    msg_count_after_first = len(state_after_first.values.get("messages", []))

    # Send second message
    await sqlite_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Second message"}]},
        config
    )

    # Verify we have more messages now
    state_after_second = await sqlite_graph.aget_state(config)
    msg_count_after_second = len(state_after_second.values.get("messages", []))
    assert msg_count_after_second > msg_count_after_first

    # Find the checkpoint from after the first message
    target_snapshot = None
    async for snapshot in sqlite_graph.aget_state_history(config):
        messages = snapshot.values.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if len(ai_messages) == 1:
            target_snapshot = snapshot
            break

    assert target_snapshot is not None, "Should find checkpoint after first message"

    # Perform rollback to that snapshot
    await sqlite_graph.aupdate_state(config, target_snapshot.values)

    # Verify the state was rolled back
    state_after_rollback = await sqlite_graph.aget_state(config)
    msg_count_after_rollback = len(state_after_rollback.values.get("messages", []))
    assert msg_count_after_rollback == msg_count_after_first
```

### 16.8 Test: The `from_conn_string` Trap

It is worth having a test that explicitly documents the `from_conn_string` trap as a regression guard:

```python
def test_from_conn_string_returns_context_manager():
    """Document that from_conn_string() returns a context manager, not a saver.

    This test exists as a regression guard and documentation: anyone reading
    it understands WHY saathi uses the explicit connection pattern.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import contextlib

    result = AsyncSqliteSaver.from_conn_string(":memory:")

    # The result is an async context manager, not an AsyncSqliteSaver
    assert not isinstance(result, AsyncSqliteSaver)
    assert isinstance(result, contextlib.AbstractAsyncContextManager)

    # It does NOT have the methods we need directly
    assert not hasattr(result, "setup")
    assert not hasattr(result, "aput")
    assert not hasattr(result, "aget_tuple")
```

### 16.9 Using `pytest-asyncio`

All the async test functions above use `pytest-asyncio` for running async tests. The `pyproject.toml` configuration for saathi includes:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Automatically detect async test functions
```

With `asyncio_mode = "auto"`, pytest-asyncio automatically wraps async test functions in an event loop. Without this setting, you would need to decorate each async test with `@pytest.mark.asyncio`.

The `sqlite_graph` fixture uses `yield` to ensure cleanup runs after each test that uses it. The `tmp_path` fixture (built into pytest) automatically cleans up the temporary directory after the test session, removing the SQLite database files.

### 16.10 Testing Considerations for Async Code

Testing async code with pytest-asyncio requires awareness of a few potential pitfalls:

**Event loop sharing**: By default, pytest-asyncio creates a new event loop for each test. This prevents cross-test contamination of async state.

**Fixture scoping**: If you define an async fixture with `scope="session"`, it shares an event loop with all tests in the session. This can cause subtle issues if tests modify shared state. For checkpointing tests, use `scope="function"` (the default) to give each test a fresh database.

**Cleanup ordering**: pytest runs fixture cleanup in reverse order of setup. If `sqlite_graph` depends on `tmp_db_path`, `tmp_db_path` is cleaned up after `sqlite_graph`. This means the `close_graph` call in `sqlite_graph`'s teardown runs while the database file still exists, which is correct.

---

## 17. Summary

This chapter covered saathi's checkpointing system from the abstract interface to the concrete implementation details. Here is a brief recap of the most important points.

### 17.1 The Architecture at a Glance

```flow
User interaction
       |
       v
saathi CLI (cli.py)
  |  +-- /checkpoints ────────────────────────────+
  |  +-- /rollback ──────────────────────────+    |
  |  +-- normal chat ────────────────+       |    |
       |                             |       |    |
       v                             |       |    |
LangGraph CompiledGraph              |       |    |
  |  +-- ainvoke(state, config) <────+       |    |
  |  +-- aget_state(config) ────────────────-+----+
  |  +-- aget_state_history(config) <────────+
  |  +-- aupdate_state(config, values) <─── /rollback
       |
       v
AsyncSqliteSaver
  |  +-- aput(config, checkpoint, ...)  <- called by LangGraph
  |  +-- aget_tuple(config)              <- called by LangGraph
  |  +-- alist(config, ...)              <- called by LangGraph
       |
       v
aiosqlite.Connection
       |
       v
.saathi/checkpoints.db (SQLite file)
```

### 17.2 The Lifecycle

1. **Startup**: `create_graph()` opens the SQLite connection, creates `AsyncSqliteSaver`, calls `setup()`, compiles the graph.
2. **Each message**: `ainvoke()` automatically saves 1–N checkpoints (one per node execution) to the database.
3. **`/checkpoints`**: `aget_state_history()` reads all checkpoints, filters to turn-level, displays them.
4. **`/rollback N`**: `aget_state_history()` finds the Nth turn, `aupdate_state()` injects that state as a new checkpoint.
5. **Shutdown**: `close_graph()` closes the SQLite connection cleanly.

### 17.3 The Key Design Decisions

**Why SQLite**: Zero configuration, single file, good enough for single-user. Would upgrade to PostgreSQL for multi-user.

**Why explicit connection, not `from_conn_string`**: `from_conn_string()` returns a context manager, not an `AsyncSqliteSaver`. Using it incorrectly gives a subtle runtime error. Explicit connection management is clearer.

**Why `aiosqlite`, not `sqlite3`**: Async programs must not block the event loop. `aiosqlite` runs SQLite in a thread pool, keeping the event loop free.

**Why store full state per checkpoint, not deltas**: Full state makes any checkpoint independently loadable without replaying from the beginning. The cost is storage size — manageable for single-user use.

**Why idempotent `close_graph`**: Multiple cleanup paths (finally block, signal handler) may both call `close_graph`. Idempotency prevents errors on the second call.

---

## 18. Key Takeaways

1. **Checkpointing saves full agent state after every node execution**, enabling persistence, rollback, and reproducibility. It is not optional for a production-quality agent tool.

2. **`BaseCheckpointSaver` is the LangGraph checkpoint interface**. You implement it (or use an existing implementation). LangGraph calls its methods automatically — you never call `put`, `get_tuple`, or `list` directly.

3. **`AsyncSqliteSaver` is the right choice for single-user CLI tools**. Zero configuration, single file, ACID guarantees, good enough performance.

4. **Never use `AsyncSqliteSaver.from_conn_string()` without `async with`**. It returns a context manager, not an `AsyncSqliteSaver` instance. The explicit pattern (`aiosqlite.connect` → `AsyncSqliteSaver(conn)` → `setup()`) is clearer and less error-prone.

5. **The `thread_id` in `config` is the session key**. All checkpoints for a session share a `thread_id`. Different `thread_id` values are completely isolated.

6. **`aget_state()` returns the current `StateSnapshot`**. Use it to read the current state without executing the graph.

7. **`aget_state_history()` is an async generator** that yields checkpoints newest-first. Use it to implement `/checkpoints` display and to find rollback targets.

8. **`aupdate_state()` injects state from outside the graph**, creating a new checkpoint. This is how `/rollback` works — it writes a new checkpoint with old state values, and the next `ainvoke` starts from there.

9. **`aiosqlite` runs SQLite in a thread pool**, keeping the async event loop unblocked. Never use synchronous `sqlite3` in an async application.

10. **Store the checkpoint database in `.saathi/checkpoints.db`** by default. Add `.saathi/` to `.gitignore`. The database grows indefinitely — future work: implement pruning.

11. **`close_graph()` should be idempotent** and called in a `finally` block to ensure the SQLite connection is always closed cleanly, even if the application exits via an exception.

12. **Tests should use `InMemorySaver` for speed**, and `AsyncSqliteSaver` with `tmp_path` fixtures only for the specific tests that verify checkpointing behavior.

13. **Upgrade to `AsyncPostgresSaver` for multi-user or production use**. The interface is identical — swap the checkpointer at graph compile time.

14. **The checkpoint database is a standard SQLite file** — inspectable with any SQLite client. The `checkpoint` BLOB column is not human-readable, but the metadata columns (`thread_id`, `checkpoint_id`, `parent_checkpoint_id`) are.

15. **Rollback is non-destructive**. Old checkpoints remain in the database. `aupdate_state` creates a *new* checkpoint with the old values — it does not delete the history between the rollback target and the current state.

---

End of Chapter 8 — Checkpointing and Rollback: Persistent Agent State

---

### Further Reading

- LangGraph documentation: [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- `langgraph-checkpoint-sqlite` source: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite)
- `aiosqlite` documentation: [aiosqlite.readthedocs.io](https://aiosqlite.readthedocs.io)
- SQLite WAL mode: [sqlite.org/wal.html](https://www.sqlite.org/wal.html)
- `langgraph-checkpoint-postgres`: for production multi-user deployments

### Next Chapter Preview

Chapter 9 covers saathi's context management system: how files and directories are added to the agent's context, how context is serialized into the system prompt, and how the agent uses context to provide accurate, project-aware responses. We will examine the `/context` command implementation, the `context_paths` state field, and the strategies saathi uses to handle large files and directory trees without exceeding LLM context limits.
