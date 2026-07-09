# Chapter 5 — LangGraph Core: State Machines for AI Agents

> "A chain is a path. A graph is a world. Agents need worlds."

This is the most important chapter in the book. LangGraph is the architectural
spine of saathi. Every message, every tool call, every streamed token, every
rollback checkpoint flows through the graph. Understanding LangGraph at a
mechanical level — not just "it runs nodes" but *how* it routes, *how* state is
merged, *how* the checkpointer persists state, and *exactly why* one missing
edge broke the ReAct loop — is the foundation for building reliable AI agents.

---

## 5.1 Why Graphs for Agents?

### 5.1.1 The Problem with Linear Chains

Imagine building an AI agent as a linear chain:

```text
user_input → llm_call → output_parser → response
```

This works for single-turn Q&A. The model reads the question, produces an
answer. Done.

Now add tools. The model needs to read a file before answering:

```text
user_input → llm_call → [model wants to call read_file]
                         → read_file(path)
                         → [tool result]
                         → llm_call → response
```

A two-step chain handles this. But what if the model needs to read three files,
then search for a pattern, then read one more file based on the search results?

```text
user → llm → read_file(a) → llm → read_file(b) → llm → read_file(c) →
       llm → search_across_files(dir, pattern) →
       llm → read_file(d) → llm → response
```

Building this as a static chain requires knowing in advance how many tool calls
the model will make. Real tasks are unpredictable. The model might make 1 tool
call or 12. The chain approach requires either:

- Building a chain with a fixed upper bound on tool calls (brittle, wastes
  tokens on empty steps)
- Using a `while` loop in Python that checks whether the last message has tool
  calls (works, but then you have reinvented a very basic version of LangGraph)

The fundamental issue is that **agents need loops**. A loop is a graph with a
cycle. Chains have no cycles. Therefore chains cannot model agents.

### 5.1.2 What a Graph Gives You

LangGraph models agent behaviour as a directed graph where:

- **Nodes** are computation steps (call the LLM, execute tools)
- **Edges** are transitions between steps (always transition, or conditional)
- **Cycles** are allowed (the tool node transitions back to the agent node)
- **State** is a typed dict that flows through the graph and is merged at
  each step

The saathi graph looks like this:

```flow
                    START
                      │
                      ▼
                  ┌───────┐
                  │ agent │ ◄──────────────────┐
                  └───────┘                    │
                      │                        │
            tools_condition                    │
                 /        \                    │
           has tools     no tools              │
               │              │                │
               ▼              ▼                │
          ┌───────┐          END               │
          │ tools │                            │
          └───────┘                            │
               │                               │
               └───────────────────────────────┘
```

The cycle is `agent → tools → agent`. The model can iterate as many times as
it needs. `tools_condition` is the exit: when the model produces a final answer
with no tool calls, the graph terminates.

---

## 5.2 The StateGraph

A `StateGraph` is LangGraph's core class. You give it a schema (a TypedDict),
add nodes and edges, then compile it into a `CompiledGraph`.

### 5.2.1 A Minimal Complete Example

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Step 1: Define the state schema
class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Step 2: Define nodes (async functions that return state updates)
async def my_llm_node(state: MyState) -> dict:
    last = state["messages"][-1]
    # In a real graph, call an LLM here
    reply = AIMessage(content=f"You said: {last.content}")
    return {"messages": [reply]}

# Step 3: Build the graph
builder = StateGraph(MyState)
builder.add_node("llm", my_llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

# Step 4: Compile
graph = builder.compile()

# Step 5: Run
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    {"configurable": {"thread_id": "test-1"}}
)
print(result["messages"][-1].content)
# => "You said: Hello"
```

This is the simplest possible LangGraph program. No tools, no cycles, no
checkpointer. But every element is here: schema, nodes, edges, compile, invoke.

### 5.2.2 StateGraph vs MessageGraph

LangGraph also ships `MessageGraph`, a convenience subclass where the state is
always a list of messages (implicitly `Annotated[list[BaseMessage], add_messages]`).
Nodes receive the list directly instead of a dict.

saathi uses `StateGraph` rather than `MessageGraph` because its state has four
fields, not just messages:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context_paths: list[str]
    mode: str
    session_id: str
```

`MessageGraph` would not accommodate `context_paths`, `mode`, and `session_id`.
`StateGraph` is the more general and usually more appropriate choice.

---

## 5.3 Nodes

A node is any callable that:

1. Takes the current state as its argument
2. Returns a `dict` of state updates

That is the entire contract. The callable can be synchronous or asynchronous.
It can be a plain function, a closure, a method, or a LangGraph prebuilt like
`ToolNode`.

### 5.3.1 The Node Signature

```python
# Synchronous node
def my_node(state: AgentState) -> dict:
    return {"messages": [...]}

# Asynchronous node (preferred in saathi)
async def my_node(state: AgentState) -> dict:
    return {"messages": [...]}

# Node that also receives the RunnableConfig
async def my_node(state: AgentState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"]["thread_id"]
    return {"messages": [...]}
```

The `config` parameter is optional. If you include it, LangGraph injects the
`RunnableConfig` that was passed to `graph.ainvoke()`. This is how a node can
access the thread ID, callbacks, or custom config values.

### 5.3.2 The Agent Node

saathi's agent node is defined in `saathi/agent/nodes.py`. It is a closure
created by `make_agent_node()`:

```python
# From saathi/agent/nodes.py (full source)
def make_agent_node(llm: LanguageModelLike, memory_store: MemoryStore):
    """Return a LangGraph node that calls the LLM with bound tools."""

    project_instructions = find_project_instructions()

    def _on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        log.warning(
            "ollama_retry",
            attempt=attempt,
            delay=delay,
            error=str(exc) or type(exc).__name__,
        )

    async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        memory_block = memory_store.format_for_prompt()
        system_prompt = build_system_prompt(
            context_paths=state.get("context_paths", []),
            memory_block=memory_block,
            mode=state.get("mode", "default"),
            project_instructions=project_instructions,
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await retry_async(
            lambda: llm.ainvoke(messages, config),
            attempts=settings.ollama_max_retries,
            base_delay=settings.ollama_retry_base_delay,
            on_retry=_on_retry,
        )
        return {"messages": [response]}

    return agent_node
```

The node:

1. Builds a fresh system prompt incorporating current `context_paths`, `mode`,
   and `memory_block`.
2. Prepends the system prompt to the message history (without storing it in
   state).
3. Calls the LLM with retry on transient errors.
4. Returns the model's response as a single-item list in `{"messages": [response]}`.

The return value `{"messages": [response]}` is a **partial state update**, not
the full new state. LangGraph merges this update into the current state using
the reducer for each field. We cover reducers in section 5.6.

### 5.3.3 The Tool Node

saathi's tool node is also a closure, defined in `saathi/agent/tool_node.py`.
It is custom rather than using LangGraph's built-in `ToolNode` for two reasons:

1. **Hooks** — saathi can block dangerous tool calls (via `HookRunner.check_block`)
   and run pre/post side effects (via `run_pre_tool` / `run("post_tool", ...)`).

2. **Parallel execution** — all tool calls in one AI message are run concurrently
   using `asyncio.gather`, bounded by a semaphore.

```python
# From saathi/agent/tool_node.py (full source)
def make_hooked_tool_node(tools: list[BaseTool], hook_runner: HookRunner):
    tools_by_name = {t.name: t for t in tools}
    semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tools))

    async def _run_one(call: dict) -> ToolMessage:
        name = call["name"]
        args = call.get("args", {})
        call_id = call["id"]

        # 1. sensitive-path guard, then pre_tool hook
        reason = hook_runner.check_block(name, args)
        if reason is None:
            reason = await hook_runner.run_pre_tool(name, args)
        if reason is not None:
            return ToolMessage(
                content=f"BLOCKED: {reason}. The tool was not executed.",
                tool_call_id=call_id,
                name=name,
            )

        # 2. execute the tool
        tool = tools_by_name.get(name)
        if tool is None:
            return ToolMessage(
                content=f"Error: unknown tool '{name}'.",
                tool_call_id=call_id,
                name=name,
            )
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:
            result = f"Error executing {name}: {exc}"
        else:
            log.debug("tool_ok", tool=name)

        # 3. post_tool hook
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

Key observations:

- The node reads `state["messages"][-1]` to get the last `AIMessage` and its
  `tool_calls`. This works because the graph guarantees the tool node only runs
  after the agent node has placed an `AIMessage` with tool calls in state.
- All tool calls run concurrently via `asyncio.gather`.
- Every call produces exactly one `ToolMessage` — even failures.
- The semaphore bounds concurrency to `settings.max_parallel_tools` (default 4).

---

## 5.4 Edges — Three Types

Edges define how the graph moves between nodes. There are three kinds.

### 5.4.1 Entry Edge

The entry edge connects the special `START` sentinel to the first node:

```python
from langgraph.graph import START

builder.add_edge(START, "agent")
```

`START` is a constant defined in `langgraph.graph`. It is the implicit entry
point of every graph. Without an entry edge the graph does not know where to
begin. Every graph must have exactly one entry edge (unless you are using
`set_entry_point()`, which is equivalent).

### 5.4.2 Unconditional Edge

An unconditional edge always transitions from one node to the next:

```python
builder.add_edge("tools", "agent")
```

After the tool node finishes, always go to the agent node. No conditions, no
routing logic.

### 5.4.3 Conditional Edge

A conditional edge calls a routing function to decide the next node:

```python
from langgraph.prebuilt import tools_condition

builder.add_conditional_edges("agent", tools_condition)
```

`tools_condition` is a function `(state) -> str | list[str]`. It returns the
name of the next node (or `END`). LangGraph calls it after the agent node
finishes and routes to whichever node the function returns.

You can also provide an explicit mapping from return values to node names:

```python
def my_router(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"

builder.add_conditional_edges(
    "agent",
    my_router,
    {"tools": "tools", "end": END}  # optional: explicit mapping
)
```

Without the mapping, LangGraph expects the routing function to return the
node name directly (which `tools_condition` does).

### 5.4.4 The Full saathi Graph

```python
# From saathi/agent/graph.py — build_graph()
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

Five lines. This is the entire graph structure. Read it as:

1. Start at `agent`.
2. After `agent`, call `tools_condition`:
   - If the last message has tool calls → go to `tools`
   - Otherwise → go to `END`
3. After `tools`, unconditionally go back to `agent`.

The cycle is `agent → tools → agent`. It repeats until the model produces a
message with no tool calls, at which point `tools_condition` routes to `END`.

---

## 5.5 tools_condition — The Built-in Router

`tools_condition` is imported from `langgraph.prebuilt`. Let us look at what
it does:

```python
# Simplified implementation of tools_condition
from langgraph.graph import END

def tools_condition(state: dict) -> str:
    """Route to 'tools' if the last message has tool_calls, else to END."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls and len(tool_calls) > 0:
        return "tools"
    return END
```

It looks at `state["messages"][-1]`. If the last message is an `AIMessage` with
non-empty `tool_calls`, it returns `"tools"`. Otherwise it returns `END`.

### 5.5.1 The Critical Bug — Why There Must Be No agent→END Edge

This is the most important edge case in saathi's development history. Early
versions of the graph had an additional edge:

```python
# WRONG — this was the original code that broke the ReAct loop
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge("agent", END)  # ← THIS LINE BROKE EVERYTHING
```

With this extra edge, the graph had two outgoing edges from the `agent` node:

- The conditional edge from `tools_condition`
- The unconditional edge to `END`

LangGraph interprets multiple outgoing edges from a node as a **fan-out**: run
all destination nodes in parallel. With two edges from `agent`, the graph was
doing:

1. Agent runs, returns message with tool calls.
2. `tools_condition` says "go to tools".
3. The unconditional `agent → END` edge *also* fires, simultaneously.
4. The graph tries to go to `tools` and `END` at the same time.
5. The `END` transition terminates the graph before `tools` can run.

Result: tool calls were silently ignored. The model would say "I'll call
`read_file` for you" and then immediately terminate without running the tool.
The symptom was the agent giving no answer (because the AIMessage with tool
calls was the final message, not an AIMessage with text content).

The fix was simply to remove the unconditional `agent → END` edge. The
`tools_condition` function already handles the `END` case — it returns `END`
when there are no tool calls. You do not need (and must not add) an explicit
`agent → END` edge.

The correct graph:

```python
# CORRECT — the current saathi graph
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # handles END case
builder.add_edge("tools", "agent")
# No agent → END edge!
```

This is annotated in the source:

```python
# From saathi/agent/graph.py
# tools_condition routes agent -> "tools" when the model emitted tool calls,
# or agent -> END when it produced a final answer. No extra agent->END edge:
# that would fan out to END on every step and break the ReAct loop.
builder.add_conditional_edges("agent", tools_condition)
```

Memorise this rule: **do not add an unconditional edge to END from any node
that also has a conditional edge that may go to END**. The conditional edge
covers the termination case. Adding the unconditional edge creates a fan-out
that will break your loop.

---

## 5.6 Reducers — How State Is Merged

When a node returns a dict, LangGraph does not simply replace the state with
the returned dict. It **merges** the update into the existing state using
a reducer for each field.

### 5.6.1 Default Reducer: Last-Write-Wins

For fields with no annotation, the default reducer is last-write-wins: the new
value replaces the old one entirely.

```python
class AgentState(TypedDict):
    mode: str          # no annotation = last-write-wins
    session_id: str    # no annotation = last-write-wins
    context_paths: list[str]  # no annotation = last-write-wins
    messages: Annotated[list[BaseMessage], add_messages]  # annotated reducer
```

When the agent node returns `{"messages": [...]}`, it does not return `mode`,
`session_id`, or `context_paths`. LangGraph leaves those fields unchanged. The
reducer is invoked only for fields that appear in the returned dict.

### 5.6.2 The add_messages Reducer

`add_messages` is imported from `langgraph.graph.message`. It is the standard
reducer for conversation history. Its behaviour:

```python
from langgraph.graph.message import add_messages

# add_messages(current_list, update_list) -> new_list

# Case 1: New messages with no matching IDs → append
current = [HumanMessage(content="Hi", id="msg1")]
update = [AIMessage(content="Hello!", id="msg2")]
result = add_messages(current, update)
# => [HumanMessage("Hi", id="msg1"), AIMessage("Hello!", id="msg2")]

# Case 2: Message with matching ID → replace (update by ID)
current = [HumanMessage(content="Hi", id="msg1"), AIMessage(content="Hello!", id="msg2")]
update = [AIMessage(content="Updated reply!", id="msg2")]  # same id as existing
result = add_messages(current, update)
# => [HumanMessage("Hi", id="msg1"), AIMessage("Updated reply!", id="msg2")]
# The old AIMessage with id="msg2" is replaced, not appended.
```

The ID-based update is used internally by LangGraph during streaming. When the
model streams tokens, each chunk has the same message ID. The partial messages
are assembled by ID in the state.

For normal tool call and response flows, Case 1 applies: new messages are
appended. This is why the state grows monotonically:

```text
After step 1 (agent runs):
  messages = [HumanMessage, AIMessage(tool_calls=[...])]

After step 2 (tools run):
  messages = [HumanMessage, AIMessage(tool_calls=[...]), ToolMessage, ToolMessage]

After step 3 (agent runs again):
  messages = [HumanMessage, AIMessage(tool_calls=[...]), ToolMessage, ToolMessage, AIMessage(final)]
```

The `add_messages` reducer is what makes the ReAct loop work correctly. Without
it, each node's return value would overwrite the entire message list, losing all
history.

### 5.6.3 Writing a Custom Reducer

You can write any function as a reducer:

```python
def keep_last_n(n: int):
    """A reducer factory that keeps only the last n items."""
    def reducer(current: list, update: list) -> list:
        combined = current + update
        return combined[-n:]
    return reducer

class BoundedState(TypedDict):
    messages: Annotated[list[BaseMessage], keep_last_n(20)]
```

saathi does not use a bounded reducer. Instead it uses the compaction system
(`/compact` command and auto-compaction) to summarise history when it grows
too large. This preserves semantic content while reducing token count.

---

## 5.7 AgentState in Depth

saathi's state is defined in `saathi/agent/state.py`:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Complete mutable state for one agent session."""

    messages: Annotated[list[BaseMessage], add_messages]
    context_paths: list[str]
    mode: str  # "default" | "explain" | "refactor" | "debug"
    session_id: str
```

### 5.7.1 messages

The full conversation history. Uses `add_messages` reducer (append / update by
ID). This is the primary data field — everything the model has seen and said,
all tool calls and their results, live here.

The system prompt is **not** stored in `messages`. It is synthesised on every
agent node call from `context_paths`, `mode`, and the `MemoryStore`. This means
you can update the system prompt without modifying stored state — just change
`mode` or `context_paths`.

### 5.7.2 context_paths

A list of file/directory paths that the user has specified as the session scope
(via the `--context` CLI flag or `/context` command). The agent node reads this
field and includes it in the system prompt:

```python
# From saathi/agent/prompts.py — build_system_prompt()
if context_paths:
    paths = "\n".join(f"  - {p}" for p in context_paths)
    parts.append(f"Context scope — prefer reading and editing these paths first:\n{paths}")
```

This does not *prevent* the model from reading other files, but it biases its
attention. When working in a large monorepo, scoping to `src/saathi/agent/`
prevents the model from wandering into unrelated directories.

### 5.7.3 mode

One of `"default"`, `"explain"`, `"refactor"`, `"debug"`. Controls which
addendum is appended to the system prompt. The user changes it with `/mode`:

```python
# From saathi/agent/prompts.py
_MODE_ADDENDA = {
    "explain": "MODE: explain\n- Read files, never modify them\n...",
    "refactor": "MODE: refactor\n- Use patch_file instead of write_file...",
    "debug": "MODE: debug\n- Reproduce the bug first before attempting a fix...",
}
```

The `mode` is stored in state so it persists across turns within a session.
Changing mode mid-session (via `/mode refactor`) updates the state and takes
effect on the next turn.

### 5.7.4 session_id

A UUID hex string. Used as the `thread_id` for the LangGraph checkpointer
(see section 5.11). This ties the state to a specific conversation thread
in the SQLite checkpoint database.

The `session_id` is generated in `cli.py` when the session starts:

```python
# From saathi/session/manager.py (inferred) / cli.py
import uuid
state = SessionState(model_id=model_id, context_paths=context_paths)
# state.session_id = uuid.uuid4().hex (generated in SessionState.__init__)
config = {"configurable": {"thread_id": state.session_id}}
```

When the user runs `/compact`, a new `session_id` is generated to start a fresh
checkpoint thread with the compacted history as the initial state.

---

## 5.8 Super-Steps — LangGraph's Execution Model

LangGraph executes graphs in **super-steps**. A super-step is one round of
execution: all nodes that have their inputs ready run in that super-step.

For a linear sequence `A → B → C`, the super-steps are:

- Super-step 1: A runs
- Super-step 2: B runs (receives A's output)
- Super-step 3: C runs (receives B's output)

For a fan-out `A → [B, C]` (A has two unconditional outgoing edges):

- Super-step 1: A runs
- Super-step 2: B and C run concurrently (both receive A's output)
- Super-step 3: (any node that depends on both B and C)

For saathi's graph:

```text
Super-step 1: agent runs
  → If tool calls: proceed to step 2
  → If no tool calls: END

Super-step 2: tools runs
  → All tool calls execute concurrently (within this super-step)

Super-step 3: agent runs again (with tool results in state)
  → If more tool calls: proceed to step 4
  → If final answer: END

... repeat until final answer
```

Each super-step corresponds to one checkpoint written to the database. After
step 1, the state is checkpointed with the agent's first response. After step 2,
the state is checkpointed with the tool results. And so on.

This checkpoint granularity enables fine-grained rollback. The user can roll
back to any step, not just the beginning of the turn. This is particularly
useful when the model called a wrong tool and modified a file incorrectly — you
can roll back to before that tool call.

### 5.8.1 The Recursion Limit

LangGraph enforces a recursion limit (default: 25 cycles) to prevent infinite
loops. Each super-step increments a counter. If the counter exceeds the limit,
LangGraph raises a `GraphRecursionError`.

```python
# You can increase the limit in the config
config = {
    "configurable": {"thread_id": "..."},
    "recursion_limit": 50,  # increase for tasks that need many tool calls
}
```

For most coding tasks, 25 is sufficient. A task that needs more than 25 tool
call cycles is either very large (e.g. refactoring an entire codebase) or stuck
in a loop (the model keeps calling the same tool with the same arguments).

---

## 5.9 Compilation

After adding all nodes and edges, you compile the graph:

```python
# From saathi/agent/graph.py
graph = builder.compile(checkpointer=checkpointer)
```

Compilation does several things:

1. **Validates the graph** — checks for disconnected nodes, missing entry
   edges, invalid conditional edge targets.

2. **Wires reducers** — for each field in the state schema, finds the
   reducer (from `Annotated` annotations or uses the default).

3. **Attaches the checkpointer** — every state transition will be persisted
   to the checkpointer's database.

4. **Returns a `CompiledGraph`** — which is itself a `Runnable` and can be
   called with `invoke`, `ainvoke`, `stream`, `astream`, and `astream_events`.

### 5.9.1 Compilation Without a Checkpointer

You can compile without a checkpointer:

```python
graph = builder.compile()  # no checkpointer
```

Without a checkpointer:

- The graph is stateless between calls (no persistence)
- `/rollback` and `/checkpoints` do not work
- Each `ainvoke` or `astream_events` call gets a fresh state (the passed-in
  `input_state` is the entire starting state)
- The `thread_id` in config is ignored

This is fine for unit tests and one-off tasks, but not suitable for an
interactive REPL where you want conversation history to persist.

### 5.9.2 The AsyncSqliteSaver

saathi uses `AsyncSqliteSaver` from `langgraph.checkpoint.sqlite.aio`:

```python
# From saathi/agent/graph.py — build_graph()
conn = await aiosqlite.connect(str(db_path))
checkpointer = AsyncSqliteSaver(conn)
await checkpointer.setup()
return builder.compile(checkpointer=checkpointer)
```

The `setup()` call creates the necessary tables in the SQLite database if they
do not exist. The tables are:

```sql
-- (Simplified schema)
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

Each row is one super-step's state. The `thread_id` is the session ID;
`checkpoint_id` is the super-step number; `parent_checkpoint_id` links to the
previous step, forming a linked list.

The connection is owned by the application, not the saver. saathi opens the
connection in `build_graph()` and closes it in `close_graph()`:

```python
async def close_graph(graph) -> None:
    """Close the SQLite connection backing a compiled graph's checkpointer."""
    checkpointer = getattr(graph, "checkpointer", None)
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        with contextlib.suppress(Exception):
            await conn.close()
```

Owning the connection directly (rather than using `AsyncSqliteSaver.from_conn_string()`
as a context manager) means the connection stays open for the entire session —
which is necessary for a long-lived REPL — and is explicitly closed on shutdown.

---

## 5.10 ainvoke vs astream vs astream_events

The `CompiledGraph` supports three primary invocation modes. Choosing the
right one matters for performance, latency, and the user experience.

### 5.10.1 ainvoke

```python
result = await graph.ainvoke(input_state, config)
# result is the final AgentState dict
```

`ainvoke` runs the graph to completion and returns the final state. No
intermediate results are available. The entire response — all tool calls,
all results, the final answer — arrives at once.

saathi uses `ainvoke` only in `--print` / `--output-format` mode, where the
goal is to emit a clean final answer to stdout for piping:

```python
# From saathi/cli.py — _print_mode()
result = await graph.ainvoke(input_state, config)
messages: list[BaseMessage] = result.get("messages", [])
response = _final_text(messages)
print(response)  # clean output for piping
```

Interactive mode does NOT use `ainvoke` because it would block the terminal
for the entire duration of the turn (potentially 30+ seconds for a complex
task). The user would see nothing until the complete answer appeared.

### 5.10.2 astream

```python
async for state_chunk in graph.astream(input_state, config):
    # state_chunk is a dict of node_name → partial state update
    for node_name, state_update in state_chunk.items():
        print(f"Node {node_name} produced: {state_update}")
```

`astream` yields partial results after each node completes. For saathi's graph:

- After the agent node: `{"agent": {"messages": [AIMessage(tool_calls=[...])]}}`
- After the tool node: `{"tools": {"messages": [ToolMessage, ToolMessage]}}`
- After the final agent node: `{"agent": {"messages": [AIMessage(content="...")]}}`

This is useful if you want to react to each node's output but do not need
token-level granularity. It does not yield individual tokens; the entire model
response appears as one chunk.

### 5.10.3 astream_events

```python
async for event in graph.astream_events(input_state, config, version="v2"):
    kind = event["event"]
    # ...
```

`astream_events` with `version="v2"` is the most granular option. It yields
an event for every significant happening in the graph:

| Event type | When it fires |
| ----------- | --------------- |
| `on_chain_start` | A chain/graph/node starts |
| `on_chain_end` | A chain/graph/node ends |
| `on_chat_model_start` | The LLM call begins |
| `on_chat_model_stream` | A token chunk arrives |
| `on_chat_model_end` | The LLM call completes |
| `on_tool_start` | A tool begins execution |
| `on_tool_end` | A tool finishes execution |

saathi uses this to:

- Print tokens as they arrive (`on_chat_model_stream`)
- Update the spinner with the current tool name (`on_tool_start`)
- Render tool results (`on_tool_end`)
- Capture the final state (`on_chain_end` for the `"LangGraph"` chain)

The `version="v2"` parameter is required for the current event schema. v1 exists
for backward compatibility but is less detailed.

---

## 5.11 The Thread Model

LangGraph's checkpointer is scoped by `thread_id`. Every graph invocation
carries a config dict:

```python
config = {"configurable": {"thread_id": "my-session-id"}}
result = await graph.ainvoke(input_state, config)
```

The `thread_id` is the key to the conversation. All checkpoints for a thread
are stored under that ID. When you invoke the graph again with the same
`thread_id`, LangGraph loads the most recent checkpoint and uses it as the
starting state, merging your new `input_state` on top.

### 5.11.1 How LangGraph Loads State at Invocation Time

When `graph.ainvoke(input_state, config)` is called with a `thread_id` that
has existing checkpoints:

1. The checkpointer loads the most recent checkpoint for that `thread_id`.
2. The loaded state becomes the base state.
3. `input_state` is merged into the base state using the reducers.
4. The graph runs from there.

This means that when saathi's `_run_turn()` passes `input_state`:

```python
input_state = {
    "messages": messages,          # the full message list including the new HumanMessage
    "context_paths": state.context_paths,
    "mode": state.mode,
    "session_id": state.session_id,
}
```

LangGraph sees the new `messages` list and merges it with the checkpointed
state using `add_messages`. Since saathi passes the complete current message
list (including the new `HumanMessage` it just appended), and `add_messages`
deduplicates by ID, this works correctly.

### 5.11.2 Thread Isolation

Different `thread_id` values are completely isolated:

```python
# Thread A
config_a = {"configurable": {"thread_id": "session-abc"}}
await graph.ainvoke({"messages": [HumanMessage(content="Hi")]}, config_a)
# Thread A state: [HumanMessage("Hi"), AIMessage("Hello!")]

# Thread B
config_b = {"configurable": {"thread_id": "session-xyz"}}
await graph.ainvoke({"messages": [HumanMessage(content="Who are you?")]}, config_b)
# Thread B state: [HumanMessage("Who are you?"), AIMessage("I'm Saathi...")]

# Thread A is unaffected by thread B
result_a = await graph.ainvoke({"messages": [HumanMessage(content="Remember me?")]}, config_a)
# Thread A state: [HumanMessage("Hi"), AIMessage("Hello!"), HumanMessage("Remember me?"), AIMessage("Yes!")]
```

This is how multiple users could share one saathi process: each user gets a
different `thread_id`. The SQLite database stores all threads; each thread is
queried independently.

saathi generates a new `session_id` for each REPL session:

```python
# In SessionState.__init__
self.session_id: str = uuid.uuid4().hex
```

And uses it as the `thread_id`:

```python
config = {"configurable": {"thread_id": state.session_id}}
```

---

## 5.12 LangGraph Prebuilt Components

LangGraph ships several prebuilt components in `langgraph.prebuilt`. We have
seen two: `tools_condition` and `ToolNode`. Let us cover all the relevant ones.

### 5.12.1 tools_condition

```python
from langgraph.prebuilt import tools_condition
```

A routing function. Returns `"tools"` if the last message has tool calls,
`END` otherwise. Used in `add_conditional_edges`. We covered it fully in
section 5.5.

### 5.12.2 ToolNode

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=ALL_TOOLS)
builder.add_node("tools", tool_node)
```

The built-in tool execution node. It reads `state["messages"][-1].tool_calls`,
executes each tool, and returns `ToolMessage` objects. It supports parallel
execution and handles errors by returning error messages as `ToolMessage`
content.

saathi uses a custom tool node (`make_hooked_tool_node`) instead of `ToolNode`
for two reasons:

1. **Hooks** — `ToolNode` does not have a hook system for pre/post execution
   side effects or path blocking.
2. **Error format** — `ToolNode` raises on errors by default. saathi's custom
   node catches all errors and returns them as `ToolMessage` content, ensuring
   the model always receives a response and can decide how to proceed.

If your use case does not need hooks, `ToolNode` is a perfectly good default:

```python
# Simple setup without hooks
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=my_tools)
builder.add_node("tools", tool_node)
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

### 5.12.3 create_react_agent

```python
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
)
```

`create_react_agent` is a convenience function that builds the standard ReAct
graph (agent → tools → agent with `tools_condition`) in one call. It is
equivalent to what saathi does manually in `build_graph()`, minus the hooks,
the custom tool node, and the retry logic.

When to use `create_react_agent`:

- Prototyping: fastest way to get a working agent
- When you do not need custom tool execution (no hooks, no path blocking)
- When you do not need custom retry behaviour

When to use a manual `StateGraph`:

- When you need hooks or other pre/post execution logic
- When you need multi-agent graphs (multiple agent nodes, complex routing)
- When you need a custom state schema (saathi's `context_paths`, `mode`)
- When you need custom error handling in the tool node

saathi does not use `create_react_agent` for all of the above reasons. But for
a quick prototype or a simple single-tool agent, it is the right starting point.

---

## 5.13 Graph Introspection

LangGraph provides tools for inspecting and visualising your graph.

### 5.13.1 get_graph()

```python
graph_viz = graph.get_graph()
```

Returns a `Graph` object with nodes, edges, and metadata. You can use it to:

```python
# Print a text representation
print(graph_viz.to_json())

# Generate a Mermaid diagram
print(graph_viz.draw_mermaid())
```

For saathi's graph, `draw_mermaid()` produces:

```text
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    agent([agent])
    tools([tools])
    __end__([<p>__end__</p>]):::last
    __start__ --> agent;
    agent -.-> tools;
    agent -.-> __end__;
    tools --> agent;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0 
    classDef last fill:#bfb6fc
```

The dashed arrows from `agent` represent the conditional edge from
`tools_condition`. Solid arrows are unconditional.

### 5.13.2 Inspecting State

```python
# Get the current state of a thread
snapshot = await graph.aget_state(config)
print(snapshot.values)         # the AgentState dict
print(snapshot.next)           # which nodes will run next
print(snapshot.config)         # the config that produced this state

# Get the full history of a thread
history = [snapshot async for snapshot in graph.aget_state_history(config)]
# history[0] is the most recent, history[-1] is the initial state
```

saathi uses `aget_state_history` in the `/checkpoints` command:

```python
# From saathi/ui/commands.py — handle_checkpoints()
async def handle_checkpoints(graph, config):
    history = [c async for c in graph.aget_state_history(config)]
    for i, checkpoint in enumerate(history[:10]):
        msg_count = len(checkpoint.values.get("messages", []))
        console.print(f"  Step {i}: {msg_count} messages")
```

---

## 5.14 Error Handling in Nodes

What happens when a node raises an exception?

LangGraph propagates the exception out of `ainvoke`, `astream`, and
`astream_events`. The caller (saathi's `_run_turn`) catches it:

```python
# From saathi/cli.py — _run_turn()
try:
    async for event in graph.astream_events(input_state, config, version="v2"):
        ...
except KeyboardInterrupt:
    spinner.stop()
    console.print("\n[yellow]Interrupted.[/yellow]")
except Exception as exc:
    spinner.stop()
    console.print(f"\n[red]Error:[/red] {exc}")
    if settings.debug:
        import traceback
        traceback.print_exc()
```

The exception does not corrupt the checkpointed state. The checkpoint that was
written before the exception is still valid. On the next invocation with the
same `thread_id`, the graph will resume from that checkpoint.

### 5.14.1 The "Absorb Errors" Pattern in Tool Nodes

Rather than propagating errors, saathi's tool node absorbs them:

```python
# From saathi/agent/tool_node.py — _run_one()
try:
    result = await tool.ainvoke(args)
except Exception as exc:
    log.error("tool_error", tool=name, error=str(exc))
    result = f"Error executing {name}: {exc}"
```

This pattern keeps the graph running. The model receives the error as a
`ToolMessage` and can respond to it. The alternatives are:

1. **Propagate** — the graph terminates, the user sees a traceback, the turn
   is lost. Bad for UX.
2. **Retry** — retry the same tool call. Appropriate for transient failures
   (network timeouts). saathi does not retry tool calls (only the LLM call is
   retried, in `retry_async`).
3. **Absorb** — the error becomes a tool result. The model sees it and can
   recover. Appropriate for expected failures (file not found, invalid regex).

The "absorb" pattern is what LangGraph's built-in `ToolNode` also does (when
`handle_tool_errors=True`).

### 5.14.2 Error Handling in the Agent Node

The agent node uses `retry_async` for transient LLM call failures:

```python
# From saathi/agent/nodes.py
response = await retry_async(
    lambda: llm.ainvoke(messages, config),
    attempts=settings.ollama_max_retries,
    base_delay=settings.ollama_retry_base_delay,
    on_retry=_on_retry,
)
```

`retry_async` retries only on connection errors (Ollama server not ready),
not on semantic errors (the model said something wrong). Retrying semantic
errors would just produce the same wrong answer, wasting tokens.

If all retry attempts fail, `retry_async` raises, which propagates out of the
agent node, through LangGraph, and is caught by the `except Exception` handler
in `_run_turn`.

---

## 5.15 The Full saathi Graph Walkthrough

This section traces a complete multi-tool turn through the graph step by step.
User says: `"Read the file src/saathi/cli.py and tell me what it does"`

### 5.15.1 Initial State

Before the turn starts, the session's state (loaded from the checkpoint) is:

```python
{
    "messages": [
        # (previous conversation history, if any)
    ],
    "context_paths": [],
    "mode": "default",
    "session_id": "a1b2c3d4e5f6..."
}
```

### 5.15.2 _run_turn() Appends the HumanMessage

```python
# From saathi/cli.py — _run_turn()
messages.append(HumanMessage(content="Read the file src/saathi/cli.py and tell me what it does"))
input_state = {
    "messages": messages,
    "context_paths": state.context_paths,
    "mode": state.mode,
    "session_id": state.session_id,
}
```

The `HumanMessage` is now in `messages`. The spinner starts.

### 5.15.3 graph.astream_events() Begins

LangGraph loads the checkpoint for this `thread_id`, merges `input_state`
using `add_messages` (appending the new `HumanMessage`), and begins execution.

**Event stream so far:**

```text
on_chain_start  name="LangGraph"
on_chain_start  name="agent"
on_chat_model_start  name="ChatOllama"
```

### 5.15.4 Super-Step 1: Agent Node Runs

The agent node calls:

```python
messages = [SystemMessage(content=system_prompt)] + state["messages"]
# state["messages"] = [...previous history..., HumanMessage("Read the file...")]
response = await llm.ainvoke(messages, config)
```

The model receives 16 tool schemas, the system prompt, and the conversation
history. It decides to call `read_file` with `path = "src/saathi/cli.py"`.

The model returns an `AIMessage`:

```python
AIMessage(
    content="",  # no text yet — just a tool call
    tool_calls=[
        {
            "id": "call_f1a2b3c4",
            "name": "read_file",
            "args": {"path": "src/saathi/cli.py"},
            "type": "tool_call",
        }
    ],
    id="msg_response_1",
)
```

Note: the model may stream tokens for the tool call. For Ollama models with
tool calling, the tool call is typically emitted as a single chunk (no
character-by-character streaming). So the `on_chat_model_stream` event fires
once with the complete `AIMessageChunk`.

**Event stream:**

```text
on_chat_model_stream  chunk=AIMessageChunk(tool_calls=[read_file(...)], content="")
on_chat_model_end     output=AIMessage(tool_calls=[read_file(...)])
on_chain_end          name="agent"
```

The agent node returns:

```python
{"messages": [AIMessage(content="", tool_calls=[...])]}
```

LangGraph applies `add_messages`: the `AIMessage` is appended to the state.

**State after super-step 1:**

```python
{
    "messages": [
        ...previous...,
        HumanMessage("Read the file src/saathi/cli.py and tell me what it does"),
        AIMessage(content="", tool_calls=[read_file(path="src/saathi/cli.py")]),
    ],
    ...
}
```

Checkpoint written to SQLite.

### 5.15.5 tools_condition Routing

LangGraph calls `tools_condition(state)`. The last message is the `AIMessage`
with `tool_calls = [read_file(...)]`. It is non-empty. `tools_condition`
returns `"tools"`.

### 5.15.6 Super-Step 2: Tool Node Runs

The tool node reads `state["messages"][-1]` (the `AIMessage` with tool calls):

```python
tool_calls = [{"id": "call_f1a2b3c4", "name": "read_file", "args": {"path": "src/saathi/cli.py"}}]
```

One tool call. It runs `_guarded(call)` for it (the semaphore is immediately
available with one call). No hook blocks it.

**Event stream:**

```text
on_chain_start  name="tools"
on_tool_start   name="read_file"  input={"path": "src/saathi/cli.py"}
```

The spinner in the CLI updates: `"→ read_file"`. `render_tool_call("read_file", {...})` fires.

`read_file("src/saathi/cli.py")` runs:

- Opens the file
- Reads ~12,000 characters of Python source
- Returns the content as a string

**Event stream:**

```text
on_tool_end     name="read_file"  output="\"\"\"Main CLI entry point using Typer.\"\"\"\n\nimport asyncio\n..."
on_chain_end    name="tools"
```

`render_tool_result("read_file", <content>)` fires. The spinner resumes.

The tool node returns:

```python
{
    "messages": [
        ToolMessage(
            content="\"\"\"Main CLI entry point using Typer.\"\"\"\n\nimport asyncio\n...",
            tool_call_id="call_f1a2b3c4",
            name="read_file",
        )
    ]
}
```

LangGraph applies `add_messages`: the `ToolMessage` is appended.

**State after super-step 2:**

```python
{
    "messages": [
        ...previous...,
        HumanMessage("Read the file src/saathi/cli.py and tell me what it does"),
        AIMessage(content="", tool_calls=[read_file(path="src/saathi/cli.py")]),
        ToolMessage(content="<file contents>", tool_call_id="call_f1a2b3c4", name="read_file"),
    ],
    ...
}
```

Checkpoint written.

### 5.15.7 Super-Step 3: Agent Node Runs Again

Back to the agent node. It builds the message list:

```python
messages = [SystemMessage(content=system_prompt)] + state["messages"]
# state["messages"] now includes:
#   HumanMessage("Read the file...")
#   AIMessage(tool_calls=[read_file(...)])
#   ToolMessage(content="<file contents>")
```

The model sees the full file contents and generates a response. This time,
no tool calls — just text:

```python
AIMessage(
    content="The `cli.py` file is the main entry point for Saathi...\n\n"
            "## Key Components\n\n"
            "### ThinkingSpinner\n...",
    tool_calls=[],  # empty
    id="msg_response_2",
)
```

This model call may produce many tokens (it is describing a 600-line file).
Each token arrives as an `on_chat_model_stream` event:

**Event stream:**

```text
on_chat_model_start
on_chat_model_stream  chunk=AIMessageChunk(content="The ")
on_chat_model_stream  chunk=AIMessageChunk(content="`cli")
on_chat_model_stream  chunk=AIMessageChunk(content=".py")
...  (many more token chunks)
on_chat_model_end     output=AIMessage(content="The `cli.py` file...")
on_chain_end          name="agent"
```

In `_run_turn()`, the first `on_chat_model_stream` event with non-empty content
triggers:

```python
if not final_answer:
    spinner.stop()              # spinner stops
    console.print(Rule(...))    # separator line printed
final_answer += chunk.content
console.print(chunk.content, end="", highlight=False)  # token printed
```

The user sees the response appear character by character.

The agent node returns:

```python
{"messages": [AIMessage(content="The `cli.py` file is the main entry point...")]}
```

**State after super-step 3:**

```python
{
    "messages": [
        ...previous...,
        HumanMessage("Read the file src/saathi/cli.py and tell me what it does"),
        AIMessage(content="", tool_calls=[read_file(path="src/saathi/cli.py")]),
        ToolMessage(content="<file contents>", ...),
        AIMessage(content="The `cli.py` file is the main entry point for Saathi..."),
    ],
    ...
}
```

Checkpoint written.

### 5.15.8 tools_condition Routing — Final Step

`tools_condition(state)` is called again. The last message is the `AIMessage`
with `tool_calls = []`. It is empty. `tools_condition` returns `END`.

### 5.15.9 Graph Terminates

LangGraph emits:

```text
on_chain_end  name="LangGraph"  output={"messages": [...all messages...]}
```

In `_run_turn()`:

```python
elif kind == "on_chain_end" and name == "LangGraph":
    output = event["data"].get("output", {})
    if "messages" in output:
        updated_messages = output["messages"]
```

`updated_messages` is captured. `_run_turn()` returns `(final_answer, updated_messages)`.

### 5.15.10 Post-Turn Processing

Back in `execute_task()`:

```python
final_answer, messages = await _run_turn(graph, config, task, state, messages)

# Capture turn snapshots for /diff
for path, original in get_turn_snapshots().items():
    session_start_snapshots.setdefault(path, original)

# Post-turn hooks
if not hook_runner.config.is_empty:
    for result in await hook_runner.run("post_turn"):
        ...
```

The turn is complete. The user sees the full response, the spinner is gone,
and the token/time stats are printed:

```text
↳ 8,432 in · 342 out · 4.7s
```

### 5.15.11 A Two-Tool Turn

Let us extend the example. User says:
`"Read cli.py and list what files are in the agent/ directory"`

The model decides to call both tools at once (it can emit multiple tool calls
in one AIMessage):

```python
AIMessage(
    content="",
    tool_calls=[
        {"id": "call_a1", "name": "read_file",      "args": {"path": "src/saathi/cli.py"}},
        {"id": "call_b2", "name": "list_directory",  "args": {"path": "src/saathi/agent"}},
    ],
)
```

The tool node receives both calls and runs them concurrently:

```python
# asyncio.gather fires both concurrently
messages = await asyncio.gather(
    _guarded({"id": "call_a1", "name": "read_file", ...}),
    _guarded({"id": "call_b2", "name": "list_directory", ...}),
)
```

Both tools execute in parallel (bounded by the semaphore). The `on_tool_start`
and `on_tool_end` events may interleave. Two `ToolMessage`s are returned:

```python
[
    ToolMessage(content="<cli.py contents>", tool_call_id="call_a1", name="read_file"),
    ToolMessage(content="[file] graph.py  (2,891 bytes)\n...", tool_call_id="call_b2", name="list_directory"),
]
```

Both are appended to state. The model then sees both results and generates its
response in a single turn. This is why parallel tool calls matter for performance:
a sequential approach would take 2x as long.

---

## 5.16 History Compaction

As a session grows, the message list grows. More messages means longer prompts,
which means more tokens, which means higher latency and higher risk of hitting
the model's context window limit.

saathi addresses this with the `/compact` command (and auto-compaction):

```python
# From saathi/cli.py — do_compact()
async def do_compact(*, auto: bool) -> None:
    nonlocal messages, config
    before = estimate_tokens(messages)
    try:
        compacted = await compact_messages(summarizer, messages)
    except Exception as exc:
        ...
        return
    if compacted is messages:  # not enough history to compact
        return
    messages = compacted
    state.session_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": state.session_id}}
    after = estimate_tokens(messages)
    console.print(f"[dim]↯ {label} history: ~{before:,} → ~{after:,} tokens[/dim]")
```

After compaction, a new `session_id` (and thus `thread_id`) is generated.
The compacted messages become the starting state for the new thread. The old
thread's history is preserved in the database (you can still inspect it if you
kept the old config) but the session moves forward on the new thread.

The reason a new thread is needed: `add_messages` only appends. You cannot
shrink an existing thread's state by returning fewer messages from a node —
`add_messages` would append to the existing list rather than replacing it.
Starting a new thread with the compacted list as the initial state is the
correct approach.

The trade-off: `/rollback` cannot cross a compaction boundary. If you compact
and then want to roll back to a point before the compaction, you would need to
switch back to the old `thread_id` (which saathi does not currently implement;
it would require saving the old config before compacting).

---

## 5.17 LangGraph vs LangChain Chains for Production

This section synthesises the architectural decision to use LangGraph.

### 5.17.1 What Chains Cannot Do

As established in section 5.1, chains cannot loop. Beyond that:

1. **Chains have no state persistence** — you are responsible for managing
   conversation history.

2. **Chains have no rollback** — if a chain step fails or produces bad output,
   you start over.

3. **Chains have no parallelism model** — parallel execution in LCEL is
   possible with `RunnableParallel`, but it is a static structure (always N
   branches, always recombined). Dynamic parallelism (the number of parallel
   tools depends on the model's output) requires a graph.

4. **Chains cannot fork and merge** — a graph can route to different paths
   based on state and merge them back.

### 5.17.2 What Graphs Add

1. **Cycles** — the core feature. Enables ReAct loops, retry loops, and any
   iterative algorithm.

2. **Persistent state** — the checkpointer stores state after every super-step.

3. **Introspection** — you can inspect and modify state at any point.

4. **Dynamic structure at invocation time** — while the graph's structure is
   fixed at compile time, the path through the graph (which nodes run, how many
   cycles) is determined at runtime by the state and the routing functions.

5. **Human-in-the-loop** — LangGraph has built-in support for interrupting the
   graph and waiting for human input before resuming. saathi does not use this
   (it is always interactive), but it is a key feature for automation workflows
   that need approval steps.

### 5.17.3 When to Use Chains

Chains (LCEL) are still appropriate for:

- **Single-turn tasks** — summarisation, translation, classification, extraction
  that do not need multiple steps.
- **Fixed pipelines** — OCR → extract fields → validate → store. The structure
  is known upfront.
- **Tests and scripts** — when you need a quick LLM call without setting up a
  full graph.

saathi uses a chain for code review:

```python
# From saathi/review.py (conceptual)
review_llm = make_llm(model_id)
# A single LLM call (no graph, no tools, no loop)
findings = await review_llm.ainvoke(review_prompt)
```

The code review task is a single-turn operation: send the diff and instructions,
get structured findings. No tools, no loops. A plain `ainvoke` is the right call.

---

## 5.18 Advanced: Modifying State from Outside the Graph

Sometimes you need to inject a state update without running through the graph
(e.g. to implement `/rollback`). LangGraph supports this via
`graph.aupdate_state()`:

```python
# Roll back to an earlier checkpoint
earlier_snapshot = history[n]

# Re-activate that checkpoint by updating to its state
await graph.aupdate_state(
    config=earlier_snapshot.config,
    values=earlier_snapshot.values,
    as_node="agent",  # which node to attribute this update to
)

# Now the next ainvoke() will resume from the earlier state
```

saathi's `/rollback` command uses this:

```python
# From saathi/ui/commands.py — handle_rollback()
async def handle_rollback(args: list[str], graph, config: dict) -> None:
    history = [c async for c in graph.aget_state_history(config)]
    # ... let user select a step n ...
    target = history[n]
    await graph.aupdate_state(target.config, target.values)
    console.print(f"[green]Rolled back to step {n}[/green]")
```

After this call, the checkpointer's pointer for this `thread_id` moves back to
the earlier checkpoint. The next graph invocation will see the earlier state.

Note the asymmetry: rolling forward (adding new messages) is what `ainvoke`
normally does. Rolling back requires `aupdate_state` to explicitly override the
current checkpoint pointer.

---

## 5.19 Summary

This chapter built a complete picture of LangGraph from the ground up:

1. **Graphs for agents** because agents need loops, and chains are linear.
   The cycle `agent → tools → agent` is the ReAct pattern.

2. **StateGraph** takes a TypedDict schema, nodes, and edges. Compilation
   validates the graph, wires reducers, and attaches the checkpointer.

3. **Nodes** are async functions that take state and return a partial update
   dict. The update is merged into state using reducers.

4. **Edges** come in three kinds: entry (`START`), unconditional, and
   conditional. `tools_condition` is the standard conditional router for
   ReAct graphs.

5. **The critical bug**: adding an unconditional `agent → END` edge alongside
   `tools_condition` creates a fan-out that terminates the graph before tools
   can run. Never add an unconditional `END` edge from a node that also has a
   conditional edge handling the `END` case.

6. **Reducers** control how state is merged. `add_messages` appends new
   messages and updates by ID, enabling the message history to grow correctly
   across the ReAct loop.

7. **AgentState** holds messages, context_paths, mode, and session_id. The
   system prompt is synthesised fresh each turn from these fields, not stored
   in messages.

8. **Super-steps** are LangGraph's execution unit. Each super-step runs
   all nodes with ready inputs, and writes a checkpoint. In saathi's sequential
   graph, each super-step runs one node.

9. **The thread model** scopes state to a `thread_id`. Different threads are
   isolated. The `session_id` serves as the thread ID.

10. **astream_events** with `version="v2"` provides token-level granularity.
    It is the right choice for interactive CLIs that need to display streaming
    tokens, tool call progress, and usage stats.

11. **The full walkthrough**: a single "read file and explain" request produces
    three super-steps — agent (emits tool call), tools (runs read_file),
    agent (generates final answer) — with two checkpoints written along the way.

12. **Compaction** starts a new thread rather than shrinking the existing one,
    because `add_messages` only appends. This is an intentional trade-off
    between simplicity and rollback granularity.

LangGraph is not magic. It is a graph execution engine with persistence and
streaming. Understanding its mechanics lets you debug failures, design better
state schemas, and build agents that are reliable in production.

---

Next: Chapter 6 — Tools in Depth: Filesystem, Shell, Search, and Git
