# Chapter 6 — Agent State and Reducers: The Heart of LangGraph

> "State is not stored — it is accumulated."
> — LangGraph design principle

---

## Table of Contents

1. [Why Immutable State?](#1-why-immutable-state)
2. [TypedDict as Schema](#2-typeddict-as-schema)
3. [The `Annotated` Pattern](#3-the-annotated-pattern)
4. [`add_messages` In Depth](#4-add_messages-in-depth)
5. [Custom Reducers](#5-custom-reducers)
6. [How Nodes Update State](#6-how-nodes-update-state)
7. [State Channels](#7-state-channels)
8. [The Messages Channel](#8-the-messages-channel)
9. [Passing Extra State](#9-passing-extra-state)
10. [State Initialization](#10-state-initialization)
11. [Partial State Updates](#11-partial-state-updates)
12. [State Serialization](#12-state-serialization)
13. [Debugging State](#13-debugging-state)
14. [State Forking](#14-state-forking)
15. [The `aupdate_state` Pattern](#15-the-aupdate_state-pattern)

---

## 1. Why Immutable State?

### The Traditional Chatbot Problem

Every chatbot tutorial begins the same way:

```python
# The naive approach — a mutable list of messages
conversation_history = []

def chat(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    response = llm.complete(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    return response
```

This works for a toy demo. It fails catastrophically in production for five reasons:

1. **No rollback.** If the model makes a mistake, you cannot go back to a known-good state without restarting the entire conversation. The only option is to manually splice out messages — a fragile, error-prone operation.

2. **No replay.** You cannot re-run a turn with different parameters (say, a different model or temperature) because there is no record of what inputs produced what outputs.

3. **No branching.** "What if I had phrased that differently?" is unanswerable. The list is a single timeline.

4. **No inspection.** You cannot pause mid-conversation and ask "what does the agent currently believe about this codebase?" because state is implicit — scattered across the list in no particular structure.

5. **No persistence.** Close the process and the list vanishes. Every session starts from zero.

LangGraph solves all five problems by treating state not as a mutable object to be edited, but as an immutable snapshot to be transformed.

### The Functional Programming Principle

The core idea is borrowed from functional programming. Instead of:

```text
state.mutate(input)
```

LangGraph does:

```text
new_state = reducer(old_state, node_output)
```

Every node in the graph takes the current state and returns a *partial update* — a dict containing only the fields it wants to change. LangGraph then merges that partial update into the existing state using *reducers* — pure functions that know how to combine old and new values.

The result is a new state snapshot that gets written to the checkpointer (a SQLite database in saathi's case) before any subsequent node runs. The old snapshot is never destroyed. It sits in the database, addressable by checkpoint ID, waiting to be recalled, replayed, or branched from.

This is the fundamental insight:

```text
state_0 --[agent_node]--> delta_1 --[reducer]--> state_1
state_1 --[tool_node]---> delta_2 --[reducer]--> state_2
state_2 --[agent_node]--> delta_3 --[reducer]--> state_3
```

Each arrow is deterministic. State_3 can always be reproduced by replaying deltas 1, 2, and 3 from state_0. This is the same principle that makes git commits reproducible and event-sourced databases auditable.

### What "Immutable" Actually Means in Practice

Python dictionaries are not immutable. LangGraph does not enforce immutability at the language level. When we say state is immutable, we mean:

- Nodes must not mutate the state dict they receive as input.
- Nodes return a new dict (the delta), which LangGraph merges.
- The merging is done by the framework, not by the node.

In practice, this discipline is easy to follow because nodes are async functions with a clear signature:

```python
async def my_node(state: AgentState) -> dict:
    # Read from state
    messages = state["messages"]
    mode = state["mode"]
    
    # Compute something
    result = await do_work(messages, mode)
    
    # Return a DELTA — only the fields you changed
    return {"messages": [result]}
    # Do NOT return the full state — only the changes
```

The framework merges your delta into the full state behind the scenes. You never need to copy state fields you didn't touch.

### Why This Enables Saathi's `/rollback` Command

When the agent makes a mistake — perhaps it wrote the wrong content to a file, or went down a dead-end reasoning path — the user types `/rollback 2`. The saathi CLI does this:

```python
# From src/saathi/ui/commands.py
async def handle_rollback(args: list[str], graph, config: dict) -> bool:
    n = int(args[0]) if args and args[0].isdigit() else 1
    history = [s async for s in graph.aget_state_history(config)]
    # history[0] is current; skip n checkpoints back
    if len(history) <= n:
        console.print("[red]Not enough history to roll back that far.[/red]")
        return False

    target = history[n]
    await graph.aupdate_state(config, target.values, as_node="agent")
    console.print(f"[green]Rolled back {n} turn(s).[/green]")
    return True
```

This is only possible because every state is checkpointed. Without immutable snapshots, there is nothing to roll back to.

---

## 2. TypedDict as Schema

### The Choice of TypedDict

When you build a LangGraph, the very first thing you define is the state schema. LangGraph requires that this schema be a `TypedDict`. Not a `dataclass`. Not a Pydantic `BaseModel`. Not a plain dict. A `TypedDict`.

This specific choice is deliberate and important. Let's examine each alternative to understand why TypedDict wins.

#### Why Not `dataclass`?

```python
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage

@dataclass
class AgentState:
    messages: list[BaseMessage] = field(default_factory=list)
    mode: str = "default"
```

Dataclasses are clean Python. But they have two problems for LangGraph:

1. **Runtime representation.** A dataclass instance is an object, not a dict. LangGraph needs to serialize state to SQLite (for checkpointing) and deserialize it. Dict-based state is trivially JSON-serializable. Object-based state requires a custom serializer for every field type.

2. **Structural typing.** LangGraph merges node outputs into state using dict-style key access. A dataclass update requires `setattr` or `replace`, which are more complex and less composable than dict `{**old, **new}`.

#### Why Not Pydantic `BaseModel`?

```python
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    messages: list[BaseMessage] = []
    mode: str = "default"
```

Pydantic is excellent for validation. But:

1. **Validation overhead.** Every time a node returns a delta, Pydantic would validate the full model. In a long agent loop with dozens of tool calls, this adds up.

2. **Immutability semantics.** Pydantic models (especially v2) have different mutation semantics than plain dicts. LangGraph's merge logic is designed for dicts.

3. **The `Annotated` pattern.** Pydantic's `Annotated` support conflicts with how LangGraph uses `Annotated` to attach reducer metadata. The two systems use the same Python feature for different purposes, which causes hard-to-debug conflicts.

#### Why TypedDict Works Perfectly

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context_paths: list[str]
    mode: str
    session_id: str
```

TypedDict has the ideal properties:

1. **It IS a dict at runtime.** `isinstance(state, dict)` is `True`. No wrapper, no overhead. JSON serialization works without custom code.

2. **It's typed at static analysis time.** Mypy, Pyright, and Pylance see the field types and give you autocomplete and type errors. But this is zero-cost at runtime — TypedDicts leave no trace in the running program.

3. **`Annotated` metadata is preserved.** TypedDict stores field annotations as raw `Annotated[...]` objects in `__annotations__`. LangGraph reads these annotations at graph compile time to extract reducer functions. Pydantic would process and consume those annotations before LangGraph can see them.

4. **No default values.** TypedDicts don't support defaults (TypedDict fields are required by default). This forces you to think explicitly about what the initial state should be, which prevents subtle bugs from implicit mutable defaults.

### Saathi's `AgentState` — The Complete Definition

The full source is in `src/saathi/agent/state.py`:

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

Four fields. That's the entire state of a saathi conversation. Let's understand each one:

| Field | Type | Reducer | Purpose |
| ------- | ------ | --------- | --------- |
| `messages` | `list[BaseMessage]` | `add_messages` | The conversation transcript |
| `context_paths` | `list[str]` | none (last write wins) | File paths to focus on |
| `mode` | `str` | none (last write wins) | Agent behaviour mode |
| `session_id` | `str` | none (last write wins) | Unique identifier for this thread |

The `messages` field has a reducer. The other three do not. Fields without reducers use "last write wins" — if a node returns `{"mode": "debug"}`, the old value of `mode` is replaced entirely.

### The Import Chain

Understanding where these types come from:

```python
# typing.TypedDict — the base class
# typing.Annotated — the generic wrapper for adding metadata to types
from typing import Annotated, TypedDict

# BaseMessage — the abstract base for all LangChain message types
# Subtypes: HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage

# add_messages — the built-in reducer for the messages channel
# Defined in langgraph itself, not langchain_core
from langgraph.graph.message import add_messages
```

The separation between `langchain_core` (messages, tools, LLM interfaces) and `langgraph` (graph, state, reducers, checkpointing) is intentional. You can use langchain_core types with any orchestration framework. LangGraph is the opinionated orchestration layer on top.

---

## 3. The `Annotated` Pattern

### Python's `Annotated` Type

`Annotated` is a standard Python typing construct introduced in Python 3.9 (backported to 3.7+ via `typing_extensions`). Its signature:

```python
Annotated[T, metadata_1, metadata_2, ...]
```

Where:

- `T` is the actual type (the part type checkers care about)
- `metadata_1`, `metadata_2`, etc. are arbitrary Python objects that type checkers **ignore** but runtime code can **inspect**

This is the key insight: `Annotated` creates a type that looks like `T` to type checkers, but carries extra information that runtime code can read via `typing.get_type_hints(include_extras=True)`.

### How LangGraph Uses `Annotated`

When you compile a `StateGraph`, LangGraph inspects the TypedDict's annotations using:

```python
import typing

hints = typing.get_type_hints(AgentState, include_extras=True)
# Returns: {
#   'messages': Annotated[list[BaseMessage], add_messages],
#   'context_paths': list[str],
#   'mode': str,
#   'session_id': str,
# }
```

For each field, LangGraph checks: does this annotation have metadata? If yes, the metadata is the reducer function for that channel. If no, the channel uses "last write wins".

Concretely, for `messages`:

```python
import typing
annotation = hints['messages']
# Annotated[list[BaseMessage], add_messages]

args = typing.get_args(annotation)
# (list[BaseMessage], <function add_messages at 0x...>)

actual_type = args[0]    # list[BaseMessage]
reducer_fn  = args[1]    # add_messages
```

LangGraph then calls `reducer_fn(old_messages, new_messages)` every time a node returns new messages.

### Other `Annotated` Reducer Examples

#### Summing Integers

```python
import operator
from typing import Annotated, TypedDict

class CounterState(TypedDict):
    total_tokens: Annotated[int, operator.add]
    turn_count: Annotated[int, operator.add]
    last_model: str  # no reducer — last write wins
```

Every time a node returns `{"total_tokens": 450, "turn_count": 1}`, LangGraph adds those to the existing values. After 5 turns with 450 tokens each, `state["total_tokens"]` is 2250 and `state["turn_count"]` is 5.

#### Merging Sets

```python
from typing import Annotated, TypedDict

class FileTrackingState(TypedDict):
    files_read: Annotated[set[str], lambda a, b: a | b]
    files_written: Annotated[set[str], lambda a, b: a | b]
```

Each node can return newly read/written files, and the sets grow without duplicates across the entire session.

#### Keeping the Maximum

```python
from typing import Annotated, TypedDict

class HighWaterState(TypedDict):
    peak_memory_mb: Annotated[int, max]
    peak_token_count: Annotated[int, max]
```

`max` is a valid reducer — it takes `(old, new)` and returns the larger value. This tracks the highest value ever seen across all turns.

#### Custom Class Instance Reducer

```python
from typing import Annotated, TypedDict

def merge_tool_calls(old: list[dict], new: list[dict]) -> list[dict]:
    """Deduplicate by tool call ID."""
    seen = {call["id"] for call in old}
    return old + [call for call in new if call["id"] not in seen]

class AuditState(TypedDict):
    all_tool_calls: Annotated[list[dict], merge_tool_calls]
```

The reducer function can be any callable with signature `(current_value, new_value) -> merged_value`. There are no other constraints.

### Type Checker Compatibility

From the type checker's perspective, `Annotated[list[BaseMessage], add_messages]` is identical to `list[BaseMessage]`. The metadata is invisible to mypy and Pyright. This means:

```python
async def my_node(state: AgentState) -> dict:
    # Type checker sees state["messages"] as list[BaseMessage] ✓
    msgs: list[BaseMessage] = state["messages"]
    
    # Type checker sees the return type correctly ✓
    return {"messages": [new_message]}
```

You get full type safety for free, with no annotation gymnastics.

---

## 4. `add_messages` In Depth

### What `add_messages` Does

`add_messages` is the built-in LangGraph reducer for the messages channel. Its source is roughly:

```python
# Simplified pseudocode — see langgraph.graph.message for the real implementation
def add_messages(
    left: list[BaseMessage],
    right: list[BaseMessage]
) -> list[BaseMessage]:
    """Merge two message lists.
    
    Rules:
    1. If a message in `right` has no ID, append it.
    2. If a message in `right` has an ID that exists in `left`,
       replace the old message at that position.
    3. If a message in `right` has an ID that doesn't exist in `left`,
       append it.
    """
    # Build an index of existing messages by ID
    existing_by_id = {
        m.id: (i, m) 
        for i, m in enumerate(left) 
        if m.id is not None
    }
    
    result = list(left)
    for new_msg in right:
        if new_msg.id is not None and new_msg.id in existing_by_id:
            # Replace in place
            idx, _ = existing_by_id[new_msg.id]
            result[idx] = new_msg
        else:
            # Append
            result.append(new_msg)
    
    return result
```

### Rule 1: New Messages Are Appended

The most common case. The agent returns an `AIMessage`, the tool node returns `ToolMessage`s — all of these have fresh IDs and get appended:

```python
# Initial state
state["messages"] = [
    HumanMessage(content="Refactor the login function", id="h1"),
]

# Agent node returns
delta = {"messages": [AIMessage(content="...", id="a1", tool_calls=[...])]}

# After add_messages:
state["messages"] = [
    HumanMessage(content="Refactor the login function", id="h1"),
    AIMessage(content="...", id="a1", tool_calls=[...]),
]
```

### Rule 2: Messages with Matching IDs Are Replaced

This is the subtle but critical rule. Why would you ever replace a message? The primary use case is **streaming**.

When a model streams its response, each chunk comes as a partial AIMessage with the same ID. Without the replace rule, you'd get hundreds of fragmented messages. With the replace rule, each chunk updates the same slot:

```python
# Chunk 1
delta = {"messages": [AIMessage(content="I will", id="stream_1")]}
# After reducer: [..., AIMessage(content="I will", id="stream_1")]

# Chunk 2  
delta = {"messages": [AIMessage(content="I will read", id="stream_1")]}
# After reducer: [..., AIMessage(content="I will read", id="stream_1")]
# (replaced in place — not appended!)

# Chunk 3
delta = {"messages": [AIMessage(content="I will read the file.", id="stream_1")]}
# After reducer: [..., AIMessage(content="I will read the file.", id="stream_1")]
```

The message list grows by one slot, not by hundreds of chunks.

### Rule 3: The Tool Call / Tool Result Protocol

The most important consequence of message IDs is how tool calls link to tool results. Here is the exact sequence that happens in every saathi agent loop:

Step 1: Agent node runs, decides to call tools

```python
# AIMessage with tool_calls
ai_msg = AIMessage(
    content="",          # no text content when calling tools
    id="ai_turn_1",
    tool_calls=[
        {
            "name": "read_file",
            "args": {"path": "src/auth.py"},
            "id": "call_abc123",    # <-- the tool call ID
            "type": "tool_call",
        },
        {
            "name": "read_file",
            "args": {"path": "src/models.py"},
            "id": "call_def456",    # <-- different ID for second call
            "type": "tool_call",
        },
    ]
)

# Delta returned by agent node
delta = {"messages": [ai_msg]}
```

After `add_messages`, the state messages list looks like:

```text
[..., HumanMessage("Refactor login"), AIMessage(tool_calls=[call_abc123, call_def456])]
```

Step 2: Tool node runs, executes both tools

```python
# ToolMessages — each references its originating tool_call_id
tool_result_1 = ToolMessage(
    content="def login(user, pwd):\n    ...",
    tool_call_id="call_abc123",   # <-- matches the call ID
    name="read_file",
    id="tm_1",
)
tool_result_2 = ToolMessage(
    content="class User(Base):\n    ...",
    tool_call_id="call_def456",   # <-- matches the OTHER call ID
    name="read_file",
    id="tm_2",
)

# Delta returned by tool node
delta = {"messages": [tool_result_1, tool_result_2]}
```

After `add_messages`, the state messages list:

```text
[
    ...,
    HumanMessage("Refactor login"),
    AIMessage(tool_calls=[call_abc123, call_def456]),
    ToolMessage(tool_call_id="call_abc123"),    # <-- linked by ID
    ToolMessage(tool_call_id="call_def456"),    # <-- linked by ID
]
```

Step 3: Agent node runs again

The LLM receives this full message history. It sees:

1. The human's request
2. Its own decision to call two files
3. The content of both files (linked by matching IDs)

The LLM can now reason about both files together and produce a final answer.

Why the IDs Are Not Optional

If the tool node returned ToolMessages without matching `tool_call_id`s, the LLM would see:

- An AIMessage that says "I'm calling tool X and tool Y"
- ToolMessages that don't reference any particular call

Most LLMs would be confused or error. The ID linkage is part of the OpenAI-compatible tool-use protocol that LangChain and Ollama both implement. It is not optional.

### Concrete Example with Message IDs in Saathi

Here is a full trace of one saathi turn, showing message IDs:

```python
# User types: "What's in requirements.txt?"

# 1. HumanMessage created in _run_turn (cli.py)
human = HumanMessage(
    content="What's in requirements.txt?",
    id="hm_7f3a9b",      # auto-generated UUID fragment
)

# 2. Agent node calls the LLM
# LLM decides to call read_file
ai_call = AIMessage(
    content="",
    id="ai_c2e1f8",
    tool_calls=[{
        "name": "read_file",
        "args": {"path": "requirements.txt"},
        "id": "tc_5d8a2c",     # tool call ID
        "type": "tool_call",
    }]
)

# 3. Tool node executes read_file
# Returns ToolMessage with matching tool_call_id
tool_result = ToolMessage(
    content="langgraph>=0.2\nlangchain-ollama>=0.1\n...",
    tool_call_id="tc_5d8a2c",  # MUST match
    name="read_file",
    id="tm_9a3b7f",
)

# 4. Agent node runs again with full context
# LLM sees: human → ai_call → tool_result
# Produces final answer:
final = AIMessage(
    content="requirements.txt contains: langgraph, langchain-ollama...",
    id="ai_ff9d3e",
)

# Final messages state:
# [human, ai_call, tool_result, final]
```

This is the complete ReAct (Reason + Act) loop in concrete form.

---

## 5. Custom Reducers

### When the Built-in Reducers Aren't Enough

`add_messages` is the right reducer for most conversation state. But there are scenarios where you want different accumulation behavior:

- Keep only the last N messages (bounded memory)
- Track which files were read this session (set accumulation)
- Record every error without duplicates (deduplicated list)
- Sum numeric counters across turns (integer addition)

### Writing a Custom Reducer

A reducer is any function with signature `(current: T, new: T) -> T`. It is called by LangGraph every time a node returns a value for that channel.

#### Example: Last-N Messages Reducer

```python
from langchain_core.messages import BaseMessage

def last_n_messages(n: int):
    """Returns a reducer that keeps only the most recent n messages."""
    def reducer(current: list[BaseMessage], new: list[BaseMessage]) -> list[BaseMessage]:
        combined = current + new
        return combined[-n:]
    return reducer

# Usage:
from typing import Annotated, TypedDict

class BoundedState(TypedDict):
    # Keep only the last 20 messages — older ones are dropped
    messages: Annotated[list[BaseMessage], last_n_messages(20)]
```

This is simple but has a catch: if you drop old messages, the conversation loses context. More subtly, if you drop an AIMessage that contained tool_calls but keep the ToolMessages that answered those calls, you create an invalid message sequence — the LLM sees tool results with no corresponding call. This is why saathi uses **compaction** instead of windowing.

#### Example: Set Union Reducer

```python
from typing import Annotated, TypedDict

class FileTrackingState(TypedDict):
    files_touched: Annotated[frozenset[str], lambda a, b: a | b]
```

#### Example: Reducer That Logs Changes

```python
import logging
from langchain_core.messages import BaseMessage

log = logging.getLogger(__name__)

def logged_add_messages(
    old: list[BaseMessage],
    new: list[BaseMessage]
) -> list[BaseMessage]:
    """add_messages with debug logging."""
    from langgraph.graph.message import add_messages
    result = add_messages(old, new)
    log.debug(
        "messages_merged",
        old_count=len(old),
        new_count=len(new),
        result_count=len(result),
    )
    return result
```

### Why Saathi Uses Compaction Instead of a Last-N Reducer

The temptation is to use a last-N reducer to automatically manage context window size. Saathi deliberately does not do this, for two reasons:

**Reason 1: Message sequence integrity.** As mentioned above, slicing the message list at an arbitrary point creates orphaned ToolMessages or split tool call chains. The agent then receives a malformed conversation history and its behavior becomes undefined.

**Reason 2: Compaction preserves meaning.** Saathi's `compact_messages` function (in `src/saathi/compaction.py`) summarizes older turns into a single SystemMessage before dropping them:

```python
async def compact_messages(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
) -> list[BaseMessage]:
    """Summarize all but the last keep_turns turns into one summary message."""
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" 
        for m in older
    )
    response = await llm.ainvoke([
        SystemMessage(content=_SUMMARY_INSTRUCTIONS),
        HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
    ])
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{_text(response)}")
    return [summary, *recent]
```

The result is `[SystemMessage(summary), *last_3_turns]` — a valid message sequence where the older context is preserved as prose, not lost entirely.

Compaction happens at a user-turn boundary (`split_for_compaction` finds the right cut point), which ensures `recent` always starts with a `HumanMessage` — never with an orphaned `ToolMessage`.

---

## 6. How Nodes Update State

### The Node Contract

A LangGraph node is a function (sync or async) with this signature:

```python
async def my_node(state: AgentState) -> dict:
    ...
```

The input is the full current state. The output is a **delta** — a dict containing only the fields being updated. LangGraph merges the delta into the state using the reducer for each field.

**Critical rule:** Only return the fields you are changing. Do not return fields you are not changing. LangGraph applies reducers only to the fields present in your delta. Fields you omit are untouched.

### The Agent Node

```python
# From src/saathi/agent/nodes.py
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
        ...
    )
    return {"messages": [response]}    # ONLY messages — nothing else
```

The agent node:

- Reads `state["messages"]`, `state["context_paths"]`, `state["mode"]`
- Returns only `{"messages": [response]}`
- Does NOT return `context_paths`, `mode`, or `session_id`

LangGraph sees the delta `{"messages": [response]}`, runs `add_messages(state["messages"], [response])`, and stores the result as the new value of the `messages` channel. The other three channels are untouched.

### The Tool Node

```python
# From src/saathi/agent/tool_node.py
async def hooked_tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_calls = list(getattr(last, "tool_calls", []) or [])
    if not tool_calls:
        return {"messages": []}

    messages = await asyncio.gather(*(_guarded(call) for call in tool_calls))
    return {"messages": list(messages)}    # ONLY messages
```

The tool node also returns only `{"messages": [...]}`. It reads `state["messages"][-1]` to find the tool calls, but it only updates the messages channel.

This is a pattern: most nodes only update one or two channels, even if they read from many.

### A Node That Updates Multiple Channels

```python
# Hypothetical node that changes mode AND adds a message
async def mode_switching_node(state: AgentState) -> dict:
    current_mode = state["mode"]
    
    if _should_switch_to_debug(state["messages"]):
        new_mode = "debug"
        notification = SystemMessage(
            content="Switching to debug mode based on error patterns detected."
        )
        return {
            "mode": new_mode,             # updates mode channel (last write wins)
            "messages": [notification],    # updates messages channel (add_messages)
        }
    
    return {}  # nothing to change — return empty dict is valid
```

When a node returns `{}`, LangGraph applies no updates. The state is unchanged. This is different from returning `{"messages": []}`, which runs the `add_messages` reducer with an empty list (a no-op, but still a reducer call).

### State Read vs. State Write

Nodes can read any field but should only write fields they intend to change:

```python
async def analysis_node(state: AgentState) -> dict:
    # READ: can access any field
    messages = state["messages"]
    mode = state["mode"]
    paths = state["context_paths"]
    session = state["session_id"]
    
    # WRITE: return only what changed
    analysis = _analyze(messages, mode, paths)
    return {
        "messages": [AIMessage(content=analysis)],
        # mode, context_paths, session_id: NOT included, therefore untouched
    }
```

---

## 7. State Channels

### The LangGraph Terminology

LangGraph calls each field in the state TypedDict a **channel**. The term comes from CSP (Communicating Sequential Processes) and actor model theory, where channels are the conduits through which actors pass messages. In LangGraph, state channels are the conduits through which nodes communicate.

Each channel has:

1. **A type** — what kind of value it holds (`list[BaseMessage]`, `str`, `list[str]`, etc.)
2. **An optional reducer** — how to merge incoming updates (from the `Annotated` metadata)
3. **A current value** — maintained by the checkpointer

When LangGraph compiles a `StateGraph`, it reads the TypedDict definition and creates a channel object for each field:

```python
# Pseudocode showing what LangGraph does internally
channels = {}
for field_name, annotation in AgentState.__annotations__.items():
    if is_annotated(annotation):
        actual_type, reducer_fn = parse_annotated(annotation)
        channels[field_name] = Channel(type=actual_type, reducer=reducer_fn)
    else:
        channels[field_name] = Channel(type=annotation, reducer=last_write_wins)
```

### Last Write Wins

For channels without an explicit reducer, LangGraph uses "last write wins": the new value simply replaces the old one.

```python
# mode channel has no reducer
state["mode"] = "default"

# Node returns:
delta = {"mode": "debug"}

# After merge (last write wins):
state["mode"] = "debug"    # replaced, not accumulated
```

This is the right behavior for scalar fields like `mode` and `session_id`. You don't want mode values to accumulate — you want the current mode to be whatever was set last.

### Channel Write Ordering

What happens when two nodes return updates to the same channel in the same "superstep"? LangGraph's edges determine the order, but within a parallel fan-out, the order of reducer application is not guaranteed.

For `add_messages`, this doesn't matter because the reducer is commutative for appends. For custom reducers, you must ensure commutativity if you use parallel nodes.

Saathi's graph has no parallel nodes — it's strictly sequential (agent → tools → agent → ...), so channel ordering is never an issue.

### The Full Channel Catalog for Saathi

```text
messages:       Annotated[list[BaseMessage], add_messages]
                Reducer: add_messages (append / replace by ID)
                Written by: agent_node, tool_node
                Read by: agent_node, tool_node, CLI

context_paths:  list[str]
                Reducer: last write wins
                Written by: handle_context (via CLI state update)
                Read by: agent_node (passed to build_system_prompt)

mode:           str
                Reducer: last write wins
                Written by: handle_mode (via CLI state update)
                Read by: agent_node (passed to build_system_prompt)

session_id:     str
                Reducer: last write wins
                Written by: CLI at startup, updated on compaction
                Read by: CLI (for display), agent_node indirectly
```

---

## 8. The Messages Channel

### The Central Artifact

The `messages` channel is the heart of every LangGraph agent. It is the complete, ordered transcript of the conversation — every input, every model response, every tool call, every tool result, in chronological order.

By the time the graph reaches `END`, `state["messages"]` contains everything that happened. This makes it trivially easy to export the conversation (saathi's `/export` command), summarize it (compaction), or replay it (rollback).

### Message Types in the Transcript

LangGraph uses LangChain's message hierarchy:

```text
BaseMessage (abstract)
├── HumanMessage      — user input
├── AIMessage         — model response (may include tool_calls)
├── SystemMessage     — system instructions (not part of conversation turns)
└── ToolMessage       — tool execution results
```

In a typical saathi session:

```text
[
    HumanMessage("What does login() do?"),           # turn 1
    AIMessage(tool_calls=[read_file(auth.py)]),       # agent: read the file
    ToolMessage(content="def login(...)...", ...),    # tool: file contents
    AIMessage(content="The login function..."),       # agent: final answer
    HumanMessage("Can you add rate limiting?"),       # turn 2
    AIMessage(tool_calls=[read_file(auth.py),         # agent: re-read + check tests
                          read_file(test_auth.py)]),
    ToolMessage(content="def login(...)...", ...),    # tool: auth.py content
    ToolMessage(content="def test_login(...)...", ..), # tool: test content
    AIMessage(tool_calls=[write_file(auth.py, ...)]), # agent: edit the file
    ToolMessage(content="Written 1234 chars...", ...), # tool: write confirmation
    AIMessage(content="Done. I added a rate limiter..."),  # agent: final answer
]
```

This is the canonical message sequence for a two-turn interaction with file reading and writing.

### The SystemMessage Is Not in `messages`

Note that the system prompt is NOT stored in `state["messages"]`. It's prepended by the agent node at call time:

```python
# From agent_node in nodes.py
system_prompt = build_system_prompt(...)
messages = [SystemMessage(content=system_prompt)] + state["messages"]
response = await llm.ainvoke(messages, config)
```

The system prompt is *ephemeral* — constructed fresh every turn from the current state fields (`mode`, `context_paths`, `memory_block`). Storing it in the messages channel would mean an old system prompt from turn 1 gets sent to the LLM in turn 10, which could be stale or contradictory.

By building the system prompt outside the messages channel, saathi ensures the LLM always receives the current, up-to-date instructions.

### Token Counting and the Messages Channel

The messages channel grows without bound in a long session. This is intentional — you want the full history. But it has a practical limit: the LLM's context window.

Saathi estimates tokens from the messages channel:

```python
# From src/saathi/compaction.py
_CHARS_PER_TOKEN = 4

def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars per token) across message contents."""
    return sum(len(_text(m)) for m in messages) // _CHARS_PER_TOKEN

def needs_compaction(messages: list[BaseMessage], budget_tokens: int) -> bool:
    return estimate_tokens(messages) > budget_tokens
```

When `estimate_tokens(messages) > settings.history_token_budget`, compaction triggers automatically before the next turn. The older messages are summarized and replaced with a compact SystemMessage, keeping the messages channel within the context window budget.

---

## 9. Passing Extra State

### Beyond Messages: Why Three More Fields?

The `messages` channel carries the conversation. But a sophisticated coding agent needs more context than just the dialogue. Saathi uses three additional state fields to pass non-message configuration through the graph:

- `context_paths`: Which files/directories should the agent focus on?
- `mode`: What behavioral mode is active?
- `session_id`: What's the unique identifier for this thread?

These fields flow through the graph without being accumulated. They're set once (or occasionally updated) and read by the agent node on every turn.

### How the Agent Node Reads Extra State

```python
# From src/saathi/agent/nodes.py
async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    memory_block = memory_store.format_for_prompt()
    system_prompt = build_system_prompt(
        context_paths=state.get("context_paths", []),   # <-- extra state
        memory_block=memory_block,
        mode=state.get("mode", "default"),              # <-- extra state
        project_instructions=project_instructions,
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await llm.ainvoke(messages, config)
    return {"messages": [response]}
```

And how `build_system_prompt` uses those fields:

```python
# From src/saathi/agent/prompts.py
def build_system_prompt(
    context_paths: list[str],
    memory_block: str,
    mode: str,
    project_instructions: str = "",
) -> str:
    parts = [BASE_PROMPT]

    if project_instructions:
        parts.append(
            "## Project instructions (from SAATHI.md)\n"
            f"{project_instructions}"
        )

    if mode and mode in _MODE_ADDENDA:
        parts.append(_MODE_ADDENDA[mode])    # e.g., debug mode instructions

    if memory_block:
        parts.append(f"Remembered facts:\n{memory_block}")

    if context_paths:
        paths = "\n".join(f"  - {p}" for p in context_paths)
        parts.append(
            f"Context scope — prefer reading and editing these paths first:\n{paths}"
        )

    return "\n\n".join(parts)
```

The `mode` field switches in a complete block of additional instructions. In `debug` mode:

```python
_MODE_ADDENDA = {
    "debug": """\
MODE: debug
- Reproduce the bug first before attempting a fix
- Read the full stack trace before reading any code
- Apply the smallest possible fix; verify it before reporting done
""",
    "refactor": """\
MODE: refactor
- Use patch_file instead of write_file for targeted changes
- Explain the reason for every modification
- Run tests after changes when a test command is available
- Prefer minimal, focused edits over full rewrites
""",
    "explain": """\
MODE: explain
- Read files, never modify them
- Cite exact file path + line number for every claim
- Use plain language; add tables and code blocks where helpful
- When in doubt, say so
""",
}
```

The `context_paths` field scopes the agent's attention. If you set context to `["src/auth/"]`, the agent's system prompt includes an explicit instruction to prefer reading and editing files under that path.

### When Extra State Fields Change

Unlike `messages` (which accumulates throughout a session), the extra fields change infrequently:

| Field | When It Changes |
| ------- | ---------------- |
| `mode` | User types `/mode debug` |
| `context_paths` | User types `/context src/auth/` |
| `session_id` | At startup, and after compaction |

The CLI's slash command handlers update `SessionState` (a separate Python object). These changes are then reflected in the next turn's input state:

```python
# From cli.py _run_turn
input_state = {
    "messages": messages,
    "context_paths": state.context_paths,   # reads from SessionState
    "mode": state.mode,                     # reads from SessionState
    "session_id": state.session_id,         # reads from SessionState
}
```

The separation between `SessionState` (Python object, in memory) and `AgentState` (dict, checkpointed) is intentional. `SessionState` is the CLI's view of what's happening. `AgentState` is what gets passed to the graph.

---

## 10. State Initialization

### The Initial State Dict

Every LangGraph run begins with an initial state dict passed to `ainvoke` or `astream_events`. This dict must provide values for every field in the TypedDict (or those fields must have defaults — TypedDicts support `total=False` for optional fields).

In saathi, the initial state is constructed in `_run_turn`:

```python
# From src/saathi/cli.py
async def _run_turn(
    graph,
    config: dict,
    task: str,
    state: SessionState,
    messages: list,
) -> tuple[str, list]:
    messages.append(HumanMessage(content=task))
    input_state = {
        "messages": messages,
        "context_paths": state.context_paths,
        "mode": state.mode,
        "session_id": state.session_id,
    }
    
    async for event in graph.astream_events(input_state, config, version="v2"):
        ...
```

Note that `messages` already contains the full conversation history from previous turns. The new `HumanMessage` was just appended. So the initial state for each turn already includes everything that happened before — the graph's checkpoint system is almost an implementation detail here, because the CLI maintains its own `messages` list separately.

Wait — this raises a question. If the CLI maintains messages separately, what does the LangGraph checkpoint do?

### The Dual Message Store

Saathi has two sources of truth for messages:

1. **The CLI's `messages` list** — a Python list maintained in `_interactive_session`. Updated by `_run_turn` at the end of each turn.

2. **The LangGraph checkpoint** — the `state["messages"]` stored in SQLite by the checkpointer.

These two should be equivalent. They're kept in sync by this code in `_run_turn`:

```python
elif kind == "on_chain_end" and name == "LangGraph":
    output = event["data"].get("output", {})
    if "messages" in output:
        updated_messages = output["messages"]  # read from graph output
```

And at the end of `_run_turn`:

```python
return final_answer, updated_messages
```

The CLI updates its `messages` list from the graph's output. So `messages` in the CLI reflects exactly what's in the checkpoint.

Why maintain both? Because `/rollback` needs the checkpoint history (many checkpoints). But the CLI needs the *current* messages list to pass to the *next* turn's `input_state`. Having both allows `/rollback` to work via the graph API while normal turns flow through the CLI's message list.

### Initial State for `--print` Mode

In non-interactive mode, the initial state is simpler:

```python
# From src/saathi/cli.py _print_mode
input_state = {
    "messages": [HumanMessage(content=task)],  # fresh start — single message
    "context_paths": context_paths,
    "mode": "default",
    "session_id": session_id,
}

result = await graph.ainvoke(input_state, config)
```

Here, the graph gets one message and runs to completion. There's no interactive loop, no message history from previous turns, and no `astream_events` — just `ainvoke` returning the final state.

### TypedDict Total vs. Partial

TypedDict supports `total=False` for fields that are optional:

```python
from typing import TypedDict

class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]  # optional
    mode: str                    # optional
```

Saathi doesn't use this — all fields are required (`total=True` is the default). This is a deliberate design choice: every state update must provide every field, preventing subtle bugs from missing initialization.

---

## 11. Partial State Updates

### The Core Mechanic

One of the most elegant aspects of LangGraph's reducer system is that nodes only need to return the fields they changed. LangGraph merges the delta with the existing state field by field:

```python
# Pseudocode for LangGraph's merge logic
def merge_state(current_state: dict, delta: dict, channels: dict) -> dict:
    new_state = dict(current_state)  # copy
    for field, new_value in delta.items():
        channel = channels[field]
        if channel.reducer:
            new_state[field] = channel.reducer(current_state[field], new_value)
        else:
            new_state[field] = new_value  # last write wins
    return new_state
```

Fields not in `delta` are untouched. `new_state[field] = current_state[field]` — no reducer call, no copy, just the existing value.

### What the Tool Node Actually Returns

The tool node in saathi returns only:

```python
return {"messages": list(messages)}
```

It touches only the `messages` channel. The `context_paths`, `mode`, and `session_id` channels are completely unaffected. This is correct — the tool node is purely about executing tools and adding their results to the conversation. It has no business changing the mode or context.

### What the Agent Node Actually Returns

Similarly:

```python
return {"messages": [response]}
```

Just the messages. The agent node reads `mode` and `context_paths` to build the system prompt, but it doesn't change them. It only produces a new AI response message.

### A More Complex Hypothetical

If saathi were extended with an autonomous mode-switching feature, the agent node might return:

```python
async def smart_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    ...
    
    # Detect if the conversation suggests a mode switch
    if _looks_like_debugging_session(state["messages"]):
        return {
            "messages": [response],
            "mode": "debug",           # switch mode automatically
        }
    
    return {"messages": [response]}   # normal case
```

This is valid LangGraph — you can return any subset of state fields from any node. The framework handles the merge.

### Empty Deltas

A node can return `{}` (an empty dict) to signal "no changes". This is valid and efficient — LangGraph skips the merge loop entirely.

```python
async def conditional_node(state: AgentState) -> dict:
    if not _needs_action(state):
        return {}  # no-op
    
    result = await _do_action(state)
    return {"messages": [result]}
```

---

## 12. State Serialization

### The Checkpointer and SQLite

LangGraph's `AsyncSqliteSaver` serializes state to a SQLite database at `.saathi/checkpoints.db`. Every time a node completes and returns a delta, LangGraph:

1. Applies the reducers to compute the new state
2. Serializes the new state to JSON
3. Writes it to the SQLite database with a checkpoint ID

This happens asynchronously and is transparent to the nodes.

The database schema (simplified) looks like:

```sql
CREATE TABLE checkpoints (
    thread_id     TEXT,
    checkpoint_id TEXT,
    parent_id     TEXT,    -- previous checkpoint ID for linked list
    type          TEXT,    -- serialization format
    checkpoint    BLOB,    -- serialized state
    metadata      BLOB,    -- extra metadata
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

Each `(thread_id, checkpoint_id)` pair is one snapshot of the full state.

### What Must Be Serializable

The rule is simple: everything in the state must be JSON-serializable (or serializable via LangChain's custom serializers).

For saathi's `AgentState`:

| Field | Serialization |
| ------- | -------------- |
| `messages` | LangChain custom serializer (handles all BaseMessage subtypes) |
| `context_paths` | Plain JSON array of strings |
| `mode` | Plain JSON string |
| `session_id` | Plain JSON string |

LangChain's message serializer handles the `BaseMessage` type hierarchy by adding a `type` discriminator field:

```json
{
  "type": "human",
  "content": "What does login() do?",
  "id": "hm_7f3a9b",
  "additional_kwargs": {}
}
```

```json
{
  "type": "ai",
  "content": "",
  "id": "ai_c2e1f8",
  "tool_calls": [
    {
      "name": "read_file",
      "args": {"path": "auth.py"},
      "id": "tc_5d8a2c",
      "type": "tool_call"
    }
  ]
}
```

```json
{
  "type": "tool",
  "content": "def login(...):\n    ...",
  "tool_call_id": "tc_5d8a2c",
  "name": "read_file"
}
```

On deserialization, the `type` field is used to reconstruct the correct Python class.

### What Cannot Be Serialized

If you accidentally put a non-serializable object in state, LangGraph will fail at checkpoint time with a serialization error. Common pitfalls:

- File handles or socket objects
- Lambda functions (can't be pickled/JSON-serialized)
- Asyncio futures or tasks
- References to live objects (database connections, etc.)

Saathi's state contains only strings, string lists, and messages — all trivially serializable.

### Building the Graph with the Checkpointer

```python
# From src/saathi/agent/graph.py
async def build_graph(
    tools: list[BaseTool],
    memory_store: MemoryStore,
    model_id: str | None = None,
    db_path: Path | None = None,
    hook_runner: HookRunner | None = None,
):
    ...
    db_path = db_path or Path(".saathi") / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(db_path))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()      # create the tables if they don't exist
    return builder.compile(checkpointer=checkpointer)
```

The `checkpointer.setup()` call creates the SQLite schema on first run. Subsequent runs find the tables already there and resume where they left off.

The `conn` is a long-lived connection. It stays open for the entire session and is closed in `close_graph`:

```python
async def close_graph(graph) -> None:
    checkpointer = getattr(graph, "checkpointer", None)
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        with contextlib.suppress(Exception):
            await conn.close()
```

---

## 13. Debugging State

### Inspecting the Current State

At any point during a session, you can inspect the current state via:

```python
# Get current state snapshot
state = await graph.aget_state(config)
print(state.values)         # the state dict
print(state.next)           # which nodes will run next
print(state.config)         # the config that produced this state
```

In saathi, the `/checkpoints` command shows all snapshots for the current thread:

```python
# From src/saathi/ui/commands.py
async def handle_checkpoints(graph, config: dict) -> None:
    history = [s async for s in graph.aget_state_history(config)]
    rows = []
    for cp in history:
        rows.append({
            "checkpoint_id": cp.config.get("configurable", {}).get("checkpoint_id", "?"),
            "message_count": len(cp.values.get("messages", [])),
        })
    render_checkpoint_table(rows)
```

Each row shows one checkpoint: its ID and how many messages were in state at that point. This gives you a timeline of the conversation's growth.

### `aget_state_history`

The full state history is a linked list of snapshots:

```python
history = [s async for s in graph.aget_state_history(config)]

# history[0]  — most recent state (current)
# history[1]  — state before last node ran
# history[2]  — state before second-to-last node ran
# ...
# history[-1] — initial state (just after first message was added)
```

Each snapshot has:

- `values`: the full state dict at that checkpoint
- `config`: the checkpoint config (includes `thread_id` and `checkpoint_id`)
- `next`: the nodes scheduled to run from this checkpoint

### Inspecting a Specific Checkpoint

```python
history = [s async for s in graph.aget_state_history(config)]

# Examine the third checkpoint back
checkpoint = history[3]
messages = checkpoint.values["messages"]

for msg in messages:
    role = msg.__class__.__name__
    content = msg.content[:100] if isinstance(msg.content, str) else str(msg.content)[:100]
    print(f"{role}: {content}...")
```

This is invaluable for debugging. You can see exactly what the agent knew at any point in time.

### Debugging Reducer Behavior

If you suspect a reducer is behaving unexpectedly, you can test it in isolation:

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Simulate two turns
turn_1_messages = [
    HumanMessage(content="Hello", id="h1"),
    AIMessage(content="Hi there", id="a1"),
]
turn_2_new_messages = [
    HumanMessage(content="How are you?", id="h2"),
]

result = add_messages(turn_1_messages, turn_2_new_messages)
# [HumanMessage("Hello"), AIMessage("Hi there"), HumanMessage("How are you?")]

# Test the replace-by-ID behavior
update = [AIMessage(content="Hi there! (updated)", id="a1")]  # same ID as above
result2 = add_messages(result, update)
# [HumanMessage("Hello"), AIMessage("Hi there! (updated)"), HumanMessage("How are you?")]
# ↑ a1 was REPLACED in place, not appended
```

---

## 14. State Forking

### The Concept

State forking means creating multiple independent timelines from the same point. In git terms: branching. In LangGraph terms: using the same `thread_id` but starting from a different checkpoint.

Every checkpoint is identified by two things:

- `thread_id`: the conversation thread
- `checkpoint_id`: the specific snapshot within that thread

To fork at checkpoint X, you create a new run that starts from checkpoint X. You don't need to create a new thread_id — you just provide the checkpoint_id in the config.

### When Forking Is Useful

1. **A/B testing prompts.** Roll back to before a particular question and try two different phrasings. Which produces better output?

2. **Exploring alternatives.** The agent is about to do something potentially destructive. Fork first, let it run, inspect the result, then either keep or discard.

3. **Teaching.** "What would have happened if the model had better context?" Fork, add context, replay.

### How `/rollback` Implements Forking

```python
# From src/saathi/ui/commands.py
async def handle_rollback(args: list[str], graph, config: dict) -> bool:
    n = int(args[0]) if args and args[0].isdigit() else 1
    history = [s async for s in graph.aget_state_history(config)]
    
    if len(history) <= n:
        console.print("[red]Not enough history to roll back that far.[/red]")
        return False

    target = history[n]
    await graph.aupdate_state(config, target.values, as_node="agent")
    console.print(f"[green]Rolled back {n} turn(s).[/green]")
    return True
```

`aupdate_state` is the key. It:

1. Takes the target checkpoint's values (the full state at that point)
2. Writes them as a new checkpoint in the SAME thread
3. The new checkpoint becomes the current state

After the rollback, the next user input will be processed from the rolled-back state. The messages list in the CLI also needs to be synchronized — saathi handles this by reloading messages from the graph state after rollback.

### The Limitations of Rollback Across Compaction Boundaries

One important limitation: `/rollback` cannot cross a compaction boundary.

When compaction runs, it creates a new `thread_id`:

```python
# From cli.py do_compact
state.session_id = uuid.uuid4().hex
config = {"configurable": {"thread_id": state.session_id}}
```

The new thread starts fresh with the compacted messages. The old thread's checkpoints are still in the database but are associated with the old `thread_id`. Since `/rollback` uses the current `config` (with the new `thread_id`), it can't reach checkpoints from before the compaction.

This is a documented limitation in saathi's architecture, noted in the `do_compact` docstring:

```python
async def do_compact(*, auto: bool) -> None:
    """Summarize old turns and continue on a fresh checkpoint thread.

    A new thread_id is used so the compacted list becomes the new baseline —
    the add_messages reducer only appends, so we can't shrink the old thread's
    state in place. Trade-off: /rollback can't cross a compaction boundary.
    """
```

---

## 15. The `aupdate_state` Pattern

### What `aupdate_state` Does

`graph.aupdate_state(config, values, as_node=None)` is the escape hatch for injecting external state into a running (or paused) graph. It:

1. Takes `values` — a partial state dict (like a node delta)
2. Applies the reducers for each field in `values`
3. Writes the resulting state as a new checkpoint

The `as_node` parameter is subtle and important. It tells LangGraph which node "virtually" produced this update. This determines which edges to follow next. If `as_node="agent"`, LangGraph treats the injected state as if the agent node just returned it, and routes accordingly (to `tools` if there are tool calls, or to `END` if not).

### `/rollback` in Detail

```python
target = history[n]
await graph.aupdate_state(config, target.values, as_node="agent")
```

`target.values` is the full state at the target checkpoint — including the messages list as it existed at that point (shorter than current). The call:

1. Runs `add_messages(current_messages, target.messages)` — but since target.messages is a SUBSET of current messages (the state was shorter then), this will result in a state that has current_messages partially overwritten

Wait — this is not quite right. Let me be more precise about what happens.

`aupdate_state` with a full state dict (all four fields) will apply the reducer for `messages`. The `add_messages` reducer with `target.values["messages"]` will:

- Find messages in `target.values["messages"]` whose IDs already exist → replace them
- Find messages with IDs that don't exist → append them

Since `target.values["messages"]` is an OLDER list (all IDs exist in the current state), every message gets replaced in place. No new messages are appended. The result is that the messages list reverts to exactly what it was at the target checkpoint.

This works because `add_messages` uses ID-based replacement — the exact property needed for rollback to work correctly.

### Injecting an External Message

`aupdate_state` can also inject a single message from outside the graph — for example, from a human reviewer:

```python
from langchain_core.messages import HumanMessage

# Inject a message as if the human had said it
await graph.aupdate_state(
    config,
    {"messages": [HumanMessage(content="Actually, don't use regex for this.")]},
    as_node="agent",  # route as if agent just processed this
)
```

After this call, the graph state has the new message appended (via `add_messages`). The next turn will include this message in context. This is useful for:

- Human-in-the-loop workflows where a reviewer can inject guidance
- Testing: inject specific messages to simulate user interactions
- Error recovery: inject a correction after detecting a model mistake

### `as_node` — Why It Matters

The `as_node` parameter controls routing after the update.

If `as_node="agent"`: LangGraph applies the conditional edge that routes agent → tools (if tool_calls present) or agent → END (if final answer). The injected state is treated as if the agent just produced it.

If `as_node="tools"`: LangGraph applies the tools → agent edge. The next node to run will be the agent.

If `as_node=None`: LangGraph applies no routing — the graph stays at its current position, awaiting the next `invoke` call.

In saathi's rollback:

```python
await graph.aupdate_state(config, target.values, as_node="agent")
```

`as_node="agent"` means the next user message will be processed by the agent node (which is the correct behavior — we're rolling back to a point where the agent is ready to receive input).

### The `aupdate_state` Signature

```python
await graph.aupdate_state(
    config: dict,           # {"configurable": {"thread_id": "..."}}
    values: dict,           # the state delta to apply
    as_node: str | None,    # which node "produced" this update
)
```

Returns the updated `StateSnapshot`, which includes the new checkpoint ID. You can use this to confirm the rollback succeeded:

```python
new_snapshot = await graph.aupdate_state(config, target.values, as_node="agent")
new_message_count = len(new_snapshot.values.get("messages", []))
print(f"Rolled back to {new_message_count} messages")
```

### A Complete Rollback Workflow

Here's the full sequence of what happens when a user types `/rollback 2`:

```text
User: /rollback 2
  ↓
handle_rollback(["2"], graph, config)
  ↓
history = [s async for s in graph.aget_state_history(config)]
# history = [checkpoint_7, checkpoint_6, checkpoint_5, checkpoint_4, ...]
#   [0] = current (7 messages)
#   [1] = 1 turn ago (5 messages)
#   [2] = 2 turns ago (3 messages)  <- target
  ↓
target = history[2]
# target.values = {"messages": [h1, a1, h2], "mode": "default", ...}
  ↓
await graph.aupdate_state(config, target.values, as_node="agent")
# New checkpoint written:
#   messages: [h1, a1, h2]  (4 messages dropped)
#   mode: "default"
#   context_paths: []
#   session_id: "..."
  ↓
console.print("Rolled back 2 turn(s).")
  ↓
# On next user input, the agent sees only [h1, a1, h2]
# The last two turns never happened (from the agent's perspective)
```

The file changes made during those two turns are NOT automatically reverted. The graph state has been rolled back, but the filesystem has not. This is why saathi also has `/diff` — to show what files changed this session, allowing the user to manually revert if needed.

---

## Summary

This chapter covered LangGraph's state management system from first principles to advanced patterns. The key takeaways:

1. **Immutable snapshots** enable rollback, replay, and branching. Every node run creates a new checkpoint.

2. **TypedDict** is the ideal state schema: dict at runtime, typed at static analysis time, transparent to LangGraph's reducer inspection.

3. **`Annotated`** attaches reducer functions to state fields. LangGraph reads these annotations at compile time to know how to merge node outputs.

4. **`add_messages`** implements append-or-replace-by-ID semantics. The ID linkage between AIMessage tool_calls and ToolMessages is the mechanism that makes the tool-use protocol work correctly.

5. **Custom reducers** let you implement set union, integer accumulation, bounded lists, or any other merge strategy.

6. **Nodes return deltas**, not full states. LangGraph merges them field by field using reducers.

7. **State channels** without reducers use last-write-wins.

8. **The messages channel** grows throughout the session and is serialized to SQLite for persistence.

9. **Extra state fields** (`mode`, `context_paths`, `session_id`) pass non-message configuration through the graph without accumulation.

10. **`aupdate_state`** is the escape hatch for external state injection and rollback.

The system is elegant: a handful of Python primitives (`TypedDict`, `Annotated`, a reducer function) implement a complete event-sourced state machine with persistence, rollback, and branching.

---

Next: Chapter 7 — Tools and the Tool Node: Parallel Execution with Hooks
