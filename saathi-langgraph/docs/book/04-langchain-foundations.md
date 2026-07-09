# Chapter 4 — LangChain Foundations: Messages, Tools, and Runnables

> "LangChain is not a framework for building apps. It is a protocol layer for the
> AI ecosystem. Everything else builds on top of it."

This chapter covers the LangChain primitives that LangGraph, saathi, and almost
every serious LLM application in the Python ecosystem depend on. By the end you
will understand why the message hierarchy is shaped the way it is, why
`bind_tools()` returns a `LanguageModelLike` instead of a `ChatOllama`, and how
the entire streaming pipeline flows from an Ollama model call down to the bytes
that land in your terminal.

---

## 4.1 LangChain's Role

When LangChain launched in 2022 it was marketed as "a framework for building
applications powered by language models." That framing caused a lot of confusion.
Developers tried to use it to build web apps, then complained that it had too
many abstractions. Others called it unnecessary glue. Neither camp was entirely
wrong, but both missed the more important thing LangChain actually did: it
standardised the vocabulary.

Before LangChain, every LLM client library had its own way of representing a
conversation. OpenAI's Python SDK used `{"role": "user", "content": "..."}`.
Anthropic used `Human:` / `Assistant:` turn markers in a raw string. Hugging
Face transformers used tokenised tensors. There was no common type that you
could hand from an OpenAI wrapper to a local model wrapper.

LangChain fixed this by defining a small, stable set of message types in
`langchain_core`. These types became the lingua franca. When LangGraph was
built on top of LangChain, it could store messages in state and route them
between nodes without caring whether the underlying model was GPT-4, Claude,
or a locally-running Llama. When saathi switched from the Anthropic API to
Ollama, the entire agent graph stayed unchanged. Only the model instantiation
line changed.

This is the key insight: **LangChain's value is in the abstractions it
defines, not the implementations it ships**. The `langchain_core` package
deliberately contains almost no concrete implementations. It is a contract.
`langchain_community` and model-specific packages like `langchain_ollama`
provide the concrete implementations, but they can all be swapped out because
they speak the same protocol.

The protocol has three main pillars:

1. **The Message Hierarchy** — a type system for conversation turns
2. **The Runnable Protocol** — a uniform interface for any computation unit
3. **The Tool Interface** — a standard way to describe callable functions to a model

We will cover each in detail.

---

## 4.2 The Message Hierarchy

A language model conversation is a sequence of messages. Different roles produce
different message types. LangChain defines a class hierarchy rooted at
`BaseMessage`:

```flow
BaseMessage
├── HumanMessage       role = "human"
├── AIMessage          role = "assistant"
├── SystemMessage      role = "system"
├── ToolMessage        role = "tool"
└── FunctionMessage    role = "function"  (legacy, prefer ToolMessage)
```

All of these live in `langchain_core.messages`. Let us look at each.

### 4.2.1 BaseMessage

```python
from langchain_core.messages import BaseMessage

# BaseMessage fields (simplified from the actual Pydantic model):
class BaseMessage(BaseModel):
    content: Union[str, List[Union[str, Dict]]]
    type: str                  # set by subclasses
    id: Optional[str]          # unique message ID; used by add_messages reducer
    name: Optional[str]        # optional display name
    additional_kwargs: Dict    # model-specific overflow
    response_metadata: Dict    # token counts, stop reason, etc.
```

The `content` field deserves special attention. For simple text it is a plain
string. For models that support multimodal content (images, files, tool results
with structured data) it can be a list of content blocks:

```python
# Simple text content
msg = HumanMessage(content="What is the capital of France?")
assert isinstance(msg.content, str)

# Multimodal content (list of blocks)
msg = HumanMessage(content=[
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
])
assert isinstance(msg.content, list)
```

saathi does not use multimodal content — all tools return strings. But you
should understand the distinction because Ollama's `vision` models and
OpenAI's `gpt-4o` both use the list form, and `AIMessage.content` from
those models may also be a list when the model returns structured content
blocks instead of plain text.

The `id` field is crucial for the `add_messages` reducer in LangGraph, which
we will cover in Chapter 5. In short: if two messages have the same `id`, the
newer one replaces the older one rather than appending. This is how streaming
works — partial tokens arrive as a stream of `AIMessage` chunks with the same
`id`, and the final assembled message updates by `id`.

### 4.2.2 HumanMessage

The message type for user input. Used in saathi every time the user presses
Enter in the REPL:

```python
# From saathi/cli.py — _run_turn()
messages.append(HumanMessage(content=task))
```

`HumanMessage` has no extra fields beyond `BaseMessage`. Its only role is to
set `type = "human"` so the model knows this is a user turn.

### 4.2.3 SystemMessage

Injected at the front of every LLM call. It is **not** stored in the graph's
`state["messages"]` — it is synthesised fresh each time the agent node runs:

```python
# From saathi/agent/nodes.py
messages = [SystemMessage(content=system_prompt)] + state["messages"]
response = await llm.ainvoke(messages, config)
```

This design decision matters. By keeping the system prompt out of state,
saathi can update it on every turn — incorporating new `mode`, `context_paths`,
or `memory_block` values without any migration of stored state. The graph state
only stores the conversation history; the system prompt is ephemeral.

### 4.2.4 AIMessage

The message type returned by the model. It has several fields that
`BaseMessage` does not:

```python
from langchain_core.messages import AIMessage

class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"
    tool_calls: List[ToolCall]              # structured tool calls (new format)
    invalid_tool_calls: List[InvalidToolCall]  # malformed calls, for error handling
    usage_metadata: Optional[UsageMetadata]    # token counts
    # inherited: content, id, additional_kwargs, response_metadata
```

The `tool_calls` field is the most important one for understanding the ReAct
loop. When the model decides to call a tool, it sets `content = ""` (or
sometimes a brief reasoning string) and populates `tool_calls`. When it decides
to give a final answer it sets `content` to the response text and leaves
`tool_calls` empty.

The `usage_metadata` field contains token usage for this response. saathi
reads it to display the `↳ 1,234 in · 567 out · 2.3s` footer after each turn:

```python
# From saathi/cli.py — _extract_usage()
def _extract_usage(output) -> tuple[int, int] | None:
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
```

The dual lookup exists because Ollama's API reports token counts under
`response_metadata` (using Ollama's own field names `prompt_eval_count` and
`eval_count`) while cloud APIs typically follow the `usage_metadata` standard.
This is an example of the abstraction leaking: the core protocol is clean, but
the concrete implementations differ in the details.

### 4.2.5 ToolMessage

After a tool executes, the tool node returns a `ToolMessage` to communicate the
result back to the model:

```python
from langchain_core.messages import ToolMessage

class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    tool_call_id: str    # MUST match the id from AIMessage.tool_calls
    name: Optional[str]  # tool name, for readability
    # inherited: content (the tool's return value as a string)
```

The `tool_call_id` field is the contract that holds the ReAct loop together.
When the model emits tool calls it assigns each one a unique ID (something like
`"call_abc123"`). The tool node must return exactly one `ToolMessage` for each
`tool_call_id`. If the model sees a `ToolMessage` without a matching
`tool_call_id` in the preceding `AIMessage`, most model APIs will reject the
request with a validation error.

This is why saathi's `make_hooked_tool_node` always produces a `ToolMessage`
even for blocked or failed calls:

```python
# From saathi/agent/tool_node.py
if reason is not None:
    log.warning("tool_blocked", tool=name, reason=reason)
    return ToolMessage(
        content=f"BLOCKED: {reason}. The tool was not executed.",
        tool_call_id=call_id,
        name=name,
    )
```

And for unknown tools:

```python
if tool is None:
    return ToolMessage(
        content=f"Error: unknown tool '{name}'.",
        tool_call_id=call_id,
        name=name,
    )
```

The model sees the error message as the tool result and can choose what to do
next — typically apologise to the user or try a different approach.

### 4.2.6 FunctionMessage (Legacy)

`FunctionMessage` is an older format that predates the current tool call
protocol. OpenAI originally used `function_calling`, then switched to `tool_use`
with a different schema. `FunctionMessage` maps to the old format. You will
encounter it in older code and documentation but should use `ToolMessage` for
all new work.

saathi does not use `FunctionMessage` anywhere.

---

## 4.3 AIMessage and Tool Calls in Depth

The `tool_calls` field on `AIMessage` contains structured data, not raw JSON
strings. LangChain normalises the model's raw output into a typed structure:

```python
# A ToolCall is a TypedDict:
class ToolCall(TypedDict):
    id: str        # unique per call, used to match ToolMessage.tool_call_id
    name: str      # the tool function name
    args: dict     # parsed arguments — already a Python dict, not a JSON string
    type: str      # always "tool_call"
```

When a model wants to call two tools in one turn it populates `tool_calls` with
two entries:

```python
# An AIMessage with two tool calls might look like this:
AIMessage(
    content="",
    tool_calls=[
        {
            "id": "call_f1a2b3",
            "name": "read_file",
            "args": {"path": "src/saathi/cli.py"},
            "type": "tool_call",
        },
        {
            "id": "call_d4e5f6",
            "name": "list_directory",
            "args": {"path": "src/saathi"},
            "type": "tool_call",
        },
    ],
    id="msg_xyz789",
)
```

The tool node iterates `tool_calls`, dispatches each call, and returns a
`ToolMessage` for each. The conversation history after this turn looks like:

```text
HumanMessage:  "What's in the cli.py file and what other files are in that dir?"
AIMessage:     content="", tool_calls=[read_file(...), list_directory(...)]
ToolMessage:   tool_call_id="call_f1a2b3", content="<file contents...>"
ToolMessage:   tool_call_id="call_d4e5f6", content="[file] cli.py  (12,400 bytes)..."
AIMessage:     content="The cli.py file contains the main entry point..."
```

Note that both `ToolMessage`s appear before the next `AIMessage`. The model
receives the entire sequence and processes the results together. This is why
parallel tool calls work: the model can issue them in one step and receive both
results before deciding what to do next.

---

## 4.4 The Runnable Protocol

Everything in LangChain that can be called is a `Runnable`. The `Runnable`
abstract base class defines a small, consistent interface:

```python
from langchain_core.runnables import Runnable

class Runnable(ABC, Generic[Input, Output]):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]:
        ...

    async def astream(self, input: Input, ...) -> AsyncIterator[Output]:
        ...

    def batch(self, inputs: List[Input], ...) -> List[Output]:
        ...

    async def abatch(self, inputs: List[Input], ...) -> List[Output]:
        ...
```

`invoke` and `ainvoke` run to completion and return the final output.
`stream` and `astream` yield partial outputs as they arrive. `batch` and
`abatch` process multiple inputs in parallel.

Everything is a `Runnable`: `ChatOllama` is a `Runnable`, the result of
`llm.bind_tools(tools)` is a `Runnable`, a `ToolNode` is a `Runnable`, a
LangGraph compiled graph is a `Runnable`.

### 4.4.1 LCEL — The Pipe Operator

LangChain Expression Language (LCEL) is a mini-DSL built on `Runnable`. The
`|` operator chains runnables:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-coder:14b")
parser = StrOutputParser()

# This creates a RunnableSequence: input → llm → parser → output
chain = llm | parser

# Equivalent to:
chain.invoke([HumanMessage(content="Hello")])
# => "Hello! How can I help you?"
```

Under the hood, `|` calls `Runnable.__or__` which wraps both sides in a
`RunnableSequence`. The sequence calls the left side, passes its output as
input to the right side, and returns the final output.

saathi does not use LCEL chains extensively because the agent's control flow
lives in the LangGraph graph structure rather than in a linear chain. But it is
important to understand LCEL because LangChain's documentation uses it
everywhere, and because `bind_tools` returns a `RunnableBinding` that is itself
composable with `|`.

### 4.4.2 RunnableConfig

Every `invoke`, `ainvoke`, `stream`, and `astream` call accepts an optional
`RunnableConfig`. This dict carries metadata that flows through the entire
call stack:

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "configurable": {
        "thread_id": "session-abc123",   # LangGraph thread identity
    },
    "callbacks": [...],                  # LangSmith, custom loggers
    "tags": ["prod", "agent"],           # for filtering in LangSmith
    "metadata": {"user_id": "u42"},      # arbitrary metadata
    "recursion_limit": 25,               # max graph cycles
    "run_name": "saathi-turn",           # name for this run in traces
}
```

In saathi, the config is set up in `cli.py` and passed to every graph
invocation:

```python
# From saathi/cli.py — _interactive_session()
config = {"configurable": {"thread_id": state.session_id}}

# And used in every turn:
async for event in graph.astream_events(input_state, config, version="v2"):
    ...
```

The `thread_id` is how LangGraph's checkpointer knows which conversation
history to load. We cover this in depth in Chapter 5.

---

## 4.5 LanguageModelLike

One of the most confusing type names in LangChain is `LanguageModelLike`. It
appears in type hints and documentation but is not a class you can instantiate.
It is a type alias:

```python
# From langchain_core.language_models.base
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage

# LanguageModelLike is any Runnable that takes a list of messages as input
# and returns a BaseMessage as output
LanguageModelLike = Runnable[List[BaseMessage], BaseMessage]
```

`ChatOllama` satisfies this type. But so does the result of
`ChatOllama(...).bind_tools(tools)`, which returns a `RunnableBinding` — a
`Runnable` that wraps the original `ChatOllama` and injects the tool schemas
into every call. The `RunnableBinding` is **not** a `ChatOllama`.

This is why `make_agent_node` in saathi accepts `LanguageModelLike` rather than
`ChatOllama`:

```python
# From saathi/agent/nodes.py
from langchain_core.language_models import LanguageModelLike

def make_agent_node(llm: LanguageModelLike, memory_store: MemoryStore):
    """
    Accepts a LanguageModelLike because ChatOllama.bind_tools(...) returns
    a Runnable binding rather than a bare ChatOllama.
    """
    async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        ...
        response = await llm.ainvoke(messages, config)
        return {"messages": [response]}

    return agent_node
```

And the call site in `build_graph` correctly binds tools before passing to the
node factory:

```python
# From saathi/agent/graph.py
llm = make_llm(model_id).bind_tools(tools)   # returns LanguageModelLike
agent_node = make_agent_node(llm, memory_store)
```

If `make_agent_node` had typed its parameter as `ChatOllama`, mypy would
report a type error here because `bind_tools()` does not return a `ChatOllama`.
Using `LanguageModelLike` is the correct, honest type annotation.

---

## 4.6 `.bind_tools()`

`bind_tools()` is a method on `BaseChatModel` (which `ChatOllama` extends).
It wraps the model in a `RunnableBinding` that injects tool schemas into every
API call. Let us trace exactly what it does.

### 4.6.1 What bind_tools() Does

When you call `llm.bind_tools(tools)`:

1. It iterates the `tools` list and calls `format_tool_to_ollama_tool(tool)`
   (or the equivalent for other backends) to convert each `BaseTool` into the
   JSON Schema format the model API expects.

2. It creates a `RunnableBinding` that stores these converted schemas.

3. Every subsequent call to `.invoke()` or `.ainvoke()` on the binding
   automatically merges the tool schemas into the API request payload.

For Ollama, the payload becomes:

```json
{
  "model": "qwen2.5-coder:14b",
  "messages": [
    {"role": "system", "content": "You are Saathi..."},
    {"role": "user",   "content": "Read the file src/saathi/cli.py"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file. Use this before editing any file.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string"}
          },
          "required": ["path"]
        }
      }
    },
    ... (14 more tools)
  ]
}
```

### 4.6.2 Tool Schema Injection Example

Here is a concrete trace for saathi's tool list:

```python
from saathi.tools import ALL_TOOLS
from saathi.agent.graph import make_llm

llm = make_llm("qwen2.5-coder:14b")
bound_llm = llm.bind_tools(ALL_TOOLS)

# bound_llm is now a RunnableBinding, not a ChatOllama
print(type(bound_llm))
# => <class 'langchain_core.runnables.base.RunnableBinding'>

# The tool schemas are stored in bound_llm.kwargs["tools"]
schemas = bound_llm.kwargs.get("tools", [])
print(f"Bound {len(schemas)} tools")
# => Bound 15 tools

# Inspect the read_file schema
import json
for schema in schemas:
    if schema["function"]["name"] == "read_file":
        print(json.dumps(schema, indent=2))
```

Output:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read the contents of a file. Use this before editing any file.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        }
      },
      "required": ["path"]
    }
  }
}
```

### 4.6.3 Why tool_choice Matters

`bind_tools()` also accepts a `tool_choice` parameter:

```python
# Force the model to call a specific tool
llm.bind_tools(tools, tool_choice="read_file")

# Force the model to call some tool (any tool)
llm.bind_tools(tools, tool_choice="any")

# Let the model decide (default)
llm.bind_tools(tools, tool_choice="auto")
```

saathi does not pass `tool_choice` and accepts the default `"auto"` behaviour,
which lets the model decide whether to call a tool or answer directly.

---

## 4.7 The @tool Decorator

The simplest way to create a LangChain tool is the `@tool` decorator from
`langchain_core.tools`:

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

The decorator does several things:

1. **Wraps the function in a `StructuredTool`** (a subclass of `BaseTool`).
2. **Sets `name`** from the function name (`read_file`).
3. **Generates `args_schema`** from the function signature using Pydantic.
4. **Sets `description`** from the function's docstring.

### 4.7.1 The Docstring is Critical

The `description` field is the most important piece of tool metadata. It is
what the model reads to decide whether to call the tool. A vague description
produces bad tool selection; a precise description produces good tool selection.

Compare these two docstrings:

```python
# Bad: too vague
@tool
def read_file(path: str) -> str:
    """Read a file."""
    ...

# Good: tells the model when to use it
@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Use this before editing any file."""
    ...
```

The phrase "Use this before editing any file" is not just documentation for
human readers. It directly shapes the model's behaviour. With this description,
a model asked to "edit the cli.py file" will reliably call `read_file` first.
Without it, the model may hallucinate the file contents and call `write_file`
with incorrect content.

Similarly for `patch_file`:

```python
@tool
def patch_file(path: str, diff: str) -> str:
    """Apply a unified diff patch to an existing file (prefer over write_file for edits)."""
    ...
```

The parenthetical `(prefer over write_file for edits)` tells the model which
tool to choose when both `write_file` and `patch_file` are available. This
single phrase dramatically reduces the rate of full-file overwrites, which are
dangerous (losing content not in the diff) and expensive (large token counts).

### 4.7.2 All saathi Tools

Here is the complete tool inventory from `saathi/tools/__init__.py`:

```python
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

ALL_TOOLS = [
    read_file,         # filesystem.py
    write_file,        # filesystem.py
    patch_file,        # filesystem.py
    list_directory,    # filesystem.py
    run_bash,          # shell.py
    search_in_file,    # search.py
    search_across_files, # search.py
    search_web,        # search.py
    save_memory,       # memory_tools.py
    recall_memory,     # memory_tools.py
    git_status,        # git.py
    git_diff,          # git.py
    git_diff_staged,   # git.py
    git_log,           # git.py
    git_commit,        # git.py
]
```

Fifteen tools in total. The model sees all fifteen when deciding how to respond.
Models with strong instruction following (like `qwen2.5-coder:14b`) do well with
this many tools. Weaker models may get confused; the usual fix is to reduce the
tool count by domain-scoping the session with `/context`.

---

## 4.8 BaseTool — Class-Based Tools

The `@tool` decorator is syntactic sugar over `BaseTool`. For more control you
can subclass `BaseTool` directly:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class WriteFileInput(BaseModel):
    path: str = Field(description="Destination path; parent directories are created")
    content: str = Field(description="Full file content to write")

class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Create or overwrite a file; parent directories are created automatically."
    args_schema: type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str) -> str:
        ...

    async def _arun(self, path: str, content: str) -> str:
        return self._run(path, content)
```

When would you use class-based tools?

1. **Dependency injection** — the tool needs access to a service (database,
   HTTP client) that you want to inject at construction time rather than via a
   global. saathi's `save_memory` and `recall_memory` tools use a module-level
   `MemoryStore` instance; a class-based tool could take `MemoryStore` as a
   constructor argument instead.

2. **Complex argument schemas** — when you need Pydantic validators, aliases,
   or nested models that cannot be expressed with simple Python type hints.

3. **Custom error handling** — `BaseTool` has `handle_tool_error` and
   `handle_validation_error` hooks that let you return a friendly error message
   instead of raising an exception.

4. **Stateful tools** — when the tool needs to maintain state between calls
   (e.g. a database cursor, a browser session).

For saathi's tools, the `@tool` decorator is sufficient because the tools are
stateless functions that take simple string arguments.

---

## 4.9 Tool Schema Generation

LangChain generates JSON Schema from Pydantic models. Understanding this
process helps you write tools that work reliably with all models.

### 4.9.1 From Python Type Hints to JSON Schema

When you write:

```python
@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file; parent directories are created automatically."""
    ...
```

LangChain internally generates a Pydantic model equivalent to:

```python
class write_file_schema(BaseModel):
    path: str
    content: str
```

And from that model it generates the JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "path": {
      "title": "Path",
      "type": "string"
    },
    "content": {
      "title": "Content",
      "type": "string"
    }
  },
  "required": ["path", "content"]
}
```

All fields are `required` by default. Optional fields must be explicitly typed
as `Optional[str]` or given a default value.

### 4.9.2 Using Field() for Richer Schemas

You can enrich the schema with `Field()` annotations:

```python
from pydantic import Field
from langchain_core.tools import tool

@tool
def search_across_files(
    directory: str,
    pattern: str,
    file_glob: str = "**/*",
) -> str:
    """Recursively search for a regex pattern across all files in a directory."""
    ...
```

The generated schema for `file_glob` will include `"default": "**/*"` and the
field will not appear in `required`. This is important: the model does not need
to supply `file_glob` unless it wants to change the default.

For the explicit class-based approach with `Field`:

```python
class SearchInput(BaseModel):
    directory: str = Field(description="Root directory to search in")
    pattern: str = Field(description="Python regex pattern")
    file_glob: str = Field(default="**/*", description="Glob to filter files, e.g. '**/*.py'")
```

Generated JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "directory": {
      "description": "Root directory to search in",
      "type": "string"
    },
    "pattern": {
      "description": "Python regex pattern",
      "type": "string"
    },
    "file_glob": {
      "default": "**/*",
      "description": "Glob to filter files, e.g. '**/*.py'",
      "type": "string"
    }
  },
  "required": ["directory", "pattern"]
}
```

The `description` fields in the schema go into the tool payload the model
receives. This means field descriptions are a second vector (after the tool
`description`) where you can guide the model. Use them.

### 4.9.3 Type Mapping Reference

| Python type      | JSON Schema type            |
| ------------------ | ----------------------------- |
| `str`            | `"string"`                  |
| `int`            | `"integer"`                 |
| `float`          | `"number"`                  |
| `bool`           | `"boolean"`                 |
| `list[str]`      | `{"type": "array", "items": {"type": "string"}}` |
| `dict`           | `{"type": "object"}`        |
| `Optional[str]`  | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` |
| `Literal["a","b"]` | `{"enum": ["a", "b"]}`   |
| Pydantic model   | `{"type": "object", "properties": {...}}` |

---

## 4.10 ChatOllama Specifics

`ChatOllama` is the concrete implementation that wraps Ollama's HTTP API. It
lives in `langchain_ollama`, a thin integration package. Understanding its
quirks is essential for debugging saathi.

### 4.10.1 Instantiation

```python
from langchain_ollama import ChatOllama

# From saathi/agent/graph.py — make_llm()
def make_llm(model_id: str) -> ChatOllama:
    return ChatOllama(
        model=model_id,
        base_url=settings.ollama_base_url,    # default: http://localhost:11434
        temperature=settings.temperature,      # default: 0
        num_ctx=settings.context_window,       # default: 8192 tokens
        num_predict=settings.max_tokens,       # default: -1 (no limit)
    )
```

`num_ctx` is the context window size. Ollama loads the model with this context
window; changing it requires unloading and reloading the model. saathi exposes
this via `SAATHI_CONTEXT_WINDOW` in the environment.

`num_predict` controls the maximum response length. The default `-1` means
unbounded. For long code generation tasks this is correct; you do not want
truncated file writes.

### 4.10.2 Tool Call Extraction

Ollama returns tool calls in a JSON format that `ChatOllama` parses into the
standard `AIMessage.tool_calls` structure. Internally, `ChatOllama` does
something like:

```python
# (Simplified; actual implementation is more robust)
def _parse_ollama_response(raw_response: dict) -> AIMessage:
    message = raw_response["message"]
    tool_calls = []
    for tc in message.get("tool_calls", []):
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": tc["function"]["name"],
            "args": tc["function"]["arguments"],  # already a dict in Ollama
            "type": "tool_call",
        })
    return AIMessage(
        content=message.get("content", ""),
        tool_calls=tool_calls,
        ...
    )
```

One subtlety: Ollama models emit tool calls when their response format
includes `"tool_calls"` in the Ollama message. If a model is not fine-tuned
for tool use, it may emit the tool call as a JSON string in the `content`
field instead. `ChatOllama` handles the standard case; the fallback case
requires a custom output parser (which saathi avoids by choosing models that
support native tool calling).

### 4.10.3 Streaming

When you call `llm.astream(messages)` on a `ChatOllama`, it streams the Ollama
SSE (Server-Sent Events) response and yields `AIMessageChunk` objects:

```python
from langchain_core.messages import AIMessageChunk

async for chunk in llm.astream(messages):
    # chunk is an AIMessageChunk, a subclass of AIMessage
    print(chunk.content, end="", flush=True)
    # chunk.tool_call_chunks also exists for streaming tool calls
```

`AIMessageChunk` supports addition: `chunk1 + chunk2` produces a merged chunk
with concatenated content. This is how the final `AIMessage` is assembled from
the stream.

saathi does not use `llm.astream()` directly. It uses `graph.astream_events()`
which yields fine-grained events including token chunks. We cover this next.

---

## 4.11 astream_events

`astream_events` is the most powerful streaming API in LangChain. Instead of
yielding intermediate state (like `astream`) or a final result (like `ainvoke`),
it yields a stream of typed event dicts. Each event has a `"event"` field that
identifies its type.

### 4.11.1 Event Types

The events saathi handles in `_run_turn()`:

```python
# From saathi/cli.py — _run_turn()
async for event in graph.astream_events(input_state, config, version="v2"):
    kind = event["event"]
    name = event.get("name", "")

    if kind == "on_chat_model_stream":
        # A token chunk from the model
        chunk = event["data"].get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            # First token: stop the spinner, print a separator
            if not final_answer:
                spinner.stop()
                console.print(Rule(style="dim cyan"))
            final_answer += chunk.content
            console.print(chunk.content, end="", highlight=False)

    elif kind == "on_chat_model_end":
        # Model call finished; extract usage metadata
        usage = _extract_usage(event["data"].get("output"))
        if usage:
            in_tokens += usage[0]
            out_tokens += usage[1]

    elif kind == "on_tool_start":
        # A tool is about to run
        spinner.update(f"→ {name}")
        render_tool_call(name, event["data"].get("input", {}))

    elif kind == "on_tool_end":
        # A tool just finished
        output = event["data"].get("output", "")
        render_tool_result(name, str(output))
        spinner.update(next(_SPINNER_PHRASES))

    elif kind == "on_chain_end" and name == "LangGraph":
        # The entire graph invocation finished; capture final state
        output = event["data"].get("output", {})
        if "messages" in output:
            updated_messages = output["messages"]
```

### 4.11.2 Event Structure

Each event is a dict with this shape:

```python
{
    "event": str,          # "on_chat_model_stream", "on_tool_start", etc.
    "name": str,           # the name of the node/tool/chain that emitted it
    "run_id": str,         # UUID for this specific run
    "parent_ids": list,    # parent run IDs for the call hierarchy
    "tags": list,          # tags from RunnableConfig
    "metadata": dict,      # metadata from RunnableConfig
    "data": dict,          # event-specific payload
}
```

The `data` dict structure varies by event type:

```python
# on_chat_model_stream
{"chunk": AIMessageChunk}

# on_chat_model_end
{"output": AIMessage, "input": [SystemMessage, HumanMessage, ...]}

# on_tool_start
{"input": {"path": "src/saathi/cli.py"}}   # the tool's arguments

# on_tool_end
{"output": "import asyncio\nimport...\n"}  # the tool's return value

# on_chain_start / on_chain_end
{"input": {...}, "output": {...}}           # chain input/output
```

### 4.11.3 Why astream_events Over astream

`astream` yields the graph's state after each node completes. For a two-node
graph (agent → tools → agent) it would yield:

```text
Step 1: {"messages": [HumanMessage, AIMessage(tool_calls=[...])]}
Step 2: {"messages": [HumanMessage, AIMessage, ToolMessage]}
Step 3: {"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage(final)]}
```

This is useful for knowing the state at each step, but it does not give you
individual tokens. The entire model response appears as one blob in the final
step.

`astream_events` with `version="v2"` provides token-level granularity. saathi
uses it to print tokens as they arrive, making the UI feel responsive even for
slow models or long responses. The spinner stays active during tool execution,
and the first token of the final answer instantly stops the spinner.

---

## 4.12 Callbacks and Tracing

LangChain's callback system is the mechanism by which observability tools like
LangSmith plug in. Every `Runnable.invoke()` call accepts a `callbacks`
parameter:

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM starting with {len(prompts)} prompts")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool starting: {serialized['name']}")

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

llm.invoke(messages, config={"callbacks": [MyCallbackHandler()]})
```

You can also set callbacks globally via environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__your_key
```

With these set, every LangChain call is automatically traced to LangSmith
without any code changes. You can see full traces, replay runs, and compare
prompt versions in the LangSmith UI.

saathi does not configure LangSmith by default, but it will automatically pick
it up if those environment variables are set. This is a good debugging strategy:
if the agent is behaving unexpectedly, set `LANGCHAIN_TRACING_V2=true` for a
session and examine the full trace in LangSmith.

### 4.12.1 How astream_events Relates to Callbacks

`astream_events` is essentially a streaming interface over the callback system.
Every callback event that fires internally is converted to a `dict` and yielded
by the async generator. This is why the event structure mirrors the callback
method names: `on_chat_model_stream` corresponds to
`BaseCallbackHandler.on_llm_new_token`, `on_tool_start` corresponds to
`BaseCallbackHandler.on_tool_start`, and so on.

The practical implication: you can either use `astream_events` (pull model,
events come to you) or callbacks (push model, your handler is called). saathi
uses `astream_events` because it integrates cleanly with `async for` and does
not require subclassing `BaseCallbackHandler`.

---

## 4.13 Output Parsers

LangChain ships a library of output parsers for extracting structured data from
LLM responses:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel

# StrOutputParser: extract content string from AIMessage
parser = StrOutputParser()
chain = llm | parser
result = chain.invoke(messages)
# result is a str, not an AIMessage

# JsonOutputParser: parse JSON from the content string
class ReviewFindings(BaseModel):
    findings: list[dict]
    score: int

parser = JsonOutputParser(pydantic_object=ReviewFindings)
chain = llm | parser
result = chain.invoke(messages)
# result is a ReviewFindings instance
```

saathi does not use these parsers in the agent loop. The primary reason is
that output parsers fail loudly when the model produces malformed output —
which happens regularly with local models. saathi's `review.py` module does
ask the model to produce JSON but uses a tolerant parser:

```python
# From saathi/review.py
import json, re

def _tolerant_json(text: str) -> dict | None:
    """Try to extract a JSON object from text that may have prose around it."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract a JSON block from markdown
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find a {...} block anywhere in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None
```

This approach is more robust for local models, which may wrap JSON in prose
("Here are my findings: `{ ... }`") or use trailing commas and other JSON5-isms
that `json.loads` rejects.

When to use LangChain output parsers:

- When you control the model (cloud API with strong instruction following)
- When you need the structured result immediately and can handle parse errors
- When using `with_structured_output()` (which uses function calling under the
  hood, not text parsing)

---

## 4.14 Memory in LangChain vs LangGraph

This section addresses a common confusion: LangChain has its own memory system
(`ConversationBufferMemory`, `ConversationSummaryMemory`, etc.). Why does saathi
use LangGraph's checkpointer instead?

### 4.14.1 LangChain Memory (The Old Way)

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

chain.predict(input="What is the capital of France?")
# memory.chat_memory now contains HumanMessage + AIMessage

chain.predict(input="And of Germany?")
# memory automatically prepends the previous conversation
```

This works for simple chatbots but has deep problems for agents:

1. **Stateful object** — the `memory` object is mutable. In an async web server
   with multiple concurrent users, you need one memory object per user, and you
   must be careful about thread safety.

2. **Not serialisable** — the memory object lives in RAM. If the process
   restarts, conversation history is lost unless you implement your own
   persistence.

3. **Not reproducible** — to replay a conversation (for debugging or testing),
   you must re-run it from scratch because there is no way to inspect or restore
   an intermediate state.

4. **No rollback** — if the agent makes a mistake, you cannot roll back to a
   previous state without starting over.

### 4.14.2 LangGraph Checkpointing (The Better Way)

LangGraph stores the entire graph state as an immutable snapshot after every
node execution. These snapshots are written to a database (SQLite in saathi's
case). Each snapshot has:

- The full `AgentState` (all messages, context_paths, mode, session_id)
- A `config` identifying the thread and the step
- A parent snapshot ID (forming a linked list of history)

This gives you:

```python
# List all checkpoints for a thread
checkpoints = [c async for c in graph.aget_state_history(config)]
# Returns: [StateSnapshot(step=5), StateSnapshot(step=4), ...]

# Roll back to a specific step
earlier_config = {"configurable": {"thread_id": session_id, "checkpoint_id": step3_id}}
await graph.aupdate_state(earlier_config, values={})  # reactivate that checkpoint

# The next graph.ainvoke() picks up from step 3, discarding steps 4 and 5
```

saathi exposes this through `/rollback` and `/checkpoints` commands:

```python
# From saathi/ui/commands.py — handle_rollback()
async def handle_rollback(args, graph, config):
    history = [c async for c in graph.aget_state_history(config)]
    # ... let user select a step ...
    target_config = history[n].config
    await graph.aupdate_state(target_config, values={"messages": history[n].values["messages"]})
```

This is fundamentally different from LangChain memory. The state is not a
mutable Python object — it is a series of immutable database rows. Concurrency
is safe (SQLite with WAL mode), restarts are safe (state is on disk), and
rollback is built in.

### 4.14.3 saathi's MemoryStore — A Separate Concern

Note that saathi also has a `MemoryStore` class, but this is **not** conversation
history memory. It is a key-value store for **facts** the model explicitly saves
between sessions:

```python
# From saathi/tools/memory_tools.py
@tool
def save_memory(scope: str, key: str, value: str) -> str:
    """
    Persist a fact for future sessions.
    scope: 'global' (user-level, ~/.saathi/) or 'project' (.saathi/ in cwd).
    """
    _store.save(scope, key, value)
    return f"Saved [{scope}] {key} = {value}"
```

This is for things like "the project's entry point is `src/saathi/cli.py`" or
"the test command is `pytest tests/`" — facts that should survive across
sessions. The LangGraph checkpointer handles within-session history; the
`MemoryStore` handles cross-session facts.

---

## 4.15 langchain_community vs langchain_core

LangChain has gone through several package restructurings. The current split:

| Package | Contents |
| --------- | ---------- |
| `langchain_core` | Protocol: `BaseMessage`, `BaseTool`, `Runnable`, `BaseCallbackHandler` |
| `langchain` | Higher-level chains, agents, memory (depends on `langchain_core`) |
| `langchain_community` | 300+ third-party integrations (vector stores, document loaders, etc.) |
| `langchain_ollama` | Ollama integration (`ChatOllama`, `OllamaEmbeddings`) |
| `langchain_openai` | OpenAI integration (`ChatOpenAI`) |
| `langgraph` | Graph engine (depends on `langchain_core`) |

### 4.15.1 Why Import from langchain_core Directly

saathi imports from `langchain_core` rather than `langchain` wherever possible:

```python
# saathi/agent/nodes.py
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

# saathi/tools/filesystem.py
from langchain_core.tools import tool

# saathi/agent/state.py
from langchain_core.messages import BaseMessage
```

The reasons:

1. **Stability** — `langchain_core` changes more slowly than `langchain`. The
   protocol types are stable; the higher-level chains in `langchain` are more
   experimental.

2. **Smaller dependency footprint** — `langchain_core` has very few
   dependencies (primarily Pydantic and `tenacity`). The full `langchain`
   package pulls in significantly more.

3. **Clarity of intent** — importing from `langchain_core` signals that you
   are relying on the protocol, not a specific implementation. This makes
   the dependency graph easier to reason about.

4. **Avoids the community package trap** — `langchain_community` is a grab-bag
   of community integrations with varying quality. Many have unmaintained
   dependencies. Only import from it when you need a specific integration;
   never import it wholesale.

### 4.15.2 The Import Hierarchy for saathi

```text
saathi
├── langchain_core        # message types, tool protocol, Runnable
├── langchain_ollama      # ChatOllama (the concrete model implementation)
└── langgraph             # graph engine, checkpointing, prebuilt nodes
    └── langchain_core    # langgraph depends on langchain_core
```

`langchain` (the main package) does not appear in saathi's dependencies at all.
This is intentional. saathi needs the protocol layer and the Ollama integration
and the graph engine. It does not need the higher-level abstractions that the
main `langchain` package provides.

---

## 4.16 Summary

This chapter traced LangChain's architecture from first principles. The key
ideas:

1. **LangChain is a protocol layer**, not a framework. Its value is in the
   types it defines: the message hierarchy, the `Runnable` protocol, the tool
   interface.

2. **The message types** encode the semantics of AI conversations. `HumanMessage`
   is user input; `AIMessage` may carry tool calls; `ToolMessage` delivers tool
   results and must match `tool_call_id`.

3. **`bind_tools()` wraps an LLM** in a `RunnableBinding` that injects tool
   schemas. The result is a `LanguageModelLike`, not the original `ChatOllama`.

4. **The `@tool` decorator** converts a Python function to a `BaseTool`. The
   docstring becomes the tool description and directly affects model quality.

5. **`astream_events`** is the streaming API saathi uses to print tokens in
   real time, display tool calls, and collect usage metadata.

6. **LangGraph checkpointing** is superior to LangChain memory for production
   agents: immutable, persistent, rollback-capable.

7. **Import from `langchain_core`** for the protocol types; use model-specific
   packages (`langchain_ollama`) only for the concrete implementations.

The next chapter takes these primitives and builds saathi's state machine on
top of them.

---

Next: Chapter 5 — LangGraph Core: State Machines for AI Agents
