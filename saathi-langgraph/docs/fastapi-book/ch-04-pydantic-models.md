# Chapter 4 — Pydantic Models

> **What you'll learn:** how Pydantic's `BaseModel` works, every `Field()` option that matters, `Literal` types, `default_factory`, nested models, optional fields, and a complete walkthrough of every model in `schemas.py`.

---

## What Pydantic is

Pydantic is a data validation library. You describe the *shape* of your data using Python class syntax and type hints, and Pydantic enforces that shape at runtime — raising clear errors when the data doesn't match.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str
```

```python
# Valid data — creates a User object
user = User(name="Ashwin", age=30, email="a@example.com")
print(user.name)   # "Ashwin"
print(user.age)    # 30

# Invalid data — raises ValidationError
user = User(name="Ashwin", age="not a number", email="a@example.com")
# pydantic_core._pydantic_core.ValidationError: 1 validation error for User
# age  Input should be a valid integer [...]
```

FastAPI uses Pydantic models for three things: deserialising request bodies, serialising responses, and generating the JSON schema that powers your API docs.

> **Experienced note:** Pydantic v2 (released 2023) rewrote the core in Rust. It is 5–50x faster than v1 for validation. The API changed somewhat — if you see old tutorials using `class Config:` inside a model or `orm_mode = True`, those are v1 patterns. FastAPI 0.100+ requires Pydantic v2.

---

## The `Field()` function

`Field()` is how you add metadata and constraints to individual model fields:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., description="Product display name", min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
    quantity: int = Field(default=0, ge=0)
```

The `...` (Ellipsis) means the field is **required** — it has no default. Common `Field()` arguments:

| Argument | Meaning |
|---|---|
| `...` (first arg) | Required — no default value |
| `default=value` | Optional with this default |
| `default_factory=callable` | Optional; default computed by calling the function |
| `description="..."` | Appears in the API docs |
| `min_length` / `max_length` | String length constraints |
| `gt` / `ge` / `lt` / `le` | Numeric constraints (greater-than, greater-or-equal, etc.) |
| `pattern="..."` | Regex the string must match |
| `examples=[...]` | Example values shown in docs |

---

## `default_factory` — computed defaults

Sometimes the default value can't be a static literal because it needs to be computed fresh each time. A UUID is a perfect example — you want each new request to get a unique ID, not all share the same one:

```python
import uuid
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

`default_factory` takes a **callable** (a function or lambda). Every time a `ChatRequest` is created without a `session_id`, Pydantic calls `lambda: str(uuid.uuid4())` to generate one. If you wrote `default=str(uuid.uuid4())` instead, all requests would share the same UUID — that value is computed once at class definition time.

> **Beginner note:** A lambda is a tiny anonymous function. `lambda: str(uuid.uuid4())` is the same as writing `def make_id(): return str(uuid.uuid4())` and passing `make_id`.

---

## `Literal` — constraining to specific values

`Literal` from Python's `typing` module restricts a field to a fixed set of allowed values:

```python
from typing import Literal

class ChatRequest(BaseModel):
    mode: Literal["default", "explain", "refactor", "debug"] = "default"
```

If a request sends `"mode": "invalid"`, Pydantic rejects it with a 422 error before your handler runs. In the docs, this field shows as an enum with exactly those four options.

---

## Optional fields with `None`

A field is optional when its type includes `None` and it has a default of `None`:

```python
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    model: str
    detail: str | None = None   # optional — omitted when Ollama is healthy
```

`str | None` is Python 3.10+ syntax for "either a string or None". In Python 3.9 and earlier you'd write `Optional[str]` from `typing`. Both mean the same thing to Pydantic.

When you return a `HealthResponse` with `detail=None`, FastAPI serialises it as `"detail": null` in JSON by default. You can suppress the null with `model_config = ConfigDict(exclude_none=True)` if you prefer cleaner output.

---

## Walking through every model in `schemas.py`

### `ChatRequest`

```python
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread/session ID — reuse to continue a conversation",
    )
    mode: Literal["default", "explain", "refactor", "debug"] = "default"
    context_paths: list[str] = Field(
        default=[],
        description="File or directory paths the agent should scope its context to",
    )
```

- `message` — required (`...`). The user's text. No constraints beyond being a string — the agent handles empty strings gracefully.
- `session_id` — auto-generated UUID if not provided. Clients can supply their own to continue a previous conversation.
- `mode` — `Literal` ensures only the four valid modes are accepted. Defaults to `"default"`.
- `context_paths` — a `list[str]`, defaulting to an empty list. `default=[]` is safe here because Pydantic creates a new list for each model instance (unlike Python function defaults, where mutable defaults are shared).

> **Experienced note:** In raw Python, `def f(x=[])` is a famous footgun — the same list is reused across calls. Pydantic avoids this: `Field(default=[])` creates a new list each time.

### `ChatResponse`

```python
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls_made: int = Field(default=0, description="Number of tool invocations during this turn")
```

Simple response shape. All three fields could have been written without `Field()` — `tool_calls_made: int = 0` is equivalent. `Field()` is used here to attach a description that appears in the docs.

### `StreamChunk`

```python
class StreamChunk(BaseModel):
    session_id: str
    delta: str
    done: bool = False
```

This is the shape of each SSE event payload in `POST /chat/stream`. `delta` is a token fragment from the LLM. `done=True` signals end of stream. Notice it is not used as a `response_model` on the streaming endpoint (because streaming responses work differently — Chapter 9 covers this).

### `SessionCreateRequest`

```python
class SessionCreateRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    mode: Literal["default", "explain", "refactor", "debug"] = "default"
```

Very similar to `ChatRequest` but just the session identity fields — no message. Used by `POST /sessions`.

### `MessageRecord`

```python
class MessageRecord(BaseModel):
    role: Literal["human", "ai", "tool", "system"]
    content: str
```

A single message in the conversation history. `Literal` on `role` is important — it ensures that whatever comes back from the LangGraph state is normalised into one of those four values, and the client can rely on that invariant.

### `SessionHistoryResponse`

```python
class SessionHistoryResponse(BaseModel):
    session_id: str
    mode: str
    messages: list[MessageRecord]
```

A **nested model** — `messages` is a list of `MessageRecord` objects. Pydantic validates the entire nested structure recursively. When FastAPI serialises this to JSON, the `messages` list becomes a JSON array of objects, each with `role` and `content`.

### `HealthResponse`

```python
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    model: str
    detail: str | None = None
```

`Literal["ok", "degraded"]` on `status` means the client can pattern-match on exactly two values without parsing a free-form string. This is a good API design practice — constrain string fields to known values wherever possible.

### `ModelInfoResponse`

```python
class ModelInfoResponse(BaseModel):
    model: str
    base_url: str
    temperature: float
    context_window: int
    max_tokens: int
    max_parallel_tools: int
```

A flat model with no optional fields. Everything is returned. This is the right choice when the response always has all fields — simpler to document and consume.

---

## Validators — adding custom rules

Sometimes the built-in constraints aren't enough. Pydantic v2 provides `@field_validator` for custom logic:

```python
from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v.strip()
```

The validator runs after basic type checking. If it raises `ValueError`, Pydantic converts it into a 422 response automatically.

> **Experienced note:** Pydantic v1 used `@validator`. v2 uses `@field_validator` with `@classmethod`. The old decorator still works in v2 but emits deprecation warnings.

---

## `model_config` — model-level settings

Pydantic v2 uses a `model_config` class variable (a `ConfigDict`) for model-level options:

```python
from pydantic import BaseModel, ConfigDict

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,  # strip leading/trailing whitespace from all strings
        frozen=True,                # make instances immutable (like a dataclass with frozen=True)
    )
    message: str
```

Saathi's models don't need `model_config` — the defaults work fine — but it's useful to know for production APIs where you want tighter control.

---

## Summary

- `BaseModel` is the base class for all Pydantic models. Fields are declared as class attributes with type hints.
- `Field(...)` adds metadata (description, constraints, examples) and is required for `default_factory`.
- `default_factory=callable` generates a fresh default on each instantiation — use it for UUIDs, lists, dicts.
- `Literal[...]` restricts a field to a fixed set of string values — great for enums, modes, roles.
- `str | None = None` makes a field optional.
- Nested models (`list[MessageRecord]`) are validated recursively.
- `@field_validator` adds custom validation logic beyond built-in constraints.

---

*Previous: [Chapter 3 — Path Operations](ch-03-path-operations.md)*  
*Next: [Chapter 5 — Request Body and Response Models](ch-05-request-response.md)*
