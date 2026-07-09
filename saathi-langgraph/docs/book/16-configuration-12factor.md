# Chapter 16 — Configuration and the 12-Factor App

> "An app's config is everything that is likely to vary between deploys (staging, production, developer environments, etc). This includes resource handles to the database, memcache, and other backing services; credentials to external services such as Amazon S3 or Twitter; and per-deploy values such as the canonical hostname for the deploy."
>
> — The 12-Factor App, Factor III

---

## 16.1 The 12-Factor App

In 2011, engineers at Heroku published a methodology for building modern, cloud-native web applications. They called it [The 12-Factor App](https://12factor.net). Twelve distinct best practices distilled from years of running thousands of applications on their platform. The methodology is language-agnostic, platform-agnostic, and has aged remarkably well.

The twelve factors are:

| # | Factor | One-line summary |
| --- | -------- | ----------------- |
| I | Codebase | One codebase tracked in revision control, many deploys |
| II | Dependencies | Explicitly declare and isolate dependencies |
| **III** | **Config** | **Store config in the environment** |
| IV | Backing services | Treat backing services as attached resources |
| V | Build, release, run | Strictly separate build and run stages |
| VI | Processes | Execute the app as one or more stateless processes |
| VII | Port binding | Export services via port binding |
| VIII | Concurrency | Scale out via the process model |
| IX | Disposability | Maximize robustness with fast startup and graceful shutdown |
| X | Dev/prod parity | Keep development, staging, and production as similar as possible |
| XI | Logs | Treat logs as event streams |
| XII | Admin processes | Run admin/management tasks as one-off processes |

Of these twelve, **Factor III — Config** is most directly relevant to saathi and to AI applications generally. Let us study it in depth.

### Factor III: Store Config in the Environment

The canonical test for whether a piece of information belongs in config is: could you open-source your codebase right now without exposing credentials or environment-specific values? If the answer is "no" because there is a hardcoded API key, a production database URL, or a flag that says `debug = False` in one place and `debug = True` in another, you have a config problem.

The 12-Factor methodology is explicit: config belongs in **environment variables**, not in config files checked into version control, not in constants defined in Python modules, and never hardcoded inline.

Why environment variables?

1. **Language-agnostic.** Every runtime—Python, Node.js, Go, Rust, Java—can read environment variables. No parsing library required.
2. **Easy to change per deploy.** CI/CD pipelines, container orchestrators, and shell scripts all make it trivial to set env vars differently for dev, staging, and production.
3. **They cannot accidentally be checked in.** You cannot accidentally `git commit` an environment variable the way you can accidentally commit a `config.py` file.
4. **Composable.** Different env vars can be set in different places (secrets manager, CI pipeline, `.env` file for local dev) and they all merge at the process level.

### Why Factor III Is Especially Important for AI Applications

Traditional web applications might have a handful of config values: a database URL, a secret key, a flag for debug mode. AI applications have a fundamentally larger and more complex configuration surface:

**Model selection.** In development you might use a lightweight local model (say, `llama3.2:3b`) that runs quickly on a laptop. In a CI pipeline you might skip LLM calls entirely (mock them). In production you might use a more capable model (`qwen2.5:14b`) running on a server with a GPU. These are not the same value; they must differ by deploy.

**Context window.** A 3-billion-parameter model might have an 8,192-token context window. A larger model might have 128k. The agent's memory management (how aggressively to compact the message history) must adapt to the window size. This is a config value.

**Concurrency limits.** A developer's laptop can run maybe 1–2 tool calls in parallel before the fan spins up and the system slows down. A production server can handle 8 or 16. This is a config value.

**Timeouts.** Local Ollama on a fast GPU responds in under a second. A cloud LLM under heavy load might take 30 seconds. Timeout values must differ by deploy.

**Logging verbosity.** You want `DEBUG`-level logs locally. You want `INFO` or `WARNING` in production. This is a config value.

**Feature flags.** "Enable streaming" might be on in production but off in a test environment. Config.

**LangSmith tracing.** You want tracing enabled for debugging, off for production privacy, and off in CI to avoid polluting trace data. Config.

None of these should be hardcoded. All of them belong in environment variables.

---

## 16.2 `pydantic-settings` — The Recommended Python Library for 12-Factor Config

Raw environment variables in Python look like this:

```python
import os

model = os.environ.get("SAATHI_MODEL", "qwen2.5:14b")
max_tokens = int(os.environ.get("SAATHI_MAX_TOKENS", "8096"))
temperature = float(os.environ.get("SAATHI_TEMPERATURE", "0.7"))
debug = os.environ.get("SAATHI_DEBUG", "false").lower() == "true"
```

This works, but it has problems:

- **No schema.** There is no single place to see all config values and their types.
- **No validation.** If someone sets `SAATHI_TEMPERATURE=banana`, your app will crash at some random point with a confusing `ValueError`.
- **Scattered.** Config reads are sprinkled throughout the codebase.
- **No documentation.** There is no canonical list of what env vars the application reads.
- **No IDE support.** No autocomplete, no type checking.

[`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) solves all of these problems. It is a Pydantic extension that lets you define your application's configuration as a typed class. Environment variables are automatically read, coerced to the correct types, and validated against your schema.

### Installation

```bash
pip install pydantic-settings
# or in pyproject.toml:
# pydantic-settings = "^2.0"
```

### The `BaseSettings` Class

`pydantic-settings` provides `BaseSettings`, a Pydantic `BaseModel` subclass that, when instantiated, automatically reads field values from environment variables.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model: str = "qwen2.5:14b"
    temperature: float = 0.7
    debug: bool = False
```

When you instantiate this class, pydantic-settings looks for environment variables named `MODEL`, `TEMPERATURE`, and `DEBUG` (by default, uppercased from the field names). If found, they are coerced to the declared type. If not found, the default is used. If the value cannot be coerced (e.g., `TEMPERATURE=banana`), a `ValidationError` is raised immediately at startup.

### Type Coercion

pydantic-settings handles type coercion automatically:

| Python type | Env var value | Result |
| ------------ | -------------- | -------- |
| `str` | `"qwen2.5:14b"` | `"qwen2.5:14b"` |
| `int` | `"8096"` | `8096` |
| `float` | `"0.7"` | `0.7` |
| `bool` | `"true"`, `"1"`, `"yes"`, `"on"` | `True` |
| `bool` | `"false"`, `"0"`, `"no"`, `"off"` | `False` |
| `list[str]` | `"a,b,c"` | `["a", "b", "c"]` |
| `Path` | `"/home/user/.saathi"` | `Path("/home/user/.saathi")` |

This is much safer than manual `os.environ.get()` and `int()` / `float()` calls scattered throughout your code.

---

## 16.3 The `SAATHI_` Prefix

When running saathi, your shell environment contains hundreds of variables: `PATH`, `HOME`, `TERM`, `LANG`, `PYTHONPATH`, and many more set by your shell configuration, virtual environment, IDE, and other tools. If saathi read a variable named simply `MODEL` or `DEBUG`, it would collide with other tools that also define variables with those names.

The solution is a **prefix**. Every environment variable that saathi reads is prefixed with `SAATHI_`. This creates a namespace that is extremely unlikely to collide with anything else.

### Setting the Prefix

In pydantic-settings, the prefix is set via `SettingsConfigDict`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAATHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = "qwen2.5:14b"
    temperature: float = 0.7
    debug: bool = False
```

With `env_prefix="SAATHI_"`, pydantic-settings looks for `SAATHI_MODEL`, `SAATHI_TEMPERATURE`, and `SAATHI_DEBUG` in the environment, not `MODEL`, `TEMPERATURE`, and `DEBUG`.

### `case_sensitive=False`

On Linux, environment variables are case-sensitive: `SAATHI_MODEL` and `saathi_model` are different variables. On Windows, they are case-insensitive. Setting `case_sensitive=False` makes pydantic-settings behaviour consistent across platforms: it will match `SAATHI_MODEL`, `saathi_model`, or `Saathi_Model` to the `model` field.

### `extra="ignore"`

By default, pydantic raises an error if you pass fields that are not defined in the model. `extra="ignore"` tells pydantic-settings to silently ignore any env vars with the `SAATHI_` prefix that it does not recognize. This is permissive but practical: if a user has an old `SAATHI_SOME_REMOVED_FIELD` in their `.env` file, they do not get a cryptic startup error.

---

## 16.4 Saathi's Full `Settings` Class

The canonical settings for saathi live in `src/saathi/config.py`. Here is the complete class:

```python
# src/saathi/config.py
"""
Application configuration via pydantic-settings.

All settings can be overridden with environment variables prefixed SAATHI_.
Example: export SAATHI_MODEL=llama3.2:latest

Settings are loaded once at module import time and cached as a module-level
singleton. Import with:
    from saathi.config import settings
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Saathi application settings.

    Loaded from environment variables (SAATHI_ prefix) and .env file.
    All fields have sensible defaults so the app works out of the box.
    """

    model_config = SettingsConfigDict(
        env_prefix="SAATHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Model settings                                                       #
    # ------------------------------------------------------------------ #

    model: str = Field(
        default="qwen2.5:14b",
        description="Ollama model to use for the LLM backbone.",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server.",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher = more creative, lower = more deterministic.",
    )

    context_window: int = Field(
        default=8096,
        gt=0,
        description="Context window size in tokens for the selected model.",
    )

    # ------------------------------------------------------------------ #
    # Agent / tool settings                                                #
    # ------------------------------------------------------------------ #

    max_parallel_tools: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of tool calls to execute in parallel.",
    )

    tool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout in seconds for individual tool calls.",
    )

    max_tool_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of times to retry a failed tool call before giving up.",
    )

    # ------------------------------------------------------------------ #
    # Memory and history settings                                          #
    # ------------------------------------------------------------------ #

    memory_dir: Path = Field(
        default=Path(".saathi/memory"),
        description="Directory where memory files are stored.",
    )

    sessions_dir: Path = Field(
        default=Path(".saathi/sessions"),
        description="Directory where session files are stored.",
    )

    commands_dir: Path = Field(
        default=Path(".saathi/commands"),
        description="Directory where custom slash command files are stored.",
    )

    history_token_budget_pct: float = Field(
        default=0.75,
        ge=0.1,
        le=0.95,
        description=(
            "Fraction of context_window to use for message history before "
            "triggering compaction. Default 75%."
        ),
    )

    compact_threshold_tokens: int = Field(
        default=6000,
        gt=0,
        description="Token count above which /compact is triggered automatically.",
    )

    # ------------------------------------------------------------------ #
    # Streaming and output settings                                        #
    # ------------------------------------------------------------------ #

    stream: bool = Field(
        default=True,
        description="Whether to stream LLM responses token by token.",
    )

    output_format: Literal["text", "json"] = Field(
        default="text",
        description="Output format for --print mode. One of: text, json.",
    )

    # ------------------------------------------------------------------ #
    # Logging and debug settings                                           #
    # ------------------------------------------------------------------ #

    debug: bool = Field(
        default=False,
        description="Enable debug logging.",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the structlog logger.",
    )

    langchain_tracing_v2: bool = Field(
        default=False,
        alias="LANGCHAIN_TRACING_V2",
        description=(
            "Enable LangSmith tracing. Note: this uses the LANGCHAIN_TRACING_V2 "
            "env var name (no SAATHI_ prefix) to match LangSmith convention."
        ),
    )

    # ------------------------------------------------------------------ #
    # Computed fields                                                      #
    # ------------------------------------------------------------------ #

    @computed_field  # type: ignore[misc]
    @property
    def history_token_budget(self) -> int:
        """Maximum tokens to use for message history (fraction of context window)."""
        return int(self.context_window * self.history_token_budget_pct)

    @computed_field  # type: ignore[misc]
    @property
    def saathi_dir(self) -> Path:
        """Root .saathi directory, derived from memory_dir parent."""
        return self.memory_dir.parent

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("temperature", mode="before")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        """Clamp temperature to the valid range [0.0, 2.0]."""
        try:
            v = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"temperature must be a number, got {v!r}") from exc
        return max(0.0, min(2.0, v))

    @field_validator("model", mode="before")
    @classmethod
    def strip_model_name(cls, v: str) -> str:
        """Strip whitespace from model name."""
        return str(v).strip()

    @field_validator("memory_dir", "sessions_dir", "commands_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Path | str) -> Path:
        """Expand ~ and environment variables in path settings."""
        return Path(v).expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Uses lru_cache so the Settings object is only created once per process,
    but can be invalidated in tests by calling get_settings.cache_clear().
    """
    return Settings()


# Module-level singleton for convenience.
# Import with: from saathi.config import settings
settings = get_settings()
```

This is the authoritative source of truth for all saathi configuration. Let us walk through each section.

---

## 16.5 `.env` Files — Local Development Without Polluting Your Shell

The `env_file=".env"` in `SettingsConfigDict` tells pydantic-settings to look for a file named `.env` in the current working directory (the project root) and load it as if its contents were environment variables.

### How `.env` Files Work

A `.env` file is a plain text file with one `KEY=VALUE` assignment per line:

```bash
# .env
SAATHI_MODEL=llama3.2:latest
SAATHI_TEMPERATURE=0.3
SAATHI_DEBUG=true
SAATHI_MAX_PARALLEL_TOOLS=2
```

Comments start with `#`. Blank lines are ignored. Values do not need to be quoted (though quotes are stripped if present). This file is loaded by pydantic-settings when the `Settings` class is instantiated—no `source .env` required.

### Why `.env` Files Are Convenient

Manually setting environment variables in your shell is tedious:

```bash
export SAATHI_MODEL=llama3.2:latest
export SAATHI_TEMPERATURE=0.3
export SAATHI_DEBUG=true
# ... and so on for every session
```

A `.env` file persists these settings across shell sessions without permanently modifying your shell's profile. You can have different `.env` files for different projects. You open it in your editor, change a value, and the next process that runs picks it up.

### `.env` Must Be in `.gitignore`

The `.env` file often contains sensitive information: API keys, database passwords, service tokens. It must never be committed to version control.

Add it to `.gitignore`:

```text
# .gitignore

# Local environment configuration (may contain secrets)
.env
.env.local

# Keep the example file
!.env.example
```

### `.env.example` — Living Documentation

The counterpart to `.env` is `.env.example`. This file is committed to version control and serves as documentation: it lists every env var the application reads, with a safe placeholder value and a comment explaining what it does.

Here is saathi's `.env.example`:

```bash
# .env.example
#
# Copy this file to .env and fill in your values.
# Lines starting with # are comments and are ignored.
# Values shown here are the application defaults.
#
# Usage:
#   cp .env.example .env
#   # edit .env with your preferred values
#

# ─────────────────────────────────────────────────────────────────────────────
# Model settings
# ─────────────────────────────────────────────────────────────────────────────

# Ollama model to use. Run `ollama list` to see available models.
# Popular choices: qwen2.5:14b, llama3.2:3b, llama3.1:8b, mistral:7b
SAATHI_MODEL=qwen2.5:14b

# URL where the Ollama server is running.
# Change this if running Ollama on a remote machine or in Docker.
SAATHI_OLLAMA_BASE_URL=http://localhost:11434

# Sampling temperature. Range: 0.0 (deterministic) to 2.0 (very creative).
# Recommended: 0.3–0.7 for coding tasks, 0.7–1.0 for creative tasks.
SAATHI_TEMPERATURE=0.7

# Context window size in tokens. Must match the model you are using.
# qwen2.5:14b: 32768. llama3.2:3b: 8192. Check `ollama show <model>`.
SAATHI_CONTEXT_WINDOW=8096

# ─────────────────────────────────────────────────────────────────────────────
# Agent / tool settings
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of tool calls to execute in parallel.
# Lower this on low-memory machines (e.g., set to 1 or 2).
SAATHI_MAX_PARALLEL_TOOLS=4

# Timeout in seconds for individual tool calls.
# Increase for slow network operations; decrease for faster failure detection.
SAATHI_TOOL_TIMEOUT_SECONDS=30

# Number of times to retry a failed tool call.
SAATHI_MAX_TOOL_RETRIES=2

# ─────────────────────────────────────────────────────────────────────────────
# Memory and history settings
# ─────────────────────────────────────────────────────────────────────────────

# Directory where memory files are stored (relative to project root).
SAATHI_MEMORY_DIR=.saathi/memory

# Directory where session files are stored.
SAATHI_SESSIONS_DIR=.saathi/sessions

# Directory where custom slash command files are stored.
SAATHI_COMMANDS_DIR=.saathi/commands

# Fraction of context window to use for message history before compaction.
# Default: 0.75 (75%). Reduce if you want more room for tool outputs.
SAATHI_HISTORY_TOKEN_BUDGET_PCT=0.75

# Token count above which /compact is triggered automatically.
SAATHI_COMPACT_THRESHOLD_TOKENS=6000

# ─────────────────────────────────────────────────────────────────────────────
# Output settings
# ─────────────────────────────────────────────────────────────────────────────

# Stream LLM responses token by token (true/false).
SAATHI_STREAM=true

# Default output format for --print mode. Options: text, json.
SAATHI_OUTPUT_FORMAT=text

# ─────────────────────────────────────────────────────────────────────────────
# Logging and debug settings
# ─────────────────────────────────────────────────────────────────────────────

# Enable debug logging (true/false). Equivalent to --debug CLI flag.
SAATHI_DEBUG=false

# Log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
SAATHI_LOG_LEVEL=INFO

# ─────────────────────────────────────────────────────────────────────────────
# LangSmith tracing (optional — see Chapter 18)
# ─────────────────────────────────────────────────────────────────────────────

# Enable LangSmith tracing. Requires LANGCHAIN_API_KEY.
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls__...
# LANGCHAIN_PROJECT=saathi-langgraph
```

Every setting documented, defaults shown, sensitive values commented out. Anyone cloning the repo can run `cp .env.example .env` and have a working configuration immediately.

---

## 16.6 Computed Fields — Derived Configuration Values

Some configuration values are not set directly by the user but are derived from other settings. Pydantic v2's `@computed_field` decorator handles these elegantly.

### The `history_token_budget` Computed Field

The maximum number of tokens saathi will use for message history is a function of two settings: `context_window` (the absolute size of the model's context) and `history_token_budget_pct` (the fraction to use for history). Computing it as a `@computed_field` means:

1. The user configures two intuitive values (window size and percentage).
2. The derived value (`history_token_budget`) is always consistent with both.
3. No code outside `config.py` needs to perform this calculation.

```python
@computed_field  # type: ignore[misc]
@property
def history_token_budget(self) -> int:
    """Maximum tokens to use for message history (fraction of context window).

    With the defaults (context_window=8096, history_token_budget_pct=0.75),
    this returns 6072. The remaining 25% (2024 tokens) is reserved for the
    system prompt, tool definitions, and the current turn's output.
    """
    return int(self.context_window * self.history_token_budget_pct)
```

Usage in the codebase:

```python
from saathi.config import settings

# In memory management logic:
if total_tokens > settings.history_token_budget:
    # trigger compaction
    ...
```

Note `# type: ignore[misc]` on the decorator line. As of Pydantic v2.5, the `@computed_field` + `@property` combination triggers a mypy false positive. The ignore comment suppresses it; the runtime behavior is correct.

### The `saathi_dir` Computed Field

```python
@computed_field  # type: ignore[misc]
@property
def saathi_dir(self) -> Path:
    """Root .saathi directory, derived from memory_dir parent."""
    return self.memory_dir.parent
```

This derives the root `.saathi` directory from `memory_dir`. If a user changes `SAATHI_MEMORY_DIR` to `/home/user/myproject/.saathi/memory`, then `saathi_dir` becomes `/home/user/myproject/.saathi` automatically. They do not need to set both.

### Why Not Just Set These in `__init__`?

Computed fields are better than overriding `__init__` or using `model_post_init` for derived values because:

1. They are validated as part of the model schema.
2. They appear in `model_fields` and `model_json_schema()`, so they show up in documentation.
3. They are re-evaluated if the model is copied with `model.model_copy(update={"context_window": 16384})`, keeping everything consistent.

---

## 16.7 Validators — Defensive Configuration

pydantic-settings inherits Pydantic's full validation system. `@field_validator` decorators run when the settings object is created and can perform arbitrary validation logic.

### Temperature Clamping

The OpenAI API accepts temperatures from 0.0 to 2.0; Ollama models generally follow the same range. Rather than crashing with a cryptic error if someone sets `SAATHI_TEMPERATURE=3.5`, saathi clamps the value:

```python
@field_validator("temperature", mode="before")
@classmethod
def clamp_temperature(cls, v: float) -> float:
    """Clamp temperature to the valid range [0.0, 2.0].

    Uses clamping rather than rejection to be permissive: a value of 2.5
    is clamped to 2.0 silently. Values that cannot be parsed as floats
    are rejected with a clear error message.

    Args:
        v: The raw value from the environment variable.

    Returns:
        The clamped temperature value.

    Raises:
        ValueError: If v cannot be parsed as a float.
    """
    try:
        v = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SAATHI_TEMPERATURE must be a number between 0.0 and 2.0, "
            f"got {v!r}"
        ) from exc
    clamped = max(0.0, min(2.0, v))
    if clamped != v:
        logger.warning(
            "SAATHI_TEMPERATURE %s is outside [0.0, 2.0]; clamped to %s",
            v,
            clamped,
        )
    return clamped
```

`mode="before"` means the validator runs on the raw string value from the environment variable, before Pydantic's own type coercion. This lets us produce a better error message than Pydantic's default.

### Model Name Sanitization

```python
@field_validator("model", mode="before")
@classmethod
def strip_model_name(cls, v: str) -> str:
    """Strip whitespace from model name.

    Prevents bugs where SAATHI_MODEL=' qwen2.5:14b' (note leading space)
    causes a confusing 'model not found' error from Ollama.
    """
    return str(v).strip()
```

Small, defensive. Stripping whitespace from a model name costs nothing and prevents a confusing failure mode.

### Path Expansion

```python
@field_validator("memory_dir", "sessions_dir", "commands_dir", mode="before")
@classmethod
def expand_path(cls, v: Path | str) -> Path:
    """Expand ~ and environment variables in path settings.

    Allows users to set paths like:
        SAATHI_MEMORY_DIR=~/work/myproject/.saathi/memory
    """
    return Path(v).expanduser()
```

A single validator handles multiple fields by listing them in the decorator. `expanduser()` expands `~` to the user's home directory.

### Reject vs Clamp vs Warn

There are three strategies for handling out-of-range values:

1. **Reject** (raise `ValueError`): Use when an invalid value would cause a hard failure. Example: `SAATHI_LOG_LEVEL=VERBOSE` is not a valid Python log level; reject it.
2. **Clamp**: Use when the intent is clear and the out-of-range value has a natural nearest valid value. Example: `SAATHI_TEMPERATURE=3.5` → clamp to `2.0`.
3. **Warn and use default**: Use when the value is clearly wrong but you can still proceed. Example: `SAATHI_TOOL_TIMEOUT_SECONDS=0` → warn and use `30`.

saathi uses a mix of these strategies. The key principle is that configuration errors should be loud and early—they should surface at startup, not silently corrupt behavior an hour into a session.

---

## 16.8 Settings in Tests — Overriding Config Per Test

The `Settings` class is instantiated once at module import time and cached:

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

This is convenient for production code (no repeated disk reads, consistent object identity) but creates a challenge for tests: if settings is a cached singleton, how do you test behavior under different configurations?

### Strategy 1: `monkeypatch.setenv` + Cache Clear

pytest's `monkeypatch` fixture allows setting environment variables for the duration of a single test:

```python
# tests/test_config.py

def test_temperature_clamping(monkeypatch):
    """Temperature above 2.0 should be clamped to 2.0."""
    monkeypatch.setenv("SAATHI_TEMPERATURE", "3.5")

    # Clear the lru_cache so a fresh Settings() is created with the new env var.
    from saathi.config import get_settings
    get_settings.cache_clear()

    from saathi.config import get_settings as gs
    s = gs()
    assert s.temperature == 2.0
```

The key steps:

1. `monkeypatch.setenv(...)` sets the env var for this test only. It is automatically undone when the test finishes.
2. `get_settings.cache_clear()` evicts the cached `Settings` instance so the next call creates a fresh one.
3. `get_settings()` returns a new `Settings` instance that reads from the patched environment.

pytest's `monkeypatch` restores the original env var automatically after each test, so tests are fully isolated.

### Strategy 2: Direct Instantiation

For simple configuration tests, create a `Settings` instance directly without going through the module singleton:

```python
def test_history_token_budget():
    """history_token_budget should be 75% of context_window."""
    s = Settings(
        SAATHI_CONTEXT_WINDOW="16384",
        SAATHI_HISTORY_TOKEN_BUDGET_PCT="0.75",
    )
    assert s.history_token_budget == 12288  # 16384 * 0.75
```

Wait—that does not look right. You cannot pass `SAATHI_CONTEXT_WINDOW` as a keyword argument to `Settings()`. The env prefix is only used when reading from the environment, not when constructing directly. When constructing directly, use the field name without the prefix:

```python
def test_history_token_budget():
    s = Settings(context_window=16384, history_token_budget_pct=0.75)
    assert s.history_token_budget == 12288
```

This is cleaner for unit testing configuration logic in isolation.

### Strategy 3: `pytest` Fixtures for Common Configurations

```python
# tests/conftest.py
import pytest
from saathi.config import Settings, get_settings


@pytest.fixture
def fast_settings():
    """Settings optimized for fast test execution."""
    return Settings(
        model="llama3.2:3b",
        context_window=4096,
        max_parallel_tools=1,
        tool_timeout_seconds=5,
        stream=False,
        debug=True,
    )


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure a fresh Settings instance for each test that modifies env vars."""
    yield
    get_settings.cache_clear()
```

The `autouse=True` on `clear_settings_cache` means it runs for every test automatically. This ensures that a test which modifies env vars and creates a new `Settings` instance does not pollute subsequent tests.

### Strategy 4: `SAATHI_` Override in CI

In CI environments, set the desired configuration via environment variables:

```yaml
# .github/workflows/test.yml
env:
  SAATHI_MODEL: "llama3.2:3b"
  SAATHI_MAX_PARALLEL_TOOLS: "1"
  SAATHI_TOOL_TIMEOUT_SECONDS: "10"
  SAATHI_DEBUG: "false"
  SAATHI_LOG_LEVEL: "WARNING"
```

These are set at the job level and apply to all test runs. No code changes, no special test fixtures. Pure 12-Factor.

---

## 16.9 All Settings Reference

The following table lists every `SAATHI_*` environment variable, its type, default value, and a description.

| Environment Variable | Type | Default | Description |
| --------------------- | ------ | --------- | ------------- |
| `SAATHI_MODEL` | `str` | `qwen2.5:14b` | Ollama model name. Run `ollama list` to see available models. |
| `SAATHI_OLLAMA_BASE_URL` | `str` | `http://localhost:11434` | URL of the Ollama server. |
| `SAATHI_TEMPERATURE` | `float` | `0.7` | Sampling temperature. Clamped to [0.0, 2.0]. |
| `SAATHI_CONTEXT_WINDOW` | `int` | `8096` | Context window in tokens. Must match the model. |
| `SAATHI_MAX_PARALLEL_TOOLS` | `int` | `4` | Max concurrent tool calls. Range: 1–32. |
| `SAATHI_TOOL_TIMEOUT_SECONDS` | `int` | `30` | Per-tool-call timeout in seconds. Range: 1–300. |
| `SAATHI_MAX_TOOL_RETRIES` | `int` | `2` | Tool call retry count on failure. Range: 0–10. |
| `SAATHI_MEMORY_DIR` | `Path` | `.saathi/memory` | Directory for memory files. `~` is expanded. |
| `SAATHI_SESSIONS_DIR` | `Path` | `.saathi/sessions` | Directory for session JSON files. |
| `SAATHI_COMMANDS_DIR` | `Path` | `.saathi/commands` | Directory for custom slash commands. |
| `SAATHI_HISTORY_TOKEN_BUDGET_PCT` | `float` | `0.75` | Fraction of context window for history. Range: 0.1–0.95. |
| `SAATHI_COMPACT_THRESHOLD_TOKENS` | `int` | `6000` | Token count triggering automatic `/compact`. |
| `SAATHI_STREAM` | `bool` | `true` | Stream LLM responses. |
| `SAATHI_OUTPUT_FORMAT` | `str` | `text` | Output format for `--print` mode. One of: `text`, `json`. |
| `SAATHI_DEBUG` | `bool` | `false` | Enable debug logging. Equivalent to `--debug` flag. |
| `SAATHI_LOG_LEVEL` | `str` | `INFO` | Python log level. One of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `LANGCHAIN_TRACING_V2` | `bool` | `false` | Enable LangSmith tracing. Note: no `SAATHI_` prefix. |

**Computed fields (not set directly):**

| Field | Type | Formula | Description |
| ------- | ------ | --------- | ------------- |
| `history_token_budget` | `int` | `context_window × history_token_budget_pct` | Max tokens for message history. |
| `saathi_dir` | `Path` | `memory_dir.parent` | Root `.saathi` directory. |

---

## 16.10 Configuration Layering — Precedence

pydantic-settings applies configuration values in the following order, from highest to lowest priority:

```text
1. Environment variables                  (highest priority)
2. .env file                              
3. Field defaults in the Settings class   (lowest priority)
```

If `SAATHI_MODEL` is set as an actual environment variable (e.g., `export SAATHI_MODEL=llama3.2:3b`), it overrides whatever is in `.env`. If `.env` contains `SAATHI_MODEL=mistral:7b`, it overrides the default of `qwen2.5:14b`.

### Practical Examples of Layering

**Example 1: Temporary override without editing `.env`**

Your `.env` file has `SAATHI_MODEL=qwen2.5:14b`. You want to try a different model for one session without changing the file:

```bash
SAATHI_MODEL=llama3.2:3b saathi
# or equivalently:
export SAATHI_MODEL=llama3.2:3b
saathi
unset SAATHI_MODEL  # restore afterwards
```

The env var takes precedence over `.env`. The `.env` file is unchanged.

Example 2: CI overrides for fast tests

In CI, you set `SAATHI_MAX_PARALLEL_TOOLS=1` as a pipeline environment variable to avoid resource contention. Developers' local `.env` files might have `SAATHI_MAX_PARALLEL_TOOLS=4`. The CI env var wins.

**Example 3: Multiple `.env` files with pydantic-settings**

pydantic-settings v2 supports multiple `.env` files with a priority order:

```python
model_config = SettingsConfigDict(
    env_file=[".env", ".env.local"],
    # .env.local takes precedence over .env
)
```

This pattern is common in Next.js projects and can be useful in Python too: `.env` contains the shared team defaults (committed to version control), and `.env.local` contains developer-specific overrides (gitignored).

### The Full Precedence Chain for `SAATHI_MODEL`

```text
shell export SAATHI_MODEL=llama3.2:3b     →  "llama3.2:3b"  ← wins
.env SAATHI_MODEL=qwen2.5:14b             →  "qwen2.5:14b"
Settings(model="qwen2.5:14b") default    →  "qwen2.5:14b"  ← loses
```

### Debugging Precedence Issues

If you cannot tell why a setting has an unexpected value:

```python
# In a Python REPL or quick debug script:
from saathi.config import settings
print(settings.model)  # actual value
print(settings.model_fields)  # field metadata
print(settings.model_config)  # the SettingsConfigDict
```

Or from the shell:

```bash
python -c "from saathi.config import settings; print(settings.model_dump_json(indent=2))"
```

This dumps all settings as JSON, making it easy to see the actual resolved values.

---

## 16.11 Secrets Management

### The Cardinal Rule: Never Hardcode Secrets

A secret is any value that:

- Grants access to a service (API keys, tokens, passwords)
- Contains personally identifiable information
- Would be embarrassing or harmful if made public

Secrets must never appear in source code. Not in `config.py` defaults, not in comments, not in test fixtures. The only place a secret should exist in your project is in a `.env` file (gitignored) or in an external secrets manager.

### Why Not `.env`?

`.env` files are convenient for local development. But they have limitations for production use:

1. They live on disk. Disk access can be audited, logged, and exfiltrated.
2. They are not encrypted at rest.
3. They do not have access controls (any process running as your user can read them).
4. They do not rotate automatically.

For production deployments, use a proper secrets manager:

- **AWS Secrets Manager** — integrates with IAM roles; supports automatic rotation.
- **HashiCorp Vault** — open-source, powerful, supports many secret backends.
- **Azure Key Vault** — for Azure deployments.
- **GCP Secret Manager** — for GCP deployments.
- **Doppler / Infisical** — modern developer-focused secrets managers with `.env`-like APIs.

The pattern in all of these is the same: the application receives secrets as environment variables (injected at deploy time by the orchestrator), and reads them via the `Settings` class. The application code does not change; only the mechanism of delivering the env vars differs.

### Saathi and API Keys

Saathi uses local Ollama, which does not require an API key. This is an advantage: there are no secrets to manage. The `SAATHI_OLLAMA_BASE_URL` is not a secret (it is a URL, not a credential).

However, if you extend saathi to use cloud LLMs, you will need API keys. The pattern is straightforward:

```python
# In Settings:
openai_api_key: str | None = Field(
    default=None,
    description="OpenAI API key. Required if using OpenAI models.",
)

anthropic_api_key: str | None = Field(
    default=None,
    description="Anthropic API key. Required if using Claude models.",
)
```

These fields would be read from `SAATHI_OPENAI_API_KEY` and `SAATHI_ANTHROPIC_API_KEY`. Set them in `.env` for local development, in a secrets manager for production. Never commit them.

### Detecting Accidental Secret Commits

Use [`git-secrets`](https://github.com/awslabs/git-secrets) or [`detect-secrets`](https://github.com/Yelp/detect-secrets) as a pre-commit hook to detect API keys before they are committed:

```bash
pip install detect-secrets
detect-secrets scan > .secrets.baseline
# Add to .pre-commit-config.yaml:
# - repo: https://github.com/Yelp/detect-secrets
#   hooks:
#   - id: detect-secrets
#     args: ['--baseline', '.secrets.baseline']
```

### The `log_safe_model_dump()` Pattern

When logging configuration at startup, be careful not to log secrets:

```python
def log_safe_model_dump(settings: Settings) -> dict:
    """Return a model dump with sensitive fields redacted."""
    sensitive_fields = {
        "openai_api_key",
        "anthropic_api_key",
        "langchain_api_key",
    }
    dump = settings.model_dump()
    for field in sensitive_fields:
        if field in dump and dump[field] is not None:
            dump[field] = "***REDACTED***"
    return dump

# At startup:
logger.info("saathi starting", config=log_safe_model_dump(settings))
```

---

## 16.12 Configuration in Docker and Container Environments

When running saathi in Docker (for example, as part of a CI pipeline or a remote development environment), environment variables are the natural configuration mechanism.

### Docker `--env` and `--env-file`

```bash
# Pass a single variable:
docker run --env SAATHI_MODEL=llama3.2:3b saathi-image saathi --print "Hello"

# Pass a file of variables (same format as .env):
docker run --env-file .env saathi-image saathi --print "Hello"
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  saathi:
    build: .
    environment:
      - SAATHI_MODEL=qwen2.5:14b
      - SAATHI_OLLAMA_BASE_URL=http://ollama:11434
      - SAATHI_MAX_PARALLEL_TOOLS=4
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

volumes:
  ollama-data:
```

Note `SAATHI_OLLAMA_BASE_URL=http://ollama:11434`—in Docker Compose, services communicate via their service names, not `localhost`. This is an example of why config must be external to code: the URL differs between a developer's laptop (`http://localhost:11434`) and a Docker Compose deployment (`http://ollama:11434`).

### Kubernetes

In Kubernetes, configuration is provided via `ConfigMap` (for non-sensitive values) and `Secret` (for sensitive values):

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: saathi-config
data:
  SAATHI_MODEL: "qwen2.5:14b"
  SAATHI_MAX_PARALLEL_TOOLS: "4"
  SAATHI_LOG_LEVEL: "INFO"

---
# deployment.yaml (excerpt)
spec:
  containers:
  - name: saathi
    envFrom:
    - configMapRef:
        name: saathi-config
    - secretRef:
        name: saathi-secrets
```

The `Settings` class does not change. The infrastructure changes how env vars are delivered.

---

## 16.13 Testing Configuration End-to-End

Here is the complete test suite for `config.py`:

```python
# tests/test_config.py
"""Tests for saathi configuration (12-factor compliance)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from saathi.config import Settings, get_settings


class TestSettingsDefaults:
    """All defaults should produce a valid Settings instance."""

    def test_creates_with_defaults(self):
        s = Settings()
        assert s.model == "qwen2.5:14b"
        assert s.temperature == 0.7
        assert s.context_window == 8096
        assert s.debug is False

    def test_computed_history_budget(self):
        s = Settings(context_window=8000, history_token_budget_pct=0.75)
        assert s.history_token_budget == 6000

    def test_computed_saathi_dir(self):
        s = Settings(memory_dir=Path(".saathi/memory"))
        assert s.saathi_dir == Path(".saathi")


class TestTemperatureValidator:
    """Temperature should be clamped to [0.0, 2.0]."""

    def test_valid_temperature(self):
        s = Settings(temperature=0.5)
        assert s.temperature == 0.5

    def test_temperature_clamped_high(self):
        s = Settings(temperature=3.0)
        assert s.temperature == 2.0

    def test_temperature_clamped_low(self):
        s = Settings(temperature=-0.5)
        assert s.temperature == 0.0

    def test_temperature_boundary_values(self):
        assert Settings(temperature=0.0).temperature == 0.0
        assert Settings(temperature=2.0).temperature == 2.0

    def test_temperature_invalid_string(self):
        with pytest.raises(ValidationError, match="temperature"):
            Settings(temperature="not-a-number")


class TestEnvVarOverride:
    """Environment variables should override defaults."""

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("SAATHI_MODEL", "llama3.2:3b")
        get_settings.cache_clear()
        s = get_settings()
        assert s.model == "llama3.2:3b"

    def test_debug_override(self, monkeypatch):
        monkeypatch.setenv("SAATHI_DEBUG", "true")
        get_settings.cache_clear()
        s = get_settings()
        assert s.debug is True

    def test_max_parallel_tools_override(self, monkeypatch):
        monkeypatch.setenv("SAATHI_MAX_PARALLEL_TOOLS", "2")
        get_settings.cache_clear()
        s = get_settings()
        assert s.max_parallel_tools == 2

    @pytest.fixture(autouse=True)
    def restore_cache(self):
        yield
        get_settings.cache_clear()


class TestCacheBehavior:
    """The lru_cache should return the same object on repeated calls."""

    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_creates_new_instance(self):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2

    @pytest.fixture(autouse=True)
    def restore_cache(self):
        yield
        get_settings.cache_clear()


class TestPathExpansion:
    """Path settings should expand ~ and env vars."""

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        expanded = str(tmp_path / ".saathi" / "memory")
        monkeypatch.setenv("HOME", str(tmp_path))
        s = Settings(memory_dir="~/.saathi/memory")
        assert "~" not in str(s.memory_dir)

    def test_relative_path(self):
        s = Settings(memory_dir=".saathi/memory")
        assert s.memory_dir == Path(".saathi/memory")
```

These tests provide comprehensive coverage of the configuration system. They run in under a second because they do not invoke Ollama or the filesystem (except `test_tilde_expansion`).

---

## 16.14 Configuration as Documentation

One underappreciated benefit of centralizing configuration in a `Settings` class is that it serves as living documentation. A new contributor to the saathi project can:

1. Open `config.py` and read every setting in one place.
2. See the type and default for each setting.
3. Read the `description` in each `Field(...)` call.
4. Open `.env.example` for a commented version with usage hints.

Without this structure, configuration is scattered: some in `argparse` defaults, some in module-level constants, some in `os.environ.get()` calls sprinkled through the code. Understanding the full configuration surface requires reading every source file.

The 12-Factor methodology, pydantic-settings, and the `SAATHI_` prefix convention together create a configuration system that is:

- **Complete** — every configurable value is in one class.
- **Typed** — each value has a declared Python type.
- **Validated** — invalid values are caught at startup.
- **Documented** — descriptions in Field and .env.example.
- **Overridable** — env vars beat .env beats defaults.
- **Testable** — monkeypatch + cache_clear makes testing easy.
- **Secure** — secrets in env vars, never in code.

This is the configuration standard for modern Python applications.

---

## Summary

- The 12-Factor App's Factor III says: store config in the environment. This is especially important for AI apps that need different models, windows, and concurrency limits per deploy.
- `pydantic-settings` implements this cleanly: define a `BaseSettings` class, and field values are automatically read from environment variables with type coercion and validation.
- The `SAATHI_` prefix namespaces all saathi env vars, preventing collisions.
- `.env` files hold developer-local settings and must be gitignored. `.env.example` is the committed, documented counterpart.
- `@computed_field` handles derived settings (e.g., `history_token_budget`). `@field_validator` handles validation and normalization (e.g., temperature clamping).
- In tests, `monkeypatch.setenv` + `get_settings.cache_clear()` lets you test different configurations in isolation.
- Secrets (API keys) belong in env vars, never in code. Use a secrets manager for production.

The full settings reference is in the table in §16.9. The canonical source is `src/saathi/config.py`.
