# Chapter 12 — History Compaction: Managing the Context Window

> "Every message costs tokens. At some point, you have to forget — but do it wisely."

---

## Overview

An AI agent lives inside a context window. Every message you send, every tool call
the model makes, every file it reads — all of it occupies token space. At first this
is irrelevant: a session that's been running for two minutes has maybe a few hundred
tokens of history. But agentic coding sessions are long. They span hours. They involve
reading large files, executing shell commands with verbose output, and iterating on
code across dozens of turns. Before long, the context window fills up.

This chapter explains exactly what happens when it does, how saathi estimates and
tracks token usage without running a full tokenizer, and the compaction strategy that
lets a session continue indefinitely without losing the accumulated understanding of
what has been done.

---

## 12.1 The Context Window Problem

Every LLM processes a fixed-length sequence of tokens. The saathi default model,
`gemma4:12b`, is configured with a 32,768-token context window
(`SAATHI_CONTEXT_WINDOW=32768`). That sounds generous until you account for everything
that has to fit inside it:

- The system prompt (saathi's instructions, tool schemas, the project's `SAATHI.md`)
- The entire conversation history: every human turn, every AI response, every tool
  call, and every tool result
- The current user message
- The model's response tokens (configured with `SAATHI_MAX_TOKENS=4096`)

32,768 tokens is roughly 24,000 words at 1.4 tokens/word, or about 96,000 characters
at the rough 4 chars/token rule. That sounds like a lot, but consider a typical agentic
turn: the user asks the model to read a source file (4,000 characters), run a shell
command (1,000 characters of output), and write an updated file (5,000 characters). That
single turn consumes roughly 2,500 tokens of history. After twelve such turns you have
used 30,000 tokens — nearly the entire window — and you have not even counted the system
prompt or tool schemas.

### What Happens When You Exceed the Context Window

When saathi sends a message list that exceeds the context window, Ollama quietly truncates
the input. It does not raise an error. It does not warn you. It simply drops messages from
the beginning of the history until the sequence fits. The model receives a truncated view
of the conversation and does not know what it is missing.

This produces a cascade of quality failures:

**Amnesia.** The model forgets the early context: the user's original goal, the
architecture decisions made in earlier turns, the constraints discovered by reading
configuration files. It answers questions as if starting fresh.

**Orphaned references.** The model may reference a variable, file, or decision that was
discussed in a message that was truncated. The response mentions "as we established
earlier..." but the earlier message no longer exists in its context.

**Tool call inconsistency.** If a tool call and its result span the truncation boundary —
the AIMessage that invoked the tool is kept but the ToolMessage result is dropped — the
model enters an illegal state. LangChain's graph will raise a validation error or the
model will hallucinate the tool result.

**Silent degradation.** Because Ollama does not announce truncation, the user does not
know why responses suddenly become inconsistent. The session looks fine from the outside
while the model is operating with a broken view of the conversation.

Saathi prevents all of this through proactive compaction: before each turn, it checks
whether the history is approaching the budget, and if so, compresses it automatically.

---

## 12.2 Token Estimation

Precise token counting requires running the model's tokenizer, which means a subprocess
call or a library import that ties saathi to a specific tokenizer. That is too heavy a
dependency for a budget check that runs before every turn.

Instead, saathi uses a simple heuristic: **one token per four characters**. This is the
rough average for English prose processed by BPE tokenizers like those used in GPT-4 and
most instruction-tuned models including Gemma. Code tokenizes slightly differently (more
tokens per character because of punctuation), but the 4:1 ratio is a conservative
enough estimate that the budget check triggers well before the actual limit.

The implementation lives in `src/saathi/compaction.py`:

```python
_CHARS_PER_TOKEN = 4


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars per token) across message contents."""
    return sum(len(_text(m)) for m in messages) // _CHARS_PER_TOKEN
```

The function iterates over all messages, extracts their text content, sums the character
counts, and divides by four. The `_text` helper normalizes content that might be a list
of content blocks (as returned by some multimodal models) to a plain string before
measuring.

This approach has known imprecision. A message full of Chinese characters (one character
per token) will be underestimated. A message full of single-character tokens like
mathematical operators will be overestimated. For the purpose of triggering compaction
before hitting the limit, a ±30% error is entirely acceptable — the budget threshold
compensates for it.

### The history_token_budget Property

The check is not made against the full context window. It is made against a budget that
reserves headroom for the current user message and the model's response. This budget is
defined in `src/saathi/config.py`:

```python
class Settings(BaseSettings):
    context_window: int = 32768
    max_tokens: int = 4096

    @property
    def history_token_budget(self) -> int:
        return int(self.context_window * 0.75)
```

With the default `context_window` of 32,768, `history_token_budget` returns 24,576
tokens. That is the maximum size the conversation history is allowed to reach before
compaction is triggered.

The check is performed in `needs_compaction`:

```python
def needs_compaction(messages: list[BaseMessage], budget_tokens: int) -> bool:
    return estimate_tokens(messages) > budget_tokens
```

And called in `cli.py` before every turn:

```python
if needs_compaction(messages, settings.history_token_budget):
    await do_compact(auto=True)
```

---

## 12.3 Why 75%?

The 75% threshold is not arbitrary. It comes from a careful accounting of what has to
fit in the remaining 25%.

A 32,768-token context window with 75% reserved for history leaves 8,192 tokens of
headroom. That headroom has to accommodate:

**The system prompt.** Saathi's system prompt includes the agent's personality, tool
schemas for all enabled tools, the project's `SAATHI.md` file (if present), any
`CLAUDE.md` or `SAATHI.md` instructions discovered by `instructions_source()`, and the
mode-specific instructions. A fully configured saathi instance with a well-documented
project and a dozen MCP tools can have a system prompt of 2,000–4,000 tokens.

**The current user message.** The user might paste a large file, a long error log, or a
multi-paragraph specification. 2,000 tokens is a reasonable upper bound for an unusually
large user message, though in practice they are much shorter.

**The model's response.** `max_tokens` is set to 4,096 by default. The model should
have enough headroom to produce a complete, detailed response without truncation.

**Tokenizer imprecision.** The 4 chars/token estimate can be off by ±30%. A 24,576-token
estimate for history might actually be 31,000 real tokens. The 25% headroom absorbs this.

Putting it together: 4,000 (system) + 2,000 (user) + 4,096 (response) + 20% safety
margin = roughly 8,000 tokens. The 25% headroom on a 32,768-token window is 8,192
tokens. The two numbers agree closely enough to justify the threshold.

The threshold can be tuned by setting `SAATHI_CONTEXT_WINDOW` to match your model's
actual context window. If you are using `llama3.1:70b` with a 128k context window, set
`SAATHI_CONTEXT_WINDOW=131072` and the budget scales accordingly.

---

## 12.4 The Compaction Strategy

When the history exceeds the budget, saathi does not truncate and does not crash. It
compacts: it summarizes the older portion of the history with the LLM, replaces those
messages with a single summary, and keeps the most recent turns verbatim.

The core logic is `compact_messages` in `src/saathi/compaction.py`:

```python
async def compact_messages(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
) -> list[BaseMessage]:
    """Summarize all but the last ``keep_turns`` turns into one summary message.

    Returns ``[summary, *recent]``, or the original list **unchanged** (same
    object) when there is not enough history to compact.
    """
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=_SUMMARY_INSTRUCTIONS),
            HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
        ]
    )
    summary_text = _text(response) if isinstance(response, BaseMessage) else str(response)
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{summary_text}")
    return [summary, *recent]
```

The function:

1. Calls `split_for_compaction` to divide the history into `older` (to be summarized)
   and `recent` (to keep verbatim).
2. If there is not enough history to compact (`split` returns `None`), returns the
   original list object unchanged — the `is messages` identity check in `do_compact`
   detects this case.
3. Formats the older turns as a transcript with type-tagged lines
   (`Human: ...`, `AI: ...`, `Tool: ...`).
4. Calls the LLM with a summary prompt to produce a concise description of what happened.
5. Wraps the summary in a `SystemMessage` with the `_SUMMARY_PREFIX` sentinel and
   prepends it to the recent messages.

The result is a list that starts with one `SystemMessage` (the summary) followed by the
last three user turns (by default) and their associated AI responses and tool calls.

### The keep_turns Default

The default `keep_turns=3` is a balance between two competing concerns. More kept turns
means better immediate context for the model at the cost of less compression. Fewer kept
turns means more aggressive compression at the risk of losing very recent context.

Three turns works well in practice because:

- The current turn plus the two preceding turns give the model enough context to
  understand the immediate thread of work.
- The summary covers everything before that, capturing the goals, decisions, and findings
  from the entire session.
- Three turns is rarely enough to overflow the budget on its own, so a freshly compacted
  history has plenty of room to grow before the next compaction is needed.

---

## 12.5 `split_for_compaction` — Splitting at User-Turn Boundaries

The split between old and new history cannot be made at an arbitrary message index.
Specifically, it must not split a tool call from its result.

Consider this sequence:

```text
[0] HumanMessage("read config.py")
[1] AIMessage(tool_calls=[{name: "read_file", id: "t1"}])
[2] ToolMessage(content="...", tool_call_id="t1")
[3] AIMessage(content="The config sets...")
[4] HumanMessage("now change the timeout")
...
```

If we cut at index 2, the `recent` slice starts with `ToolMessage`. That is an orphaned
result: the AI message that requested the tool call (index 1) has been summarized away.
LangChain's message validation will reject this sequence, and even if it didn't, the
model would be confused: it sees a tool result with no preceding request.

The safe cut point is always at a `HumanMessage` boundary. Every valid conversation
segment starts with a human turn. The function enforces this:

```python
def split_for_compaction(
    messages: list[BaseMessage], keep_turns: int
) -> tuple[list[BaseMessage], list[BaseMessage]] | None:
    """Split into ``(older, recent)`` at a user-turn boundary.

    Keeps the last ``keep_turns`` user turns (and everything after them) intact.
    Returns ``None`` when there aren't more than ``keep_turns`` turns — i.e. there
    is nothing worth compacting yet.
    """
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= keep_turns:
        return None
    cut = human_idxs[-keep_turns]
    return messages[:cut], messages[cut:]
```

The logic:

1. Collect the indices of all `HumanMessage` objects in the list. These are the turn
   boundaries.
2. If there are `keep_turns` or fewer human turns total, there is nothing to compact —
   keeping the last `keep_turns` turns means keeping everything. Return `None`.
3. Otherwise, find the index of the `(-keep_turns)`-th human message (counting from the
   end). That is the cut point.
4. Return `(messages[:cut], messages[cut:])`.

The `recent` slice always starts at a `HumanMessage`. The `older` slice always ends just
before a `HumanMessage`. No tool call/result pair is ever split.

### Why Not Split on AIMessage?

Some approaches to context compression split on assistant turns: keep the last N
assistant responses. This is problematic for agentic conversations because assistant
turns come in pairs: one `AIMessage` with `tool_calls` (requesting a tool) and one
`AIMessage` with `content` (delivering the final answer for that turn). Between them are
one or more `ToolMessage` objects. Splitting on assistant turn boundaries requires
correctly identifying which AIMessages are "final" versus which are "intermediate tool
callers" — a non-trivial classification that depends on whether the message has content
versus tool calls.

Splitting on `HumanMessage` boundaries is unambiguous. There is exactly one human message
per user turn. Everything between two consecutive human messages — all the tool calls,
tool results, and intermediate AI messages — forms a coherent unit that must not be split.

---

## 12.6 LLM Summarization

The summary is generated by invoking the same LLM that the agent uses for its normal
conversation. This is a deliberate choice: the summarizer understands the domain, the
tools, and the codebase in the same way the agent does.

The summarization prompt is a constant in `compaction.py`:

```python
_SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context window. "
    "Write a concise summary capturing: the user's goals, key decisions, files "
    "read or modified, important findings, and any unresolved threads. Preserve "
    "concrete details a developer would need to continue. Output only the summary."
)
```

This prompt has several important properties:

**Domain specificity.** It tells the model it is compacting a coding assistant
conversation, so the model knows to emphasize files, code changes, and technical
decisions.

**Information hierarchy.** It explicitly lists what to preserve: goals, decisions, files
modified, findings, unresolved threads. A generic summarization prompt would produce
a vague overview. This prompt produces a structured, actionable summary.

**Output discipline.** "Output only the summary" prevents the model from wrapping the
summary in preamble like "Here is a concise summary of the conversation:" — which wastes
tokens and adds noise.

The transcript fed to the summarizer uses human-readable prefixes:

```python
transcript = "\n".join(
    f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
)
```

For a message sequence, this produces:

```text
Human: explain the compaction module
AI: tool_calls=[read_file(path='src/saathi/compaction.py')]
Tool: """Conversation history compaction..."""
AI: The compaction module lives in src/saathi/compaction.py. It defines...
Human: how does split_for_compaction work?
AI: split_for_compaction takes...
```

The class name manipulation (`replace('Message', '')`) strips the `Message` suffix so
the labels are `Human`, `AI`, `Tool`, and `System` — readable without being verbose.

### The Summary as a SystemMessage

The summary is wrapped in a `SystemMessage`, not a `HumanMessage` or `AIMessage`:

```python
_SUMMARY_PREFIX = "Summary of earlier conversation:"

summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{summary_text}")
return [summary, *recent]
```

`SystemMessage` is the right container for several reasons:

**It is not attributed to either party.** A `HumanMessage` would imply the user said it.
An `AIMessage` would imply the model generated it as part of its response. A
`SystemMessage` sits above the conversation as background knowledge.

**Most LLMs respect system message content.** The model is trained to treat system
messages as ground truth. Putting the summary in a system message biases the model to
trust it as accurate context rather than treating it as something a human or assistant
claimed.

**It is visually distinct in debug output.** When inspecting conversation history during
debugging, a `SystemMessage` with `_SUMMARY_PREFIX` is immediately recognizable as a
compaction artifact.

The `_SUMMARY_PREFIX` sentinel also serves a practical purpose: tests can check
`result[0].content.startswith(_SUMMARY_PREFIX)` to verify that compaction produced the
expected output without depending on the specific summary text.

---

## 12.7 Why a Fresh Thread ID?

After compaction, the session continues on a new `thread_id`:

```python
async def do_compact(*, auto: bool) -> None:
    nonlocal messages, config
    before = estimate_tokens(messages)
    try:
        compacted = await compact_messages(summarizer, messages)
    except Exception as exc:
        log.warning("compaction_failed", error=str(exc))
        if not auto:
            console.print(f"[yellow]Compaction failed:[/yellow] {exc}")
        return
    if compacted is messages:  # not enough history to compact
        if not auto:
            console.print("[dim]Not enough history to compact yet.[/dim]")
        return
    messages = compacted
    state.session_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": state.session_id}}
    after = estimate_tokens(messages)
    label = "auto-compacted" if auto else "compacted"
    log.info("history_compacted", before=before, after=after, auto=auto)
    console.print(f"[dim]↯ {label} history: ~{before:,} → ~{after:,} tokens[/dim]")
```

The new `thread_id` is critical. To understand why, you need to understand how
LangGraph's checkpointer works.

### The Checkpointer's Append-Only Semantics

LangGraph persists conversation state in a SQLite database via the `SqliteSaver`
checkpointer. Each agent invocation appends new state to the checkpoint store under
the current `thread_id`. The `add_messages` reducer is how messages accumulate: each
turn adds new messages to the state without replacing the old ones.

This design is intentional. It enables `/rollback`, which works by restoring an earlier
checkpoint for the current `thread_id`. The checkpoint store is the ground truth for the
conversation history.

But it creates a fundamental problem for compaction. If we want to replace the old
messages with `[summary, *recent]`, we cannot do so within the current `thread_id`.
The `add_messages` reducer only knows how to add; it does not know how to delete or
replace. Even if we tried to update the state directly, every past checkpoint would still
contain the full history, and a `/rollback` would restore it.

The only clean solution is to start a new `thread_id` whose initial state is the
compacted message list. The new thread has no history before the compaction point. The
old thread remains intact in the database but is no longer the active thread.

### The Old Thread Is Not Deleted

The old thread is never deleted from the SQLite database. This is a deliberate choice:

1. It allows emergency recovery. If the summary loses crucial information, a developer
   can examine the SQLite database directly and find the full old history under the
   previous `thread_id` (which is logged at `info` level as `history_compacted`).

2. It avoids write-heavy database operations. Deleting checkpoints requires careful
   transaction management to avoid partial state corruption.

3. The database is on the user's local machine and is small. A few extra checkpoints
   for old threads do not matter.

---

## 12.8 Trade-off: Rollback Cannot Cross a Compaction Boundary

The new `thread_id` means that `/rollback` can only undo turns within the current
thread. It cannot restore the state to before the compaction happened.

This is an explicit, documented trade-off. When a user runs `/rollback` after a
compaction, saathi shows them the checkpoints of the current (post-compaction) thread.
The oldest visible checkpoint is the first turn after compaction, not the very beginning
of the session.

Why is this acceptable?

**Compaction is rare.** In normal use, compaction happens once per session at most —
usually toward the end of a very long session. The user is unlikely to want to roll back
past two hours of work to the beginning.

**The summary captures the gist.** Even if a specific early decision is lost from the
active context, the compaction summary records it. The user can ask "what did we decide
about X?" and the model will answer from the summary.

**The alternative is worse.** The alternative to starting a new thread is to store the
compacted message list in some parallel structure and manage its interaction with the
checkpointer manually. That complexity is not worth the marginal benefit of enabling
rollback past a compaction boundary.

**It is transparent.** The user sees the compaction message (`↯ compacted history:
~X → ~Y tokens`) and knows a boundary has been crossed. Rollback attempts after that
will correctly show the post-compaction checkpoints.

---

## 12.9 Auto-Compaction

Compaction is transparent in normal use. Before every agent turn, `execute_task` checks
whether compaction is needed:

```python
async def execute_task(task: str) -> None:
    """Run one agent turn for a task and record its result + post_turn hooks."""
    nonlocal messages
    if needs_compaction(messages, settings.history_token_budget):
        await do_compact(auto=True)
    clear_turn_snapshots()

    final_answer, messages = await _run_turn(graph, config, task, state, messages)
    ...
```

The check runs on every turn. When `needs_compaction` returns `True`, `do_compact` is
called with `auto=True`. The `auto` flag controls two things:

1. **Failure handling.** When `auto=True` and compaction fails (e.g., the LLM call
   errors out), the error is logged but not shown to the user — the session continues
   with the original messages. When `auto=False` (i.e., the user ran `/compact`
   explicitly), the failure is shown as a yellow warning.

2. **The console message.** Auto-compaction shows `↯ auto-compacted history: ...`.
   Manual compaction shows `↯ compacted history: ...`.

The `auto=True` path is designed to be invisible when everything works and silent when
it fails. The user should never have their session interrupted by compaction mechanics.

### The Compaction Sequence

When auto-compaction triggers, the sequence is:

1. `needs_compaction(messages, settings.history_token_budget)` returns `True`.
2. `do_compact(auto=True)` is called.
3. `compact_messages(summarizer, messages)` is called. This involves an LLM invocation
   (typically 2–5 seconds for a local model).
4. The local `messages` list and `config` are updated in place (via `nonlocal`).
5. Control returns to `execute_task`.
6. `_run_turn` is called with the compacted messages. The user's task is executed
   against the compacted context.

The user sees the compaction message, then a brief additional pause for the actual task,
then the response. In practice the compaction is not noticeable because the LLM call
overlaps with the time it takes the user to read the compaction notice.

---

## 12.10 `/compact` — Manual Compaction

The `/compact` slash command gives the user explicit control over compaction timing:

```python
if cmd == "compact":
    await do_compact(auto=False)
    continue
```

Manual compaction is useful in several scenarios:

**Preemptive compaction.** The user knows they are about to start a long sub-task
(e.g., a refactor that will touch many files) and wants to free up as much context
as possible before starting. Running `/compact` before the sub-task gives the session
the maximum possible runway.

**Quality reset.** After a long meandering session with many false starts, the history
may be cluttered with dead-end explorations. Running `/compact` distills it down to
the actionable context.

**Debugging compaction.** The `/compact` command shows the before/after token counts
and any failure messages. It is the easiest way to verify that compaction is working
correctly.

When the user runs `/compact` and there is not enough history to compact (fewer than
`keep_turns + 1` user turns), saathi shows:

```text
Not enough history to compact yet.
```

This comes from the identity check in `do_compact`:

```python
if compacted is messages:  # not enough history to compact
    if not auto:
        console.print("[dim]Not enough history to compact yet.[/dim]")
    return
```

`compact_messages` returns the same list object (`return messages`) when
`split_for_compaction` returns `None`. The identity check (`is`) rather than equality
check (`==`) is intentional: it avoids comparing potentially large message lists and
catches the "no-op" case exactly.

---

## 12.11 `needs_compaction` — The Check

The full `needs_compaction` function is simple but central:

```python
def needs_compaction(messages: list[BaseMessage], budget_tokens: int) -> bool:
    return estimate_tokens(messages) > budget_tokens
```

It is a single comparison. The simplicity is intentional: this function is called on
every turn, and it should be as fast as possible. The `estimate_tokens` function itself
is O(N) in the total character count of all messages — linear in the data it has to
process, with no sorting, hashing, or allocation beyond the sum.

The `budget_tokens` parameter comes from `settings.history_token_budget`, which is
computed as a property of the `Settings` object:

```python
@property
def history_token_budget(self) -> int:
    return int(self.context_window * 0.75)
```

For a 32,768-token context window, this returns 24,576. This value is recomputed on each
call (property, not a cached attribute) so it reflects any runtime changes to
`settings.context_window` — useful for testing.

In the `execute_task` call site, the check is:

```python
if needs_compaction(messages, settings.history_token_budget):
    await do_compact(auto=True)
```

Note that the check uses the current `messages` list, not the list that will be sent to
the model. The list sent to the model includes the current user message
(appended in `_run_turn`) and the system prompt (embedded in the graph's LLM
configuration). The check therefore fires somewhat early — when the history alone exceeds
75% of the window — which is exactly the desired behavior, because we want to compact
before the margin disappears.

---

## 12.12 Testing Compaction

The test suite for compaction lives in `tests/test_compaction.py`. It covers seven
scenarios systematically.

### The Test Conversation Fixture

All tests that need a representative conversation use a shared fixture:

```python
def _conversation() -> list[BaseMessage]:
    """4 user turns; turns 1 and 3 include a tool-call / tool-result pair."""
    return [
        HumanMessage(content="turn 1"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "t1"}]),
        ToolMessage(content="file contents", tool_call_id="t1", name="read_file"),
        AIMessage(content="answer 1"),
        HumanMessage(content="turn 2"),
        AIMessage(content="answer 2"),
        HumanMessage(content="turn 3"),
        AIMessage(content="", tool_calls=[{"name": "run_bash", "args": {}, "id": "t3"}]),
        ToolMessage(content="cmd output", tool_call_id="t3", name="run_bash"),
        AIMessage(content="answer 3"),
        HumanMessage(content="turn 4"),
        AIMessage(content="answer 4"),
    ]
```

This fixture has four user turns with tool-call/result pairs interspersed. It is
specifically designed to test that the split respects tool-call boundaries: turns 1 and 3
each have an intermediate tool call that must not be orphaned.

### Token Estimation

```python
def test_estimate_tokens() -> None:
    msgs = [HumanMessage(content="a" * 40)]  # 40 chars / 4 = 10
    assert estimate_tokens(msgs) == 10
```

40 characters divided by 4 gives exactly 10 tokens. This is the simplest possible
verification of the formula. The test uses a single message to eliminate any ambiguity
about how multiple messages interact.

### Needs Compaction Threshold

```python
def test_needs_compaction() -> None:
    msgs = [HumanMessage(content="x" * 400)]  # ~100 tokens
    assert needs_compaction(msgs, budget_tokens=50) is True
    assert needs_compaction(msgs, budget_tokens=500) is False
```

400 characters / 4 = 100 tokens. The test verifies both sides of the threshold:
a 100-token history should exceed a 50-token budget, and not exceed a 500-token budget.

### No-Op When Too Few Turns

```python
def test_split_returns_none_when_too_few_turns() -> None:
    msgs = _conversation()  # 4 turns
    assert split_for_compaction(msgs, keep_turns=4) is None
    assert split_for_compaction(msgs, keep_turns=5) is None
```

With 4 user turns and `keep_turns=4`, there is nothing to split — keeping 4 turns means
keeping all of them. The function returns `None`. With `keep_turns=5`, the same: asking
to keep more turns than exist is also a no-op.

### Splitting at User-Turn Boundary

```python
def test_split_cuts_at_turn_boundary() -> None:
    msgs = _conversation()
    split = split_for_compaction(msgs, keep_turns=2)
    assert split is not None
    older, recent = split
    # recent keeps the last 2 turns and must start with a HumanMessage
    assert isinstance(recent[0], HumanMessage)
    assert recent[0].content == "turn 3"
    assert older[-1].content == "answer 2"
```

With 4 turns and `keep_turns=2`, the split keeps turns 3 and 4. The `recent` slice
starts at "turn 3" (a `HumanMessage`) and the `older` slice ends at "answer 2" (an
`AIMessage`). The tool-call/result pair in turn 3 is entirely within the `recent` slice
and is not orphaned.

### No LLM Call When Nothing to Compact

```python
async def test_compact_returns_unchanged_when_too_few_turns() -> None:
    msgs = _conversation()
    llm = FakeLLM()
    result = await compact_messages(llm, msgs, keep_turns=4)
    assert result is msgs  # same object, no LLM call
    assert llm.calls == []
```

When `keep_turns` equals the number of turns, `compact_messages` should return the
original list without calling the LLM. The identity check (`is`) is critical: it confirms
that no new list was constructed, which would have required copying all messages.

### Summary Format and Recent Tail

```python
async def test_compact_summarizes_and_keeps_recent() -> None:
    msgs = _conversation()
    llm = FakeLLM(summary="condensed history")
    result = await compact_messages(llm, msgs, keep_turns=2)

    # summary first, as a SystemMessage carrying the model's text
    assert isinstance(result[0], SystemMessage)
    assert result[0].content.startswith(_SUMMARY_PREFIX)
    assert "condensed history" in result[0].content

    # the retained tail is exactly the last 2 turns, starting on a HumanMessage
    assert isinstance(result[1], HumanMessage)
    assert result[1].content == "turn 3"
    assert [m.content for m in result[-2:]] == ["turn 4", "answer 4"]

    # no orphaned ToolMessage leads the retained tail
    assert not isinstance(result[1], ToolMessage)
    assert llm.calls, "summarizer should have been invoked once"
```

This is the most comprehensive test. It verifies:

- The first message is a `SystemMessage` with the `_SUMMARY_PREFIX`.
- The summary text is present in the `SystemMessage`.
- The retained tail starts with the correct `HumanMessage`.
- The last two messages are the last turn's human and AI messages.
- No `ToolMessage` orphan leads the retained tail.
- The LLM was actually called (not a no-op).

### Token Reduction

```python
async def test_compact_shrinks_token_estimate() -> None:
    msgs = [
        HumanMessage(content="x" * 400),
        AIMessage(content="y" * 400),
        HumanMessage(content="turn 2 " * 10),
        AIMessage(content="answer 2"),
        HumanMessage(content="turn 3"),
        AIMessage(content="answer 3"),
        HumanMessage(content="turn 4"),
        AIMessage(content="answer 4"),
    ]
    before = estimate_tokens(msgs)
    result = await compact_messages(FakeLLM(summary="short"), msgs, keep_turns=2)
    assert estimate_tokens(result) < before
```

The first two messages are large (400 chars each). The `FakeLLM` returns "short" as the
summary. After compaction, the large messages are replaced by the short summary and the
recent turns. The estimated token count should be significantly smaller.

---

## 12.13 The Fake LLM in Tests

All `compact_messages` tests use a `FakeLLM` — a minimal test double that does not
require a running Ollama instance:

```python
class FakeLLM:
    """Minimal async LLM stub that returns a fixed summary."""

    def __init__(self, summary: str = "THE SUMMARY") -> None:
        self.summary = summary
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content=self.summary)
```

The design follows the minimal interface principle: `compact_messages` needs an object
with an `ainvoke` method that returns something with a `.content` attribute. `FakeLLM`
provides exactly that and nothing more.

The `calls` list is a built-in call recorder. Tests can assert:

- `llm.calls == []` — the LLM was not invoked.
- `llm.calls` — the LLM was invoked at least once.
- `len(llm.calls) == 1` — the LLM was invoked exactly once.
- `llm.calls[0]` — inspect the exact messages sent to the summarizer.

The `summary` parameter lets each test control what the fake LLM returns:

```python
# Verify summary text appears in the output
llm = FakeLLM(summary="condensed history")
result = await compact_messages(llm, msgs, keep_turns=2)
assert "condensed history" in result[0].content

# Verify token reduction with a short summary
result = await compact_messages(FakeLLM(summary="short"), msgs, keep_turns=2)
assert estimate_tokens(result) < before
```

Note that `FakeLLM` does not inherit from `BaseChatModel` or any LangChain base class.
It is a duck-typed stub. `compact_messages` accepts a `LanguageModelLike` type hint —
which is a protocol type in LangChain, not a concrete class — so any object with the
right interface works. This makes test setup minimal and fast.

### Extending the Fake LLM for Richer Tests

For tests that need more sophisticated behavior, `FakeLLM` can be subclassed:

```python
class CountingFakeLLM(FakeLLM):
    """Track which messages were passed to the summarizer."""

    async def ainvoke(self, messages, config=None):
        # Save a copy of the messages for inspection
        self.calls.append(list(messages))
        return await super().ainvoke(messages, config)


def test_summarizer_sees_correct_transcript():
    """The summarizer should see the older turns, not the recent ones."""
    msgs = _conversation()
    llm = CountingFakeLLM(summary="summary")
    asyncio.run(compact_messages(llm, msgs, keep_turns=2))

    # The summarizer received a SystemMessage (instructions) and a HumanMessage (transcript)
    assert len(llm.calls[0]) == 2
    assert isinstance(llm.calls[0][0], SystemMessage)
    assert isinstance(llm.calls[0][1], HumanMessage)
    transcript = llm.calls[0][1].content
    assert "turn 1" in transcript
    assert "turn 2" in transcript
    # Recent turns should NOT be in the transcript sent to the summarizer
    assert "turn 3" not in transcript
    assert "turn 4" not in transcript
```

This pattern — calling `asyncio.run` on an async test in a synchronous test function —
works for simple cases. For the full test suite, saathi uses `pytest-anyio` so test
functions can be `async def` directly.

---

## 12.13b Deep Dive: What the Summarizer Actually Receives

It is worth examining the exact prompt the LLM receives during compaction, because the
quality of the summary depends entirely on the quality of the input.

Consider the 4-turn test fixture from `test_compaction.py` with `keep_turns=2`. The
`older` slice is:

```text
HumanMessage("turn 1")
AIMessage(content="", tool_calls=[{name: "read_file", args: {}, id: "t1"}])
ToolMessage(content="file contents", tool_call_id="t1", name="read_file")
AIMessage(content="answer 1")
HumanMessage("turn 2")
AIMessage(content="answer 2")
```

The `_text` function applied to each message:

```python
transcript = "\n".join(
    f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
)
```

Produces the transcript string:

```text
Human: turn 1
AI: [{'name': 'read_file', 'args': {}, 'id': 't1', 'type': 'tool_call'}]
Tool: file contents
AI: answer 1
Human: turn 2
AI: answer 2
```

The second line reveals something important: when an `AIMessage` has only `tool_calls`
and no textual content, `_text` falls back to `str(content)` where `content` is an
empty string `""` — but wait, let us trace through the actual code more carefully.

The `_text` helper is:

```python
def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)
```

For `AIMessage(content="", tool_calls=[...])`, `message.content` is `""` — an empty
string. So `_text` returns `""`. The transcript line becomes:

```text
AI: 
```

Not very informative for the summarizer. The tool_calls are not surfaced in the
transcript. This is a known limitation: the summarizer sees an empty AI turn followed
by a Tool result, which may be confusing without the tool call context.

A future improvement would be to render tool-calling turns more explicitly:

```python
def _render_message(m: BaseMessage) -> str:
    name = m.__class__.__name__.replace("Message", "")
    if isinstance(m, AIMessage) and m.tool_calls:
        calls = ", ".join(f"{tc['name']}({tc.get('args', {})})" for tc in m.tool_calls)
        return f"{name}: [calls: {calls}]"
    return f"{name}: {_text(m)}"
```

This would produce:

```text
AI: [calls: read_file({})]
Tool: file contents
AI: answer 1
```

Which gives the summarizer accurate context about what tools were called. This is a
documented area for improvement in the compaction module.

### The Summarizer Call Stack

The full chain for a compaction:

1. `do_compact` calls `compact_messages(summarizer, messages)`.
2. `compact_messages` calls `split_for_compaction(messages, keep_turns=3)`.
3. If split is possible, formats the transcript and calls:

   ```python
   response = await llm.ainvoke([
       SystemMessage(content=_SUMMARY_INSTRUCTIONS),
       HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
   ])
   ```

4. The `llm` here is `summarizer = make_llm(model_id)` — a fresh LLM instance bound
   to no tools. This is important: we do not want the summarizer to make tool calls.
   It should only read its context and produce a text summary.
5. The response is wrapped in a `SystemMessage` and prepended to the recent tail.
6. `do_compact` swaps the local `messages` reference and generates a new `thread_id`.

The summarizer uses the same model as the agent, but a separate instance. Using the
same model ensures the summarizer understands the domain (code, tools, technical
language). Using a separate instance prevents the summarization prompt from corrupting
the agent's context.

### Summary Quality and Its Impact

The quality of the compaction summary directly affects the agent's performance for the
rest of the session. A good summary:

- States the user's goal ("User wants to refactor the authentication module to use JWT")
- Records files modified ("Modified `auth/token.py` and `auth/middleware.py`")
- Captures decisions ("Decided to use `python-jose` library instead of `authlib`")
- Notes unresolved issues ("TODO: handle token refresh — not implemented yet")

A poor summary:

- Is vague ("Discussed code changes")
- Omits file names ("Read and modified several files")
- Misrepresents decisions ("Considered using JWT")

The `_SUMMARY_INSTRUCTIONS` prompt is carefully worded to elicit good summaries, but
the actual quality depends on the model. Smaller, less capable models may produce
vaguer summaries. This is another argument for using a capable model for compaction
even if you use a smaller model for routine turns.

---

## 12.14 Compaction vs Truncation vs Sliding Window

Three main strategies exist for keeping context within the window. Understanding their
trade-offs explains why compaction was chosen for saathi.

### Strategy 1: Truncation

**How it works.** Drop messages from the beginning of the history until the total fits
within the budget.

**Implementation:**

```python
def truncate_messages(
    messages: list[BaseMessage], budget_tokens: int
) -> list[BaseMessage]:
    while estimate_tokens(messages) > budget_tokens and messages:
        messages = messages[1:]  # drop oldest
    return messages
```

**Advantages.** Trivially simple. Zero LLM overhead. Guaranteed to produce a result
within the budget.

**Disadvantages.** Loses information abruptly and invisibly. The model has no knowledge
that context was dropped. It may confidently answer questions based on the truncated
context without knowing the answer was in a dropped message. More dangerously, truncation
can split tool call/result pairs: if the budget forces dropping everything up to index 2,
and index 2 is a `ToolMessage`, the remaining sequence starts with an orphaned result.

For a coding agent that needs to maintain accurate understanding of the codebase across
a long session, truncation is unacceptably lossy.

### Strategy 2: Sliding Window

**How it works.** Keep the last N complete turns (human message + all associated AI
messages and tool calls) regardless of token count. If a turn exceeds the budget
individually, truncate it.

**Implementation:**

```python
def sliding_window(
    messages: list[BaseMessage], keep_turns: int
) -> list[BaseMessage]:
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= keep_turns:
        return messages
    cut = human_idxs[-keep_turns]
    return messages[cut:]
```

**Advantages.** Preserves recent context perfectly. No LLM overhead. Simple to
implement and reason about.

**Disadvantages.** The dropped messages are simply gone — no summary, no residue. The
model forgets everything before the window. After 10 turns, it has no idea what the
original goal was. For a short conversation this is fine; for a two-hour coding session
it is disastrous. The model will start making suggestions that contradict decisions made
in the early session.

Additionally, a sliding window makes the context window effective size equal to the
window size, not the full context window. This wastes available context for recent turns
that are short.

### Strategy 3: Compaction (Saathi's Approach)

**How it works.** Summarize the older turns using the LLM, keep recent turns verbatim,
start a new thread. The summary preserves the semantic content of the dropped messages.

**Implementation:** As described throughout this chapter.

**Advantages.** Preserves semantic content. The model knows the goals, decisions, and
findings from the early session, even if it does not have the exact text. The session
can continue indefinitely with full contextual awareness.

**Disadvantages.** Requires an LLM call (adds 2–5 seconds of latency). The summary may
lose specific details (exact file contents, precise error messages). Rollback cannot
cross the compaction boundary.

### Why Compaction Is Best for Agents

The key difference between a conversational chatbot and a coding agent is the nature
of context continuity. A chatbot user who sees their first message scroll off the screen
does not care much — they remember what they said. A coding agent that loses the memory
of which files it modified, which bugs it found, and what architectural constraints were
established is actively dangerous. It may re-introduce bugs that were just fixed, re-read
files it already analyzed, or make changes that contradict the established architecture.

Compaction maintains the semantic thread even as the token budget forces compression. The
agent always knows where it is, what it has done, and what remains. The latency cost of
the LLM call is small compared to the cost of an agent that has lost its bearings.

---

## 12.15 Production Considerations

### Compaction Latency

The compaction LLM call typically takes 2–10 seconds depending on the model and the
length of the history being summarized. This adds noticeable latency to the turn that
triggers compaction.

Mitigation strategies:

**Use the same model.** Saathi uses the session's LLM for both the agent and the
summarizer (`summarizer = make_llm(model_id)` in `_interactive_session`). This avoids
the overhead of loading a second model.

**Compact earlier.** The 75% threshold triggers compaction while there is still plenty
of budget left. The compaction summary is typically much shorter than the messages it
replaced, so multiple turns can pass before the next compaction is needed.

**Consider a smaller summarizer model.** For a production deployment, you might use a
smaller, faster model for summarization (e.g., `gemma3:4b` for summaries, `gemma4:12b`
for the main agent). This is a trivial configuration change:

```python
summarizer = make_llm(settings.compact_model or model_id)
```

This feature is not in the current codebase but the architecture supports it.

### Failure Handling

Compaction is best-effort. If the LLM call fails for any reason, `do_compact` logs a
warning and returns without modifying `messages` or `config`:

```python
try:
    compacted = await compact_messages(summarizer, messages)
except Exception as exc:
    log.warning("compaction_failed", error=str(exc))
    if not auto:
        console.print(f"[yellow]Compaction failed:[/yellow] {exc}")
    return
```

The session continues with the original (uncompacted) messages. This means the context
window may overflow on the next turn — but the model handles that by receiving a large
context, which is better than crashing or losing work.

### Monitoring Compaction Events

Saathi logs every compaction event with structlog:

```python
log.info("history_compacted", before=before, after=after, auto=auto)
```

In a production deployment, this log event can be monitored:

```python
# In your log aggregation system, alert if:
# 1. Compaction is happening every turn (context grows faster than compaction shrinks it)
# 2. Compaction latency exceeds 30 seconds (model is overloaded)
# 3. Compaction failure rate exceeds 5% (network or model issues)
```

The `before` and `after` token counts allow you to track the compression ratio over time.
A healthy compaction typically reduces the token count by 70–85% (a 3,000-token summary
replacing 20,000 tokens of history). If the ratio is lower (say 50%), the model is
producing verbose summaries that should be prompted to be more concise.

### Memory for Compaction: What Gets Lost

Compaction is a lossy operation. The summary cannot capture everything in the older
turns. Specifically, things that are typically lost:

**Exact file contents.** If the model read a file early in the session and later modified
it, the summary captures "read and modified `config.py`" but not the original contents.
This is usually fine because the model can re-read the current file if needed.

**Intermediate error messages.** A long debugging session may produce many error traces.
The summary captures "debugged import error in `tools/__init__.py`, resolved by adding
`__all__`" but not the exact traceback.

**Discarded approaches.** Early turns often involve exploring multiple approaches before
settling on one. The summary captures the chosen approach but may not record the rejected
alternatives and why they were rejected. This can cause the model to re-explore the same
dead ends in a very long session.

For most coding sessions, these losses are acceptable. The model can re-read files,
re-run commands, and re-derive conclusions. The summary provides the high-level context
that would be hardest to re-derive: goals, constraints, and decisions.

### Tuning for Your Use Case

The compaction parameters can be tuned via environment variables:

```bash
# Larger model with more context: less frequent compaction
SAATHI_CONTEXT_WINDOW=131072

# More conservative: compact at 60% instead of 75%
# (Requires adding a settings field, currently hardcoded at 75%)

# More recent context preserved after compaction (default: 3 turns)
# (Requires passing keep_turns through settings, currently hardcoded)
```

For future extensibility, the `compact_messages` function accepts `keep_turns` as a
keyword argument. Wiring this to a settings field is a minimal change:

```python
# In config.py:
compact_keep_turns: int = 3

# In cli.py:
compacted = await compact_messages(
    summarizer, messages, keep_turns=settings.compact_keep_turns
)
```

---

## 12.15a The Auto-Compact Flow in Detail

To make the auto-compaction logic concrete, let us trace through a realistic scenario
step by step.

### Scenario Setup

The user has been working with saathi for 45 minutes. The session has accumulated 20
turns. The conversation history contains:

- The early turns: exploring the repository structure, reading several files, discussing
  the architecture. Each turn consumed roughly 1,200 tokens of history.
- The middle turns: iterating on a refactor, reading and writing code files. Roughly
  1,800 tokens per turn.
- The most recent 3 turns: testing the refactor, running commands, reviewing output.

Rough token count: 5 early × 1,200 + 12 middle × 1,800 + 3 recent × 1,400 = 6,000 +
21,600 + 4,200 = 31,800 estimated tokens.

The `history_token_budget` is 24,576 (75% of 32,768). The budget was exceeded sometime
during the middle phase, but compaction was deferred until the next `execute_task` call.

### Step 1: User Types a Message

The user types: "Now let's add unit tests for the refactored module."

The REPL receives this and calls `execute_task("Now let's add unit tests for the
refactored module.")`.

### Step 2: Compaction Check

```python
async def execute_task(task: str) -> None:
    nonlocal messages
    if needs_compaction(messages, settings.history_token_budget):
        await do_compact(auto=True)
```

`needs_compaction(messages, 24576)` is called. `estimate_tokens(messages)` sums all
character counts and divides by 4, returning approximately 31,800. Since 31,800 >
24,576, `needs_compaction` returns `True`.

### Step 3: do_compact Called

```python
async def do_compact(*, auto: bool) -> None:
    nonlocal messages, config
    before = estimate_tokens(messages)  # ~31,800
    try:
        compacted = await compact_messages(summarizer, messages)
    except Exception as exc:
        log.warning("compaction_failed", error=str(exc))
        return  # continue with original messages
```

`before` is recorded (31,800). `compact_messages` is called with the full message list
and the default `keep_turns=3`.

### Step 4: split_for_compaction

Inside `compact_messages`, `split_for_compaction(messages, keep_turns=3)` is called.

The 20 turns have human messages at indices 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
44, 48, 52, 56, 60, 64, 68, 72, 76 (rough indices — each turn varies).

`human_idxs = [0, 4, 8, 12, ..., 72, 76]`  (20 elements)

With `keep_turns=3`, `cut = human_idxs[-3]` — the index of the 18th human message
(turn 18 out of 20). Everything before that index is `older`; everything from there on
is `recent`.

`older` contains turns 1–17 (85% of the history).
`recent` contains turns 18–20 (the last three turns).

### Step 5: Transcript Formatting

The older turns are formatted as a transcript. For a 17-turn session, this transcript
is roughly 25,000 characters of text. The summarizer receives this as a `HumanMessage`.

The LLM call takes approximately 3 seconds (for `gemma4:12b` on a modern GPU). The
spinner is already visible to the user with "thinking…" — the compaction delay is
invisible, absorbed into what looks like normal model latency.

### Step 6: Summary Generation

The model produces a summary. A good summary for 17 turns of coding session might be:

```text
User is refactoring the authentication module (src/auth/) to separate token
generation from middleware validation. Key decisions: use python-jose for JWT
(not authlib — rejected due to maintenance concerns), store token metadata in
Redis (key format: "token:{user_id}:{jti}"), 24-hour expiry. Files modified:
src/auth/token.py (new JWT generation), src/auth/middleware.py (updated to call
verify_token()), src/auth/models.py (added TokenMetadata dataclass). Tests for
token.py are still TODO. Config changes: added REDIS_URL to Settings, added
SAATHI_JWT_SECRET env var requirement.
```

This summary captures all the durable information from 17 turns in approximately 600
characters (~150 tokens).

### Step 7: Compacted Message List

The result is:

```python
[
    SystemMessage("Summary of earlier conversation:\nUser is refactoring..."),
    HumanMessage("turn 18 content"),
    AIMessage("turn 18 response"),
    HumanMessage("turn 19 content"),
    AIMessage("turn 19 response"),
    HumanMessage("turn 20 content"),
    AIMessage("turn 20 response"),
]
```

Seven messages instead of roughly 80. The estimated token count is now approximately
150 (summary) + 3 × 1,400 (recent turns) = 4,350 tokens. Compression ratio: ~87%.

### Step 8: New Thread ID

```python
messages = compacted        # replace local reference
state.session_id = uuid.uuid4().hex    # e.g., "a3f7b2c1..."
config = {"configurable": {"thread_id": state.session_id}}
```

The thread ID changes. The user sees:

```text
↯ auto-compacted history: ~31,800 → ~4,350 tokens
```

### Step 9: The Actual Task

Control returns to `execute_task`. The task ("Now let's add unit tests...") is appended
to the compacted messages and the agent runs with 4,350 + (task tokens) ≈ 4,500 tokens
of context. The model has plenty of room.

Crucially, the model's first context includes the summary. It knows about the
authentication refactor, the file names, the decisions, and the TODO for tests. The
user's request to "add unit tests" lands on a model that understands exactly what needs
to be tested — even though the files that were read and the decisions that were made
all happened in messages that no longer exist as verbatim text.

---

## 12.15b Advanced: Compaction Across Multiple Sessions

The compaction mechanism is designed for a single interactive session, but the
principles extend to multi-session workflows.

### The Session Continuation Problem

When a user closes saathi and reopens it the next day, they can `/session load` to
restore a previous session. The restored session has the full message history from
the previous session, including any messages that were already compacted.

If the restored session is large (close to the budget already), the very first auto-
compaction of the new session will trigger immediately. The user may be surprised to
see compaction happen before they even type their first message.

The solution is pre-emptive compaction at session save time:

```python
# Future enhancement: compact before saving so the restored session is lean
async def save_and_compact(session_mgr, state, messages, summarizer):
    if needs_compaction(messages, settings.history_token_budget // 2):
        # Use half the budget as the threshold for pre-save compaction
        # so the saved session has plenty of room to grow
        compacted = await compact_messages(summarizer, messages)
        if compacted is not messages:
            messages = compacted
    session_mgr.save(state, messages)
    return messages
```

This is not in the current codebase but is a natural evolution.

### Compaction in Non-Interactive Mode (`--print`)

The `--print` mode runs a single task and exits. It does not do compaction: the session
is too short to need it, and the overhead of starting a fresh thread would exceed the
benefit.

```python
async def _print_mode(model_id, context_paths, task, output_format):
    # Note: no compaction logic here
    # Single turn, single session, no persistent history
    graph = await build_graph([*ALL_TOOLS, *mcp_tools], memory_store, model_id, ...)
    result = await graph.ainvoke(input_state, config)
```

If you are using `--print` for pipeline scripts that might pass very large inputs, be
mindful of the context window. A single message with 20,000 characters already uses
5,000 tokens — 15% of a 32k window. Adding tool results and the response, and you
could hit the limit in a single turn. In that case, consider using the interactive mode
with manual compaction, or breaking the task into smaller sub-tasks.

### Compaction and the LangGraph Checkpointer

After compaction, the new thread_id has no history other than the compacted messages.
When LangGraph persists the first turn on the new thread, it creates a fresh set of
checkpoints. The SQLite database now contains:

```text
thread_id_1 → [checkpoint_0, checkpoint_1, ..., checkpoint_N]  (old thread)
thread_id_2 → [checkpoint_0, checkpoint_1, ...]  (new thread, post-compaction)
```

The `state.session_id` tracks the current thread. `config["configurable"]["thread_id"]`
is always set to `state.session_id`. The graph uses this to load the correct checkpoint
on each invocation.

`/checkpoints` (the slash command) shows only the checkpoints of the current
thread:

```python
await handle_checkpoints(graph, config)
```

Since `config` now points to the new thread, `/checkpoints` shows the post-compaction
history. The pre-compaction checkpoints are still in the database but are not surfaced
through normal commands. This is the intended behavior.

### Long-Running Agents and Repeated Compaction

For a very long session (say, 8 hours of continuous use), compaction may happen multiple
times. Each compaction:

1. Summarizes the history up to the current point.
2. Starts a new thread.
3. The new thread's initial "history" is the summary plus the last 3 turns.

After several compactions, the history might look like:

```text
SystemMessage("Summary of earlier conversation: [summary 3 — itself a summary of summary 2...]")
HumanMessage("turn N-2")
...
HumanMessage("turn N")
```

The summary at the top is a summary of summaries — the model has been through multiple
compaction cycles. Each cycle potentially loses more information than the previous cycle.

For most practical coding sessions, this is acceptable: the most important information
(current goals, recent files, immediate problems) is always in the recent turns, and the
summary captures the high-level arc. But for very long sessions, consider using `/compact`
manually at natural breakpoints (end of a feature, before starting a new sub-task) rather
than relying entirely on auto-compaction.

### Database Maintenance

Over time, old thread checkpoints accumulate in the SQLite database. A typical long-running
saathi installation might have hundreds of old threads. The database stays small
(each checkpoint is a few kilobytes of JSON), but periodic cleanup is good hygiene:

```python
# Future utility: prune threads that are not the current active thread
# and are older than N days
async def prune_old_threads(graph, current_thread_id: str, max_age_days: int = 30):
    """Remove checkpoints for threads older than max_age_days."""
    # This would require direct access to the SqliteSaver's database connection
    # — not currently exposed through the LangGraph public API.
    pass
```

The LangGraph team is aware of this maintenance need and checkpoint management tooling
is expected to improve in future releases.

---

## 12.15c The Compaction Module: Complete Source Reference

For reference, here is the complete `src/saathi/compaction.py` as it exists in the
repository, annotated with cross-references to the sections of this chapter:

```python
"""Conversation history compaction to stay within the context window.

When history grows large, we summarize the older turns into a single message and
keep the most recent turns verbatim. The cut is made at a **user-turn boundary**
so the retained tail is always a valid message sequence — it never begins with an
orphaned ``ToolMessage`` whose ``AIMessage`` (tool call) was summarized away.
"""
# See §12.1 for the context window problem this solves.

from __future__ import annotations

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

_SUMMARY_PREFIX = "Summary of earlier conversation:"
# _SUMMARY_PREFIX is used as a sentinel in tests (§12.12) and for display.

_CHARS_PER_TOKEN = 4
# The 4 chars/token heuristic is discussed in §12.2.

_SUMMARY_INSTRUCTIONS = (
    "You are compacting a coding-assistant conversation to save context window. "
    "Write a concise summary capturing: the user's goals, key decisions, files "
    "read or modified, important findings, and any unresolved threads. Preserve "
    "concrete details a developer would need to continue. Output only the summary."
)
# The summarization prompt is analyzed in §12.6.


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate (~4 chars per token) across message contents."""
    # Used in §12.2 (budget tracking), §12.11 (needs_compaction), 
    # §12.12 (test_compact_shrinks_token_estimate).
    return sum(len(_text(m)) for m in messages) // _CHARS_PER_TOKEN


def needs_compaction(messages: list[BaseMessage], budget_tokens: int) -> bool:
    # Called in cli.py before every turn (§12.9) and exposed via /compact (§12.10).
    return estimate_tokens(messages) > budget_tokens


def split_for_compaction(
    messages: list[BaseMessage], keep_turns: int
) -> tuple[list[BaseMessage], list[BaseMessage]] | None:
    """Split into ``(older, recent)`` at a user-turn boundary.

    Keeps the last ``keep_turns`` user turns (and everything after them) intact.
    Returns ``None`` when there aren't more than ``keep_turns`` turns — i.e. there
    is nothing worth compacting yet.
    """
    # The user-turn boundary requirement is explained in §12.5.
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= keep_turns:
        return None
    cut = human_idxs[-keep_turns]
    return messages[:cut], messages[cut:]


async def compact_messages(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
) -> list[BaseMessage]:
    """Summarize all but the last ``keep_turns`` turns into one summary message.

    Returns ``[summary, *recent]``, or the original list **unchanged** (same
    object) when there is not enough history to compact.
    """
    # The identity-return is checked with `is` in do_compact (§12.10).
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
    )
    # Transcript format is discussed in §12.6 and §12.13b.
    response = await llm.ainvoke(
        [
            SystemMessage(content=_SUMMARY_INSTRUCTIONS),
            HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
        ]
    )
    summary_text = _text(response) if isinstance(response, BaseMessage) else str(response)
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}\n{summary_text}")
    # SystemMessage rationale is in §12.6.
    return [summary, *recent]
```

This annotated listing serves as a map from source code to the explanatory sections.
When reviewing or modifying the compaction logic, use the section references to find
the relevant analysis.

---

## 12.16 Extending the Compaction System

The compaction module is intentionally small and composable. Several enhancements are
straightforward to build on top of the existing design.

### Hierarchical Summaries

For very long sessions that have already been compacted multiple times, the summary
at the top of the context is itself a summary of summaries. Information can degrade
with each compaction cycle. A hierarchical approach can mitigate this:

```python
async def compact_messages_hierarchical(
    llm: LanguageModelLike,
    messages: list[BaseMessage],
    *,
    keep_turns: int = 3,
    summary_depth: int = 0,
) -> list[BaseMessage]:
    """Like compact_messages, but appends the depth to the summary prefix.

    Depth 0: normal compaction.
    Depth 1: compacting a session that was already compacted once.
    Depth 2: third-generation compaction.
    """
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return messages
    older, recent = split

    # If older starts with an existing summary, extract its depth
    existing_depth = 0
    if older and isinstance(older[0], SystemMessage):
        content = older[0].content
        if content.startswith(_SUMMARY_PREFIX):
            # Try to detect if this is already a summary-of-summary
            if "Prior summary:" in content:
                existing_depth = content.count("Prior summary:")

    transcript = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {_text(m)}" for m in older
    )
    response = await llm.ainvoke([
        SystemMessage(content=_SUMMARY_INSTRUCTIONS),
        HumanMessage(content=f"Conversation so far:\n\n{transcript}"),
    ])
    summary_text = _text(response) if isinstance(response, BaseMessage) else str(response)

    depth_label = f" (compaction generation {existing_depth + 1})" if existing_depth > 0 else ""
    summary = SystemMessage(content=f"{_SUMMARY_PREFIX}{depth_label}\n{summary_text}")
    return [summary, *recent]
```

This is not in the current codebase but illustrates how the module can be extended
without breaking the existing interface.

### Selective Preservation

Some messages are more valuable than others and should not be summarized. For example:

- A `SystemMessage` with project instructions (the `SAATHI.md` content) should never
  be summarized away.
- A very recent tool result that the user explicitly asked about should be preserved
  verbatim.

This could be implemented by marking messages with metadata:

```python
# Conceptual — not in current codebase
critical_message = HumanMessage(
    content="Remember: the database schema is in schema.sql",
    additional_kwargs={"compaction_preserve": True},
)
```

And then modifying `split_for_compaction` to always keep preserved messages in the
`recent` slice regardless of `keep_turns`.

### Compression Ratio Monitoring

The `do_compact` function already computes before/after token counts. Adding a
compression ratio check prevents degenerate cases where the summary is nearly as long
as the original:

```python
after = estimate_tokens(messages)
ratio = after / before if before > 0 else 1.0
if ratio > 0.6:
    # Summary is more than 60% of original — not worth the compaction latency
    log.warning("compaction_poor_ratio", before=before, after=after, ratio=ratio)
    # Optionally: retry with a stricter summary prompt
```

A ratio above 60% suggests the model is producing verbose summaries. The fix is usually
to add "Be as concise as possible." to `_SUMMARY_INSTRUCTIONS`, or to reduce the
maximum response tokens for the summarizer call.

### Compaction Hooks

The hook system in saathi supports arbitrary hooks. A `post_compact` hook could be
added to run after every compaction:

```python
# In hooks.json:
{
  "post_compact": [
    {"run": "echo 'Compacted at $(date)' >> session_log.txt"}
  ]
}
```

Or a Python hook that saves the summary to a file for later review:

```python
# Conceptual
async def save_summary_hook(summary: str, session_id: str) -> None:
    path = Path(f".saathi/summaries/{session_id}.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"# Session Summary\n\n{summary}\n", encoding="utf-8")
```

These are extension points that the architecture supports but the current codebase does
not implement.

### Async Compaction

Currently, compaction blocks the turn: the user's message is not processed until
compaction completes. For models with slow inference, this could add noticeable latency.

An async compaction approach would start the summarization call in the background while
the current turn executes, and apply the compaction result before the next turn:

```python
# Conceptual
_pending_compaction: asyncio.Task | None = None

async def execute_task(task: str) -> None:
    global _pending_compaction
    # Apply pending compaction result if ready
    if _pending_compaction and _pending_compaction.done():
        compacted = await _pending_compaction
        if compacted is not messages:
            messages = compacted
            # new thread_id, etc.
        _pending_compaction = None

    # Schedule compaction for next turn if needed
    if needs_compaction(messages, settings.history_token_budget):
        _pending_compaction = asyncio.create_task(
            compact_messages(summarizer, messages)
        )

    # Run this turn normally
    final_answer, messages = await _run_turn(...)
```

This hides compaction latency by doing it concurrently with the current turn, at the
cost of additional complexity and the risk that the compaction happens on messages that
are being modified by the current turn.

The current synchronous approach is simpler and more correct. Async compaction is a
future optimization for environments where compaction latency is unacceptable.

---

## Summary

History compaction is the mechanism that lets saathi sessions run indefinitely without
hitting the context window limit. The key design decisions are:

1. **Token estimation via 4 chars/token.** Fast and good enough for budget tracking.
2. **75% budget threshold.** Reserves headroom for the system prompt, user message, and
   response.
3. **Split at HumanMessage boundaries.** Never orphans tool call/result pairs.
4. **LLM summarization.** Preserves semantic content rather than dropping it.
5. **New thread_id after compaction.** Clean slate for the checkpointer without
   corrupting the existing thread.
6. **Best-effort failure handling.** A failed compaction never stops the session.
7. **Transparent auto-compaction.** The user sees a one-line notice; no manual
   intervention required.

These decisions collectively ensure that a saathi coding session can span as many hours
and as many turns as the user needs, without quality degradation from context overflow
and without interrupting the user's flow.

---

### Next: Chapter 13 — Model Context Protocol: Extending Agents with External Tools
