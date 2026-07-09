# Chapter 14 — The Code Review Workflow: Concurrent LLM Specialists

> *"One reviewer sees one thing. Four reviewers see everything — and they finish at the same time."*

---

## Table of Contents

1. [The Problem with Single-Reviewer LLM Code Review](#1-the-problem-with-single-reviewer-llm-code-review)
2. [The Multi-Reviewer Architecture](#2-the-multi-reviewer-architecture)
3. [`asyncio.gather` for Concurrent Reviews](#3-asynciogather-for-concurrent-reviews)
4. [The `Finding` Pydantic Model](#4-the-finding-pydantic-model)
5. [Why NOT `format="json"` (Ollama JSON Mode)](#5-why-not-formatjson-ollama-json-mode)
6. [Tolerant JSON Extraction](#6-tolerant-json-extraction)
7. [`parse_findings`](#7-parse_findings)
8. [Confidence Filtering](#8-confidence-filtering)
9. [Severity Ranking](#9-severity-ranking)
10. [`review_one` — The Per-Reviewer Function](#10-review_one--the-per-reviewer-function)
11. [Rich Display — `render_review`](#11-rich-display--render_review)
12. [Using a Separate LLM Instance](#12-using-a-separate-llm-instance)
13. [The `/code-review` CLI Flow](#13-the-code-review-cli-flow)
14. [Testing the Review System](#14-testing-the-review-system)
15. [Extending the Review System](#15-extending-the-review-system)
16. [Multi-Agent Patterns](#16-multi-agent-patterns)

---

## 1. The Problem with Single-Reviewer LLM Code Review

Code review is one of the highest-value tasks an LLM can assist with. A developer opens a pull request, pastes the diff, and asks "what's wrong with this?" The LLM scans thousands of tokens and produces a list of concerns.

In practice, single-prompt code review has a well-documented failure mode: **mediocrity through breadth**. You hand the model a 400-line diff and ask it to think about correctness, security, error handling, and design simultaneously. The model tries to do all of these at once, gives each dimension a passing glance, and produces results that are:

- **Too general.** "You should add error handling to this function" rather than "the `subprocess.run` call on line 42 will raise `FileNotFoundError` if the binary is absent, and there is no caller-visible indication of that."
- **Unevenly weighted.** The model tends to generate items until it feels it has enough. Style nits are easy to generate; subtle concurrency bugs are hard. You get five style nits and miss the race condition.
- **Cluttered with speculation.** A single broad prompt invites the model to include uncertain observations, because nothing in the prompt tells it to be confident before speaking.

The deeper issue is cognitive. Humans have long known that good code review is actually several different mental activities running at different layers of abstraction:

- *Does it do what it claims?* (correctness, logic, off-by-one errors)
- *What happens when it goes wrong?* (error handling, validation, resource cleanup)
- *Is this the right shape?* (type design, interfaces, abstractions, SOLID principles)
- *Can it be exploited?* (injection, path traversal, privilege escalation, secret exposure)

These are not just different concerns — they are different *modes of reading*. A security review involves looking for places where user-controlled data touches sensitive APIs. A bug review involves tracing data flow and checking boundary conditions. An error-handling review involves asking "what is not caught?" at every external call site.

Asking a single model invocation to hold all four mental modes simultaneously, over a long diff, produces worse results than asking four focused reviewers to read the same diff one at a time — and much worse than asking four focused reviewers to read it *in parallel*.

### The Cognitive Specialization Principle

There is a parallel to how expert teams actually work. A security-conscious engineer reading code sees different things than a senior backend engineer looking for algorithmic correctness bugs. The same diff; different mental models; different findings.

LLMs exhibit something analogous. A system prompt that orients the model toward one specific concern dramatically increases the quality and specificity of findings in that concern. The model stops hedging. It commits to the task it has been given. It digs deeper.

This is the motivation for saathi's `/code-review` command: run four specialist reviewers, each with a focused system prompt, over the same diff, concurrently.

### What Makes a Good Reviewer Prompt?

A reviewer prompt needs to do several things:

1. **Assign a role.** "You are the 'security' code reviewer." This activates a persona — the model knows what kind of entity it is playing.
2. **Define the scope precisely.** "You hunt security issues: injection, unsafe shell/eval, path traversal, hardcoded secrets, and missing input sanitization." This tells the model exactly what to look for and (implicitly) what to ignore.
3. **Set a quality bar.** "Report ONLY issues you are genuinely confident about — no style nits, no speculation." This filters out noise before the model produces it.
4. **Define the output format.** Tell the model exactly what JSON shape to produce. Include the schema in the prompt. Remove ambiguity about structure.

When all four elements are present, you get findings that are specific, actionable, and low-noise.

---

## 2. The Multi-Reviewer Architecture

Saathi defines four specialist reviewers in `src/saathi/review.py`. Each is an entry in a dictionary that maps reviewer names to their focus instructions:

```python
DEFAULT_REVIEWERS: dict[str, str] = {
    "bugs": (
        "You hunt correctness bugs: logic errors, off-by-one mistakes, wrong "
        "conditions, unhandled None/empty cases, incorrect return values, and "
        "broken control flow."
    ),
    "error-handling": (
        "You hunt error-handling gaps: swallowed or overly broad exceptions, "
        "silent failures, missing validation, and resources that aren't closed."
    ),
    "design": (
        "You review type and interface design: weak or wrong type hints, "
        "inconsistent signatures, leaky abstractions, and needless complexity."
    ),
    "security": (
        "You review security: injection, unsafe shell/eval, path traversal, "
        "hardcoded secrets, and missing input sanitization."
    ),
}
```

The key architectural decision is: **each reviewer gets the same diff, but a completely different system prompt**. There is no routing, no decomposition of the diff, no attempt to split the work. All four reviewers look at the full diff. Their specialization comes entirely from the system prompt.

This is intentional. You do not want a routing layer that decides "this part of the diff is a security concern." You want each reviewer to independently scan the full picture from their own lens. A variable that is used insecurely is also often a design problem and a bug. You want all three reviewers to see it.

### Why a Dictionary?

The `dict[str, str]` shape — name to instructions — has several practical benefits:

- **Easy to extend.** Add a new reviewer by adding a key. No class hierarchy, no registration system, no decorators. One line.
- **Easy to override.** `run_review()` accepts an optional `reviewers` parameter. Tests can pass a single-reviewer dictionary to isolate behavior.
- **Self-documenting.** The dictionary is readable at a glance. Anyone can understand the full reviewer setup in 20 lines.
- **Easy to serialize.** If you wanted to store reviewer configurations in a file, a dict of strings is trivial to serialize to JSON or TOML.

### The System Prompt Template

Each reviewer's full system prompt is assembled in `review_one()`:

```text
You are the '{reviewer}' code reviewer. {instructions}

Report ONLY issues you are genuinely confident about — no style nits,
no speculation. Respond with a JSON object of this exact shape:
{"findings": [{"severity": "high|medium|low", "confidence": 0-100,
"file": "path", "line": <number or null>, "issue": "what is wrong",
"suggestion": "how to fix"}]}
If you find nothing, respond with {"findings": []}. Output only JSON.
```

Every reviewer gets the same structured output requirement. The only variation is the reviewer name and the focus instructions. This uniformity is what makes aggregation possible: all four reviewers produce the same JSON schema, which the same parser can handle.

### The Human Message

The human message is deliberately simple:

```text
Review this diff:

{diff}
```

Nothing more. The reviewer's role, focus, quality bar, and output format are all in the system message. The human message is just "here is the material."

This separation — persona and instructions in system, material in human — is a general best practice for structured LLM workflows. It makes the system message reusable and the human message swappable.

---

## 3. `asyncio.gather` for Concurrent Reviews

The most important performance decision in the review system is concurrency. Without it, four serial reviewer calls over a large diff with a 7B–14B parameter model could take 30–60 seconds. With concurrency, you pay only the cost of the slowest reviewer.

```python
async def run_review(
    llm: LanguageModelLike,
    diff: str,
    *,
    reviewers: dict[str, str] | None = None,
    min_confidence: int = 70,
) -> list[Finding]:
    """Run all reviewers concurrently; return findings >= min_confidence, ranked."""
    reviewers = reviewers or DEFAULT_REVIEWERS
    results = await asyncio.gather(
        *(review_one(llm, name, instr, diff) for name, instr in reviewers.items())
    )
    findings = [f for group in results for f in group if f.confidence >= min_confidence]
    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 1), -f.confidence))
    return findings
```

### The Generator Expression Inside `gather`

```python
*(review_one(llm, name, instr, diff) for name, instr in reviewers.items())
```

This is a generator expression unpacked with `*` into `asyncio.gather()`'s positional arguments. Each `review_one(...)` call creates a coroutine. The `*` unpacks all coroutines as separate arguments. `asyncio.gather` schedules all of them concurrently on the event loop and awaits all of them.

The result is a tuple of lists (one `list[Finding]` per reviewer). The list comprehension flattens them:

```python
findings = [f for group in results for f in group if f.confidence >= min_confidence]
```

This is a flat list of all findings from all reviewers, with the confidence filter applied in the same pass.

### Performance Characteristics

With four reviewers and a model that takes 6 seconds per review:

- **Serial execution:** 4 × 6s = 24s total
- **Concurrent execution:** max(6s, 6s, 6s, 6s) = 6s total (plus scheduling overhead, ~0.1s)

In practice, reviewers do not all take exactly the same time. The "bugs" reviewer on a complex diff might take 10 seconds while the "security" reviewer takes 4 seconds. Concurrent execution means you pay 10 seconds, not 14.

For an interactive CLI tool — where the user is waiting — this difference is the line between "usable" and "frustrating."

### Why `asyncio.gather` and Not a Thread Pool?

The LLM calls are I/O-bound. The CPU is idle while the HTTP request to Ollama is in flight and while Ollama's GPU is generating tokens. `asyncio.gather` handles this well: all four requests are in flight simultaneously, and the event loop handles their completion as they arrive.

A thread pool (`concurrent.futures.ThreadPoolExecutor`) would also work, but it is heavier-weight and introduces thread management overhead. Since the rest of saathi is already async (LangGraph, the tool node, the CLI loop), `asyncio.gather` is the natural choice.

### Graceful Degradation

`asyncio.gather` has a default `return_exceptions=False` behavior: if any coroutine raises, the exception propagates. But `review_one` is designed to never raise — it catches all exceptions internally and returns an empty list. This means `run_review` cannot crash due to a reviewer failure. We will examine this in detail in Section 10.

---

## 4. The `Finding` Pydantic Model

The `Finding` model is the currency of the review system. Every reviewer produces a list of findings; every rendering function consumes a list of findings; every test asserts on findings.

```python
class Finding(BaseModel):
    reviewer: str = ""
    severity: str = "medium"
    confidence: int = 50
    file: str = ""
    line: int | None = None
    issue: str = ""
    suggestion: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 50
        return max(0, min(100, n))

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v: Any) -> str:
        s = str(v).lower().strip()
        return s if s in _SEVERITY_RANK else "medium"

    @field_validator("line", mode="before")
    @classmethod
    def _coerce_line(cls, v: Any) -> int | None:
        if v in (None, "", "null"):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
```

### Field Design

Every field has a default. This is intentional: LLMs sometimes omit fields, produce partial objects, or give unexpected types. A `Finding` with defaults is always constructable from partial data. The validators fill in safe defaults when the LLM's output is ambiguous.

**`reviewer: str = ""`** — The reviewer name is injected by `parse_findings`, not expected from the LLM's JSON. The LLM does not know its own reviewer name; the code that called it does.

**`severity: str = "medium"`** — A reasonable middle ground. If the LLM forgets to specify severity, "medium" is a conservative guess.

**`confidence: int = 50`** — 50 is the explicit midpoint of uncertainty. Not confident, not dismissable.

**`file: str = ""`** — Empty string rather than `None` because the display code always does string operations on the file field.

**`line: int | None = None`** — `None` means "unknown line" rather than a wrong line number. The display code gracefully omits line numbers for `None`.

**`issue: str = ""`** — The description. Empty string means "nothing to show."

**`suggestion: str = ""`** — The fix. Empty string is fine — not every finding has a clear mechanical fix.

### Validator: Confidence Clamping

```python
@field_validator("confidence", mode="before")
@classmethod
def _clamp_confidence(cls, v: Any) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 50
    return max(0, min(100, n))
```

The `mode="before"` means this validator runs on the raw input before Pydantic's type coercion. This lets us handle strings (`"85"`) and out-of-range values (`150`, `-5`) uniformly.

The behavior:

- `150` → `100` (clamp to max)
- `-5` → `0` (clamp to min)
- `"85"` → `85` (string to int)
- `"nonsense"` → `50` (fallback default)
- `None` → `50` (fallback default)

### Validator: Severity Normalization

```python
@field_validator("severity", mode="before")
@classmethod
def _normalize_severity(cls, v: Any) -> str:
    s = str(v).lower().strip()
    return s if s in _SEVERITY_RANK else "medium"
```

LLMs commonly produce `"High"`, `"HIGH"`, `"high"`, and occasionally `"critical"`. The `_SEVERITY_RANK` dict defines exactly three valid values: `"high"`, `"medium"`, `"low"`. Anything else — including `"critical"`, which the prompt does not mention — falls through to `"medium"`.

This is a design choice worth noting: the system intentionally uses three severity levels, not five. "Critical" and "info" are omitted because:

1. Few local code reviews actually warrant "critical" — that word creates urgency that a local model may not be able to justify.
2. "Info" is just noise. If it is important enough to review, it is at least "low."

The three-level scale (high/medium/low) produces cleaner output and easier sorting.

### Validator: Line Coercion

```python
@field_validator("line", mode="before")
@classmethod
def _coerce_line(cls, v: Any) -> int | None:
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
```

LLMs often produce line numbers as strings (`"42"`) or as the JSON literal `null`. They sometimes produce the string `"null"`. This validator handles all of these cases:

- `None` → `None` (JSON null)
- `""` → `None` (empty string)
- `"null"` → `None` (string "null")
- `"42"` → `42` (string to int)
- `"abc"` → `None` (unparseable → unknown)

---

## 5. Why NOT `format="json"` (Ollama JSON Mode)

Ollama supports a `format="json"` parameter that enables **grammar-constrained decoding**. When enabled, Ollama ensures that every token it generates is valid JSON — at every step, it restricts the token vocabulary to only those tokens that can extend the current partial JSON.

This sounds appealing: guaranteed valid JSON output with no parsing needed. In practice, it is significantly slower and saathi deliberately does not use it.

### How Grammar-Constrained Decoding Works

At each token generation step, a language model computes a probability distribution over its full vocabulary (typically 32,000–128,000 tokens). It then samples from that distribution to select the next token.

Grammar-constrained decoding modifies this process: before sampling, it computes a **mask** over the vocabulary. Tokens that would produce invalid partial JSON are assigned zero probability. Only tokens that keep the output valid JSON are eligible.

The mask computation is not free. For each token step, the system must evaluate which of thousands of tokens are valid continuations of the current partial JSON. This is done by running a JSON parser in "predict" mode. On complex or deeply nested JSON, this can become substantial work.

### The Performance Impact

The slowdown varies by model size, hardware, and diff complexity. In saathi's testing:

- **Without `format="json"`:** A reviewer call on a 150-line diff takes roughly 4–8 seconds with a 12B model on a mid-range GPU.
- **With `format="json"`:** The same call takes roughly 18–35 seconds — 4–5× slower.

The slowdown grows with the complexity of the output. A review with multiple findings, each with multiple fields, generates a complex JSON tree. The grammar-constrained decoder is doing significant work at every step.

For an interactive CLI that runs four reviewers concurrently, the difference is:

- Without JSON mode: 4 × 8s concurrent ≈ 8s total
- With JSON mode: 4 × 30s concurrent ≈ 30s total

A 30-second code review feels slow. An 8-second code review feels acceptable.

### The Alternative: Tolerant Parsing

The reason `format="json"` exists is to prevent invalid JSON. But in practice, capable LLMs rarely produce truly invalid JSON when the prompt clearly specifies the expected format. They sometimes produce:

- Valid JSON surrounded by prose ("Here are my findings:\n```json\n{...}\n```")
- Valid JSON after a brief preamble ("I found the following issues:\n{...}")
- Valid JSON with minor formatting differences

None of these require grammar-constrained decoding to fix. They require a tolerant parser that can find and extract valid JSON from a response that contains prose.

Saathi's `_extract_json()` function (covered in Section 6) handles all of these cases without any performance cost. The LLM generates freely; the parser finds the JSON afterward.

### The CLI Comment

The code comment in `cli.py` makes this explicit:

```python
# No json_format: Ollama's grammar-constrained JSON mode is very
# slow on larger models; the tolerant parser recovers JSON anyway.
review_llm = make_llm(model_id)
```

This is the canonical documentation for this architectural decision. When someone wonders why `json_format=True` is not used here, the comment explains.

### When Grammar-Constrained Decoding IS Appropriate

Grammar-constrained decoding has legitimate uses:

- **Extremely small models** (1B–3B parameters) that frequently produce malformed JSON. Grammar constraints compensate for weaker instruction following.
- **Latency-insensitive batch tasks** where correctness is paramount and speed is not a user concern.
- **Complex schemas** where tolerant parsing might produce ambiguous results (e.g., schemas with optional nested arrays where an empty array vs. absent array matters semantically).

None of these apply to saathi's code review use case. The models saathi targets (7B–14B) follow instructions reliably. The task is interactive and latency-sensitive. The schema is simple. Tolerant parsing is the right choice.

---

## 6. Tolerant JSON Extraction

`_extract_json()` is the function that bridges the LLM's text response and structured data. It is designed to handle the full range of real-world LLM outputs, from clean JSON to JSON buried in prose.

```python
def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from an LLM response (tolerates prose/fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
    return None
```

### Strategy 1: Fence Stripping

The first thing the function does is check for code fences:

```python
if text.startswith("```"):
    text = text.strip("`")
    if text.lstrip().lower().startswith("json"):
        text = text.lstrip()[4:]
```

Many LLMs, when asked to produce JSON, produce:

````text
```json
{"findings": [...]}
```
````

The fence stripping handles this. The backticks are stripped. The `json` language hint is stripped. What remains is the raw JSON.

This handles the most common non-plain-JSON case.

### Strategy 2: Direct Parse

After fence stripping, try to parse the entire text as JSON:

```python
try:
    parsed = json.loads(text)
    return parsed if isinstance(parsed, (dict, list)) else None
except json.JSONDecodeError:
    pass
```

If the response is well-behaved — clean JSON, possibly after fence stripping — this succeeds immediately. The `isinstance` check rejects JSON primitives (numbers, strings, booleans) that are technically valid JSON but not what we want.

### Strategy 3: Find-and-Parse

If the text is not parseable as a whole, find the first occurrence of `{` or `[` and the last occurrence of the matching closer, and try to parse the slice:

```python
for open_c, close_c in (("{", "}"), ("[", "]")):
    start, end = text.find(open_c), text.rfind(close_c)
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
```

This handles the case where the model produces:

```text
I found the following issues in your code:
{"findings": [{"severity": "high", ...}]}
Let me know if you have any questions.
```

The `text.find("{")` finds the `{` at the start of the JSON object. `text.rfind("}")` finds the `}` at the end. The slice `text[start:end+1]` is the JSON object. This parses successfully.

The loop tries `{}` before `[]`, because reviewer responses are expected to be objects (the `{"findings": [...]}` form), not bare arrays. But both are tried, so bare array responses also work.

### Why `rfind` for the Closer?

`text.rfind("}")` finds the *last* `}` in the text. This is correct for nested JSON: a JSON object might contain many `}` characters as part of nested structures. The last `}` is the one that closes the outermost object.

`text.find("}")` would find the first `}`, which closes the first nested object — not the outer wrapper. `rfind` is the right choice.

### Failure Mode

If all three strategies fail, the function returns `None`. The callers (`parse_findings`) handle `None` gracefully — it produces an empty findings list.

In practice, `None` is rare. If the LLM produces any JSON at all, one of the three strategies finds it. The only case where `None` occurs is if the LLM produces no JSON whatsoever — for example, a complete refusal or a pure English response. This is handled gracefully; it produces zero findings for that reviewer.

---

## 7. `parse_findings`

`parse_findings` converts the raw text of an LLM response into a list of validated `Finding` objects.

```python
def parse_findings(text: str, reviewer: str) -> list[Finding]:
    """Parse a reviewer's JSON response into Findings, skipping malformed items."""
    data = _extract_json(text)
    if isinstance(data, dict):
        items = data.get("findings", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    findings: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            findings.append(
                Finding(
                    reviewer=reviewer,
                    severity=item.get("severity", "medium"),
                    confidence=item.get("confidence", 50),
                    file=str(item.get("file", "")),
                    line=item.get("line"),
                    issue=str(item.get("issue", "")),
                    suggestion=str(item.get("suggestion", "")),
                )
            )
        except Exception:  # noqa: BLE001 — skip anything that won't validate
            continue
    return findings
```

### Handling Both JSON Forms

The prompt tells the LLM to produce `{"findings": [...]}`. This is the expected form. But LLMs sometimes produce bare arrays — `[{...}, {...}]` — especially when the prompt is processed through a different model version or the system prompt is truncated.

The code handles both:

```python
if isinstance(data, dict):
    items = data.get("findings", [])
elif isinstance(data, list):
    items = data
else:
    items = []
```

If the data is a dict, look for a `"findings"` key. If there is no such key, default to an empty list (the reviewer found nothing, or produced a malformed response).

If the data is a list, use it directly.

If `_extract_json` returned `None`, use an empty list.

### Non-Dict Item Filtering

```python
if not isinstance(item, dict):
    continue
```

This handles the case where the `findings` array contains non-objects. For example:

```json
{"findings": ["this is a nit", {"severity": "high", "issue": "real bug"}]}
```

The string `"this is a nit"` is not a dict and is silently skipped. The `{"severity": "high", "issue": "real bug"}` is a dict and is processed normally.

### Per-Finding Exception Handling

```python
try:
    findings.append(Finding(...))
except Exception:  # noqa: BLE001 — skip anything that won't validate
    continue
```

Even after passing the `isinstance(item, dict)` check, a finding might fail Pydantic validation for reasons beyond what the validators handle. The broad `except Exception` means: if a single finding is unprocessable, skip it and continue. Never let one bad finding abort the processing of all subsequent findings.

The `# noqa: BLE001` comment acknowledges that broad exception catching is intentional here (the linter rule BLE001 flags "blind exception" catches).

### The `reviewer` Injection

Notice that the `reviewer` parameter to `parse_findings` is not read from the JSON — it is the string passed by the caller. The LLM's JSON does not include a `reviewer` field. The code injects it:

```python
Finding(
    reviewer=reviewer,  # injected by caller, not from LLM JSON
    severity=item.get("severity", "medium"),
    ...
)
```

This is correct: the LLM does not know its own reviewer name. The reviewer name is a concept from the orchestration layer, not from the model. Injecting it from the outside keeps the JSON schema simple and prevents the LLM from producing incorrect or inconsistent reviewer names.

---

## 8. Confidence Filtering

After aggregating findings from all reviewers, `run_review` applies a confidence filter:

```python
findings = [f for group in results for f in group if f.confidence >= min_confidence]
```

The default `min_confidence` is 70, configurable via the `review_min_confidence` setting (which can be overridden with `SAATHI_REVIEW_MIN_CONFIDENCE` in the environment).

### Why Confidence Filtering Works

When you ask a model to self-rate its confidence on a 0–100 scale, the ratings correlate meaningfully with actual accuracy. This is a well-studied phenomenon in LLM calibration research. Models with good calibration (most modern models above 7B parameters) tend to give lower confidence scores when they are less certain.

In the reviewer prompt, the instruction is: "Report ONLY issues you are genuinely confident about — no style nits, no speculation." But sometimes the model includes a borderline observation with a confidence score of 55. The confidence filter gives the model a second channel to express uncertainty without having to decide binary "include/exclude."

### The 70% Threshold

70% is a practical threshold that works well across a range of diff sizes and model capabilities:

- It admits findings where the model has clear evidence of a problem.
- It rejects findings that the model itself considers uncertain.
- It does not require perfect confidence (100), which would be almost nothing.

### Configuring the Threshold

```python
# In .env or environment:
SAATHI_REVIEW_MIN_CONFIDENCE=80  # stricter
SAATHI_REVIEW_MIN_CONFIDENCE=50  # more permissive
```

A team doing a first-pass noise reduction might set it to 50 to see everything. A team that wants only high-signal findings might set it to 85. The default of 70 is calibrated for typical use.

---

## 9. Severity Ranking

After confidence filtering, findings are sorted by severity (most severe first) and within the same severity, by confidence (most confident first):

```python
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 1), -f.confidence))
```

### The Sort Key Explained

Python's `sort` is stable and comparisons are element-wise on tuples. For each finding:

- `_SEVERITY_RANK.get(f.severity, 1)`: convert severity to a number. `"high"` = 0, `"medium"` = 1, `"low"` = 2. The `.get(f.severity, 1)` default of 1 means unknown severities sort as medium — a safe fallback.
- `-f.confidence`: negate confidence so that higher confidence sorts first (Python sorts ascending, so `-90` < `-70` means 90% sorts before 70%).

The tuple comparison in Python sorts on the first element first. So a high-severity finding always sorts before a medium-severity finding, regardless of confidence. Within the same severity, higher confidence sorts first.

### Why This Order Matters

Users read reviews top-to-bottom. The findings they see first are the ones they should act on first. Putting high-severity findings at the top, and the most confident high-severity findings before less certain ones, ensures that the most actionable content appears first.

A high-severity finding at 75% confidence might be more urgent than a medium-severity finding at 95% confidence. The sort respects that.

### The `_SEVERITY_RANK` Dictionary

```python
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
```

This is defined at module level, not inside a function, so it is computed once and reused. It is used in two places: the `_normalize_severity` validator (to check if a severity value is valid) and the sort key.

---

## 10. `review_one` — The Per-Reviewer Function

`review_one` is the per-reviewer coroutine. It takes the LLM, the reviewer's name and instructions, and the diff, and returns a list of findings.

```python
async def review_one(
    llm: LanguageModelLike, reviewer: str, instructions: str, diff: str
) -> list[Finding]:
    """Run a single specialist reviewer over the diff."""
    system = SystemMessage(
        content=(
            f"You are the '{reviewer}' code reviewer. {instructions}\n\n"
            "Report ONLY issues you are genuinely confident about — no style nits, "
            "no speculation. Respond with a JSON object of this exact shape:\n"
            '{"findings": [{"severity": "high|medium|low", "confidence": 0-100, '
            '"file": "path", "line": <number or null>, "issue": "what is wrong", '
            '"suggestion": "how to fix"}]}\n'
            'If you find nothing, respond with {"findings": []}. Output only JSON.'
        )
    )
    human = HumanMessage(content=f"Review this diff:\n\n{diff}")
    try:
        response = await llm.ainvoke([system, human])
    except Exception as exc:  # noqa: BLE001
        log.warning("reviewer_failed", reviewer=reviewer, error=str(exc))
        return []
    return parse_findings(_text(response), reviewer)
```

### Message Construction

The function builds a two-message conversation: a `SystemMessage` and a `HumanMessage`. This is the canonical pattern for single-turn LLM tasks in LangChain.

The system message is constructed per-reviewer — the reviewer name and instructions are interpolated. The human message is always the same structure: "Review this diff:\n\n{diff}".

### The JSON Schema in the Prompt

The system prompt includes the exact JSON schema:

```text
{"findings": [{"severity": "high|medium|low", "confidence": 0-100,
"file": "path", "line": <number or null>, "issue": "what is wrong",
"suggestion": "how to fix"}]}
```

This is inline in the prompt string, not a separate template. It is a real JSON object with placeholder values. The model sees it as an example and follows it.

Including the schema inline rather than describing it in natural language significantly improves compliance. "Respond with a JSON object containing a 'findings' array where each element has a 'severity' string, a 'confidence' integer..." is harder for the model to parse than seeing the actual shape.

### Error Handling: Never Raise

```python
try:
    response = await llm.ainvoke([system, human])
except Exception as exc:
    log.warning("reviewer_failed", reviewer=reviewer, error=str(exc))
    return []
```

This is the critical design principle: **a reviewer that fails returns an empty list, never raises**. If the LLM is unavailable, if the network times out, if Ollama returns an error, the reviewer gracefully produces zero findings.

Why is this important? Because `review_one` is called by `asyncio.gather`. If a reviewer raises an unhandled exception, `asyncio.gather` would propagate the exception from `run_review`. But a code review with three reviewers instead of four is still useful. The user should see three reviewers' findings, not an error message.

The structured log event `reviewer_failed` records the failure for debugging without exposing it to the user.

### The `_text` Helper

```python
def _text(message: BaseMessage | str) -> str:
    if isinstance(message, str):
        return message
    content = message.content
    return content if isinstance(content, str) else str(content)
```

This helper extracts the text content from a `BaseMessage`. LangChain message content can be a string or a list (for multi-modal messages). This function normalizes both cases to a string.

---

## 11. Rich Display — `render_review`

The display layer uses `rich` to render findings as colored panels. Each finding gets its own panel, color-coded by severity.

```python
_SEVERITY_COLOR = {"high": "red", "medium": "yellow", "low": "blue"}


def render_review(findings: list[Finding], min_confidence: int) -> None:
    """Print review findings as severity-colored panels, most severe first."""
    if not findings:
        console.print(f"[green]✓ No findings at or above {min_confidence}% confidence.[/green]")
        return
    console.print(f"\n[bold]Code review — {len(findings)} finding(s)[/bold]\n")
    for f in findings:
        color = _SEVERITY_COLOR.get(f.severity, "yellow")
        loc = f.file + (f":{f.line}" if f.line else "")
        title = (
            f"[{color}]{f.severity.upper()}[/{color}] "
            f"[dim]{f.confidence}%[/dim] [cyan]{loc}[/cyan] [dim]({f.reviewer})[/dim]"
        )
        body = f.issue
        if f.suggestion:
            body += f"\n[dim]→ {f.suggestion}[/dim]"
        console.print(Panel(body, title=title, title_align="left", border_style=color))
```

### The Panel Format

Each finding renders as a Rich `Panel` with:

- **Border color:** red for high severity, yellow for medium, blue for low
- **Title:** severity label (colored), confidence percentage (dimmed), file:line (cyan), reviewer name (dimmed in parentheses)
- **Body:** the issue description, followed by the suggestion prefixed with `→` (dimmed)

For example, a high-severity bug finding in `calc.py` at line 2, 90% confidence from the bugs reviewer, looks like:

```text
╭─ HIGH 90% calc.py:2 (bugs) ──────────────────────────────────╮
│ Division by zero is unhandled when b == 0.                   │
│ → Guard against b == 0 or catch ZeroDivisionError.          │
╰───────────────────────────────────────────────────────────────╯
```

### The Location String

```python
loc = f.file + (f":{f.line}" if f.line else "")
```

If `f.line` is `None`, the location is just the file name. If `f.line` is an integer, the location is `filename:linenum`. The ternary is compact and handles both cases.

### The Empty State

```python
if not findings:
    console.print(f"[green]✓ No findings at or above {min_confidence}% confidence.[/green]")
    return
```

An empty findings list means either the review found nothing, or all findings were below the confidence threshold. The message mentions the threshold so the user understands what "no findings" means: not necessarily that the code is perfect, but that no reviewer was confident enough to flag anything.

### Why Rich Panels?

Rich panels provide visual separation between findings. When a review produces 8 findings, panels make it easy to read one finding at a time, see at a glance which are most severe (by color), and skip to the relevant location information.

The alternative — a plain text list — is harder to scan. The visual distinction between high (red border), medium (yellow border), and low (blue border) findings lets the user triage at a glance.

---

## 12. Using a Separate LLM Instance

The code review system creates its own LLM instance, separate from the agent's LLM:

```python
review_llm = make_llm(model_id)
```

This is not the same `llm` object that the agent uses. It is a fresh instance of the same model.

### Why a Separate Instance?

**No `json_format=True`** — the agent LLM might be configured with `json_format=True` for structured tool calls (in some configurations). The review LLM explicitly does not use JSON mode, for the performance reasons described in Section 5. Using a separate instance ensures the review LLM's configuration cannot accidentally inherit the agent LLM's configuration.

**No graph integration** — the agent LLM is bound to the LangGraph graph. It participates in the graph's message routing, checkpointing, and state management. The review system operates completely outside the graph. It is a code-orchestrated workflow (a fancy name for "regular async Python code"), not a graph node. Using a separate LLM instance makes this architectural boundary explicit.

**No tool binding** — the agent LLM has tools bound to it (the LangChain tools). The review LLM does not need tool calling capability. A separate instance without bound tools is simpler and has no risk of accidental tool invocation.

**Independent lifecycle** — the review LLM is created when `/code-review` is invoked and used only for the duration of that review. It is not held across turns. This keeps the review system stateless and reproducible.

### The `make_llm` Function

`make_llm` is defined in `saathi.agent` and creates a `ChatOllama` instance with the project's configured model and settings. The function signature is:

```python
def make_llm(model_id: str, *, json_format: bool = False) -> ChatOllama:
    ...
```

The `json_format=False` default is what makes the review LLM fast. When the CLI dispatches `/code-review`, it calls `make_llm(model_id)` without `json_format=True`, relying on the default.

---

## 13. The `/code-review` CLI Flow

The command is dispatched in `cli.py`'s main interaction loop. When the user types `/code-review` (or the alias `/review`), the following sequence executes:

```python
if cmd in ("code-review", "review"):
    diff = get_working_diff()
    if not diff:
        console.print("[dim]No uncommitted changes to review.[/dim]")
        continue
    # No json_format: Ollama's grammar-constrained JSON mode is very
    # slow on larger models; the tolerant parser recovers JSON anyway.
    review_llm = make_llm(model_id)
    spinner = ThinkingSpinner()
    spinner.update("reviewing changes…")
    spinner.start()
    try:
        findings = await run_review(
            review_llm, diff, min_confidence=settings.review_min_confidence
        )
    finally:
        spinner.stop()
    render_review(findings, settings.review_min_confidence)
    continue
```

### Step 1: Get the Diff

`get_working_diff()` calls `git diff HEAD` to get all uncommitted changes (staged and unstaged). If the repo is clean, it returns an empty string and the CLI prints a message and does nothing.

```python
def get_working_diff() -> str:
    """Return the uncommitted diff vs HEAD (falling back to the unstaged diff)."""
    diff = _git("diff", "HEAD")
    if diff.startswith("Error") or "not a git repository" in diff.lower():
        return ""
    if diff == "(no output)":
        diff = _git("diff")
        if diff == "(no output)" or diff.startswith("Error"):
            return ""
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n… (diff truncated for review)"
    return diff
```

The fallback from `git diff HEAD` to `git diff` handles the case where there are staged but not committed changes. The truncation at `_MAX_DIFF_CHARS = 24_000` prevents extremely large diffs from overwhelming the LLM's context window.

### Step 2: Create the Review LLM

A separate `ChatOllama` instance is created without JSON mode. This is the explicit, intentional separation described in Section 12.

### Step 3: Show a Spinner

The review is asynchronous and takes a few seconds. A `ThinkingSpinner` provides visual feedback so the user knows something is happening. The spinner is stopped in a `finally` block so it always stops, even if `run_review` raises (though `run_review` is designed not to raise — belt and suspenders).

### Step 4: Run the Review

`run_review(review_llm, diff, min_confidence=settings.review_min_confidence)` runs all four reviewers concurrently and returns a filtered, ranked list of findings.

### Step 5: Render

`render_review(findings, settings.review_min_confidence)` displays the findings as colored panels. The `min_confidence` is passed again for the "no findings" message.

### The `continue` Statement

After rendering, `continue` returns to the top of the CLI's input loop. This is the standard pattern in saathi's CLI: each command either `continue`s (returns to input loop) or invokes `await execute_task(...)` (which runs an agent turn). Code review is a direct code-orchestrated workflow, not an agent task, so it uses `continue`.

---

## 14. Testing the Review System

`tests/test_review.py` contains 10 tests covering the full review stack. It uses two local fake LLMs instead of Ollama, making all tests offline and fast.

### The Test Doubles

```python
class FakeLLM:
    """Returns a fixed response for every reviewer call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        return AIMessage(content=self.response)


class RaisingLLM:
    async def ainvoke(self, messages, config=None):
        raise RuntimeError("model unavailable")
```

`FakeLLM` returns the same pre-set response for every call, and counts invocations. `RaisingLLM` always raises — it simulates an unavailable model or network error.

Neither class inherits from `BaseChatModel` or any LangChain base class. They only need to implement `async def ainvoke(...)`, which is the single method `review_one` calls. Python's duck typing means they are fully compatible with the `LanguageModelLike` type annotation.

### Test 1–3: Finding Validation

```python
def test_confidence_is_clamped() -> None:
    assert Finding(confidence=150).confidence == 100
    assert Finding(confidence=-5).confidence == 0
    assert Finding(confidence="nonsense").confidence == 50


def test_severity_is_normalized() -> None:
    assert Finding(severity="High").severity == "high"
    assert Finding(severity="CRITICAL").severity == "medium"  # unknown -> medium


def test_line_is_coerced() -> None:
    assert Finding(line="12").line == 12
    assert Finding(line="null").line is None
    assert Finding(line="abc").line is None
```

These tests validate the Pydantic validators directly by constructing `Finding` objects with edge-case inputs. They are pure unit tests with no dependencies.

Notice `Finding(severity="CRITICAL").severity == "medium"` — this documents the intended behavior that "CRITICAL" is not a valid severity and falls through to the default. A reader seeing this test understands exactly what the normalizer does with unknown values.

### Test 4: JSON Extraction

```python
def test_extract_json_plain_and_fenced() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose before {"a": 1} prose after') == {"a": 1}
    assert _extract_json("no json here") is None
```

Four cases in one test: plain JSON, fenced JSON, prose-wrapped JSON, and no JSON. This covers all three extraction strategies plus the failure case.

### Tests 5–6: `parse_findings`

```python
def test_parse_object_and_array_forms() -> None:
    obj = parse_findings('{"findings": [{"issue": "x", "confidence": 80}]}', "r")
    assert len(obj) == 1 and obj[0].reviewer == "r" and obj[0].confidence == 80
    arr = parse_findings('[{"issue": "y"}]', "r")
    assert len(arr) == 1 and arr[0].confidence == 50  # default


def test_parse_skips_non_dict_and_garbage() -> None:
    assert parse_findings('{"findings": ["oops", {"issue": "ok"}]}', "r") == [
        Finding(reviewer="r", issue="ok")
    ]
    assert parse_findings("totally not json", "r") == []
```

These tests verify both the object-wrapper form and the bare-array form. The second test's first assertion uses `==` with a `Finding` object — Pydantic models support equality by value, so `Finding(reviewer="r", issue="ok")` equals another `Finding` with the same fields.

### Tests 7–10: `run_review` Aggregation

```python
async def test_run_review_filters_below_confidence() -> None:
    llm = FakeLLM(
        _resp([
            {"severity": "high", "confidence": 90, "file": "a.py", "issue": "bug"},
            {"severity": "low", "confidence": 40, "file": "b.py", "issue": "nit"},
        ])
    )
    findings = await run_review(llm, "diff", reviewers={"solo": "look"}, min_confidence=70)
    assert len(findings) == 1
    assert findings[0].confidence == 90
    assert findings[0].reviewer == "solo"
```

This test passes a single-reviewer `reviewers` dict (`{"solo": "look"}`) to isolate the behavior. It verifies that the 40-confidence finding is filtered out.

```python
async def test_run_review_sorts_by_severity_then_confidence() -> None:
    llm = FakeLLM(
        _resp([
            {"severity": "medium", "confidence": 95, "issue": "m"},
            {"severity": "high", "confidence": 75, "issue": "h"},
        ])
    )
    findings = await run_review(llm, "d", reviewers={"solo": "x"}, min_confidence=70)
    assert [f.severity for f in findings] == ["high", "medium"]
```

This verifies that severity ranks above confidence in the sort: the high-severity 75% finding sorts before the medium-severity 95% finding.

```python
async def test_run_review_aggregates_across_reviewers() -> None:
    llm = FakeLLM(_resp([{"severity": "high", "confidence": 90, "issue": "i"}]))
    findings = await run_review(
        llm, "d", reviewers={"r1": "a", "r2": "b", "r3": "c"}, min_confidence=70
    )
    assert len(findings) == 3
    assert {f.reviewer for f in findings} == {"r1", "r2", "r3"}
    assert llm.calls == 3
```

This is the concurrency test. With three reviewers and `FakeLLM` returning one finding per call, the result should have three findings (one from each reviewer), and the LLM should have been called three times. The `llm.calls == 3` assertion verifies that all three reviewers actually ran.

```python
async def test_failed_reviewer_yields_no_findings() -> None:
    findings = await run_review(RaisingLLM(), "d", reviewers={"solo": "x"})
    assert findings == []
```

The graceful degradation test: `RaisingLLM` always raises, but `run_review` returns an empty list instead of propagating the exception.

---

## 15. Extending the Review System

### Adding a New Reviewer

To add a new reviewer, add a single entry to `DEFAULT_REVIEWERS`:

```python
DEFAULT_REVIEWERS: dict[str, str] = {
    "bugs": "...",
    "error-handling": "...",
    "design": "...",
    "security": "...",
    # New: performance reviewer
    "performance": (
        "You review performance: O(n²) algorithms where O(n) is possible, "
        "unnecessary memory allocations in hot paths, blocking I/O in async "
        "contexts, and missing database indices. Report only clear, measurable "
        "performance problems — not micro-optimizations."
    ),
}
```

That is the complete change. No other code needs to touch. The new reviewer is automatically included in `run_review`, runs concurrently with the others, and its findings are aggregated, filtered, and ranked with the rest.

### Changing the Confidence Threshold

In `.env` or the environment:

```bash
SAATHI_REVIEW_MIN_CONFIDENCE=60   # lower threshold, more findings
SAATHI_REVIEW_MIN_CONFIDENCE=85   # higher threshold, fewer but more certain findings
```

The `settings.review_min_confidence` value is used in the CLI dispatch and passed to `run_review`. No code changes needed.

### Adding a New Finding Field

Add the field to `Finding` with a default:

```python
class Finding(BaseModel):
    ...
    category: str = ""       # e.g., "async", "memory", "api"
    references: list[str] = []  # e.g., ["CWE-22", "OWASP-A01"]
```

Update the `review_one` system prompt to include the new fields in the JSON schema. Update `parse_findings` to extract them:

```python
Finding(
    ...
    category=str(item.get("category", "")),
    references=item.get("references", []),
)
```

Update `render_review` to display them if non-empty.

### Overriding Reviewers at Call Time

For testing or custom workflows, pass a custom `reviewers` dict to `run_review`:

```python
findings = await run_review(
    llm,
    diff,
    reviewers={"custom": "You focus exclusively on async/await correctness."},
    min_confidence=60,
)
```

This does not modify `DEFAULT_REVIEWERS`. The custom dict is used only for this call.

---

## 16. Multi-Agent Patterns

Saathi's code review system demonstrates several general-purpose patterns for multi-agent LLM workflows that apply far beyond code review.

### Pattern 1: Specialist Agents with Focused Prompts

The core insight: one agent with a broad prompt is worse than multiple agents with focused prompts. The improvement is not just incremental — it is qualitative. Focused prompts enable a different kind of engagement from the model.

When implementing a multi-agent system, ask: can this task be decomposed into aspects that can be reviewed independently? If yes, specialize.

### Pattern 2: Fan-Out / Fan-In

The fan-out/fan-in pattern:

1. Receive a single input (the diff)
2. Fan out to N parallel agents (the reviewers)
3. Collect N outputs
4. Fan in: aggregate, filter, rank

This is structurally identical to map-reduce. The "fan out" is the `asyncio.gather` call. The "fan in" is the list comprehension that flattens and filters.

The pattern is broadly applicable:

- N summarizers, each summarizing a different section of a document
- N validators, each checking a different constraint
- N translators, each translating to a different language
- N evaluators, each scoring a proposal from a different criterion

### Pattern 3: Aggregation and Ranking

Raw multi-agent output is a collection of items from multiple sources. Aggregation means combining them into a unified list. Ranking means ordering them by relevance.

Without ranking, the user sees findings in arbitrary order (order of reviewer completion, which varies). With ranking, the user sees the most important findings first.

When designing an aggregation step, ask:

- What dimensions matter for ranking? (Severity, confidence, relevance, freshness?)
- What should be filtered out? (Low confidence, duplicate issues, out-of-scope?)
- What should be de-duplicated? (Findings that are essentially the same issue from different reviewers?)

Saathi filters by confidence and sorts by severity-then-confidence. A more sophisticated system might also de-duplicate semantically similar findings across reviewers.

### Pattern 4: Graceful Degradation

Multi-agent systems are only reliable if agent failures are isolated. An exception in one agent should not cascade to the others or abort the workflow.

The rule: **every agent should catch its own exceptions and return an empty or default result instead of raising**. The orchestrator (here, `asyncio.gather`) should receive results, not exceptions.

This pattern is essential in production systems. Networks fail. Models time out. GPU memory gets exhausted. An agent that never raises means these failures are logged and handled gracefully, not user-visible crashes.

### Pattern 5: Structured Output as the Lingua Franca

In a multi-agent system, agents need to communicate. The most practical format for structured communication between LLM agents and downstream code is JSON, validated by Pydantic.

The `Finding` model is the lingua franca of the review system. All four reviewers produce it; the aggregation, filtering, ranking, and display code consume it. New reviewers are automatically compatible because they produce the same schema.

When designing a multi-agent system:

1. Define the shared output type first (the Pydantic model)
2. Write the system prompt around that type
3. Write tolerant extraction and validation
4. Trust the validators to normalize edge cases

### Pattern 6: Separation of Agent and Orchestration

The review system is orchestration code — regular Python async functions calling LLMs directly. It is not a LangGraph graph, not an agent loop, not a tool. It runs completely outside the agent's graph.

This is the right choice when:

- The task has a fixed structure (always 4 reviewers, always aggregate)
- The task does not need checkpointing or resumability
- The task does not need tool calling
- The task's flow does not need to be dynamically determined by the LLM

Graphs are for dynamic workflows where the LLM's responses determine the next step. Orchestration code is for workflows where the structure is fixed and only the content varies.

---

## Summary

Saathi's `/code-review` command is a compact but fully-realized example of multi-agent LLM orchestration. In about 200 lines of code, it demonstrates:

- **Specialist agents** — four reviewers, each with a focused system prompt
- **Concurrent execution** — `asyncio.gather` for parallel LLM calls
- **Structured output** — `Finding` Pydantic model with defensive validators
- **Tolerant parsing** — `_extract_json` handles the full range of real-world LLM outputs
- **Quality filtering** — confidence threshold cuts noise
- **Ranked output** — severity-then-confidence sort puts the most important findings first
- **Graceful degradation** — reviewer failures return empty lists, never propagate
- **Performance awareness** — explicitly avoids Ollama JSON mode's grammar-constrained penalty
- **Separate LLM instance** — review LLM is independent of the agent LLM
- **Clean CLI integration** — the command fits the CLI's existing pattern in 20 lines
- **Testability** — 10 offline tests using simple fake LLMs cover the full stack

These patterns scale from a four-reviewer code review to a hundred-agent document analysis pipeline. The code is small enough to understand fully; the patterns are large enough to apply broadly.

---

*Next chapter: [Chapter 15 — Testing LLM Applications: Strategies and Patterns](./15-testing-llm-apps.md)*
