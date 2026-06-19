# 03 — Prompt Engineering Is a Design Problem

*Part 3 of 5 — From Neuron to Agent*
*Series index: [README.md](./README.md)*

---

## Overview

This document covers `prompts.py` — the most underestimated file in the research agent project.

Prompt engineering is often described as a creative skill. In an agentic pipeline it is a software design problem. The model's output must be parseable by the next step in the pipeline. A prompt that produces inconsistent output format does not just return a bad answer — it breaks the loop entirely.

This document covers:

- The difference between a system prompt and a user prompt
- Why `apply_chat_template()` must be used instead of manual string construction
- Output contracts and why the planner prompt needs a concrete example
- Thinking token stripping
- The full `prompts.py` implementation

---

## System prompt vs user prompt

Gemma 4 12B is an instruction-tuned model. It expects inputs structured as a conversation with explicit roles. The two roles used in this project are `system` and `user`.

| Role | Purpose | Changes at runtime |
| --- | --- | --- |
| `system` | Defines model behaviour, constraints, output format | No — fixed per pipeline step |
| `user` | Injects dynamic content — the question, the search results | Yes — changes every call |

In `generate_response()`, both are passed as structured message dicts:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": user_message}]
    }
]
```

The system prompt is a module-level constant. The user prompt is a function that returns a string. They are kept separate because the pipeline calls `generate_response()` twice with different system prompts — one for the planner, one for the synthesizer.

---

## Why `apply_chat_template()` and not manual string construction

Gemma 4 12B has a custom chat template defined in its `tokenizer_config.json`. The template specifies the exact special tokens, role markers, and delimiters the model was trained to expect.

Manual string construction gets this wrong:

```python
# Wrong — do not do this
prompt = f"<system>{system_message}</system>\n<user>{user_message}</user>"
```

The correct approach:

```python
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

`apply_chat_template()` reads the template from the model's tokenizer config and formats the messages exactly as the model was trained to receive them. If Google updates the template in a future revision, the code requires no changes — the processor handles it automatically.

`add_generation_prompt=True` appends the model-turn start token at the end of the formatted input, signalling to the model that it should begin generating a response.

References:

- <https://huggingface.co/docs/transformers/main/en/chat_templating>

---

## The planner prompt

The planner has one job: take a research question and return a numbered list of 2–3 sub-questions. The output of this call is parsed directly by `parse_sub_questions()`. If the model returns anything other than a numbered list — an introduction, an explanation, a closing remark — the parser either silently drops content or returns fewer sub-questions than expected.

### System prompt

```python
PLANNER_SYSTEM_PROMPT = """You are a research planner. Your only job is to break a complex question into 2 or 3 focused sub-questions that can each be answered by a single web search.

Rules you must follow without exception:
- Output ONLY a numbered list. No introduction, no explanation, no closing remarks.
- Each sub-question must be specific and self-contained.
- Each sub-question must be on its own line.
- Do not number beyond 3.
- Do not output anything other than the numbered list.

Example output format:
1. What is the history of quantum computing?
2. What are the latest breakthroughs in quantum computing in 2025?
3. Which companies are leading quantum computing research today?"""
```

Three design decisions in this prompt:

**1. "Rules you must follow without exception"** — instruction-tuned models respond to imperative framing. Softer phrasing like "please try to" produces less consistent adherence.

**2. Explicit negative constraints** — "No introduction, no explanation, no closing remarks" is stated separately from the positive instruction. Without this, the model frequently adds preamble such as "Sure, here are your sub-questions:" which breaks the parser.

**3. Concrete output example** — the example shows the exact format required. This is the single most effective technique for constraining output structure. The model has seen numbered lists in its training data in many formats. The example eliminates ambiguity about which format is expected here.

### User prompt

```python
def build_planner_user_prompt(user_question: str) -> str:
    return f"Break this research question into 2 or 3 focused sub-questions:\n\n{user_question}"
```

The user prompt restates the task in imperative form and injects the dynamic content. It mirrors the instruction from the system prompt intentionally — repetition increases adherence.

### Verified output

Input question: `"How do transformer models handle long-range dependencies?"`

Planner output:

```text
1. What is the role of the self-attention mechanism in capturing dependencies between tokens?
2. What are the limitations of the standard transformer architecture regarding sequence length and memory complexity?
3. What techniques, such as sparse attention or sliding windows, are used to optimize long-range dependency handling?
```

Clean numbered list. No preamble. No closing remark. Parser-ready.

---

## The synthesizer prompt

The synthesizer receives the original question plus all formatted search results concatenated together. Its job is to write a structured answer grounded only in those results.

### System_prompt

```python
SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. You receive a research question and a set of web search results. Your job is to write a clear, well-structured answer based strictly on the search results provided.

Rules you must follow:
- Base your answer only on the search results provided. Do not add information from outside the results.
- Structure your answer with a short introduction, key findings, and a brief conclusion.
- When referencing a specific result, cite it by its result number, for example: [1], [2].
- If the search results do not contain enough information to answer the question, say so clearly.
- Write in plain, precise language. No filler phrases."""
```

The critical constraint is "Do not add information from outside the results." Without this, the model blends its pre-training knowledge with the search results and the citations become unreliable — a result number may appear in the output but not correspond to the actual source of that information.

### User_prompt

```python
def build_synthesizer_user_prompt(
    user_question: str,
    all_search_results_text: str
) -> str:
    return (
        f"Research question: {user_question}\n\n"
        f"Search results:\n{all_search_results_text}\n\n"
        f"Write a structured answer to the research question based on the search results above."
    )
```

`all_search_results_text` is the concatenated output of `format_search_results_for_prompt()` called once per sub-question. Each block is labeled with its query header and result numbers `[1]` through `[5]`, giving the model clear anchors for citation.

---

## Output parsing

`parse_sub_questions()` lives in `prompts.py`, not in `agent.py`. Parsing is a prompt concern. If the planner output format changes, the prompt and the parser change together in the same file.

```python
def parse_sub_questions(planner_response: str) -> list[str]:
    sub_questions = []

    for line in planner_response.strip().splitlines():
        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        # Strip leading numbering patterns: "1.", "1)", "1 "
        if cleaned_line[0].isdigit():
            stripped = cleaned_line.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if stripped:
                sub_questions.append(stripped)

    return sub_questions
```

The parser handles three numbering variants — `1.`, `1)`, `1` — because instruction-tuned models occasionally vary punctuation even with an explicit example. The parser does not assume a fixed number of sub-questions and returns whatever is found, including zero. The agent loop checks for an empty return and aborts with an explicit message rather than passing an empty list to the search step.

Verified:

```python
mock_output = (
    "1. What is the attention mechanism in transformer models?\n"
    "2. How do transformers differ from RNNs in handling long sequences?\n"
    "3. What are the limitations of transformers with very long contexts?"
)

parse_sub_questions(mock_output)
# ["What is the attention mechanism in transformer models?",
#  "How do transformers differ from RNNs in handling long sequences?",
#  "What are the limitations of transformers with very long contexts?"]
```

---

## Thinking token stripping

Gemma 4 supports a thinking mode where internal reasoning is wrapped in delimiters:

```text
<|channel>thought
... internal reasoning ...
<channel|>
```

These blocks appear in the raw decoded output before the final answer. For the agent pipeline, thinking tokens must be stripped before the planner output is passed to the parser and before the synthesizer output is returned as the final answer.

```python
def _strip_thinking_tokens(text: str) -> str:
    cleaned = re.sub(
        r"<\|channel>thought.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()
```

`re.DOTALL` is required because the thinking block spans multiple lines. Without it, `.` does not match newline characters and the regex fails to remove the block.

References:

- <https://ai.google.dev/gemma/docs/capabilities/text/basic>

---

## Complete `prompts.py`

```python
"""
prompts.py — Gemma 4 Research Agent
Prompt templates for the planner and synthesizer steps of the agent loop.

References:
    Gemma 4 system prompt support:
        https://huggingface.co/google/gemma-4-12B-it
    apply_chat_template:
        https://huggingface.co/docs/transformers/main/en/chat_templating
    Gemma 4 thinking mode:
        https://ai.google.dev/gemma/docs/capabilities/text/basic
"""

import re


PLANNER_SYSTEM_PROMPT = """You are a research planner. Your only job is to break a complex question into 2 or 3 focused sub-questions that can each be answered by a single web search.

Rules you must follow without exception:
- Output ONLY a numbered list. No introduction, no explanation, no closing remarks.
- Each sub-question must be specific and self-contained.
- Each sub-question must be on its own line.
- Do not number beyond 3.
- Do not output anything other than the numbered list.

Example output format:
1. What is the history of quantum computing?
2. What are the latest breakthroughs in quantum computing in 2025?
3. Which companies are leading quantum computing research today?"""


SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. You receive a research question and a set of web search results. Your job is to write a clear, well-structured answer based strictly on the search results provided.

Rules you must follow:
- Base your answer only on the search results provided. Do not add information from outside the results.
- Structure your answer with a short introduction, key findings, and a brief conclusion.
- When referencing a specific result, cite it by its result number, for example: [1], [2].
- If the search results do not contain enough information to answer the question, say so clearly.
- Write in plain, precise language. No filler phrases."""


def build_planner_user_prompt(user_question: str) -> str:
    return f"Break this research question into 2 or 3 focused sub-questions:\n\n{user_question}"


def build_synthesizer_user_prompt(
    user_question: str,
    all_search_results_text: str
) -> str:
    return (
        f"Research question: {user_question}\n\n"
        f"Search results:\n{all_search_results_text}\n\n"
        f"Write a structured answer to the research question based on the search results above."
    )


def parse_sub_questions(planner_response: str) -> list[str]:
    sub_questions = []
    for line in planner_response.strip().splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        if cleaned_line[0].isdigit():
            stripped = cleaned_line.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if stripped:
                sub_questions.append(stripped)
    return sub_questions


def _strip_thinking_tokens(text: str) -> str:
    cleaned = re.sub(
        r"<\|channel>thought.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()
```

---

## References

- `apply_chat_template`: <https://huggingface.co/docs/transformers/main/en/chat_templating>
- Gemma 4 thinking mode: <https://ai.google.dev/gemma/docs/capabilities/text/basic>
- Gemma 4 model card: <https://huggingface.co/google/gemma-4-12B-it>
- `GenerationConfig` (generate() parameters): <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation>
- Agent source: [../Research_Agent_Gemma4/prompts.py](../Research_Agent_Gemma4/prompts.py)

---

*[← 02 — Loading a 12B Model](./02-loading-a-12b-model.md) | [Series Index](./README.md) | [Next: 04 — The ReAct Loop →](./04-react-loop-from-scratch.md)*
