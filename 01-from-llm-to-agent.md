# 01 — From LLM to Agent: What Actually Changes

*Part 1 of 5 — From Neuron to Agent*
*Series index: [README.md](./README.md)*

---

## Overview

This document traces the architectural and conceptual gap between a mini language model (Project 7, 4 stacked Transformer blocks, 159,558 parameters) and a production research agent built on Gemma 4 12B (48 Transformer blocks, 7.5 GB in 4-bit NF4 quantization).

The model architecture does not change. The way the model is used does.

---

## What a language model does

A language model has one function: given a sequence of tokens, predict the probability distribution over the next token.

```text
P(token_n | token_1, token_2, ..., token_{n-1})
```

In Project 7, this looked like:

```python
# 4 weather words in → 5th word predicted
training_sequences  → shape (batch_size, sequence_length=4)
training_targets    → shape (batch_size,)

# vocabulary_size = 198
# embedding_dim   = 64
# number_of_heads = 4
# number_of_blocks = 4
# total parameters = 159,558
```

Training result:

```text
Epoch   1 | Loss: 5.0600
Epoch 500 | Loss: 0.0000
```

In Gemma 4 12B the function is identical. The sequence length is 128,000 tokens instead of 4. The vocabulary is 256,000 tokens instead of 198. The blocks are 48 instead of 4. The mechanism is the same.

---

## What an agent is

An agent is not a different model architecture. It is an orchestration pattern — a loop that calls the model multiple times, routes tool outputs back into the model's context, and terminates when a goal is reached.

### Single model call (Projects 1–11)

```flow
input_tokens → model.forward() → output_tokens
```

One call. One response. No external state.

### Agent loop (Project 12)

```flow
user_question
      │
      ▼
model.generate()          ← Call 1: Planner
      │
parse_sub_questions()     ← structured output parsing
      │
      ▼
search_web() × N          ← Tool call: ddgs web search
      │
format_results_for_prompt()
      │
      ▼
model.generate()          ← Call 2: Synthesizer
      │
      ▼
final_answer
```

Two model calls. One tool invocation loop. The output of Call 1 determines the inputs to the tool. The tool outputs determine the input to Call 2.

---

## Implementation

### `generate_response()` — the shared primitive

Both planner and synthesizer calls go through the same function. The model does not change between calls. Only the prompt changes.

```python
def generate_response(
    model,
    processor,
    user_message: str,
    system_message: str = "You are a helpful research assistant."
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user",   "content": [{"type": "text", "text": user_message}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_token_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_token_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True
        )

    response_token_ids = output_token_ids[0][input_token_length:]
    raw_response = processor.decode(response_token_ids, skip_special_tokens=True)
    return _strip_thinking_tokens(raw_response).strip()
```

### `run_research_agent()` — the orchestration loop

```python
def run_research_agent(model, processor, user_question: str) -> str:

    # --- Call 1: Planner ---
    planner_response = generate_response(
        model=model,
        processor=processor,
        user_message=build_planner_user_prompt(user_question),
        system_message=PLANNER_SYSTEM_PROMPT
    )

    sub_questions = parse_sub_questions(planner_response)

    # --- Tool calls: Web search ---
    all_formatted_results = []

    for sub_question in sub_questions:
        search_results = search_web(sub_question)
        formatted_results = format_search_results_for_prompt(
            query=sub_question,
            results=search_results
        )
        all_formatted_results.append(formatted_results)

    combined_search_results = "\n\n".join(all_formatted_results)

    # --- Call 2: Synthesizer ---
    final_answer = generate_response(
        model=model,
        processor=processor,
        user_message=build_synthesizer_user_prompt(
            user_question=user_question,
            all_search_results_text=combined_search_results
        ),
        system_message=SYNTHESIZER_SYSTEM_PROMPT
    )

    return final_answer
```

---

## Build series to agent: direct mapping

| Build series concept | Agent project equivalent |
| --- | --- |
| `transformer_block.py` — single block | `model.language_model.layers.0` through `layers.47` |
| 4 stacked blocks in `mini_llm.py` | 48 stacked blocks in Gemma 4 12B |
| `word_to_index` vocabulary, 198 tokens | SentencePiece vocabulary, 256,000 tokens |
| `loss 5.06 → 0.00` on weather corpus | Pre-trained on web-scale data, instruction-tuned |
| Raw `model.forward()` call | `generate_response()` with `apply_chat_template` |
| Single training loop | Two-call orchestration loop with tool use |
| No external state | Web search results injected into context |

---

## Key parameters

| Parameter | Project 7 (mini LLM) | Gemma 4 12B |
| --- | --- | --- |
| Transformer blocks | 4 | 48 |
| Embedding dim | 64 | 2048 |
| Attention heads | 4 | 8 |
| Vocabulary size | 198 | 256,000 |
| Context window | 4 tokens | 128,000 tokens |
| Total parameters | 159,558 | 12,000,000,000 |
| Active parameters | 159,558 (dense) | 12B (dense) |
| Memory footprint | negligible | 7.5 GB (4-bit NF4) |

---

## References

- Gemma 4 model card: <https://huggingface.co/google/gemma-4-12B-it>
- `Gemma4UnifiedForConditionalGeneration` API: <https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified>
- `apply_chat_template` docs: <https://huggingface.co/docs/transformers/main/en/chat_templating>
- ReAct pattern (Yao et al., 2022): <https://arxiv.org/abs/2210.03629>
- Project 7 source: [../07-mini-LLM/](/07-mini-LLM/)
- Agent source: [../Research_Agent_Gemma4/agent.py](../Research_Agent_Gemma4/agent.py)

---

*[← Series Index](./README.md) | [Next: 02 — Loading a 12B Model →](./02-loading-a-12b-model.md)*
