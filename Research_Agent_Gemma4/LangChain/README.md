# LangChain Implementation — Phase 2

This directory is a **second implementation of the same Gemma 4 research agent** from the parent directory, rebuilt using [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/).

The parent `../agent.py` is Phase 1: raw Python loops with direct model calls.  
This `agent.py` is Phase 2: the same logic expressed as composable LCEL chains.

---

## What's different

| Aspect | Phase 1 (`../agent.py`) | Phase 2 (this directory) |
| --- | --- | --- |
| **Chain pattern** | Explicit `generate_response()` calls | `prompt \| model \| parser \| lambda` pipes |
| **Model wrapper** | Direct `processor` + `model.generate()` | `ChatHuggingFace(HuggingFacePipeline(...))` |
| **Prompts** | String-building functions in `prompts.py` | `ChatPromptTemplate` objects |
| **Search tool** | Direct `ddgs` library calls | `DuckDuckGoSearchRun` from `langchain_community` |
| **Output parsing** | Manual `clean_response()` | `StrOutputParser()` + `RunnableLambda` |
| **Observability** | `print()` statements | Optional [LangSmith](https://smith.langchain.com/) tracing |

The pipeline logic — Planner → Search → Synthesizer — is identical in both implementations.

---

## When to use each

**Use Phase 1 (raw Python) when:**

- You want to understand exactly what's happening at every step
- You're debugging the model's reasoning or output format
- You're teaching or learning how agents work under the hood
- You don't need LangChain ecosystem integrations

**Use Phase 2 (LCEL) when:**

- You want to swap components (e.g. switch model, add memory, change tools) with minimal code changes
- You need production observability via LangSmith tracing
- You're building on top of a larger LangChain application
- You prefer declarative, composable pipelines over explicit loops

---

## How to run

### 1. Install dependencies

```bash
pip install transformers bitsandbytes torch langchain-core langchain-huggingface langchain-community python-dotenv
```

### 2. Set up environment variables

Copy the example file and fill in your token:

```bash
cp ../.env.example .env
```

Edit `.env`:

```text
HF_TOKEN=your_huggingface_token_here

# Optional: enable LangSmith tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_key_here
# LANGCHAIN_PROJECT=gemma4-research-agent
```

### 3. Run

```bash
python agent.py
```

The agent will:

1. Accept a research question (hardcoded in `agent.py` — edit `main()` to change it)
2. Use the Planner to break it into 2–3 sub-questions
3. Search the web for each sub-question via DuckDuckGo
4. Use the Synthesizer to write a grounded answer from search results

---

## File overview

| File | Role |
| --- | --- |
| `agent.py` | Orchestration — builds LCEL chains, runs the pipeline |
| `model.py` | Loads Gemma 4 12B (4-bit NF4) and wraps it in `ChatHuggingFace` |
| `prompts.py` | `ChatPromptTemplate` objects for Planner and Synthesizer |
| `tools.py` | Web search via `DuckDuckGoSearchRun` |

---

## Hardware requirements

Same as Phase 1: Gemma 4 12B at 4-bit quantization requires ~8 GB VRAM.  
Tested on Kaggle T4 × 2 (2 × 16 GB). A single A100 or equivalent also works.
