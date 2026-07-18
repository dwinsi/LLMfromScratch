# Research Agent with LangChain

This folder contains a second version of the Gemma 4 research agent from the parent directory. The logic is identical: given a research question, the agent breaks it into sub-questions, searches the web for each, and writes a grounded answer. What changes is how that logic is expressed in code.

The parent `../agent.py` builds the pipeline using plain Python: explicit function calls, loops, and string handling. This version builds the same pipeline using **LangChain**, a framework that provides reusable building blocks for connecting language models to tools and data.

---

## What is LangChain?

LangChain is a Python library that gives you pre-built components for common AI application patterns. Instead of writing the same boilerplate every time (load a model, format a prompt, call the model, parse the output, call a tool, repeat), LangChain provides standard pieces that snap together.

Think of it like plumbing connectors. You have a source (a prompt), a processor (a model), and a drain (an output). LangChain defines the standard pipe fittings that connect them. Once you learn the standard fittings, you can assemble new pipelines quickly and swap components without rewriting the connections.

The specific part of LangChain this project uses is called **LCEL** (LangChain Expression Language).

---

## What is LCEL?

LCEL is a way of writing pipelines using the pipe operator `|`, the same symbol used in Unix shell commands.

In a Unix shell, `cat file.txt | grep "word" | wc -l` means: take the file, pass it through grep, pass the result through wc. Each step receives the output of the previous step.

LCEL works the same way for AI pipelines:

```python
chain = prompt | model | output_parser
result = chain.invoke({"question": "What is photosynthesis?"})
```

This means: format the prompt, pass it to the model, parse the output. Each component receives the output of the previous one. The entire pipeline is one composable object that can be invoked, streamed, or run in batch mode.

The equivalent in plain Python (Phase 1 approach) looks like:

```python
formatted_prompt = format_prompt(question)
raw_output = model.generate(formatted_prompt)
result = clean_response(raw_output)
```

Both do the same thing. The LCEL version is more composable and integrates with LangChain's ecosystem of tools and observability features.

---

## How this agent works

The pipeline has three stages, same as Phase 1:

```text
Stage 1: Planner
  Input:  a research question from the user
  Action: ask the language model to break it into 2 or 3 focused sub-questions
  Output: a list of sub-questions

Stage 2: Search
  Input:  each sub-question from the Planner
  Action: search the web via DuckDuckGo and collect the top results
  Output: a collection of search results, one per sub-question

Stage 3: Synthesizer
  Input:  the original question + all search results
  Action: ask the language model to write a grounded answer citing the sources
  Output: the final answer
```

The LCEL version expresses each stage as a chain built from components. The components in this project are:

**ChatPromptTemplate.** A reusable prompt object that knows how to format itself. You define the template once with placeholder variables, then fill in the variables at runtime. This replaces the string-building functions in the Phase 1 `prompts.py`.

**ChatHuggingFace wrapped in HuggingFacePipeline.** A LangChain-compatible wrapper around the Gemma 4 model. The model loaded in `model.py` is the same 4-bit quantized Gemma 4 12B from Phase 1, but wrapped so it speaks LangChain's interface. This means any LangChain component that expects "a chat model" will work with it.

**DuckDuckGoSearchRun.** A LangChain-compatible wrapper around DuckDuckGo search. Instead of calling the `ddgs` library directly, you use this tool object. It fits into the LangChain tool interface, which means you can swap it for a different search provider later by changing one line.

**StrOutputParser.** A simple parser that takes the model's output object and extracts the plain text string. In Phase 1 this was the manual `clean_response()` function.

---

## Files

| File | Role |
| --- | --- |
| `agent.py` | Builds the LCEL chains and runs the full pipeline |
| `model.py` | Loads Gemma 4 12B at 4-bit quantization and wraps it for LangChain |
| `prompts.py` | ChatPromptTemplate objects for the Planner and Synthesizer stages |
| `tools.py` | Web search via DuckDuckGoSearchRun |

---

## Setup and running

### Step 1: install dependencies

```bash
pip install transformers bitsandbytes torch langchain-core langchain-huggingface langchain-community python-dotenv
```

The extra packages compared to Phase 1 are `langchain-core`, `langchain-huggingface`, and `langchain-community`. These provide the LangChain base interfaces, the HuggingFace wrappers, and the community tools (DuckDuckGo search) respectively.

### Step 2: set up environment variables

Copy the example file from the parent directory:

```bash
cp ../.env.example .env
```

Edit `.env` and fill in your Hugging Face token:

```text
HF_TOKEN=your_huggingface_token_here

# Optional: enable LangSmith tracing (see below)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_key_here
# LANGCHAIN_PROJECT=gemma4-research-agent
```

### Step 3: run

```bash
python agent.py
```

The research question is hardcoded in the `main()` function at the bottom of `agent.py`. Edit it there to ask a different question.

---

## Optional: LangSmith tracing

Phase 1 uses `print()` statements to observe what the agent is doing at each step. Phase 2 has an optional upgrade: LangSmith tracing.

LangSmith is a web dashboard (from the LangChain team) that automatically records every step of every chain run. When tracing is enabled, each agent run appears in the dashboard with a full breakdown: what prompt was sent, what the model returned, how long each step took, and the token counts. You can drill into any step and inspect the exact inputs and outputs.

To enable it, uncomment the three `LANGCHAIN_` lines in your `.env` file and create a free account at smith.langchain.com to get an API key.

This is the main observability advantage of Phase 2 over Phase 1. For debugging a single run, `print()` statements are often enough. For comparing runs, tracking latency, or sharing traces with others, LangSmith is more useful.

---

## Phase 1 vs Phase 2: when to use each

**Use Phase 1 (raw Python, `../agent.py`) when:**

- You are learning how agents work and want to see every step explicitly
- You are debugging the model's reasoning and want to inspect raw outputs
- You do not need LangChain-specific integrations or ecosystem tools
- You want the minimum possible dependencies

**Use Phase 2 (LangChain, this folder) when:**

- You want to swap components with minimal code changes (different model, different search tool, different prompt format)
- You want production observability through LangSmith
- You are building this as part of a larger LangChain application
- You prefer the declarative pipeline style over explicit loops

---

## Hardware requirements

Same as Phase 1. Gemma 4 12B at 4-bit NF4 quantization requires approximately 8 GB of GPU VRAM. This project was tested on Kaggle's dual T4 setup (2 x 16 GB). A single A100 or equivalent also works.
