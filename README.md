# From Neuron to Agent

I started this repository with a single neuron, a sigmoid function, and a weather prediction example.

Twelve projects later, I have a research agent built on Gemma 4 12B that takes a question, breaks it into sub-questions, searches the web for each, and synthesises a structured answer — all orchestrated with plain Python and no frameworks.

This is not a tutorial on how to use Gemma 4. It is a record of what 11 projects of building from scratch taught me when I finally loaded a production model and gave it a goal. Every concept in the article series connects back to something I built by hand first.

---

## The 12-project journey

| # | What I built | Key concept introduced |
| --- | --- | --- |
| 01 | A single neuron from scratch | Sigmoid, weighted sum, forward pass |
| 02 | A network that learns via backpropagation | MSE loss, gradient descent |
| 03 | The same network rebuilt in PyTorch | `nn.Module`, autograd, SGD |
| 04 | A word-level RNN on a custom corpus | Hidden state, sequence prediction, cross-entropy |
| 05 | An RNN with hand-rolled attention (Q, K, V) | Scaled dot-product attention |
| 06 | A full Transformer block in NumPy and PyTorch | Multi-head attention, residual connections, positional encoding |
| 07 | A mini LLM with 4 stacked Transformer blocks | Stacking, temperature sampling |
| 08 | BPE tokenisation via HuggingFace tokenizers | Subword tokens, vocabulary compression |
| 09 | A Llama-style mini LLM | RMSNorm, SwiGLU, RoPE |
| 10 | The same model with Grouped Query Attention | KV head sharing |
| 11 | Mixture of Experts architecture from scratch | Top-K routing, load balancing loss |
| 12 | A research agent built on Gemma 4 12B | ReAct loop, quantisation, orchestration |

Projects 1–7 built the foundation. Projects 8–11 added the exact techniques that make production LLMs efficient at scale. Project 12 was the question: what happens when you give that architecture a goal?

---

## Getting started

```bash
# Projects 1–11 (no large model downloads required)
pip install -r requirements-core.txt

# All 12 projects including the Gemma 4 agent
pip install -r requirements.txt
```

Every project has a `config.json` with all hyperparameters. Change a value there and rerun — no code edits needed.

Each project from 04 onwards produces a training **and** validation loss curve. Projects 05, 06, and 11 also generate visualisations: attention heatmaps and an MoE router utilisation plot.

---

## What's been added beyond the build series

### Ablation studies — `ablations/`

Three standalone scripts that each train two model variants for 500 epochs to isolate the contribution of a single architectural change:

| Script | What it compares |
| --- | --- |
| `ablation_rope_vs_sinusoidal.py` | Sinusoidal positional encoding vs RoPE |
| `ablation_gqa_vs_mha.py` | Standard multi-head attention vs Grouped Query Attention |
| `ablation_dense_vs_moe.py` | Dense SwiGLU FFN vs Mixture of Experts |

### LangChain comparison — `Research_Agent_Gemma4/LangChain/`

A second implementation of the same research agent using LangChain Expression Language (LCEL). The pipeline logic is identical to the main `agent.py`; only the orchestration layer changes. See [LangChain/README.md](Research_Agent_Gemma4/LangChain/README.md) for a side-by-side comparison of the two approaches and guidance on when to use each.

---

## The five articles

### [Part 1 — From LLM to Agent: What Actually Changes](./01-from-llm-to-agent.md)

Seven projects to build a mini LLM. One more to give it a goal. This article bridges the build series and the agent project — what changes when a language model gets tools and a reasoning loop instead of just a prompt.

### [Part 2 — Loading a 12B Model: What I Learned](./02-loading-a-12b-model.md)

My mini LLM had 4 Transformer blocks. Gemma 4 12B has 48. This article covers what changes at scale — 4-bit quantisation, device mapping across two GPUs, and every wrong class name and deprecated API I hit along the way.

### [Part 3 — Prompt Engineering Is a Design Problem](./03-prompt-engineering-as-design.md)

The most underestimated file in the project was `prompts.py`. Writing a prompt that a model must parse reliably is not a creative exercise. It is a software design problem with contracts, failure modes, and consequences.

### [Part 4 — The ReAct Loop, Built From Scratch](./04-react-loop-from-scratch.md)

The pattern that every agentic system uses — reason, act, observe, repeat — explained through the pipeline I built. No LangChain, no abstractions, just the loop written in plain Python so the mechanism is visible.

### [Part 5 — Dense vs MoE: The Architecture Beyond](./05-dense-vs-moe.md)

My mini LLM was dense — every parameter active on every token. Gemma 4 26B activates only 4 billion parameters out of 26 billion per token using Mixture of Experts routing. This article explains why that matters and what Project 11 taught me before I understood it in a production model.

---

## The principle behind all of it

The problem always comes first. The language follows.

I did not learn Gemma 4 by reading the documentation first. I learned it by spending 11 projects understanding what was inside it, and then loading it.

That is the only way I know how to learn something properly.

---

github.com/dwinsi/LLMfromScratch
