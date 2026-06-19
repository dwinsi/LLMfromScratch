# From Neuron to Agent

*A five-part series. The capstone of a 12-project journey.*

---

I started this repository with a single neuron, a sigmoid function, and a weather prediction example.

Twelve projects later, I have a research agent built on Gemma 4 12B that takes a question, breaks it into sub-questions, searches the web for each, and synthesises a structured answer, all orchestrated with plain Python and no frameworks.

This article series is not a tutorial on how to use Gemma 4. It is a record of what 11 projects of building from scratch taught me when I finally loaded a production model and gave it a goal. Every concept in these articles connects back to something I built by hand first.

---

## The 12-project journey

| Project | What I built |
| --- | --- |
| 01 | A single neuron from scratch |
| 02 | A neural network that learns via backpropagation |
| 03 | The same network rebuilt in PyTorch |
| 04 | A word-level RNN on a custom corpus |
| 05 | An RNN with hand-rolled attention (Q, K, V) |
| 06 | A full Transformer block in NumPy and PyTorch |
| 07 | A mini LLM with 4 stacked Transformer blocks |
| 08 | BPE tokenisation via HuggingFace tokenizers |
| 09 | A Llama-style mini LLM with RMSNorm and SwiGLU |
| 10 | The same model improved with Grouped Query Attention |
| 11 | Mixture of Experts architecture from scratch |
| 12 | A research agent built on Gemma 4 12B |

Projects 1–7 built the foundation. Projects 8–11 added the exact techniques that make production LLMs work at scale. Project 12 was the question: what happens when you give that architecture a goal?

---

## The five articles

### [Part 1 — From LLM to Agent: What Actually Changes](./01-from-llm-to-agent.md)

Seven projects to build a mini LLM. One more to give it a goal. This article bridges the build series and the agent project, what changes when a language model gets tools and a reasoning loop instead of just a prompt.

### [Part 2 — Loading a 12B Model: What I Learned](./02-loading-a-12b-model.md)

My mini LLM had 4 Transformer blocks. Gemma 4 12B has 48. This article covers what changes at scale, 4-bit quantisation, device mapping across two GPUs, and every wrong class name and deprecated API I hit along the way.

### [Part 3 — Prompt Engineering Is a Design Problem](./03-prompt-engineering-as-design.md)

The most underestimated file in the project was `prompts.py`. Writing a prompt that a model must parse reliably is not a creative exercise. It is a software design problem with contracts, failure modes, and consequences.

### [Part 4 — The ReAct Loop, Built From Scratch](./04-react-loop-from-scratch.md)

The pattern that every agentic system uses, reason, act, observe, repeat, explained through the pipeline I built. No LangChain, no abstractions, just the loop written in plain Python so the mechanism is visible.

### [Part 5 — Dense vs MoE: The Architecture Beyond](./05-dense-vs-moe.md)

My mini LLM was dense, every parameter active on every token. Gemma 4 26B activates only 4 billion parameters out of 26 billion per token using Mixture of Experts routing. This article explains why that matters and what Project 11 taught me before I understood it in a production model.

---

## Getting started

```bash
# Projects 1–11 only (no GPU required for early projects)
pip install -r requirements-core.txt

# All 12 projects including the Gemma 4 agent
pip install -r requirements.txt
```

Every project directory contains a `config.json` with all hyperparameters. Change a value there and rerun — no code edits needed.

---

## Repository structure

```text
LLMfromScratch/
├── requirements.txt              ← all 12 projects
├── requirements-core.txt         ← projects 1–11 only
│
├── 01-nuron/          config.json + neuron.py
├── 02-network/        config.json + neural_network.py
├── 03-pytorch/        config.json + weather_predictor.py
├── 04-rnn/            config.json + 04-rnn.py
├── 05-attention/      config.json + 05-rnn_attention_pytorch.py
├── 06-transformer/    config.json + 06-transformer_pytorch.py
├── 07-mini-LLM/       config.json + mini_llm.py
├── 08-BPE_tokenisation/  config.json + 08_mini_llm_bpe.py
├── 09-mini_llm_llama_style/  config.json + mini_llm_llama_style.py
├── 10-mini_llm_GQA/   config.json + mini_llm_gqa.py
├── 11-Mixture_of_Expert/  config.json + mini_llm_moe.py
│
├── ablations/
│   ├── ablation_rope_vs_sinusoidal.py   ← sinusoidal PE vs RoPE
│   ├── ablation_gqa_vs_mha.py           ← standard MHA vs GQA
│   └── ablation_dense_vs_moe.py         ← dense FFN vs MoE
│
└── Research_Agent_Gemma4/
    ├── config.json
    ├── agent.py
    ├── model.py
    ├── prompts.py
    ├── tools.py
    ├── notebook.ipynb
    └── LangChain/      ← same agent rebuilt with LCEL (see LangChain/README.md)
```

Projects 4–11 each produce a training **and** validation loss curve. Projects 5, 6, and 11 also produce visualisations: attention heatmaps and an MoE router utilisation plot.

The `ablations/` directory contains three standalone scripts that each train two model variants side by side (500 epochs) to isolate the contribution of a single architectural change.

---

## The principle behind all of it

The problem always comes first. The language follows.

I did not learn Gemma 4 by reading the documentation first. I learned it by spending 11 projects understanding what was inside it, and then loading it.

That is the only way I know how to learn something properly.

---

### github.com/dwinsi/LLMfromScratch
