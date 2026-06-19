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

## Repository structure

```Folder
LLMfromScratch/
├── 01-nuron/
├── 02-network/
├── 03-pytorch/
├── 04-rnn/
├── 05-attention/
├── 06-transformer/
├── 07-mini-LLM/
├── 08-BPE_tokenisation/
├── 09-mini_llm_llama_style/
├── 10-mini_llm_GQA/
├── 11-Mixture_of_Expert/
└── Research_Agent_Gemma4/
    ├── agent.py
    ├── model.py
    ├── prompts.py
    ├── tools.py
    ├── notebook.ipynb
    └── articles/
        ├── README.md          ← you are here
        ├── 01-from-llm-to-agent.md
        ├── 02-loading-a-12b-model.md
        ├── 03-prompt-engineering-as-design.md
        ├── 04-react-loop-from-scratch.md
        └── 05-dense-vs-moe.md
```

---

## The principle behind all of it

The problem always comes first. The language follows.

I did not learn Gemma 4 by reading the documentation first. I learned it by spending 11 projects understanding what was inside it, and then loading it.

That is the only way I know how to learn something properly.

---

### github.com/dwinsi/LLMfromScratch
