# jamba — Mamba + Attention + Mixture-of-Experts, from scratch

A standalone project that combines the three ideas the modern LLM frontier
converged on, in one configurable model you can train on a MacBook Air M2 (or any
machine):

- **Mamba blocks** — selective state space, `O(L)` memory, fixed-size state
- **Attention blocks** — exact recall, `O(L²)`, inserted sparingly
- **Mixture-of-Experts** — many expert FFNs, only the top-k run per token

This mirrors what AI21's **Jamba** ships: a mostly-Mamba backbone with a little
attention to patch recall, plus MoE for capacity. Everything is written from
scratch and heavily commented.

---

## ⚠️ Read this first — how to run (avoids the "No module named 'model'" error)

The scripts import each other (`from model import ...`). Python only finds those
imports if it knows where the files are. This project handles that automatically
via `_bootstrap.py` (every script imports it first, which adds the project folder
to Python's path) — **so it works no matter which directory you run from.**

Still, the clean way to run is:

```bash
cd path/to/jamba        # go INTO the project folder
python smoke_test.py    # then run
```

If you use an IDE (VS Code / PyCharm), open the **jamba folder itself** as your
project/workspace so the working directory matches. If you ever still hit an
import error, it means `model.py` and the script aren't in the same folder — make
sure all files below live together in one `jamba/` folder.

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | The `JambaLM` model — mamba/attention mixers + dense/MoE FFNs, plus the char tokenizer |
| `utils.py` | Shared helpers: device pick, data loading, batching, LR schedule |
| `_bootstrap.py` | Tiny import-path fix imported by every script (prevents ModuleNotFoundError) |
| `train.py` | Train any architecture (`--arch mamba/transformer/hybrid/jamba`) |
| `demo.py` | Compare all four (params, active/token, layout) + MoE routing — **no training** |
| `generate.py` | Load a checkpoint and generate text |
| `visualize_model.py` | torchinfo / torchview / Netron + MoE routing heatmap |
| `smoke_test.py` | ~15-second check that everything runs — **run first** |
| `requirements.txt` | `torch` + `matplotlib` |
| `requirements-viz.txt` | Optional visualization libraries |

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install torch matplotlib

python smoke_test.py     # confirm everything runs
python demo.py           # see the four architectures compared (instant)
python train.py          # train the full jamba model on Shakespeare
python generate.py --seed "ROMEO:"    # generate from the trained model
```

---

## Why this architecture

Each ingredient fixes a specific limitation of the others:

| Ingredient | Strength | Weakness |
|---|---|---|
| Mamba (SSM) | linear cost, fixed memory, long context | blurry exact recall |
| Attention | perfect exact recall / in-context lookup | quadratic cost, growing KV cache |
| MoE | huge capacity at low compute per token | more total parameters to store |

Jamba's insight: use Mamba for *most* layers (cheap), sprinkle in attention to
recover exact recall, and use MoE to add capacity without paying for it on every
token. You get long-context efficiency **and** recall **and** capacity.

---

## The four presets

All the same width and depth (8 layers, d_model=128) — only the block mix changes:

```
mamba        [m. m. m. m. m. m. m. m.]   pure SSM, dense FFN
transformer  [A. A. A. A. A. A. A. A.]   pure attention, dense FFN
hybrid       [m. m. m. A. m. m. m. A.]   mostly mamba + attention, dense FFN
jamba        [m. mE m. AE m. mE m. AE]   hybrid mixers + MoE      <- the full thing

legend:  m = mamba   A = attention   . = dense FFN   E = MoE FFN
```

`demo.py` prints exactly this plus the parameter accounting.

---

## The MoE bargain (the thing to actually look at)

Run `python demo.py` and compare the `jamba` row to the others:

- **Total parameters**: `jamba` has by far the most — MoE adds 8 expert FFNs per
  MoE layer, so lots of capacity.
- **Active params per token**: stays close to the others — each token only runs
  **top-2 of 8** experts (25% of expert params).

That's the whole point of Mixture-of-Experts: **more knowledge capacity, nearly the
same compute per token.**

### How the MoE layer works (in `model.py`)

1. A small **router** (one linear layer) scores all 8 experts for each token.
2. Each token keeps its **top-2** experts; their gate weights renormalize to sum to 1.
3. Only those 2 experts run; outputs are combined by the gate weights.
4. An **auxiliary load-balancing loss** is added to training to stop the router
   collapsing onto a few experts. It's minimized when tokens spread evenly.

Each expert is a **SwiGLU** FFN (`silu(gate(x)) * up(x) -> down`).

---

## Visualizing the network

```bash
pip install -r requirements-viz.txt
brew install graphviz        # macOS — needed by torchview
```

- **torchinfo** — per-layer table. `--tool torchinfo`
- **torchview** — architecture diagram PNG. `--tool torchview`
- **Netron** — interactive ONNX explorer. `--tool netron`
- **MoE routing heatmap** — how tokens spread across experts. `--tool moe`
- **TensorBoard** — wired into `train.py`; logs loss + **MoE aux loss**.
  `tensorboard --logdir runs`

```bash
python visualize_model.py --arch jamba --tool all
python visualize_model.py --arch jamba --tool moe   # -> moe_routing_jamba.png
```

The MoE routing heatmap is the one to look at: on an untrained model the router
spreads tokens roughly uniformly; visualize after training to see whether load
balancing held.

---

## Configuration (`CONFIG` in `train.py`)

| Setting | Default | Meaning |
|---|---|---|
| `n_layers` | 8 | total layers |
| `attn_every` | 4 | attention every Nth layer (hybrid/jamba) |
| `moe_every` | 2 | MoE FFN every Nth layer (jamba only) |
| `n_experts` | 8 | experts per MoE layer |
| `top_k` | 2 | experts used per token |
| `d_model` | 128 | model width |
| `context_len` | 256 | sequence length |
| `batch_size` | 32 | lower to 16 if memory is tight on the 8GB Air |

---

## Notes & honest caveats

- **Sequential scan.** The Mamba scan is a readable Python loop — correct and
  MPS-friendly, but not as fast as a fused parallel-scan kernel. Fine at this scale.
- **MoE dispatch.** Experts run via a per-expert masking loop: clear and correct;
  production MoE uses grouped GEMM + expert parallelism for speed.
- **MPS quirks.** If you hit an unsupported op, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Character-level.** Uses a char tokenizer for simplicity.

---

## How this connects to the bigger picture

```
Legendre polynomials  →  optimal projection basis
HiPPO                 →  stable A matrix for compressing history
S4 / S4D              →  makes it fast (diagonal A)
Mamba (selective SSM) →  input-dependent B, C, Δ
Jamba (this project)  →  Mamba backbone + sparse attention + MoE
```

This is the current frontier recipe in miniature: an efficient sequence mixer
(attention variants or SSMs) plus a sparse MoE feed-forward — the two axes every
2026 frontier model is built on.
