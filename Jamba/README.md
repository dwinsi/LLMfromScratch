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
| ------ | --------- |
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

## Project folder layout

After training, the project produces two runtime directories alongside the source files:

```folder
jamba/
├── checkpoints/          # saved model weights (one .pt file per arch)
│   ├── jamba.pt
│   ├── mamba.pt
│   ├── transformer.pt
│   └── hybrid.pt
└── runs/                 # TensorBoard event logs (one subfolder per arch)
    ├── jamba/
    ├── mamba/
    ├── transformer/
    └── hybrid/
```

### `checkpoints/` — model snapshots

`train.py` evaluates the model every 250 steps and saves a checkpoint whenever a new **best validation loss** is achieved:

```python
torch.save({
    "model":  model.state_dict(),   # all learned weights
    "config": CONFIG,               # full hyperparameter dict
    "arch":   arch,                 # "jamba", "mamba", etc.
    "vocab":  (tok.stoi, tok.itos)  # char ↔ index mappings
}, ckpt)
```

Only the best checkpoint per architecture is kept (overwritten on improvement). This means `checkpoints/jamba.pt` always holds the lowest-val-loss weights seen so far. `generate.py` loads this file to reconstruct the model and produce text without retraining.

**Why save only the best?** During cosine-decay training the loss isn't monotonically decreasing — checkpointing only on improvement prevents accidentally restoring a worse mid-run snapshot.

### `runs/` — TensorBoard training logs

`train.py` creates a `SummaryWriter` pointed at `runs/{arch}/`. At every evaluation step it logs four scalars:

| Tag | What it measures |
| --- | ---------------- |
| `loss/train` | mean cross-entropy over 50 random training batches |
| `loss/val` | mean cross-entropy over 50 random validation batches |
| `lr` | current learning rate (post warmup/cosine) |
| `moe/aux_loss` | MoE load-balancing penalty (should decrease if routing stays balanced) |

To open the dashboard:

```bash
tensorboard --logdir runs
# then open http://localhost:6006
```

The `moe/aux_loss` curve is the most informative one for Jamba specifically — it tells you whether the router is spreading tokens evenly across experts or collapsing onto a few.

---

## Why this architecture

Each ingredient fixes a specific limitation of the others:

| Ingredient | Strength | Weakness |
| --- | --- | --- |
| Mamba (SSM) | linear cost, fixed memory, long context | blurry exact recall |
| Attention | perfect exact recall / in-context lookup | quadratic cost, growing KV cache |
| MoE | huge capacity at low compute per token | more total parameters to store |

Jamba's insight: use Mamba for *most* layers (cheap), sprinkle in attention to
recover exact recall, and use MoE to add capacity without paying for it on every
token. You get long-context efficiency **and** recall **and** capacity.

---

## The four presets

All the same width and depth (8 layers, d_model=128) — only the block mix changes:

```text
mamba        [m. m. m. m. m. m. m. m.]   pure SSM, dense FFN
transformer  [A. A. A. A. A. A. A. A.]   pure attention, dense FFN
hybrid       [m. m. m. A. m. m. m. A.]   mostly mamba + attention, dense FFN
jamba        [m. mE m. AE m. mE m. AE]   hybrid mixers + MoE      <- the full thing

legend:  m = mamba   A = attention   . = dense FFN   E = MoE FFN
```

`demo.py` prints exactly this plus the parameter accounting.

---

## Architecture diagrams

### High-level forward pass

```mermaid
flowchart TD
    A["Input tokens\n(B, L)"] --> B["Token Embedding\n(B, L, d_model)"]
    B --> C["+ Positional Embedding\n(attention arches only)"]
    C --> D["Dropout"]
    D --> E["Block 1"]
    E --> F["Block 2"]
    F --> G["..."]
    G --> H["Block 8"]
    H --> I["LayerNorm"]
    I --> J["Linear head → logits\n(B, L, vocab_size)"]
    J --> K["Cross-Entropy Loss\n+ MoE Aux Loss"]

    style A fill:#6366f1,color:#fff,stroke:#4f46e5
    style B fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style C fill:#a78bfa,color:#fff,stroke:#7c3aed
    style D fill:#c4b5fd,color:#1e1b4b,stroke:#7c3aed
    style E fill:#06b6d4,color:#fff,stroke:#0891b2
    style F fill:#06b6d4,color:#fff,stroke:#0891b2
    style G fill:#06b6d4,color:#fff,stroke:#0891b2
    style H fill:#06b6d4,color:#fff,stroke:#0891b2
    style I fill:#10b981,color:#fff,stroke:#059669
    style J fill:#10b981,color:#fff,stroke:#059669
    style K fill:#f43f5e,color:#fff,stroke:#e11d48
```

### Single Block (pre-norm + residual)

Every one of the 8 blocks has the same two-sublayer structure regardless of which mixer or FFN type is used:

```mermaid
flowchart LR
    x["x (B,L,d)"] --> n1["LayerNorm"]
    n1 --> mix["Mixer\n(Mamba or Attention)"]
    mix --> add1(("+"))
    x --> add1
    add1 --> n2["LayerNorm"]
    n2 --> ffn["FFN\n(Dense or MoE)"]
    ffn --> add2(("+"))
    add1 --> add2
    add2 --> out["x' (B,L,d)"]

    style x fill:#6366f1,color:#fff,stroke:#4f46e5
    style n1 fill:#f59e0b,color:#fff,stroke:#d97706
    style mix fill:#3b82f6,color:#fff,stroke:#2563eb
    style add1 fill:#e2e8f0,color:#1e293b,stroke:#94a3b8
    style n2 fill:#f59e0b,color:#fff,stroke:#d97706
    style ffn fill:#10b981,color:#fff,stroke:#059669
    style add2 fill:#e2e8f0,color:#1e293b,stroke:#94a3b8
    style out fill:#6366f1,color:#fff,stroke:#4f46e5
```

### MambaBlock — selective state space

The SSM compresses history into a fixed-size state `h` of shape `(d_inner, d_state)`. The key innovation over classic SSMs is that `B`, `C`, and `Δ` are **input-dependent** — the model learns *what* to remember per token.

```mermaid
flowchart TD
    x["x (B,L,d_model)"] --> ip["in_proj → split"]
    ip --> xssm["x_ssm (d_inner)"]
    ip --> z["gate z (d_inner)"]

    xssm --> conv["Causal Conv1d\n(depthwise, kernel=4)"]
    conv --> silu1["SiLU"]
    silu1 --> xp["x_proj"]
    xp --> B["B  (d_state)"]
    xp --> C["C  (d_state)"]
    xp --> dt["Δt (scalar)"]

    dt --> dtp["dt_proj → softplus\n→ delta (d_inner)"]
    dtp --> scan["selective_scan loop\nh_t = exp(Δ·A)·h_{t-1} + Δ·B·u_t\ny_t = C·h_t + D·u_t"]
    B --> scan
    C --> scan
    scan --> y["y (d_inner)"]

    y --> gate["y * silu(z)"]
    z --> gate
    gate --> op["out_proj → (d_model)"]

    style x fill:#6366f1,color:#fff,stroke:#4f46e5
    style ip fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style xssm fill:#06b6d4,color:#fff,stroke:#0891b2
    style z fill:#f59e0b,color:#fff,stroke:#d97706
    style conv fill:#06b6d4,color:#fff,stroke:#0891b2
    style silu1 fill:#06b6d4,color:#fff,stroke:#0891b2
    style xp fill:#0ea5e9,color:#fff,stroke:#0284c7
    style B fill:#10b981,color:#fff,stroke:#059669
    style C fill:#10b981,color:#fff,stroke:#059669
    style dt fill:#10b981,color:#fff,stroke:#059669
    style dtp fill:#0ea5e9,color:#fff,stroke:#0284c7
    style scan fill:#f59e0b,color:#fff,stroke:#d97706
    style y fill:#06b6d4,color:#fff,stroke:#0891b2
    style gate fill:#ec4899,color:#fff,stroke:#db2777
    style op fill:#6366f1,color:#fff,stroke:#4f46e5
```

**Why the causal Conv1d?** It mixes a short local window (4 tokens) before the SSM, acting like a lightweight position-aware feature extractor that feeds better inputs into the recurrence.

**Why input-dependent B, C, Δ?** Classic SSMs (S4) use fixed A/B/C matrices — the same transition for every token. Mamba makes them functions of the input, so the model can slow down its state update on informative tokens and coast through filler tokens.

### MultiHeadAttention — exact recall

```mermaid
flowchart LR
    x["x (B,L,d)"] --> qkv["QKV proj\n→ 3×(B,H,L,d/H)"]
    qkv --> Q & K & V
    Q & K & V --> sdp["scaled_dot_product_attention\n(causal mask)"]
    sdp --> proj["out proj → (B,L,d)"]

    style x fill:#6366f1,color:#fff,stroke:#4f46e5
    style qkv fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style Q fill:#3b82f6,color:#fff,stroke:#2563eb
    style K fill:#06b6d4,color:#fff,stroke:#0891b2
    style V fill:#10b981,color:#fff,stroke:#059669
    style sdp fill:#f59e0b,color:#fff,stroke:#d97706
    style proj fill:#6366f1,color:#fff,stroke:#4f46e5
```

Uses `F.scaled_dot_product_attention` with `is_causal=True` — PyTorch fuses the softmax and masking into a single kernel (Flash-Attention style on CUDA/MPS).

**Why sparse attention (every 4th layer)?** Full attention on every layer costs O(L²) per layer. Mamba handles the bulk of sequence mixing cheaply; attention is only inserted to anchor exact recall that SSMs smear.

### MoE layer — routing + experts

```mermaid
flowchart TD
    x["x (B·L, d_model)\ntoken vectors"] --> router["Router linear\n→ (B·L, 8 experts)"]
    router --> softmax["softmax scores"]
    softmax --> topk["top-2 expert indices\n+ renormalize weights"]

    topk --> e0["Expert 0\nSwiGLU FFN"]
    topk --> e1["Expert 1\nSwiGLU FFN"]
    topk --> edot["..."]
    topk --> e7["Expert 7\nSwiGLU FFN"]

    e0 & e1 & edot & e7 --> acc["weighted sum\n(only 2 experts fire\nper token)"]
    acc --> out["output (B·L, d_model)"]

    softmax --> aux["Aux load-balance loss\n= n_experts · Σ(tokens_per_expert\n× mean_router_prob)"]

    style x fill:#6366f1,color:#fff,stroke:#4f46e5
    style router fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style softmax fill:#a78bfa,color:#fff,stroke:#7c3aed
    style topk fill:#f59e0b,color:#fff,stroke:#d97706
    style e0 fill:#10b981,color:#fff,stroke:#059669
    style e1 fill:#10b981,color:#fff,stroke:#059669
    style edot fill:#6ee7b7,color:#065f46,stroke:#059669
    style e7 fill:#10b981,color:#fff,stroke:#059669
    style acc fill:#06b6d4,color:#fff,stroke:#0891b2
    style out fill:#6366f1,color:#fff,stroke:#4f46e5
    style aux fill:#f43f5e,color:#fff,stroke:#e11d48
```

Each **Expert** is a SwiGLU FFN:

```text
output = down( silu(gate(x)) * up(x) )
```

`d_hidden` is sized as `4 * d_model // top_k` so that running 2 experts costs roughly the same FLOPs as one dense FFN.

**Why the aux loss?** Without it the router collapses — a few experts get all the tokens and the rest are never trained. The Switch-Transformer-style auxiliary loss penalizes any uneven load distribution, keeping all 8 experts useful.

### Full layer-by-layer network view (default Jamba, 8 layers)

```mermaid
flowchart TD
    TOK["CharTokenizer\n(65-char vocab)"]
    TOK --> EMB["Token Embedding\n65 → 128"]
    EMB --> POS["+ Pos Embedding\n(256 positions)"]
    POS --> DROP["Dropout 0.1"]

    DROP --> L1["Layer 1\nMamba + Dense FFN"]
    L1 --> L2["Layer 2\nMamba + MoE FFN ×8 experts"]
    L2 --> L3["Layer 3\nMamba + Dense FFN"]
    L3 --> L4["Layer 4\nAttention (4 heads) + MoE FFN ×8"]
    L4 --> L5["Layer 5\nMamba + Dense FFN"]
    L5 --> L6["Layer 6\nMamba + MoE FFN ×8 experts"]
    L6 --> L7["Layer 7\nMamba + Dense FFN"]
    L7 --> L8["Layer 8\nAttention (4 heads) + MoE FFN ×8"]

    L8 --> LN["LayerNorm"]
    LN --> HEAD["Linear 128 → 65\n(weight-tied to embedding)"]
    HEAD --> LOSS["Cross-Entropy Loss\n+ 0.01 × MoE Aux Loss"]

    style TOK fill:#6366f1,color:#fff,stroke:#4f46e5
    style EMB fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style POS fill:#a78bfa,color:#fff,stroke:#7c3aed
    style DROP fill:#c4b5fd,color:#1e1b4b,stroke:#7c3aed

    style L1 fill:#06b6d4,color:#fff,stroke:#0891b2
    style L2 fill:#10b981,color:#fff,stroke:#059669
    style L3 fill:#06b6d4,color:#fff,stroke:#0891b2
    style L4 fill:#f59e0b,color:#fff,stroke:#d97706
    style L5 fill:#06b6d4,color:#fff,stroke:#0891b2
    style L6 fill:#10b981,color:#fff,stroke:#059669
    style L7 fill:#06b6d4,color:#fff,stroke:#0891b2
    style L8 fill:#f59e0b,color:#fff,stroke:#d97706

    style LN fill:#64748b,color:#fff,stroke:#475569
    style HEAD fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style LOSS fill:#f43f5e,color:#fff,stroke:#e11d48
```

**Weight tying**: the output linear head shares weights with the token embedding. This is a standard trick that reduces parameter count and regularizes the embedding space.

---

## Training pipeline

```mermaid
flowchart TD
    DATA["tiny-Shakespeare\n(~1MB text, auto-downloaded)"]
    DATA --> SPLIT["90/10 train/val split\n(character indices)"]
    SPLIT --> BATCH["get_batch: random offset sampling\n(B=32, L=256)"]
    BATCH --> FWD["forward pass\n+ MoE aux loss accumulation"]
    FWD --> LOSS["total loss = CE + 0.01·aux"]
    LOSS --> BACK["backward + grad clip (1.0)"]
    BACK --> OPT["AdamW\nlr=3e-3, β=(0.9,0.95), wd=0.1"]
    OPT --> LR["LR schedule\nlinear warmup (100 steps)\n→ cosine decay → 1e-4"]
    LR --> EVAL{"step % 250 == 0?"}
    EVAL -- yes --> EST["estimate_loss\n(50 batches train+val)"]
    EST --> TB["TensorBoard log\nruns/{arch}/"]
    EST --> CKPT{"val_loss < best?"}
    CKPT -- yes --> SAVE["save checkpoints/{arch}.pt"]
    CKPT -- no --> NEXT
    SAVE --> NEXT["next step"]
    EVAL -- no --> NEXT

    style DATA fill:#6366f1,color:#fff,stroke:#4f46e5
    style SPLIT fill:#8b5cf6,color:#fff,stroke:#7c3aed
    style BATCH fill:#06b6d4,color:#fff,stroke:#0891b2
    style FWD fill:#3b82f6,color:#fff,stroke:#2563eb
    style LOSS fill:#f59e0b,color:#fff,stroke:#d97706
    style BACK fill:#f97316,color:#fff,stroke:#ea580c
    style OPT fill:#ec4899,color:#fff,stroke:#db2777
    style LR fill:#a78bfa,color:#fff,stroke:#7c3aed
    style EVAL fill:#64748b,color:#fff,stroke:#475569
    style EST fill:#0ea5e9,color:#fff,stroke:#0284c7
    style TB fill:#3b82f6,color:#fff,stroke:#2563eb
    style CKPT fill:#64748b,color:#fff,stroke:#475569
    style SAVE fill:#10b981,color:#fff,stroke:#059669
    style NEXT fill:#e2e8f0,color:#1e293b,stroke:#94a3b8
```

---

## The MoE bargain (the thing to actually look at)

Run `python demo.py` and compare the `jamba` row to the others:

- **Total parameters**: `jamba` has by far the most — MoE adds 8 expert FFNs per
  MoE layer, so lots of capacity.
- **Active params per token**: stays close to the others — each token only runs
  **top-2 of 8** experts (25% of expert params).

That's the whole point of Mixture-of-Experts: **more knowledge capacity, nearly the
same compute per token.**

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
| --- | --- | --- |
| `d_model` | 128 | embedding / hidden width throughout the network |
| `n_layers` | 8 | total blocks |
| `n_heads` | 4 | attention heads (d_model/n_heads = 32 per head) |
| `d_state` | 16 | SSM state size — how much history Mamba compresses into |
| `context_len` | 256 | max sequence length |
| `dropout` | 0.1 | applied after embeddings and in dense FFN |
| `attn_every` | 4 | insert attention at every Nth layer (layers 4, 8 in default) |
| `moe_every` | 2 | use MoE FFN at every Nth layer (layers 2,4,6,8 in default) |
| `n_experts` | 8 | total experts per MoE layer |
| `top_k` | 2 | experts activated per token (25% of capacity) |
| `batch_size` | 32 | lower to 16 if memory-constrained |
| `max_steps` | 3000 | total training steps |
| `eval_every` | 250 | evaluate + maybe checkpoint every N steps |
| `lr` | 3e-3 | peak learning rate |
| `min_lr` | 1e-4 | floor after cosine decay |
| `warmup` | 100 | linear warmup steps before cosine decay begins |
| `grad_clip` | 1.0 | gradient norm clip threshold |

---

## Notes & honest caveats

- **Sequential scan.** The Mamba scan is a readable Python loop — correct and
  MPS-friendly, but not as fast as a fused parallel-scan kernel. Fine at this scale.
- **MoE dispatch.** Experts run via a per-expert masking loop: clear and correct;
  production MoE uses grouped GEMM + expert parallelism for speed.
- **MPS quirks.** If you hit an unsupported op, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Character-level.** Uses a char tokenizer for simplicity (65-char vocab on Shakespeare).

---

## How this connects to the bigger picture

```text
Legendre polynomials  →  optimal projection basis
HiPPO                 →  stable A matrix for compressing history
S4 / S4D              →  makes it fast (diagonal A)
Mamba (selective SSM) →  input-dependent B, C, Δ
Jamba (this project)  →  Mamba backbone + sparse attention + MoE
```

This is the current frontier recipe in miniature: an efficient sequence mixer
(attention variants or SSMs) plus a sparse MoE feed-forward — the two axes every
2026 frontier model is built on.
