# The Selective SSM (Mamba) — Implementation

This builds the **actual Mamba mechanism** on top of the HiPPO foundation:
making $B$, $C$, and the step size $\Delta$ **input-dependent** (the "selectivity"),
and running the recurrence with a **parallel scan**.

Three files, from clearest-to-read → most-practical:

| File | Runs without torch? | What it shows |
|------|:---:|---------------|
| `selective_ssm_numpy.py` | ✅ yes | The bare mechanism: input-dependent B/C/Δ + parallel scan, with a proof that the parallel scan equals the sequential recurrence |
| `mamba_train_numpy.py` | ✅ yes | A **trainable** selective SSM with hand-derived gradients, learning a content-based memory task from scratch |
| `mamba_block_torch.py` | needs torch | The **full production-style Mamba block** (conv + gate + selective scan), a drop-in replacement for a Transformer block |

---

## Run it

```bash
# these two run anywhere — pure NumPy
python selective_ssm_numpy.py
python mamba_train_numpy.py

# this one needs PyTorch (the form you'd use on your machine)
pip install torch
python mamba_block_torch.py
```

---

## The one idea: selectivity

A vanilla SSM (Stages 3–4) is **Linear Time-Invariant** — $A$, $B$, $C$, $\Delta$ are
fixed, so every token is processed identically. It cannot decide that one token
matters more than another.

Mamba's single change: make $B$, $C$, and $\Delta$ **functions of the current input**.

```
vanilla SSM:   B, C, Δ   are fixed parameters
Mamba (S6):    B = Linear_B(x_t)
               C = Linear_C(x_t)
               Δ = softplus(Linear_Δ(x_t))
```

Now the model can, per token, choose to **write hard** (large Δ), **skip** (small Δ),
or **read** specific state dimensions (via C). This is "content-based reasoning" —
the ability that lets Mamba solve tasks vanilla SSMs fail, like selective copying.

The cost: input-dependent parameters break the convolution trick (the kernel is no
longer fixed), so training needs the **parallel scan** instead.

---

## The parallel scan

The recurrence $h_t = a_t\, h_{t-1} + b_t$ looks inherently sequential. But the
combine operation is **associative**:

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\; a_2 b_1 + b_2)
$$

Associativity means we can compute all $h_t$ with a **prefix scan** in $O(\log L)$
parallel steps instead of $O(L)$ sequential ones. `selective_ssm_numpy.py`
implements the Hillis–Steele scan and verifies it matches the sequential loop to
machine precision (~1e-16). Real Mamba uses a hardware-aware version fused into a
single GPU kernel — same math, faster execution.

---

## What the training demo proves

`mamba_train_numpy.py` trains on a **selective-recall task**: a sequence of random
values with one position "marked"; the model must output the marked value at the
end. This *requires* selectivity — the model has to learn to write the marked value
into memory and hold it while ignoring everything else.

Result (from an actual run):

```
step        train MSE
     0        0.3229   (random init)
   500        0.0694
  1000        0.0101
  4000        0.0034
Final MSE: 0.0036   vs baseline (predict-the-mean) 0.3204
```

A ~90× improvement over the baseline. The model genuinely learned content-based
memory — the core capability behind Mamba.

**A subtle but crucial detail** the demo makes concrete: $A$ must be *small* in
magnitude. With $A = -(1,2,\dots,N)$ the discretized $\bar A = e^{\Delta A}$ decays
to zero within a couple of steps, so memory can't survive a 20-step sequence. Using
gentle values ($A \approx -0.05 \ldots -0.5$) lets the state persist, and $\Delta$
(driven by the mark) does the selective writing. This is exactly why real Mamba
initializes $A$ with small-magnitude values.

---

## The full block (`mamba_block_torch.py`)

The production-style block wraps the selective SSM in the complete Mamba layer:

```
x → LayerNorm
  → in_proj         (expand to 2·d_inner: an SSM path and a gate path)
  → causal Conv1d   (depthwise — cheap local context before the SSM)
  → SiLU
  → selective SSM   (input-dependent B, C, Δ; the S6 core)
  → × SiLU(gate)    (gated output, like a GLU)
  → out_proj        (back to d_model)
  → + residual
```

Key design points, all in the code:
- **`A_log` parameterization.** We store $\log A$ and use $A = -e^{A_{\log}}$, so $A$ is
  always negative → the recurrence is always stable, no matter what training does.
- **Depthwise causal conv.** Each channel gets a tiny local window before the SSM;
  `padding=d_conv-1` then trimming to length `L` keeps it causal (no peeking ahead).
- **Drop-in shape.** Input and output are both `(batch, seq_len, d_model)`, so a
  `MambaBlock` can replace a `TransformerBlock` directly. `MambaLM` stacks them into
  a working language model with weight-tied embeddings.

The self-test confirms: correct output shape, all $A$ values negative (stable), and
a 2-layer `MambaLM` learning long-range recall (predicting the first token at the
final position, far above chance).

---

## How this connects to everything before it

```
Legendre polynomials  →  give the optimal projection basis
        ↓
HiPPO                 →  derives a stable A matrix for compressing history
        ↓
S4 / S4D              →  makes it fast (diagonal A, convolution training)
        ↓
Mamba (this code)     →  makes B, C, Δ input-dependent (selectivity)
                          + parallel scan for training
```

The HiPPO-style stable, structured $A$ is the backbone; selectivity is what turns
it into a language model that can reason about content. You've now built the whole
chain from orthogonal polynomials to a working Mamba block.

---

## Where to go next

- Swap `MambaBlock` into the TinyGPT project from earlier (it has the same interface
  as a Transformer block) and train on Shakespeare — compare loss curves and speed.
- Replace the sequential scan in `selective_scan` with a true parallel-scan CUDA
  kernel (or `torch.associative_scan` where available) for real speed.
- Read the official implementation at `github.com/state-spaces/mamba` and the
  minimal reference at `github.com/johnma2006/mamba-minimal` — this code deliberately
  mirrors their structure so the comparison is direct.
