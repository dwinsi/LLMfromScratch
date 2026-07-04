# mamba-vs-transformer — a from-scratch Transformer vs Mamba comparison, on your MacBook Air

A complete, runnable project that trains **both** a Transformer and a **Mamba**
language model on the same data, so you can compare them head-to-head. Tuned to
run on a MacBook Air M2 using Apple's **MPS GPU backend**.

This is the capstone of the whole journey: everything from Legendre polynomials →
HiPPO → S4 → the selective SSM comes together here as a working language model you
can train in a few minutes and generate text from.

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | Both architectures (`TransformerBlock`, `MambaBlock`) behind one `LanguageModel` interface, plus the char tokenizer |
| `train.py` | Train a single model (`--arch mamba` or `--arch transformer`) |
| `compare.py` | Train **both** and plot loss curves + speed — the payoff |
| `generate.py` | Load a checkpoint and generate text interactively |
| `smoke_test.py` | 10-second check that MPS works and both models run — **run this first** |
| `visualize_model.py` | Visualize either architecture (torchinfo / torchview / Netron) |
| `requirements.txt` | Just `torch` + `matplotlib` |
| `requirements-viz.txt` | Optional visualization libraries |

---

## Setup on a MacBook Air M2

```bash
# 1. Make a fresh environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with Apple Silicon support (this is the default wheel now)
pip install torch matplotlib

# 3. Confirm everything works — this checks MPS and runs both models once
python smoke_test.py
```

You should see `✓ Apple MPS GPU backend is available and will be used`.
If you see CPU instead, training still works — just slower.

---

## Quick start

```bash
# Train a Mamba model on Shakespeare (auto-downloads ~1MB of text)
python train.py

# Train a Transformer instead
python train.py --arch transformer

# Train BOTH and compare (produces comparison.png)
python compare.py

# Generate text from a trained model
python generate.py --seed "ROMEO:"
```

---

## What to expect on an M2 Air (8GB)

With the default config (d_model=128, 4 layers, context=256, 3000 steps):

| | Transformer | Mamba |
|---|---|---|
| Parameters | ~0.4M | ~0.5M |
| Train time (3000 steps) | ~3–5 min | ~5–8 min* |
| Final val loss | ~1.5 | ~1.5 |
| Memory | comfortable | comfortable |

\* The Mamba block here uses a **sequential** scan (a readable Python loop), which is
slower than the fused CUDA kernel real Mamba uses. On this small scale it's totally
fine; for large-scale work you'd swap in a parallel-scan kernel. The point here is
clarity and correctness, not raw speed.

Both should produce recognizably Shakespeare-flavored text after training:

```
ROMEO:
What say you to my suit? I am content,
And yet I would that I had stay'd awhile...
```

---

## The comparison you're running

The interesting question this project lets you explore firsthand:

- **Transformer** uses attention — every token can look directly at every other
  token. Great at exact recall, but cost grows as $O(L^2)$ with sequence length.
- **Mamba** uses a selective state space — it keeps a fixed-size memory it updates
  token by token. Cost grows only as $O(L)$, and inference uses constant memory per
  token (no growing KV cache).

On short Shakespeare sequences they perform similarly. The Mamba advantage shows up
at **long context** — try bumping `context_len` to 512 or 1024 in `train.py` and
watch how each scales. That's the whole reason SSMs matter.

---

## Config knobs (`CONFIG` in `train.py`)

| Setting | Default | Try |
|---|---|---|
| `d_model` | 128 | 256 for a bigger, better model |
| `n_layers` | 4 | 6 for more depth |
| `context_len` | 256 | 512 or 1024 to stress-test scaling |
| `d_state` | 16 | 32 or 64 for more Mamba memory |
| `batch_size` | 32 | drop to 16 if memory is tight |
| `max_steps` | 3000 | 6000 for a sharper model |

If you hit memory pressure on the 8GB Air, lower `batch_size` first, then `context_len`.

---

## Visualizing the network

Four complementary tools, each answering a different question. Install them with:

```bash
pip install -r requirements-viz.txt
brew install graphviz        # macOS — needed by torchview
```

**1. torchinfo — "how many params, what shapes?"** The fastest sanity check. Prints a
per-layer table with input/output shapes and parameter counts.

```bash
python visualize_model.py --arch mamba --tool torchinfo
```

**2. torchview — "what does the architecture look like?"** Draws the module tree with
tensor shapes as a PNG. The clearest way to *see* how a MambaBlock differs from a
TransformerBlock. Uses `device='meta'` so it costs no memory.

```bash
python visualize_model.py --arch mamba --tool torchview   # -> arch_mamba.png
python visualize_model.py --arch transformer --tool torchview
```

**3. Netron — "let me click through every layer."** Exports to ONNX and opens an
interactive explorer you can zoom, pan, and inspect node by node.

```bash
python visualize_model.py --arch mamba --tool netron      # -> mamba.onnx + launches viewer
# or drag the .onnx file onto https://netron.app
```

**4. TensorBoard — "how is training going?"** Already wired into `train.py`. Logs
loss curves, learning rate, weight histograms, and the compute graph automatically.

```bash
python train.py --arch mamba          # logging happens during training
tensorboard --logdir runs             # open http://localhost:6006 in a browser
```

Do everything at once:

```bash
python visualize_model.py --arch mamba --tool all
```

**A note on visualizing Mamba specifically.** The selective scan is a `for` loop over
the sequence, so torchview and ONNX *unroll* it into one block per timestep. To keep
the diagram readable, `visualize_model.py` uses a small config (2 layers, seq_len 32).
torchinfo doesn't trace the loop at all, so it's unaffected. All the ops we use
(einsum, conv1d, softplus, matmul) export cleanly to ONNX opset 17.

---



```
Legendre polynomials   →   the optimal projection basis
HiPPO                  →   a stable, structured A matrix for compressing history
S4 / S4D               →   makes it fast (diagonal A)
Selective SSM (Mamba)  →   input-dependent B, C, Δ  ← the MambaBlock in model.py
mamba-vs-transformer   →   a full trainable LM comparing Mamba vs Transformer
```

The `MambaBlock` in `model.py` is the same block developed step by step earlier —
the `A = -exp(A_log)` stable parameterization, the input-dependent `B/C/Δ`, the
causal depthwise conv, and the gated output. Now it's wired into a real language
model you can train on your laptop.

---

## Notes & honest caveats

- **Sequential scan.** The `selective_scan` here is a clear Python loop over the time
  dimension. It's correct and MPS-compatible but not the fastest possible. Production
  Mamba fuses this into a hardware-aware parallel scan. For learning and for models
  this size, the loop is the right call — you can read every line.
- **MPS quirks.** Apple's MPS backend is excellent but occasionally a specific op
  falls back to CPU with a warning. That's harmless. If you ever hit an unsupported
  op, set the env var `PYTORCH_ENABLE_MPS_FALLBACK=1` before running.
- **Character-level.** This uses a char tokenizer for simplicity. To go further, swap
  in the BPE tokenizer from the earlier TinyGPT project — the interface is the same.
- **Reproducibility.** MPS doesn't guarantee bit-identical runs; expect small
  variation in the loss between runs. The trends and comparisons hold.

---

## Next steps

1. **Scale the context.** Set `context_len=1024` and compare how Transformer vs Mamba
   training time and memory behave. This is where you *see* the $O(L^2)$ vs $O(L)$
   difference on your own machine.
2. **Swap in BPE.** Replace `CharTokenizer` with the BPE tokenizer from the earlier
   project for word-level modeling.
3. **A parallel scan.** Replace the loop in `selective_scan` with an associative scan
   for a real speedup — a great exercise now that you understand the math.
4. **Read the references.** `github.com/state-spaces/mamba` (official) and
   `github.com/johnma2006/mamba-minimal` (minimal) — this code mirrors their structure.
```
