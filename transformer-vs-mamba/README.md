# Transformer vs Mamba: A Side-by-Side Comparison

This project trains both a Transformer and a Mamba language model on the same data and lets you compare them directly: loss curves, training speed, memory usage, and generated text. Both models are written from scratch so you can read every line of code.

It is designed to run on a MacBook Air M2 using Apple's MPS GPU backend. It also works on any machine with CUDA or just a CPU (slower, but it works).

---

## What you are comparing

Both models learn the same task: given a sequence of characters from Shakespeare, predict the next character. The difference is in how each model reads and remembers the sequence.

**The Transformer** uses attention. At every position, the model can look directly at every previous character and decide how much to weight it. This gives it precise recall but at a cost: the attention computation grows quadratically with sequence length. A sequence twice as long takes four times as long to process.

**Mamba** uses a selective state space model. Instead of looking back at all previous characters directly, it maintains a fixed-size hidden state that is updated one character at a time. The entire history is compressed into those N numbers. The cost of processing each new character is constant regardless of how long the sequence is so far.

On short sequences (the default 256-character context), both models perform similarly and reach roughly the same loss. The difference becomes visible when you increase the context length: Transformer training time scales up steeply while Mamba stays nearly flat. That is the core comparison this project lets you see for yourself.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Both architectures (TransformerBlock and MambaBlock) behind one shared LanguageModel interface, plus the character tokenizer |
| `train.py` | Train a single model with `--arch mamba` or `--arch transformer` |
| `compare.py` | Train both architectures back to back and produce a comparison plot |
| `generate.py` | Load a saved checkpoint and generate text |
| `smoke_test.py` | A 10-second check that the GPU backend is available and both models run. Run this first. |
| `visualize_model.py` | Visualize either architecture using torchinfo, torchview, or Netron |
| `requirements.txt` | Core dependencies: torch, matplotlib |
| `requirements-viz.txt` | Optional visualization libraries |

---

## Setup

```bash
# Create a fresh virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install PyTorch and matplotlib
pip install torch matplotlib

# Verify everything works (takes about 10 seconds)
python smoke_test.py
```

On Apple Silicon you should see a message confirming the MPS backend is active. On other machines it will use CUDA (if available) or CPU. Training works on CPU but takes longer.

---

## Running the comparison

```bash
# Train both models and produce a comparison plot (comparison.png)
python compare.py

# Train just the Mamba model
python train.py

# Train just the Transformer
python train.py --arch transformer

# Generate text from a trained model
python generate.py --seed "ROMEO:"
```

`compare.py` is the main script. It trains both models with identical settings, plots their validation loss curves side by side, and prints a summary of training time and final loss. The plot is saved as `comparison.png`.

---

## What to expect on an M2 Air (8 GB)

With the default settings (128-dimensional model, 4 layers, 256-character context, 3000 training steps):

| | Transformer | Mamba |
| --- | --- | --- |
| Parameters | about 0.4 million | about 0.5 million |
| Training time (3000 steps) | 3 to 5 minutes | 5 to 8 minutes |
| Final validation loss | around 1.5 | around 1.5 |
| Memory | comfortable | comfortable |

Both should produce recognizably Shakespeare-flavored text after training:

```text
ROMEO:
What say you to my suit? I am content,
And yet I would that I had stay'd awhile...
```

The Mamba model is slower here because its scan is implemented as a readable Python loop rather than a fused GPU kernel. At this small scale that is the right tradeoff: you can read every line of the scan. For production use you would swap in a parallel-scan kernel.

---

## The key experiment: scale the context length

The default context length is 256 characters. Both models handle this comfortably. To see the architectural difference clearly, increase it:

```text
In train.py or compare.py, change:
    context_len = 256
to:
    context_len = 512   (or 1024)
```

Watch what happens:

- The Transformer takes noticeably longer per step and uses more memory. Attention cost scales with `seq_len^2`: doubling the context roughly quadruples the attention computation.
- Mamba's per-step cost barely changes. The state update at each position is always the same size regardless of how far back the sequence goes.

This is not a theoretical claim. You can see it on your own machine in a few minutes of training. That is why SSMs were considered an important development: constant-memory inference and near-linear training cost for arbitrarily long sequences.

---

## Inside the MambaBlock

The Mamba block in `model.py` implements the full selective SSM mechanism. Here is what happens to each token:

```text
Input: x with shape (batch, seq_len, d_model)

Step 1: in_proj
  Expand the input to 2 * d_inner, split into two paths:
  - x_ssm: the main path that goes through the SSM
  - gate:  a parallel path that will filter the output

Step 2: Causal Conv1d (on the x_ssm path)
  Apply a depthwise convolution with a 4-token window.
  This mixes a short local context before the SSM processes the sequence.
  Causal: only looks at the current position and the 3 positions before it.

Step 3: Compute the selective parameters from the current token
  B     = linear projection of x     (controls what gets written into memory)
  C     = linear projection of x     (controls what gets read from memory)
  Delta = softplus(linear projection) (controls the step size of the update)

  These three values are different for every token. This is the "selectivity":
  each token decides for itself how strongly to update the state.

Step 4: Selective scan
  For each position in the sequence:
    A_bar = exp(Delta * A)              (how much old state persists)
    B_bar = Delta * B                   (how much new input enters)
    state = A_bar * state + B_bar * x   (update the hidden state)
    y     = C * state + D * x           (read from the state)

Step 5: Gated output
  final output = y * silu(gate)
  The gate path modulates which information passes through.

Step 6: out_proj
  Project back to d_model.
```

The matrix A is stored as `A_log` and reconstructed as `A = -exp(A_log)`. This guarantees A is always negative, which guarantees the state always decays rather than growing unboundedly. Stability is enforced structurally: no matter what gradient descent does to `A_log`, the reconstructed A can never become positive.

---

## Inside the TransformerBlock

The Transformer block uses standard multi-head causal self-attention:

```text
Input: x with shape (batch, seq_len, d_model)

Step 1: LayerNorm
Step 2: Multi-head attention (causal)
  Project x to Q, K, V for each attention head
  Compute attention scores: Q * K^T / sqrt(d_head)
  Apply causal mask: each position can only attend to itself and earlier positions
  Compute weighted sum: softmax(scores) * V
  Project back to d_model
Step 3: Residual: add original x
Step 4: LayerNorm
Step 5: Feed-forward network (two linear layers with a GELU activation)
Step 6: Residual: add x from after the attention
```

The causal mask is what makes this a language model rather than a general encoder: at training time all positions are processed in parallel, but each position only sees its own past.

---

## Configuration

All settings are in the `CONFIG` dictionary in `train.py`. The most useful ones to adjust:

| Setting | Default | What it does |
| --- | --- | --- |
| `d_model` | 128 | Width of all internal representations |
| `n_layers` | 4 | Number of stacked blocks |
| `n_heads` | 4 | Attention heads in the Transformer (d_model / n_heads = 32 per head) |
| `d_state` | 16 | Size of Mamba's hidden state per channel |
| `context_len` | 256 | Maximum sequence length (increase this to see scaling behavior) |
| `batch_size` | 32 | Reduce to 16 if you run out of memory |
| `max_steps` | 3000 | Total training steps |
| `lr` | 3e-3 | Peak learning rate |

To get a better model at the cost of more training time, increase `d_model` to 256 and `n_layers` to 6. To stress-test the scaling comparison, increase `context_len` to 512 or 1024.

If you run out of memory on an 8 GB machine, reduce `batch_size` first, then `context_len`.

---

## Visualizing the architectures

```bash
pip install -r requirements-viz.txt
brew install graphviz        # macOS only, needed by torchview

# Per-layer parameter table and shapes
python visualize_model.py --arch mamba --tool torchinfo
python visualize_model.py --arch transformer --tool torchinfo

# Architecture diagram saved as a PNG
python visualize_model.py --arch mamba --tool torchview
python visualize_model.py --arch transformer --tool torchview

# Interactive ONNX explorer in the browser
python visualize_model.py --arch mamba --tool netron

# Run all visualization tools at once
python visualize_model.py --arch mamba --tool all
```

TensorBoard is also wired into `train.py` automatically. To view loss curves during or after training:

```bash
tensorboard --logdir runs
# then open http://localhost:6006
```

One thing to know about visualizing Mamba: the selective scan is a loop over the sequence length, so diagram tools like torchview unroll it into one block per timestep. To keep the diagram readable, `visualize_model.py` uses a smaller config (2 layers, 32-character sequence) when generating Mamba diagrams.

---

## Caveats

**Sequential scan.** The `selective_scan` function is a Python loop over the time dimension. It is correct, readable, and MPS-compatible, but slower than a fused parallel-scan GPU kernel. For a model this size on a laptop, the loop is fine. For larger-scale work, replacing it with a parallel scan would be the next step.

**MPS quirks.** Apple's MPS backend works well but occasionally a specific operation falls back to CPU with a warning printed to the console. This is harmless. If you see an actual error about an unsupported operation, set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

**Character-level tokenizer.** The model uses a 65-character vocabulary covering all characters in Shakespeare. This is the simplest possible tokenizer and requires no external library. Production language models use byte-pair encoding (BPE) with vocabularies of 32,000 tokens or more.

**Run-to-run variation.** MPS does not guarantee bit-identical results between runs. Expect small differences in the loss values between separate training runs. The trends and relative comparisons between Transformer and Mamba are stable.

---

## Where this fits in the learning series

This project is the practical capstone of a series that builds up from the mathematical foundations:

```text
Legendre polynomials    the orthogonal reference shapes used to compress a history

HiPPO                   derives the unique A and B matrices that optimally
                        compress a signal into Legendre coefficients

S4 and S4D              reparameterizes A so the recurrence runs fast

Mamba                   makes B, C, and Delta input-dependent (selectivity)
                        and uses a parallel scan for efficient training

transformer-vs-mamba    wires both architectures into real language models
                        you can train and compare on your own machine
```

The `MambaBlock` in `model.py` is the same block built step by step in the HiPPO SSM folder: the `A = -exp(A_log)` stable parameterization, the input-dependent B, C, and Delta, the causal depthwise convolution, and the gated output. Here it is wired into a language model trained end to end on real text.

---

## Next steps

**Scale the context length.** Set `context_len = 1024` in `compare.py` and run the comparison again. You will see the `O(L^2)` vs `O(L)` scaling difference concretely: watch the Transformer slow down while Mamba stays nearly constant.

**Swap in BPE tokenization.** Replace the `CharTokenizer` in `model.py` with the BPE tokenizer from the earlier projects. The interface is the same. This gives word-level modeling with a much richer vocabulary.

**Implement a parallel scan.** Replace the Python loop in `selective_scan` with an associative parallel scan. The math is in `README_mamba.md` in the HiPPO SSM folder. This is a good exercise now that you understand what the scan is computing.

**Read the source.** The official Mamba implementation is at `github.com/state-spaces/mamba`. A minimal readable version is at `github.com/johnma2006/mamba-minimal`. This code deliberately mirrors their structure so the comparison is direct.
