# Jamba: Mamba + Attention + Mixture of Experts

Jamba is a small language model that combines three ideas from modern AI research into one configurable architecture you can train on a laptop. It is named after AI21's Jamba model, which uses the same combination at production scale.

This project is the final step in a learning series. By the time you reach this folder, you have already built:

- A plain Transformer with attention
- A Mamba block with selective state space memory
- A Mixture of Experts (MoE) feed-forward layer

Jamba puts all three together and lets you compare them side by side.

---

## Why three components instead of one?

Each component is good at something the others are not:

| Component | Good at | Weak at |
| --- | --- | --- |
| Mamba (SSM) | Long sequences, constant memory cost, fast inference | Blurry exact recall of specific past tokens |
| Attention | Exact recall: can look up any past token precisely | Cost grows quadratically with sequence length |
| Mixture of Experts | Large knowledge capacity without proportional compute cost | More total parameters to store on disk and in memory |

Jamba's idea is to use all three together so each one covers the others' weaknesses:

- **Mamba handles most layers cheaply.** Processing the bulk of the sequence with fixed-cost SSM blocks keeps inference fast and memory constant.
- **Attention is sprinkled in rarely.** Every few layers, one attention block handles the cases where exact recall matters. You get the benefit without paying the quadratic cost on every layer.
- **MoE replaces the feed-forward network in some layers.** More total capacity (knowledge stored in expert weights) without proportional compute per token, because each token only activates 2 of the 8 experts.

The result: long-context efficiency from Mamba, exact recall from attention, and large capacity from MoE, all in one model.

---

## Before you run: the import path fix

The scripts in this folder import from each other (`from model import ...`). Python needs to know where those files are. The `_bootstrap.py` file handles this automatically: every script imports it first, which adds the project folder to Python's search path.

The safest way to run is to be inside the folder:

```bash
cd path/to/jamba
python smoke_test.py
```

If you use VS Code or PyCharm, open the `jamba/` folder itself as your workspace root (not its parent). If you see a `ModuleNotFoundError`, it means you are running from a different directory.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | The full Jamba model: Mamba blocks, attention blocks, dense FFN, MoE FFN, character tokenizer |
| `utils.py` | Shared helpers: device selection, data loading, batch generation, learning rate schedule |
| `_bootstrap.py` | Adds the project folder to Python's path so imports work from any directory |
| `train.py` | Train any of the four architectures with `--arch mamba`, `transformer`, `hybrid`, or `jamba` |
| `demo.py` | Instantly compare all four architectures: parameter counts, active parameters per token, layer layout |
| `generate.py` | Load a saved checkpoint and generate text |
| `visualize_model.py` | Visualize the model structure and MoE routing behavior |
| `smoke_test.py` | A 15-second check that everything runs correctly. Run this first. |
| `requirements.txt` | Core dependencies: torch, matplotlib |
| `requirements-viz.txt` | Optional visualization libraries: torchinfo, torchview, netron |

---

## Quick start

```bash
# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch matplotlib

# Verify everything works (takes about 15 seconds)
python smoke_test.py

# Compare the four architectures without training
python demo.py

# Train the full Jamba model on Shakespeare
python train.py

# Generate text from the trained model
python generate.py --seed "ROMEO:"
```

---

## The four architecture presets

All four presets use the same width (128-dimensional embeddings) and the same depth (8 layers). The only difference is which block type appears at each layer:

```text
mamba       [ m  m  m  m  m  m  m  m ]   pure SSM, dense FFN every layer
transformer [ A  A  A  A  A  A  A  A ]   pure attention, dense FFN every layer
hybrid      [ m  m  m  A  m  m  m  A ]   mostly Mamba, attention at layers 4 and 8
jamba       [ m  mE m  AE m  mE m  AE ]  hybrid mixers + MoE at layers 2, 4, 6, 8

Key: m = Mamba block   A = Attention block   E = MoE FFN   (no E = dense FFN)
```

Run `python demo.py` to see the parameter counts for each. You will notice that `jamba` has by far the most total parameters but active parameters per token stay close to the others, because each token only activates 2 of the 8 MoE experts.

---

## Inside each block

Every block in the network, regardless of type, follows the same two-sublayer pattern:

```text
Input x
  |
  +---> LayerNorm --> Mixer (Mamba or Attention) --> + (add back x)
  |                                                  |
  +---> LayerNorm --> FFN (Dense or MoE) ----------> + (add back x)
  |
Output x'
```

This is called pre-norm with residual connections. It is the same structure used in modern Transformers. The LayerNorm stabilizes the values before each sublayer, and the residual additions ensure gradients can flow back through many layers without vanishing.

### The Mamba block

The Mamba block processes each token using a selective state space model. The key property is that it compresses the entire history into a fixed-size hidden state, so memory cost does not grow with sequence length.

The internal structure:

```text
Input x (batch, seq_len, d_model)
  |
  in_proj: expand to 2 * d_inner, split into x_ssm and gate
  |
  x_ssm path:
    Causal Conv1d (mixes a 4-token local window)
    SiLU activation
    x_proj: compute B, C, and Delta from the current token
    Selective scan:
      Delta = softplus(linear projection of x)    <- controls step size
      B = linear projection of x                  <- controls what gets written in
      C = linear projection of x                  <- controls what gets read out
      h_t = exp(Delta * A) * h_{t-1} + Delta * B * u_t
      y_t = C * h_t + D * u_t
  |
  gate path:
    SiLU activation
  |
  y = (SSM output) * (gate)    <- gated output, filters what passes through
  out_proj: back to d_model
```

The three input-dependent parameters (B, C, Delta) are what make Mamba "selective." A classic SSM uses fixed B, C, and Delta: every token gets processed identically. Mamba computes fresh values for each token, so the model can write an important token strongly into memory (large Delta) and let a filler word pass through almost unchanged (small Delta).

The causal Conv1d before the SSM mixes a short local window of 4 tokens, giving the SSM better local context to work from. The "causal" part means it only looks at the current and past tokens, never future ones.

### The Attention block

The attention block is standard multi-head self-attention with a causal mask:

```text
Input x (batch, seq_len, d_model)
  |
  QKV projection: x --> Q, K, V   each shape (batch, heads, seq_len, d_head)
  |
  Scaled dot-product attention with causal mask:
    scores = Q * K^T / sqrt(d_head)
    mask: each position can only attend to itself and earlier positions
    weights = softmax(scores)
    output = weights * V
  |
  Output projection: back to (batch, seq_len, d_model)
```

Attention is expensive: computing the scores matrix costs time proportional to `seq_len^2`. Jamba uses it only every 4th layer (layers 4 and 8 in the default setup). Mamba handles the other 6 layers cheaply. The occasional attention layer provides the exact recall capability that SSMs alone cannot match.

### The MoE feed-forward layer

The MoE layer replaces the dense feed-forward network in some layers. Instead of one FFN that all tokens pass through, there are 8 independent FFNs (called experts), and each token is processed by only the top 2.

```text
Input: all tokens flattened to (batch * seq_len, d_model)
  |
  Router: a small linear layer produces 8 scores per token
  Softmax: convert scores to probabilities
  Top-2 selection: pick the 2 highest-probability experts for each token
  Renormalize: the 2 selected weights sum to 1
  |
  For each of the 8 experts:
    Find which tokens chose this expert
    Run only those tokens through this expert's SwiGLU FFN
    Weight the output by the router probability
    Add to the result
  |
  Output: (batch * seq_len, d_model) reshaped back to (batch, seq_len, d_model)
```

Each expert is a SwiGLU feed-forward network:

```text
output = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

The hidden dimension of each expert is set to `4 * d_model / top_k`. With `top_k = 2`, this means each token activates `2 * (4 * d_model / 2) = 4 * d_model` total hidden units, which is the same as a single dense FFN. Two experts, same compute, but the model has 8 times the knowledge capacity spread across 8 specialist networks.

### The router collapse problem and the auxiliary loss

If the router is left to its own devices, it will collapse: a few popular experts get all the tokens, the others get none, and the "unused" experts never receive gradient updates and stay permanently useless. This defeats the purpose of MoE.

To prevent this, training adds a second loss term called the load-balancing auxiliary loss:

```text
aux_loss = n_experts * sum over each expert of:
               (fraction of tokens routed to this expert)
             * (mean router probability assigned to this expert)
```

When routing is perfectly balanced, this equals 1.0. When all tokens go to one expert, it spikes to `n_experts`. The training loss is:

```text
total loss = cross-entropy loss + 0.01 * aux_loss
```

The small coefficient 0.01 means prediction accuracy is still the main goal, but the penalty is large enough to keep all experts in use.

---

## What gets saved during training

After training, two directories appear alongside the source files:

```text
jamba/
├── checkpoints/        saved model weights
│   ├── jamba.pt
│   ├── mamba.pt
│   ├── transformer.pt
│   └── hybrid.pt
└── runs/               TensorBoard training logs
    ├── jamba/
    ├── mamba/
    ├── transformer/
    └── hybrid/
```

**Checkpoints** are saved whenever the validation loss improves. Only the best checkpoint per architecture is kept. The saved file contains the model weights, the full configuration dictionary, the architecture name, and the character-to-index vocabulary mapping. Loading this file is enough to reconstruct the model and generate text without retraining.

**TensorBoard logs** record four values at every evaluation step:

| Metric | What it means |
| --- | --- |
| `loss/train` | Average cross-entropy over 50 random training batches |
| `loss/val` | Average cross-entropy over 50 random validation batches |
| `lr` | Current learning rate (warming up then decaying) |
| `moe/aux_loss` | Load-balancing penalty (should stay near 1.0) |

To view the training curves:

```bash
tensorboard --logdir runs
# then open http://localhost:6006 in your browser
```

The `moe/aux_loss` curve is the most interesting one for Jamba specifically. A value near 1.0 means the router is distributing tokens evenly. A rising value means router collapse is beginning.

---

## Training details

The training pipeline:

```text
1. Download tiny-Shakespeare (~1MB of text, downloaded automatically)
2. Split 90% train / 10% validation by character position
3. Sample random batches: batch size 32, sequence length 256
4. Forward pass: compute cross-entropy loss + MoE aux loss
5. Backward pass: compute gradients, clip gradient norm to 1.0
6. AdamW optimizer: lr=3e-3, betas=(0.9, 0.95), weight decay=0.1
7. Learning rate schedule: linear warmup for 100 steps, cosine decay to 1e-4
8. Every 250 steps: evaluate on both train and val, log to TensorBoard,
   save checkpoint if val loss improved
9. Run for 3000 steps total (a few minutes on a laptop)
```

---

## Configuration reference

All settings live in the `CONFIG` dictionary in `train.py`. The most useful ones to adjust:

| Setting | Default | What it controls |
| --- | --- | --- |
| `d_model` | 128 | Embedding width throughout the network |
| `n_layers` | 8 | Total number of blocks |
| `n_heads` | 4 | Attention heads (each head is 32-dimensional) |
| `d_state` | 16 | SSM state size: how many numbers Mamba uses to summarize history |
| `context_len` | 256 | Maximum sequence length the model can process |
| `attn_every` | 4 | Insert an attention block every N layers |
| `moe_every` | 2 | Use a MoE FFN every N layers |
| `n_experts` | 8 | Total experts per MoE layer |
| `top_k` | 2 | Experts activated per token |
| `batch_size` | 32 | Reduce to 16 if you run out of memory |
| `max_steps` | 3000 | Total training steps |
| `lr` | 3e-3 | Peak learning rate after warmup |
| `warmup` | 100 | Steps of linear warmup before cosine decay begins |

---

## Visualizing the model

```bash
pip install -r requirements-viz.txt

# Show a per-layer parameter table
python visualize_model.py --arch jamba --tool torchinfo

# Save a diagram of the architecture as a PNG
python visualize_model.py --arch jamba --tool torchview

# Open an interactive ONNX explorer in the browser
python visualize_model.py --arch jamba --tool netron

# Show a heatmap of how tokens are distributed across MoE experts
python visualize_model.py --arch jamba --tool moe

# Run all of the above
python visualize_model.py --arch jamba --tool all
```

The MoE routing heatmap is the most informative visualization. On an untrained model the router distributes tokens roughly uniformly. After training, you can see whether the load-balancing loss held: if it did, all 8 experts still receive a roughly equal share of tokens.

---

## Caveats and known limitations

**Sequential scan.** The Mamba scan is written as a Python loop. It is correct and runs fine at this small scale, but it is slower than a fused parallel-scan GPU kernel. The `README_mamba.md` file in the HiPPO SSM folder explains the parallel scan algorithm.

**MoE dispatch.** Experts run via a per-expert masking loop: iterate over each of the 8 experts, select the tokens that chose it, run them through, and accumulate. This is clear and correct, but production MoE implementations use grouped matrix multiplications and expert parallelism across GPUs for speed.

**MPS compatibility.** On Apple Silicon (MPS backend), some operations may not be supported. If you see an error, set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

**Character-level tokenizer.** The model uses a 65-character vocabulary (all characters in Shakespeare). This is simple and requires no external library, but it is not how production language models work. Production models use byte-pair encoding (BPE) with vocabularies of 32,000 to 100,000 tokens.

---

## How this fits the full learning series

This project is the final step in a series that builds the ideas from the ground up:

```text
Legendre polynomials   orthogonal reference shapes for compressing a signal

HiPPO                  derives the unique A and B matrices that optimally
                       compress a history into Legendre coefficients

S4 and S4D             reparameterizes A so the recurrence can be computed
                       fast using a convolutional trick

Mamba                  makes B, C, and Delta input-dependent (selectivity)
                       and uses a parallel scan to train efficiently

Jamba (this project)   combines a Mamba backbone with sparse attention and
                       Mixture of Experts, the recipe behind frontier models
```

Jamba is a miniature version of what large-scale models like AI21's Jamba (51B parameters) and other hybrid SSM-Transformer architectures do in production. The same three ingredients (selective SSM, sparse attention, MoE) appear in most frontier architectures released in 2024 and 2025.
