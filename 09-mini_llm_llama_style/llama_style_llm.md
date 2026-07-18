# 09: LLaMA-Style Architecture

Project 8 used BPE tokenisation but kept the same Transformer architecture from Projects 6 and 7. This project keeps BPE tokenisation and replaces three components inside the Transformer block with their modern equivalents.

The result is architecturally equivalent to LLaMA 1: the open-source model Meta released in 2023 that sparked a wave of open-source large language model development.

The three changes are:

1. **RMSNorm** replaces LayerNorm
2. **RoPE** (Rotary Positional Encoding) replaces sinusoidal positional encoding
3. **SwiGLU** replaces the GELU feed-forward network

Each change is a focused improvement. The overall structure of the Transformer block stays the same. After reading this project you will understand not just what these components are, but why they exist and why they were adopted by every modern open-source language model.

---

## Why bother changing the Transformer?

The Transformer architecture from Project 6 works. GPT-2 was built with it. So why did LLaMA change things?

The answer is scale. When a model has billions of parameters and trains for weeks, even a 10 percent speedup saves days of compute time and thousands of dollars in GPU costs. Each of the three changes in this project was adopted because it is either faster, more expressive, or better at generalising to longer sequences than the original it replaced.

On our small weather corpus, you will not see a dramatic difference in loss curves or generated text quality. The corpus is too small for these improvements to matter much. What matters here is understanding the ideas: you are learning the components that are inside GPT-4, Gemma, Mistral, and every other modern large language model.

---

## Change 1: RMSNorm

### What LayerNorm does

Recall from Project 6 that Layer Normalisation was added to stabilise training. Without it, the activations inside the network can grow very large or very small, which makes gradients explode or vanish.

LayerNorm takes a vector of activations and normalises it to have mean zero and variance one:

```text
Step 1: mean      = average of all values in the vector
Step 2: variance  = average of squared deviations from the mean
Step 3: normalise = (x - mean) / sqrt(variance + epsilon)
Step 4: scale     = learned_scale * normalised + learned_shift
```

The `epsilon` is a tiny constant (like `1e-6`) to prevent division by zero when the variance is very small. The `learned_scale` and `learned_shift` are parameters the model trains, allowing it to undo the normalisation to any extent it finds useful.

### What RMSNorm does

RMSNorm is simpler. It skips the mean subtraction and the shift parameter:

```text
Step 1: rms       = sqrt(mean(x^2) + epsilon)
Step 2: normalise = x / rms
Step 3: scale     = learned_scale * normalised
```

RMS stands for Root Mean Square: the square root of the average of the squared values. It is a measure of the magnitude of the vector, not its spread around a mean.

### Why remove the mean subtraction?

The mean subtraction in LayerNorm was added to make the output shift-invariant: if you add the same constant to every element of the input, the output does not change. This seems like a useful property.

But it turns out the learned `learned_scale` and `learned_shift` parameters already provide this flexibility. The explicit mean subtraction is doing work that the learned parameters duplicate. RMSNorm removes the redundant computation.

In practice, RMSNorm produces similar or identical quality to LayerNorm while being measurably faster (fewer operations, one fewer learned parameter per layer).

```python
class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, epsilon=1e-6):
        super(RMSNorm, self).__init__()
        self.epsilon      = epsilon
        self.learned_scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms        = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        normalised = x / rms
        return self.learned_scale * normalised
```

`torch.mean(x ** 2, dim=-1, keepdim=True)` computes the mean of the squared activations along the last dimension (the embedding dimension), keeping the shape so the division broadcasts correctly across the whole vector.

Used in: LLaMA 1/2/3, Gemma, Mistral, Falcon.

---

## Change 2: RoPE (Rotary Positional Encoding)

This is the most technically interesting change in the project. To understand it, we need to first revisit how positional encoding worked before.

### The problem: Transformers do not know word order

A Transformer processes all tokens in parallel. If you shuffled the order of the input tokens, the self-attention computation would produce the same result (just with swapped rows). The model has no built-in sense of which token came first, second, or third.

To fix this, Projects 6, 7, and 8 used **sinusoidal positional encoding**: a fixed pattern of sine and cosine waves was added to each token embedding before the first Transformer block. Token at position 0 got one pattern, token at position 1 got a slightly different pattern, and so on.

```text
Projects 6-8:
  token_representation = word_embedding + positional_encoding
  (this happens once, before any Transformer block)
```

This works but has two weaknesses:

First, the position information is injected once at the start and must survive unchanged through every attention layer, every residual connection, and every feed-forward network. By the time it reaches the fourth Transformer block, the positional signal has been diluted by everything that happened in between.

Second, sinusoidal encoding uses absolute positions (position 0, position 1, etc.). The attention mechanism would benefit more from knowing the relative distance between tokens (this token is 3 positions before that one) rather than their absolute positions in the sequence.

### How RoPE works

RoPE solves both problems by encoding position directly into the Query and Key vectors, inside every attention layer.

Instead of adding a vector to the embeddings, RoPE **rotates** the Query and Key vectors by an angle that depends on their position. Two vectors that are close in position will be rotated to similar orientations. Two vectors that are far apart will be rotated to very different orientations.

When you take the dot product of a rotated Q at position `m` with a rotated K at position `n`, the result naturally depends on the difference `m - n`, not on the absolute values of `m` and `n`. This gives the model relative position information for free.

### The rotation in practice

Every attention head dimension is paired up: dimension 0 with dimension 1, dimension 2 with dimension 3, and so on. Each pair is treated as a 2D point and rotated by an angle:

```text
angle = position * frequency

rotated_dim_0 = dim_0 * cos(angle) - dim_1 * sin(angle)
rotated_dim_1 = dim_0 * sin(angle) + dim_1 * cos(angle)
```

Different pairs of dimensions use different frequencies. The first pair uses a high frequency (rotates quickly as position increases). The last pair uses a very low frequency (rotates slowly). This is the same multi-frequency idea as sinusoidal encoding, but applied as a rotation rather than an additive offset.

The frequencies follow a geometric progression:

```text
frequency_for_pair_i = 1 / (10000 ^ (2i / head_dim))
```

For `head_dim = 16`:

```text
Pair 0: freq = 1 / 10000^0.000 = 1.000    (rotates once per token)
Pair 1: freq = 1 / 10000^0.125 = 0.562
Pair 2: freq = 1 / 10000^0.250 = 0.316
...
Pair 7: freq = 1 / 10000^0.875 = 0.018    (rotates once per 56 tokens)
```

The first few pairs encode fine-grained local position. The last few pairs encode coarse long-range position.

### The code

```python
def compute_rope_frequencies(head_dim, max_seq_len, base=10000):
    # One frequency per dimension pair
    i         = torch.arange(0, head_dim, 2).float()
    freqs     = 1.0 / (base ** (i / head_dim))

    # Compute angle for each position
    positions = torch.arange(max_seq_len).float()
    angles    = torch.outer(positions, freqs)   # (max_seq_len, head_dim/2)

    return torch.cos(angles), torch.sin(angles)
```

`torch.outer(positions, freqs)` produces a matrix where entry `[pos, i]` is `position * frequency_i`: the angle for dimension pair `i` at sequence position `pos`.

```python
def apply_rope(query_or_key, cos_table, sin_table):
    # Split into even and odd dimensions
    x_even = query_or_key[..., 0::2]   # dimensions 0, 2, 4, ...
    x_odd  = query_or_key[..., 1::2]   # dimensions 1, 3, 5, ...

    # Rotate each dimension pair
    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd  = x_even * sin_vals + x_odd * cos_vals

    # Interleave back
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    return x_rotated.flatten(-2)
```

`0::2` means "every other element starting at 0": dimensions 0, 2, 4, ... `1::2` means dimensions 1, 3, 5, ... These are the even/odd dimensions in each pair.

The cos and sin tables are precomputed once per `TransformerBlock` and stored as non-learnable buffers:

```python
cos_table, sin_table = compute_rope_frequencies(self.head_dim, max_seq_len)
self.register_buffer('rope_cos', cos_table)
self.register_buffer('rope_sin', sin_table)
```

`register_buffer` makes the tables part of the model (moved to GPU with `.to(device)`, saved in checkpoints) without treating them as learned parameters.

### What disappears

Because RoPE handles position inside each attention layer, the entire sinusoidal positional encoding buffer from the `MiniLanguageModel` class is gone:

```text
Projects 6-8:
  self.positional_encoding = self._build_sinusoidal_encoding(...)

Project 9:
  (nothing here — RoPE is inside each TransformerBlock)
```

Used in: LLaMA 1/2/3, Gemma, Mistral, Falcon, GPT-NeoX, PaLM 2.

---

## Change 3: SwiGLU

### What the feed-forward network did before

The Transformer block in Projects 6-8 contained a two-step feed-forward network after the attention sublayer:

```text
Step 1: expand   = GELU(x @ W_expand)     (64 -> 128 dimensions)
Step 2: compress = expanded @ W_compress   (128 -> 64 dimensions)
```

The expansion creates room for the network to compute richer intermediate representations. The non-linearity (GELU) introduces the ability to represent non-linear patterns. The compression brings everything back to the original embedding dimension so the next block receives the same shape.

### What SwiGLU does instead

SwiGLU replaces the two-step expand-compress with a three-step gated mechanism:

```text
Step 1: gate    = SiLU(x @ W_gate)         (64 -> hidden dimensions)
Step 2: value   = x @ W_value              (64 -> hidden dimensions)
Step 3: hidden  = gate * value             (element-wise product)
Step 4: compress = hidden @ W_compress     (hidden -> 64 dimensions)
```

The key difference is the element-wise product `gate * value`. This is a **gating mechanism**.

### What does gating mean?

Think of the gate as a filter. For each dimension of the intermediate representation:

- If `gate[i]` is close to 1: `value[i]` passes through almost unchanged
- If `gate[i]` is close to 0: `value[i]` is suppressed to near zero

The gate is learned: different inputs produce different gate patterns, so the network can selectively emphasise different aspects of the information for different contexts.

In a plain GELU network, the activation is applied uniformly across all dimensions. In SwiGLU, the gating is learned and input-dependent. Some dimensions can be shut off entirely for certain inputs while remaining active for others. This makes the feed-forward layer more expressive.

### SiLU: the activation inside the gate

SiLU stands for Sigmoid Linear Unit. It is defined as:

```text
SiLU(x) = x * sigmoid(x)
```

A few values for comparison:

```text
x       sigmoid(x)    SiLU(x)    GELU(x)
-3       0.047         -0.143     -0.004
-2       0.119         -0.238     -0.045
-1       0.269         -0.269     -0.159
 0       0.500          0.000      0.000
 1       0.731          0.731      0.841
 2       0.881          1.762      1.955
 3       0.953          2.858      2.996
```

SiLU and GELU are very similar in shape. Both are smooth, non-monotonic (they dip slightly negative before rising), and approximate the identity function for large positive values. The main practical difference is that SiLU is slightly cheaper to compute.

### Why three weight matrices instead of two?

SwiGLU uses three matrices: `W_gate`, `W_value`, and `W_compress`. This would increase parameter count compared to the two-matrix standard FFN if the hidden dimension stayed the same.

To keep parameter counts comparable, the hidden dimension is reduced. The standard FFN used `4 * embedding_dim` for the hidden dimension. SwiGLU uses approximately `2/3 * 4 * embedding_dim = 2.67 * embedding_dim`. This is the ratio LLaMA uses.

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, embedding_dim, feedforward_hidden_dim):
        super(SwiGLUFeedForward, self).__init__()
        self.gate_projection     = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.value_projection    = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.compress_projection = nn.Linear(feedforward_hidden_dim, embedding_dim, bias=False)
        self.dropout             = nn.Dropout(0.1)

    def forward(self, x):
        gate   = F.silu(self.gate_projection(x))
        value  = self.value_projection(x)
        hidden = gate * value
        hidden = self.dropout(hidden)
        return self.compress_projection(hidden)
```

Note `bias=False` on all three linear layers. LLaMA removes biases from most weight matrices. Biases add parameters without much benefit when the model is large and properly normalised. At our small scale this is a minor detail but worth knowing.

Used in: LLaMA 1/2/3, PaLM, Gemma, likely GPT-4.

---

## The updated Transformer block

The three changes slot into the existing Transformer block structure. The overall shape is unchanged:

```text
Projects 6, 7, 8:             Project 9 (LLaMA-style):
---------------------         --------------------------
LayerNorm                     RMSNorm
   |                             |
Multi-head attention          Multi-head attention
(sinusoidal pos outside)      (RoPE inside Q and K)
   |                             |
Residual connection           Residual connection
   |                             |
LayerNorm                     RMSNorm
   |                             |
Linear -> GELU -> Linear      W_gate x SiLU + W_value -> W_compress
   |                             |
Residual connection           Residual connection
```

The TransformerBlock forward method shows where each piece fits:

```python
def forward(self, token_representations, causal_mask):
    # Pre-norm with RMSNorm
    normed = self.rms_norm_before_attention(token_representations)

    # Q, K, V projections
    Q = self.query_projection(normed)
    K = self.key_projection(normed)
    V = self.value_projection(normed)

    # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
    Q = Q.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)
    K = K.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)
    V = V.view(batch_size, seq_len, self.number_of_attention_heads, self.head_dim)

    # Apply RoPE to Q and K (not V)
    Q = apply_rope(Q, self.rope_cos, self.rope_sin)
    K = apply_rope(K, self.rope_cos, self.rope_sin)

    # Reshape for batched matrix multiply: (batch, num_heads, seq_len, head_dim)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # Scaled dot-product attention with causal mask
    attention_scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attention_scores  = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output  = torch.matmul(attention_weights, V)

    # Reshape back and project
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
    attention_output = self.output_projection(attention_output)

    # Residual connection
    token_representations = token_representations + attention_output

    # Pre-norm before feed forward
    normed = self.rms_norm_before_feedforward(token_representations)

    # SwiGLU feed forward
    feedforward_output = self.swiglu_feedforward(normed)

    # Residual connection
    token_representations = token_representations + feedforward_output

    return token_representations
```

RoPE is applied to Q and K but not V. That is intentional: the rotation encodes position into the score computation (Q times K), not into the values that are retrieved.

---

## The MiniLanguageModel: what changed from Project 8

The model class becomes slightly simpler because the sinusoidal encoding buffer disappears:

```python
class MiniLanguageModel(nn.Module):
    def __init__(self, ...):
        self.word_embedding    = nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # No positional encoding here: RoPE handles position inside each block

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(..., max_seq_len=max_sequence_length)
            for _ in range(number_of_blocks)
        ])

        self.final_rms_norm    = RMSNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size, bias=False)
```

`final_rms_norm` is applied to the output of the last Transformer block before the output projection. This is the same pre-norm pattern used inside each block but applied one final time at the end of the stack.

---

## Parameter count comparison

```text
Component                            Project 8         Project 9
-----------------------------------------------------------------
word embedding (256 x 64)               16,384            16,384
positional encoding buffer                 512              none  (RoPE is inside blocks)
per block: RMSNorm / LayerNorm             256               128  (one scale, no shift)
per block: Q, K, V, O projections       16,384            16,384
per block: feed forward                 16,384            ~16,000  (adjusted hidden dim)
final norm                                 256               128
output projection (64 x 256)            16,384            16,384
-----------------------------------------------------------------
Total (4 blocks)                      ~167,040          ~165,000  (approximate)
```

The parameter counts are very similar. These changes are not primarily about reducing parameters. They improve training stability (RMSNorm, SwiGLU) and sequence length generalisation (RoPE).

---

## Training

Everything outside the model architecture is identical to Project 8: BPE tokenisation, sliding window sequences of length 8, `DataLoader` with batch size 32, Adam optimiser, cosine annealing scheduler, gradient clipping at `max_norm=1.0`.

```python
loss_function = nn.CrossEntropyLoss()
optimiser     = optim.Adam(model.parameters(), lr=learning_rate)
scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=number_of_epochs)
```

The training loop:

```python
for epoch in range(number_of_epochs):
    model.train()
    for batch_sequences, batch_targets in training_loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets   = batch_targets.to(device)

        optimiser.zero_grad()
        output_scores = model(batch_sequences)
        loss          = loss_function(output_scores, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

    scheduler.step()
```

Gradient clipping prevents the parameter updates from being too large in a single step. With `max_norm=1.0`, if the norm of all gradients combined exceeds 1.0, every gradient is scaled down proportionally so the total norm is exactly 1.0. This is especially useful in early training when the model's predictions are poor and the gradients can spike.

---

## Where each idea came from

These three changes did not all arrive at the same time. Here is a brief history:

**RMSNorm** was proposed in "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019). It showed that the mean subtraction in LayerNorm contributed almost nothing to performance while adding computational cost.

**RoPE** was proposed in "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021). It noted that existing positional encoding schemes did not efficiently encode relative position and proposed the rotation approach.

**SwiGLU** was introduced in "GLU Variants Improve Transformer" (Noam Shazeer, 2020). The paper tested many gating variants and found that SwiGLU consistently outperformed the standard GELU FFN.

LLaMA combined all three into a single architecture in 2023. The combination became the de facto standard for open-source language models.

---

## Architecture comparison across the series

```text
Project    Architecture                           Key innovation
------------------------------------------------------------------------
01         Single neuron, sigmoid, numpy          weights, bias, gradient
02         Two-layer network, backprop, numpy      chain rule, hidden layer
03         PyTorch basics                          autograd, nn.Module
04         Word-level RNN, numpy                  sequences, tanh, cross-entropy
05         RNN with attention, PyTorch             Q/K/V, nn.Embedding, Adam
06         Transformer block                       multi-head attention, causal mask,
                                                   layer norm, residual, positional enc
07         Mini LLM (4 blocks)                     nn.ModuleList, DataLoader,
                                                   cosine annealing, GELU
08         BPE tokenisation                        subword tokens, HuggingFace tokenizers
09         LLaMA-style (this project)              RMSNorm, RoPE, SwiGLU
10         Grouped Query Attention                 KV cache efficiency, GQA
```

---

## Running

```bash
pip install torch tokenizers matplotlib
python mini_llm_llama_style.py
```

The script loads the weather corpus, trains a BPE tokeniser, builds training sequences, trains the model for the configured number of epochs, generates sample text, and saves the loss curve to `loss_curve_llama_style.png`.

## Files

```text
mini_llm_llama_style.py     full training script
weather_corpus_v2.txt       shared corpus with Project 8
config.json                 hyperparameters
loss_curve_llama_style.png  training vs validation loss (generated on run)
```

## What is next

Project 10 adds **Grouped Query Attention (GQA)**: instead of every attention head having its own separate Key and Value matrices, multiple Query heads share a single Key and Value head. This brings the architecture to LLaMA 2 level and introduces the KV cache efficiency that makes large models practical at inference time.
