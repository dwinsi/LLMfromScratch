# 10: Grouped Query Attention (GQA)

This project adds **Grouped Query Attention** to the LLaMA-style architecture built in Project 9. It is a small but important change: the Key and Value projections now use fewer heads than the Query projection, which cuts memory usage during inference. After this project, the architecture is functionally equivalent to LLaMA 2.

If you are reading this without having followed the earlier projects, do not worry. This document explains everything from the ground up.

---

## Start here: what is attention?

Before talking about GQA specifically, you need to understand what attention is doing.

Every token in a sequence (a word, a sub-word fragment, a punctuation mark) is represented as a vector of numbers called an **embedding**. Attention is the mechanism that lets each token look at other tokens in the sequence and decide how much to "pay attention" to each one.

The way it works is through three projections of that embedding vector:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information should I pass along?"

For each token, you compute a dot product between its Query and every other token's Key. High dot product = high relevance. You then use those relevance scores as weights to sum up the Value vectors. The result is a new representation of that token that has been informed by its context.

This happens in parallel across the entire sequence, hence the name "self-attention".

---

## Multi-Head Attention: the baseline

Rather than doing attention once with one large Q, K, V, the model splits the embedding dimension into several smaller **heads** and runs attention independently in each one. Each head can specialize in a different aspect of the relationship between tokens; one might track grammatical agreement, another might track co-reference, and so on.

With 4 heads and an embedding dimension of 64, each head gets dimension 16 (64 / 4 = 16). In **Multi-Head Attention (MHA)**, every head has its own Q, K, and V projection:

```text
Head 0:  Q₀ (dim 16), K₀ (dim 16), V₀ (dim 16)
Head 1:  Q₁ (dim 16), K₁ (dim 16), V₁ (dim 16)
Head 2:  Q₂ (dim 16), K₂ (dim 16), V₂ (dim 16)
Head 3:  Q₃ (dim 16), K₃ (dim 16), V₃ (dim 16)
```

Total Q dimension: 4 × 16 = 64
Total K dimension: 4 × 16 = 64
Total V dimension: 4 × 16 = 64

MHA is maximally expressive. Every head attends with its own independent keys and values. Projects 7 through 9 used MHA.

---

## The KV cache problem

During text generation, the model produces one token at a time. To produce token number 100, it needs the Key and Value vectors for all 99 previous tokens. Recomputing those from scratch at every step is extremely slow: it means re-running the entire model over all previous tokens at each step.

The standard solution is the **KV cache**: store the K and V vectors as they are computed and reuse them at each subsequent step. The Q vector for the new token is computed fresh, but the cached K and V vectors for all previous tokens are already available.

The catch: the KV cache consumes memory that grows with both sequence length and the number of K/V heads.

```text
KV cache size = sequence_length × number_of_kv_heads × head_dim × 2 × bytes_per_value
```

For a model like LLaMA 2 70B deployed to serve thousands of simultaneous users, this is a serious constraint. At full MHA with 64 heads, the KV cache at a sequence length of 4096 tokens is enormous. Reducing the number of KV heads directly reduces the cache size, which determines how many users can be served in parallel.

For our small training experiment this barely registers. At production scale it is the difference between serving one user and serving thousands.

---

## The three attention variants

Once you understand the KV cache problem, the three variants are simply different tradeoffs between quality and memory:

### Multi-Head Attention (MHA)

Every query head has its own key and value heads. Maximum expressivity, maximum KV cache cost.

```text
Q heads: 4   K heads: 4   V heads: 4

Q projection:  64 -> 64 dimensions
K projection:  64 -> 64 dimensions
V projection:  64 -> 64 dimensions
```

### Multi-Query Attention (MQA)

All query heads share a single key head and a single value head. Maximum KV cache savings, but a significant reduction in expressivity; all heads attend to the same keys and values.

```text
Q heads: 4   K heads: 1   V heads: 1

Q projection:  64 -> 64 dimensions
K projection:  64 -> 16 dimensions   (just one head's worth)
V projection:  64 -> 16 dimensions
```

### Grouped Query Attention (GQA): this project

The balanced middle ground. Query heads are divided into groups, and each group shares one key head and one value head. With 4 query heads and 2 KV heads, we get 2 groups of 2 query heads each:

```text
Q heads: 4   K heads: 2   V heads: 2

Group 0:  Q0, Q1 share K0 and V0
Group 1:  Q2, Q3 share K1 and V1

Q projection:  64 -> 64 dimensions
K projection:  64 -> 32 dimensions   (two heads x 16 dim each)
V projection:  64 -> 32 dimensions
```

GQA captures most of the memory savings of MQA while maintaining most of the quality of MHA. This is why it became the standard choice in modern LLMs: LLaMA 2, Mistral, Gemma, and others all use it.

---

## How the expansion works

After projecting K and V to the reduced size, the K and V tensors have a different shape from Q:

```text
Q shape:  (batch, num_query_heads, seq_len, head_dim)  ->  (batch, 4, seq_len, 16)
K shape:  (batch, num_kv_heads,   seq_len, head_dim)  ->  (batch, 2, seq_len, 16)
V shape:  (batch, num_kv_heads,   seq_len, head_dim)  ->  (batch, 2, seq_len, 16)
```

To compute attention, Q and K must have compatible shapes. The solution is to temporarily expand K and V by repeating each KV head for every query head that belongs to its group:

```python
K_expanded = K.repeat_interleave(self.queries_per_kv_head, dim=1)
V_expanded = V.repeat_interleave(self.queries_per_kv_head, dim=1)
```

With `queries_per_kv_head = 2`, `repeat_interleave` takes each of the 2 KV heads and repeats it twice, producing 4 heads total:

```text
Before expansion (2 KV heads):      After expansion (4 heads):

  KV head 0                           KV head 0  (copy 1)  <- serves Q head 0
  KV head 1                           KV head 0  (copy 2)  <- serves Q head 1
                                      KV head 1  (copy 1)  <- serves Q head 2
                                      KV head 1  (copy 2)  <- serves Q head 3
```

The expanded K and V now have the same shape as Q, and standard scaled dot-product attention proceeds normally.

**This expansion is temporary.** It exists only during the attention calculation for the current step. In the KV cache, K and V are stored at the smaller size (2 heads, not 4). That is where the memory savings come from.

---

## Walking through the code

Here is the GQA `forward` method annotated step by step:

```python
def forward(self, token_representations, causal_mask):
    batch_size = token_representations.shape[0]
    seq_len    = token_representations.shape[1]

    # Step 1: Normalise before attention (Pre-norm LLaMA style)
    normed = self.rms_norm_before_attention(token_representations)

    # Step 2: Project to Q, K, V
    # Q gets the full embedding_dim; K and V get the smaller kv_projection_dim
    Q = self.query_projection(normed)    # shape: (batch, seq_len, 64)
    K = self.key_projection(normed)      # shape: (batch, seq_len, 32)  <- smaller
    V = self.value_projection(normed)    # shape: (batch, seq_len, 32)  <- smaller

    # Step 3: Reshape into heads
    Q = Q.view(batch_size, seq_len, self.number_of_query_heads, self.head_dim)
    # shape: (batch, seq_len, 4, 16)

    K = K.view(batch_size, seq_len, self.number_of_kv_heads, self.head_dim)
    V = V.view(batch_size, seq_len, self.number_of_kv_heads, self.head_dim)
    # shape: (batch, seq_len, 2, 16)

    # Step 4: Apply RoPE positional encoding to Q and K
    Q = apply_rope(Q, self.rope_cos, self.rope_sin)
    K = apply_rope(K, self.rope_cos, self.rope_sin)

    # Step 5: Transpose so the head dimension is second
    # Attention operates over (seq_len, head_dim) for each head
    Q = Q.transpose(1, 2)   # (batch, 4, seq_len, 16)
    K = K.transpose(1, 2)   # (batch, 2, seq_len, 16)
    V = V.transpose(1, 2)   # (batch, 2, seq_len, 16)

    # Step 6: Expand K and V to match the number of query heads
    K_expanded = K.repeat_interleave(self.queries_per_kv_head, dim=1)
    V_expanded = V.repeat_interleave(self.queries_per_kv_head, dim=1)
    # Both now (batch, 4, seq_len, 16), same shape as Q

    # Step 7: Scaled dot-product attention
    attention_scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # Step 8: Apply causal mask (tokens can only attend to earlier positions)
    attention_scores = attention_scores.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
    )

    # Step 9: Softmax -> attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_weights = self.attention_dropout(attention_weights)

    # Step 10: Weighted sum of values
    attention_output = torch.matmul(attention_weights, V_expanded)

    # Step 11: Reshape heads back into a single vector and project to embedding_dim
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
    attention_output = self.output_projection(attention_output)

    # Step 12: Residual connection: add attention output to input
    token_representations = token_representations + attention_output

    # Step 13: Feed-forward with residual (SwiGLU, unchanged from Project 9)
    normed             = self.rms_norm_before_feedforward(token_representations)
    feedforward_output = self.swiglu_feedforward(normed)
    token_representations = token_representations + feedforward_output

    return token_representations
```

---

## Why the assertion exists

```python
assert number_of_query_heads % number_of_kv_heads == 0, \
    "number_of_query_heads must be divisible by number_of_kv_heads"
```

For groups to be uniform, the number of query heads must divide evenly by the number of KV heads. With 4 query heads, the only valid KV head counts are 1 (MQA), 2 (GQA), and 4 (MHA). Trying to use 3 KV heads with 4 query heads would leave one group with 2 queries and another with just 1, which would require special-casing throughout the code. The assertion catches this configuration error immediately instead of producing a silent shape mismatch.

---

## Parameter savings

The only change from MHA is in the K and V projections. Q and the output projection are unchanged.

```text
Attention weights comparison (per transformer block):

Component          MHA (Project 9)       GQA (Project 10)      Saved
----------------------------------------------------------------------
Q projection       64 x 64 = 4,096       64 x 64 = 4,096           0
K projection       64 x 64 = 4,096       64 x 32 = 2,048       2,048
V projection       64 x 64 = 4,096       64 x 32 = 2,048       2,048
O projection       64 x 64 = 4,096       64 x 64 = 4,096           0
----------------------------------------------------------------------
Per block savings:                                               4,096
Total (4 blocks):                                               16,384
```

At our scale, 16,384 parameters saved is a modest 6% reduction. At LLaMA 2 70B scale (64 query heads, 8 KV heads), the same ratio eliminates billions of parameters and, more importantly, shrinks the KV cache by 8x at inference time.

---

## Comparison across projects

```text
Project    Attention type     Q heads    KV heads    KV params per block    Cumulative additions
------------------------------------------------------------------------------------------------
7          MHA                4          4           8,192                  Transformer, causal mask
8          MHA                4          4           8,192                  BPE tokeniser
9          MHA + RoPE         4          4           8,192                  RMSNorm, RoPE, SwiGLU
10         GQA + RoPE         4          2           4,096                  GQA (this project)
```

Everything else (RMSNorm, RoPE, SwiGLU, BPE tokenisation, batching, cosine annealing, gradient clipping) carries over from Project 9 unchanged.

---

## Running

```bash
pip install torch tokenizers matplotlib
python mini_llm_gqa.py
```

The script trains on `weather_corpus_v2.txt` (shared with Projects 8 and 9), plots the loss curve, and generates a short text sample at the end.

## Files

```text
mini_llm_gqa.py          training script with GQA implementation
weather_corpus_v2.txt    shared training corpus from Projects 8 and 9
config.json              model and training hyperparameters
images/
  loss_curve_gqa.png     training and validation loss
```

---

## What is next

Project 11 replaces the feed-forward network with a **Mixture of Experts (MoE)** layer. Instead of one FFN that runs for every token, there are multiple "expert" FFNs and a router that selects only a few of them per token. The model gains more total capacity but spends the same compute per token; this is the architecture used in Gemma 4 and likely GPT-4.
