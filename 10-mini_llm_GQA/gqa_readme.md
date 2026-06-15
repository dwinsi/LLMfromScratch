# 10 — Grouped Query Attention (GQA)

Adds Grouped Query Attention to the LLaMA-style architecture from Project 9. Multiple query heads now share a single set of key and value heads, reducing the KV projection parameters and the memory required for the KV cache during inference.

After this project, the model is architecturally equivalent to LLaMA 2.

---

## The three attention variants

Understanding GQA requires knowing where it sits between the two extremes.

**Multi-Head Attention (MHA)** — what Projects 7 through 9 used. Every attention head has its own Q, K and V projection. With 4 heads of dimension 16 each:

```
Q heads: 4  ->  4 × 16 = 64 dimensions of queries
K heads: 4  ->  4 × 16 = 64 dimensions of keys
V heads: 4  ->  4 × 16 = 64 dimensions of values
```

**Multi-Query Attention (MQA)** — the extreme reduction. All query heads share a single K and V head.

```
Q heads: 4  ->  4 × 16 = 64 dimensions of queries
K heads: 1  ->  1 × 16 = 16 dimensions of keys
V heads: 1  ->  1 × 16 = 16 dimensions of values
```

**Grouped Query Attention (GQA)** — the balanced middle ground. Query heads are divided into groups, each group sharing one K and V head.

```
Q heads: 4  ->  4 × 16 = 64 dimensions of queries
K heads: 2  ->  2 × 16 = 32 dimensions of keys    (2 groups of 2 query heads)
V heads: 2  ->  2 × 16 = 32 dimensions of values
```

GQA is the standard choice in modern LLMs because it captures most of the memory savings of MQA while maintaining most of the quality of MHA.

---

## Why this matters: the KV cache

During text generation, the model predicts one token at a time. At each step it needs the K and V vectors for all previous tokens. Recomputing them from scratch at every step would be extremely slow.

The solution is the KV cache: store the K and V vectors as they are computed and reuse them at subsequent steps.

The problem: the KV cache grows with both sequence length and number of K/V heads.

```
KV cache size = sequence_length × number_of_kv_heads × head_dim × 2 × bytes_per_value
```

For LLaMA 2 70B serving thousands of users simultaneously, the KV cache at full MHA would be enormous. GQA with 8 KV heads instead of 64 reduces the KV cache by 8×.

For our small model this barely matters. At production scale it is the difference between serving one user and serving thousands.

---

## The repeat_interleave approach

After projecting K and V to the reduced size, we need to expand them so every query head has a K and V to attend to:

```python
K_expanded = K.repeat_interleave(self.queries_per_kv_head, dim=1)
V_expanded = V.repeat_interleave(self.queries_per_kv_head, dim=1)
```

With `queries_per_kv_head = 2`, this takes K of shape `(batch, 2, seq_len, head_dim)` and repeats each entry twice to produce `(batch, 4, seq_len, head_dim)`.

```
KV head 0  ->  serves query heads 0 and 1
KV head 1  ->  serves query heads 2 and 3
```

The expansion is temporary, only during the attention calculation. K and V are stored at the smaller size. This is where the memory savings come from.

---

## Parameter savings

```
Attention weights comparison:

Component          MHA (Project 9)    GQA (Project 10)    Saved
Q projection       64 × 64 = 4,096    64 × 64 = 4,096        0
K projection       64 × 64 = 4,096    64 × 32 = 2,048    2,048
V projection       64 × 64 = 4,096    64 × 32 = 2,048    2,048
O projection       64 × 64 = 4,096    64 × 64 = 4,096        0

Per block savings:                                        4,096
Total savings (4 blocks):                                16,384
```

At our scale this is a modest reduction. At LLaMA 2 70B scale (64 query heads, 8 KV heads), GQA saves billions of parameters and dramatically reduces inference memory.

---

## The assertion

The code includes an assertion to catch configuration errors immediately:

```python
assert number_of_query_heads % number_of_kv_heads == 0, \
    "number_of_query_heads must be divisible by number_of_kv_heads"
```

Query heads must divide evenly into groups. With 4 query heads, valid KV head counts are 1, 2 and 4. GQA with 2 KV heads means each group has 2 query heads.

---

## Comparison across projects

```
Project    Attention type    Q heads    KV heads    KV params per block
7          MHA               4          4           8,192
8          MHA               4          4           8,192
9          MHA + RoPE        4          4           8,192
10         GQA + RoPE        4          2           4,096
```

---

## Running

```
pip install torch tokenizers matplotlib
python mini_llm_gqa.py
```

## Files

```
mini_llm_gqa.py          the training script
weather_corpus_v2.txt    shared with Projects 8 and 9
images/
  loss_curve_gqa.png
```

## What is next

Project 11 adds Mixture of Experts (MoE). The feed forward network is replaced with multiple expert networks, only a few of which activate per token. This is the architecture used in Gemma 4 and likely GPT-4. More capacity, same compute per token.