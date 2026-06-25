# Concepts Behind the Dense vs MoE Ablation Study

> **Who this is for:** Anyone — whether you are a researcher, a student, or someone entirely new to machine learning. Technical details are included for practitioners, but every concept is first explained in plain language so anyone can follow the story.

---

## What Is This Script Doing?

Imagine you are trying to teach a computer to predict the next word in a sentence. You show it thousands of sentences, and it gradually learns patterns — after "the sky is", the next word is probably "blue", not "elephant".

This script does exactly that, but it compares **two different internal designs** for the brain doing the learning:

- **Dense model** — one single "thinking unit" that processes every word the same way, every time.
- **MoE model (Mixture of Experts)** — a panel of 8 specialist "thinking units", where each word is dynamically routed to the 2 most relevant specialists.

The comparison is called an **ablation study** — a controlled experiment where you change exactly one thing and measure the effect. Here, the one thing changed is the feed-forward network (FFN) inside each transformer block. Everything else — attention, normalization, positional encoding — is held constant.

---

## The Full Architecture at a Glance

Before diving into each concept, here is the full pipeline a batch of text goes through:

```
Raw text
   │
   ▼
[1] BPE Tokenizer ──► converts words into integer IDs
   │
   ▼
[2] Token Embedding ──► turns each ID into a vector of 64 numbers
   │
   ▼
[3] Dropout ──► randomly zeros out 10% of values (training only)
   │
   ▼
[4 × 4 Transformer Blocks]
   ├── RMSNorm
   ├── GQA Attention with RoPE  ──► token talks to other tokens
   ├── Residual connection (+)
   ├── RMSNorm
   ├── FFN: SwiGLU (Dense) OR MoE (8 experts, top-2)  ──► token thinks independently
   └── Residual connection (+)
   │
   ▼
[5] Final RMSNorm
   │
   ▼
[6] LM Head (Linear layer) ──► produces a score for every vocab token
   │
   ▼
[7] Cross-Entropy Loss (+ Aux Loss for MoE)
   │
   ▼
[8] Adam Optimizer + Cosine LR Decay + Gradient Clipping
```

---

## Part 1 — Turning Text into Numbers

### Concept 1: BPE Tokenizer (Byte-Pair Encoding)

**Plain language:**
A computer cannot read words. It can only work with numbers. A tokenizer is a translator — it converts human text into a list of numbers, and can convert those numbers back to text.

BPE works like this: start with the alphabet (A, B, C…), then find the most common pair of letters that appear next to each other in your entire text, and merge them into a new symbol. Keep merging the most frequent pairs until you have a vocabulary of a target size (256 here). The result is a vocabulary of common *subwords* — full words that appear often, plus fragments for rare words.

**Example (simplified):**

```
"weather" → ["weath", "er"]    (if "weath" was a common fragment)
"rainy"   → ["rain", "y"]
"fog"     → ["fog"]            (common enough to be its own token)
```

**Byte-Level BPE** goes one step further — it starts from individual bytes (raw computer memory values, 0–255) rather than letters. This means it can tokenize *any* text in any language, emoji included, without ever producing an "unknown" token.

**Why it matters:**
The vocabulary size controls the trade-off between precision and efficiency. A large vocabulary means each token carries more meaning but requires a bigger model to handle it. Here, a vocabulary of 256 is very small — chosen to keep the model lightweight for this educational experiment. Real LLMs like GPT-4 use ~100,000 tokens.

**Technical detail:**

```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()   # work on bytes, not characters
trainer = BpeTrainer(vocab_size=256, min_frequency=1)
tokenizer.train(files=[corpus_path], trainer=trainer)
```

---

### Concept 2: Sliding Window Dataset

**Plain language:**
Once the text is tokenized into a list of numbers, we need to create training examples. We use a sliding window: take 8 consecutive tokens as input, and the same 8 tokens shifted by one position as the target. The model learns to predict each next token given all previous ones.

**Visual:**

```
Full token sequence:  [5, 23, 7, 41, 12, 8, 3, 19, 6, 44, ...]

Window 1:  Input  = [5, 23, 7, 41, 12, 8, 3, 19]
           Target = [23, 7, 41, 12, 8, 3, 19, 6]   ← shifted by 1

Window 2:  Input  = [23, 7, 41, 12, 8, 3, 19, 6]
           Target = [7, 41, 12, 8, 3, 19, 6, 44]
```

At every position within a window, the model is being asked: "given everything you've seen so far in this sequence, what comes next?" This creates 8 prediction tasks per window, making training very data-efficient.

---

## Part 2 — Representing Meaning as Vectors

### Concept 3: Token Embeddings

**Plain language:**
A number like `42` (the token ID for "rain") means nothing by itself to a neural network — it's just an arbitrary label. An embedding converts each token ID into a list of 64 floating-point numbers (a vector). These numbers encode *meaning* — tokens that are semantically similar end up with similar vectors.

Think of it like a map. Every word gets assigned a point in a 64-dimensional space. Words with related meanings cluster together. The model learns these positions during training.

**Visual analogy:**

```
Token ID 42 ("rain")  → [0.12, -0.83, 0.44, 0.07, ..., -0.21]  (64 numbers)
Token ID 91 ("snow")  → [0.09, -0.79, 0.51, 0.03, ..., -0.18]  (similar — weather words)
Token ID 3  ("the")   → [0.94,  0.02, -0.11, 0.88, ...,  0.63]  (very different)
```

**Technical detail:**
`nn.Embedding(vocab_size, embed_dim)` is just a matrix of shape `(256, 64)`. Looking up token 42 is literally indexing row 42 of this matrix. The matrix values are learned via backpropagation.

---

## Part 3 — Normalization

### Concept 4: RMSNorm (Root Mean Square Normalization)

**Plain language:**
As numbers flow through the layers of a neural network, they can grow very large or very small. This causes the network to become unstable — gradients either explode or vanish, and learning stops. Normalization is like a volume knob: it rescales the values back to a manageable range after each step.

RMSNorm asks: "how loud is this signal overall?" It computes the loudness (the RMS — Root Mean Square, basically the average size of all values), then divides everything by that loudness to bring it back to a standard scale. A learnable parameter `weight` then lets the network decide how loud it wants to be after normalization.

**The formula:**

```
RMS(x) = √( mean(x²) + tiny_number )
RMSNorm(x) = (x / RMS(x)) × weight
```

**Why RMSNorm instead of the older LayerNorm?**
LayerNorm also subtracts the mean before dividing. RMSNorm skips that step — it's cheaper to compute, and research found that the mean-subtraction isn't necessary for transformers. LLaMA, Gemma, and Mistral all switched to RMSNorm.

**Where it appears in this script:**
Applied *before* every sub-layer (this is called pre-norm). So the pattern per block is:

```
output = input + sublayer(RMSNorm(input))
```

The residual connection (`+ input`) means the raw signal always passes through. The sublayer only needs to learn a *correction*, not a full transformation — this is far more stable.

---

## Part 4 — The Feed-Forward Network

### Concept 5: SwiGLU Feed-Forward Network

**Plain language:**
After the attention mechanism lets tokens talk to each other, the feed-forward network (FFN) lets each token "think" independently — transforming its own representation without looking at neighbors. It's like each word going off to consult its own reference book.

SwiGLU is a gated version of this network. Imagine two parallel paths:

- **Gate path** — decides *which parts of the information matter* (produces values between 0 and 1 using a smooth activation called SiLU)
- **Content path** — holds the actual information

The gate multiplies the content, suppressing irrelevant dimensions and amplifying relevant ones. Then a final projection maps back to the original size.

**Visual:**

```text
Input x (64 dims)
   ├──► gate_proj → SiLU activation  ─────┐
   │                                      × multiply
   └──► up_proj ─────────────────────────┘
                                          │
                                     down_proj
                                          │
                                    Output (64 dims)
```

**SiLU (Sigmoid Linear Unit):**

```text
SiLU(x) = x × sigmoid(x)
```

Unlike ReLU which hard-clips negative values to zero, SiLU smoothly suppresses them while still allowing small negative values through. This makes gradients flow more smoothly during training.

**Why no bias terms?**
All linear layers in this model use `bias=False`. Since RMSNorm is applied before each sublayer and has its own learnable scale, a separate bias would be redundant — it would just shift the distribution that normalization immediately re-centers.

**Why SwiGLU over standard FFN?**
Google's PaLM, Meta's LLaMA, and Google's Gemma all use SwiGLU. It consistently achieves lower loss than ReLU or GELU FFNs at the same parameter count. The gate mechanism gives the network more expressive power per parameter.

---

## Part 5 — Positional Encoding

### Concept 6: RoPE (Rotary Positional Embeddings)

**Plain language:**
A transformer's attention mechanism, by itself, is completely position-blind — it treats the sequence like a bag of tokens with no sense of order. "The cat sat on the mat" and "mat the on sat cat the" would look identical without positional information.

Positional encoding injects *where* each token is in the sequence. RoPE does this in an elegant way: it rotates the query and key vectors by an angle that depends on their position. The further apart two tokens are, the more their vectors have been rotated relative to each other — and when you compute the dot product between a query and a key, the result naturally encodes relative distance.

**Clock analogy:**
Imagine each token's query vector is like a clock hand. Position 1 starts at 12 o'clock, position 2 is at 1 o'clock, position 3 at 2 o'clock, and so on. When two clock hands point at each other, they are "aligned" (attending strongly). As positions get further apart, the clock hands point in increasingly different directions, reducing their alignment.

**The math:**

```text
Split the head dimension in half: x → (x1, x2)
For position p, compute angle θ using frequencies: θ = p / 10000^(2i/d)
Rotated output: [ x1·cos(θ) - x2·sin(θ),  x1·sin(θ) + x2·cos(θ) ]
```

This is the same as multiplying by a 2D rotation matrix — a geometric rotation in the embedding space.

**Why not just add position numbers to embeddings (like older models)?**
Learned absolute position embeddings (used in GPT-2) don't generalize to sequences longer than what was seen in training. RoPE encodes *relative* positions — the relationship between token at position 5 and token at position 8 is the same regardless of where in the sequence they are. This generalizes far better to longer contexts.

**Where in the code:**

```python
cos, sin = precompute_rope_freqs(head_dim, T, device)   # compute rotation tables once
q = apply_rope(q, cos, sin)   # rotate queries
k = apply_rope(k, cos, sin)   # rotate keys
```

Only Q and K are rotated — V is not, because V holds values to be aggregated, not positions to be compared.

---

## Part 6 — Attention

### Concept 7: GQA — Grouped Query Attention

**Plain language:**
Attention is the mechanism that allows tokens to "look at" each other. Each token creates three things:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I have to offer?"
- **Value (V)** — "what information should be passed along if someone attends to me?"

The attention score between token A and token B is computed as `Q_A · K_B` (dot product). High score = strong attention. After softmax, these scores become weights that determine how much of each token's value gets mixed into the output.

**Multi-head attention** runs this process in parallel across multiple "heads" — each head can learn to attend to different kinds of relationships (syntax, semantics, co-reference, etc.).

**The problem with standard multi-head attention at scale:**
At inference time, every token's K and V vectors need to be stored in memory (the KV cache) so that when generating token 1000, the model doesn't have to recompute all previous K/V pairs. With 4 heads and a large model, this KV cache becomes enormous.

**Grouped Query Attention (GQA) solution:**
Instead of each Q head having its own K and V head, multiple Q heads *share* a single K/V head.

```text
Standard MHA (4Q, 4KV):         GQA used here (4Q, 2KV):
  Q1 ──► K1, V1                   Q1 ──► K1, V1  ─┐
  Q2 ──► K2, V2                   Q2 ──► K1, V1  ─┘  (share)
  Q3 ──► K3, V3                   Q3 ──► K2, V2  ─┐
  Q4 ──► K4, V4                   Q4 ──► K2, V2  ─┘  (share)
```

This halves the size of the KV cache at inference while keeping model quality close to full MHA. LLaMA 2/3, Mistral, and Gemma all use GQA.

**In the code — the `repeat_interleave` trick:**

```python
k = k.repeat_interleave(self.groups, dim=1)   # expand 2 KV heads → 4
v = v.repeat_interleave(self.groups, dim=1)
```

This simply duplicates each KV head so the shape matches the Q heads, and standard matrix multiplication can proceed normally.

### Concept 8: Causal Masking

**Plain language:**
When training a language model, all 8 tokens in a sequence are processed simultaneously (for speed). But the model must not be allowed to "cheat" by seeing future tokens when predicting the current one. A causal mask enforces this rule.

Imagine the attention score matrix as a grid where row = "which token is asking" and column = "which token is being attended to":

```text
           pos 0  pos 1  pos 2  pos 3  pos 4 ...
pos 0  →  [  ✓     ✗     ✗     ✗     ✗  ]   (can only see itself)
pos 1  →  [  ✓     ✓     ✗     ✗     ✗  ]   (can see pos 0 and 1)
pos 2  →  [  ✓     ✓     ✓     ✗     ✗  ]   (can see pos 0, 1, 2)
pos 3  →  [  ✓     ✓     ✓     ✓     ✗  ]
...
```

The ✗ positions are filled with `-infinity` before the softmax. `softmax(-∞) = 0`, so those positions contribute nothing to the output.

**In the code:**

```python
causal = torch.triu(torch.full((T, T), float("-inf"), diagonal=1))
attn = (q @ k.T) * scale + causal   # mask added before softmax
```

### Concept 9: Residual Connections (Skip Connections)

**Plain language:**
Each transformer block adds its output *on top of* its input rather than replacing it:

```text
output = input + what_the_block_learned(input)
```

Think of it this way: the block only needs to learn a small *correction* or *refinement* to the representation. The original signal always passes through unchanged. This is like an editor marking up a document — the original text is preserved, and only the edits are added.

**Why this matters for training deep networks:**
In a network with 4 blocks stacked, the gradient (the error signal flowing backwards during training) can flow straight through the `+` operations all the way to the first layer without having to pass through any learned weights. Without residual connections, gradients in deep networks shrink exponentially (vanishing gradient problem) and early layers stop learning. Residual connections were introduced by ResNet (2015) for image models and became universal in transformers.

---

## Part 7 — Mixture of Experts

### Concept 10: MoE — Mixture of Experts FFN

**Plain language:**
In the dense model, every token is processed by the same FFN every time. In a Mixture of Experts model, there are 8 different FFNs (the "experts"), and a small "router" network decides which 2 experts are most relevant for each token.

Think of it like a hospital. Instead of one general practitioner seeing every patient, there are 8 specialists (cardiologist, neurologist, etc.). A triage nurse (the router) looks at each patient and decides which 2 specialists are most relevant. Each specialist sees a different subset of patients, allowing them to develop deep expertise in their domain.

**The key insight — conditional computation:**
All 8 experts have the same number of parameters as a single dense FFN. But at any given token, only 2 are active. This means:

- The model has 8× more total parameters (more potential knowledge)
- But uses only 2× the compute per token compared to a single expert

This is the MoE trade-off: more knowledge at the same computational cost.

**Step-by-step walkthrough:**

**Step 1 — The Router**
The Router

```text
Token vector (64 dims)
    ↓
Linear layer (64 → 8)   ← one score per expert
    ↓
Softmax
    ↓
[0.32, 0.05, 0.18, 0.03, 0.27, 0.06, 0.07, 0.02]   ← probability over 8 experts
```

**Step 2 — Top-K Selection**
Pick the 2 highest probabilities:

```text
Expert 0: 0.32  ← selected
Expert 4: 0.27  ← selected
(others discarded)
```

**Step 3 — Renormalization**
The 2 selected weights are rescaled to sum to 1:

```text
Expert 0: 0.32 / (0.32 + 0.27) = 0.54
Expert 4: 0.27 / (0.32 + 0.27) = 0.46
```

**Step 4 — Dispatch and Compute**
Send this token to Expert 0 and Expert 4. Each expert processes it independently and returns an output vector.

**Step 5 — Weighted Combination**
combination

```text
final_output = 0.54 × expert_0_output + 0.46 × expert_4_output
```

**Real-world use:**
Mixtral 8×7B (Mistral AI), GPT-4 (widely rumored), and Switch Transformer (Google) all use MoE. It allows building models with hundreds of billions of total parameters while keeping inference compute manageable.

---

### Concept 11: Auxiliary Load-Balancing Loss

**Plain language:**
There's a natural failure mode for MoE: the router collapses. It finds one or two "safe" experts that work well on average and always routes to them. The other experts receive no tokens, get no gradient, and never improve. You end up with a model that effectively has 1–2 experts, wasting 6–7 experts' worth of capacity.

The auxiliary (helper) loss is a penalty specifically designed to prevent this. It is added to the main loss and pushes the router to distribute tokens more evenly across all experts.

**The formula:**

```tetx
aux_loss = N × Σᵢ (fraction_i × mean_prob_i)
```

Where:

- `fraction_i` = fraction of tokens actually assigned to expert i (e.g., 0.3 if 30% of tokens go to expert 3)
- `mean_prob_i` = average router probability for expert i across all tokens
- `N` = number of experts (8)

**Why this works — the clever trick:**
`fraction_i` is computed from the top-k argmax, which is not differentiable — you can't compute gradients through it. But `mean_prob_i` IS differentiable. The product of a non-differentiable "load indicator" with a differentiable "probability signal" creates a usable gradient:

- If expert 3 is over-loaded (`fraction_3` is high), the loss increases when `mean_prob_3` is also high.
- The gradient flows back through `mean_prob_3`, pushing the router to lower its probability for expert 3 and redistribute to others.

**In the training loop:**

```python
ce_loss = CrossEntropyLoss(logits, targets)          # main objective
total_loss = ce_loss + 0.01 × aux_loss               # 0.01 keeps aux from dominating
```

**The `0.01` weight matters a lot.** Too low and the router ignores it (which is exactly what happened in this run — aux_loss stayed flat at ~4.0, indicating expert collapse). Too high and the model cares more about distributing tokens than predicting the next word.

---

## Part 8 — Training Machinery

### Concept 12: Cross-Entropy Loss

**Plain language:**
After the model produces scores (logits) for every possible next token, we need to measure how wrong it was. Cross-entropy loss answers: "how much probability mass did the model put on the correct answer?"

If the model was very confident and correct (assigned 90% probability to the right token), the loss is low. If it was confident and wrong (assigned 90% to the wrong token), the loss is very high. If it was uncertain (spread probability evenly across all 256 tokens), the loss is moderate.

**Mathematical intuition:**

```text
loss = -log(p_correct_token)

If p_correct = 0.9  →  loss = -log(0.9) = 0.105   (good, low loss)
If p_correct = 0.5  →  loss = -log(0.5) = 0.693   (uncertain)
If p_correct = 0.1  →  loss = -log(0.1) = 2.303   (wrong, high loss)
```

The model is penalized more severely for being confidently wrong than for being uncertain.

---

### Concept 13: Adam Optimizer

**Plain language:**
After computing the loss, we know *how wrong* the model was. The optimizer uses the gradient (the direction of steepest improvement) to update the model's parameters. Adam is the most widely used optimizer for training transformers.

Adam keeps track of two running averages for each parameter:

- **Momentum (m)** — the average direction of recent gradients. Like a ball rolling downhill, it keeps moving in a consistent direction rather than bouncing around.
- **Adaptive scale (v)** — the average *size* of recent gradients. Parameters that historically get large gradient updates receive smaller step sizes; parameters that rarely update get larger step sizes.

**Plain analogy:**
Imagine tuning 180,000 dials on a mixing board (each dial = one model parameter). SGD turns every dial by the same amount regardless of their history. Adam remembers which dials have been spinning wildly (large, noisy gradients) and turns those more cautiously, while being bolder with dials that haven't moved much.

---

### Concept 14: Cosine Annealing Learning Rate Scheduler

**Plain language:**
The learning rate controls how big each parameter update step is. A high learning rate explores broadly but might overshoot good solutions. A low learning rate converges precisely but slowly.

Cosine annealing starts with a high learning rate and smoothly decreases it to zero following a cosine curve. Early in training, the model makes large exploratory updates. As training progresses, updates become smaller and more precise, settling the model into a good minimum.

**Visual:**

```graph
LR
│╲
│ ╲
│  ╲___
│      ╲___________
│                  ╲_________
└──────────────────────────── Epoch
1                            500
```

**Why cosine rather than step-decay?**
Step-decay drops the LR abruptly (e.g., divide by 10 at epoch 300). This can cause the loss to spike momentarily. Cosine decay is smooth — no sudden changes, no instability.

---

### Concept 15: Gradient Clipping

**Plain language:**
Occasionally during training, a particularly "surprising" batch causes a very large gradient update — a spike that could send the model's parameters flying to a bad region, undoing hours of training. Gradient clipping is a safety valve.

Before applying any update, we measure the total size of all gradients. If it exceeds a threshold (1.0 here), we scale all gradients down proportionally so the total magnitude equals exactly 1.0. The *direction* of the update is preserved; only the magnitude is capped.

**Analogy:** If you're driving and suddenly slam the gas pedal, your cruise control (gradient clipping) doesn't brake — it just caps your acceleration so you don't skid off the road.

---

### Concept 16: Dropout

**Plain language:**
Dropout is a regularization technique that prevents overfitting. During training, it randomly "turns off" 10% of neurons at each forward pass — their output is set to zero. This forces the network to learn redundant representations, because it cannot rely on any single neuron always being present.

During inference (evaluation/generation), all neurons are active, but their outputs are scaled down to compensate for the fact that during training only 90% were active on average.

**Analogy:** Training a sports team where, at random, 10% of players sit out each practice. The team learns to play well regardless of who is absent. In the actual game (inference), everyone plays — but the team is more robust because no single player is a single point of failure.

---

## Part 9 — The Ablation Results Explained

### What the numbers mean

| Model | Parameters | Final CE Loss |
| ------- | ----------- | --------------- |
| Dense SwiGLU | 180,800 | 0.2820 |
| MoE (8 experts, top-2) | 870,976 | 0.2732 |

**CE Loss** is the cross-entropy loss. Lower is better. A loss of 0.28 means the model assigns roughly `e^(-0.28) ≈ 76%` of probability mass to the correct next token on average.

### Reading the result

The MoE model achieves **~3% lower loss** (0.2732 vs 0.2820) but uses **4.8× more parameters** (870K vs 181K). For this small experiment on a weather corpus, that's a poor trade-off — the benefit is marginal relative to the cost.

However, this does NOT mean MoE is bad. The MoE advantage shows up at scale. In large models (billions of parameters), MoE allows you to have 8× the knowledge at the same *inference compute cost*. Mixtral 8×7B performs like a 47B model while computing like a 14B model.

### The flat aux loss — what went wrong

The auxiliary loss stayed at ~4.0 throughout the entire 500 epochs (it barely moved). This is a strong signal of **expert collapse** — the router is consistently routing to the same 2–3 experts and ignoring the rest. The theoretical maximum of the aux loss formula when experts are perfectly balanced is 1.0 (`N × 1/N × 1/N = 1`). A value of 4.0 indicates highly uneven routing.

**Root causes for this experiment:**

1. **`AUX_LOSS_WEIGHT = 0.01` is too small** — the router's collapse is not penalized heavily enough to change behavior
2. **Very small model and dataset** — there's not enough diversity in the weather corpus to incentivize specialization across 8 experts
3. **Short sequences (length 8)** — not enough tokens per batch to establish meaningful routing statistics

In production MoE systems, the aux loss weight is tuned carefully, and techniques like expert-choice routing or token dropping are used to enforce balance.

---

## Glossary

| Term | Plain English Definition |
| ------ | -------------------------- |
| **Token** | The smallest unit of text the model works with — could be a word, part of a word, or a punctuation mark |
| **Embedding** | A list of numbers that represents a token's meaning in a geometric space |
| **Vector** | An ordered list of numbers — like coordinates in multi-dimensional space |
| **Dimension** | One of the 64 numbers in each embedding vector |
| **Gradient** | The mathematical direction in which to adjust each parameter to reduce the loss |
| **Backpropagation** | The process of computing gradients by working backwards through the network |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A small subset of training examples processed together (32 sequences here) |
| **Logits** | Raw unnormalized scores output by the model before softmax converts them to probabilities |
| **Softmax** | A function that converts a list of arbitrary numbers into probabilities that sum to 1 |
| **Loss** | A single number measuring how wrong the model's predictions are — lower is better |
| **Overfitting** | When a model memorizes training data rather than learning general patterns |
| **Ablation study** | A controlled experiment that changes exactly one component of a model to measure its isolated effect |
| **Parameters** | The learnable numbers inside a neural network — the "knobs" that training adjusts |
| **Inference** | Using a trained model to make predictions (as opposed to training it) |
| **KV Cache** | Memory used during text generation to avoid recomputing keys and values for past tokens |
| **Conditional computation** | Only activating a subset of the model's parameters for each input (the core MoE idea) |

---

## Further Reading

| Concept | Foundational Paper |
| --------- | -------------------- |
| Transformer architecture | *Attention Is All You Need* — Vaswani et al., 2017 |
| RoPE | *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021 |
| GQA | *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al., 2023 |
| SwiGLU | *GLU Variants Improve Transformer* — Noam Shazeer, 2020 |
| RMSNorm | *Root Mean Square Layer Normalization* — Zhang & Sennrich, 2019 |
| Mixture of Experts | *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer* — Shazeer et al., 2017 |
| MoE Load Balancing | *Switch Transformers* — Fedus et al., 2021 |
| BPE Tokenization | *Neural Machine Translation of Rare Words with Subword Units* — Sennrich et al., 2016 |

---

## Ablation Study 3: RoPE vs Sinusoidal Positional Encoding

> **Script:** `ablation_rope_vs_sinusoidal.py`
> **Question asked:** Does RoPE (Rotary Positional Embeddings) produce better language models than the original Sinusoidal positional encoding from the 2017 "Attention Is All You Need" paper?

---

## What Study 3 Is Testing

Every previous ablation used RoPE for positional encoding. This study steps back to ask: is RoPE actually better than the encoding method transformers were born with — sinusoidal positional encoding?

Everything is held constant: same tokenizer, same dataset, same RMSNorm, same SwiGLU FFN, same training loop. Only the positional encoding strategy changes.

---

## The Problem Both Methods Solve

**Plain language:**
A transformer's attention mechanism is, at its core, a set function — it has no built-in sense of order. Whether "the dog bit the man" or "the man bit the dog", the raw token representations are the same. The model needs to be told *where* each word is in the sequence.

Positional encoding solves this by injecting position information into each token's representation. The two approaches differ fundamentally in *when* and *how* they inject it.

---

## The Two Methods Compared

### Sinusoidal Positional Encoding (the original)

**Plain language:**
The original transformer (2017) adds a fixed pattern of sine and cosine waves to each token's embedding *before* any attention is computed. Think of it like a watermark: each position gets a unique fingerprint made of waves, and that fingerprint is stamped onto the token before it enters the network.

**How the watermark is built:**
Each position gets a vector of the same size as the token embedding (64 numbers here). Each number in that vector is a sine or cosine value at a specific frequency — low-frequency waves for coarse position, high-frequency waves for fine position:

```text
Position 0:  [sin(0/1),   cos(0/1),   sin(0/100), cos(0/100),   ...]
Position 1:  [sin(1/1),   cos(1/1),   sin(1/100), cos(1/100),   ...]
Position 2:  [sin(2/1),   cos(2/1),   sin(2/100), cos(2/100),   ...]
```

No two positions produce the same vector, and nearby positions produce similar vectors — the geometry encodes proximity.

**The formula:**

```text
PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
```

Where `pos` is the position (0, 1, 2…), `i` is the dimension index, and `d` is the embedding size (64).

**In code:**

```python
pe = torch.zeros(max_len, embed_dim)
position = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sine
pe[:, 1::2] = torch.cos(position * div_term)   # odd dims: cosine
x = x + pe[:, :x.size(1)]                       # add watermark to token embedding
```

**Key properties:**

- **Fixed** — the values are computed once and never learned or updated during training
- **Absolute** — each position gets a unique vector independent of other tokens
- **Applied once** — added to the embedding before the first layer, then untouched

**The limitation — absolute vs relative positions:**
Sinusoidal encoding tells the model "token X is at position 5." But for language, what often matters is the *relationship* — "these two tokens are 3 apart." Absolute position 5 in sentence A carries no information about position 5 in sentence B. More importantly, if the model is trained on sequences of length 8, it has never seen a sinusoidal pattern for position 9 or 100 — it cannot generalize to longer contexts.

---

### RoPE (Rotary Positional Embeddings)

**Plain language:**
Instead of stamping a watermark onto tokens *before* attention, RoPE injects position information *inside* the attention computation itself — by rotating the Query and Key vectors by an angle that depends on their position. The relative distance between two tokens is encoded directly in the dot product between their rotated Q and K.

Think of it like two compasses. Each token's query vector is a compass needle pointing in a direction that depends on its position. Token at position 3 points at 3 o'clock; token at position 7 points at 7 o'clock. The dot product between two compass needles encodes how far apart they are on the clock — regardless of where on the clock face they are in absolute terms.

**How the rotation is applied:**
The head dimension (16 here) is split into pairs. Each pair `(x1, x2)` is rotated by an angle `θ = position × frequency`:

```text
Rotated pair = [ x1·cos(θ) - x2·sin(θ),   x1·sin(θ) + x2·cos(θ) ]
```

This is the standard 2D rotation formula from geometry — rotating a 2D point `(x1, x2)` by angle `θ`.

**Why this encodes relative position:**
The dot product between a rotated Query at position `p` and a rotated Key at position `q` depends only on `(p - q)` — the *difference* in positions. The absolute positions cancel out mathematically. This is the key insight.

**In code:**

```python
cos, sin = precompute_rope_freqs(head_dim, T, device)   # compute once per sequence
q = apply_rope(q, cos, sin)   # rotate queries by their positions
k = apply_rope(k, cos, sin)   # rotate keys by their positions
# V is NOT rotated — values carry content, not position
attn = q @ k.transpose(-2, -1)   # dot product now encodes relative distance
```

**Key properties:**

- **No extra parameters** — the rotation angles are computed from a formula, not learned
- **Relative** — dot products encode token *distance*, not absolute position
- **Applied per-layer, inside attention** — every attention layer sees fresh positional information
- **Extrapolates to longer sequences** — since only relative differences matter, the model handles positions it never saw in training

---

## Side-by-Side Architecture Comparison

```text
SINUSOIDAL MODEL                    RoPE MODEL
========================            ========================
Token IDs                           Token IDs
    ↓                                   ↓
nn.Embedding (64-dim)               nn.Embedding (64-dim)
    ↓                                   ↓
+ SinusoidalPE  ← position          Dropout
  added HERE,                           ↓
  before all layers              [Transformer Blocks ×4]
    ↓                               ├── RMSNorm
Dropout                             ├── RoPEAttention
    ↓                               │     ├── Q_proj → rotate(Q, pos)
[Transformer Blocks ×4]             │     ├── K_proj → rotate(K, pos)
├── RMSNorm                         │     └── V_proj  (no rotation)
├── nn.MultiheadAttention           └── SwiGLU FFN
│     (position already baked in)
└── SwiGLU FFN
```

Notice: the sinusoidal model uses PyTorch's built-in `nn.MultiheadAttention` (a complete, optimized module), while the RoPE model uses a hand-written attention module so that Q and K can be rotated before the dot product. The sinusoidal model cannot use the hand-written module directly because position is already baked into the embedding — there is nothing left to inject inside the attention.

---

## Study 3 Results

| Model      | Final Loss | Positional Method                      | Parameters |
|------------|------------|----------------------------------------|------------|
| Sinusoidal | 0.2808     | Added to embedding, fixed              | ~197K      |
| RoPE       | 0.2768     | Injected into attention, formula-based | ~197K      |

Both models have the same parameter count — sinusoidal encoding adds no learnable parameters (it's a fixed precomputed table), and RoPE adds no parameters either (it's a formula applied at runtime).

**RoPE wins by 0.0040 loss** — a small but consistent margin across all 500 epochs.

### Reading the result

Looking at the epoch-by-epoch progression:

| Epoch | Sinusoidal | RoPE | RoPE advantage |
|-------|-----------|------|----------------|
| 100 | 0.3409 | 0.3303 | **−0.0106** (larger early) |
| 200 | 0.3115 | 0.3155 | +0.0040 (sinusoidal briefly better) |
| 300 | 0.2948 | 0.2958 | roughly equal |
| 400 | 0.2895 | 0.2859 | **−0.0036** |
| 500 | 0.2808 | 0.2768 | **−0.0040** |

RoPE starts with a bigger advantage (better early convergence), loses it in the middle, then pulls ahead again in the final stretch. This pattern suggests RoPE's relative-position encoding is particularly useful when the model needs to make fine-grained distinctions — which tends to happen in the later stages of training when the model is fitting harder examples.

### Why the difference is small here

On a short-sequence task (length 8), absolute and relative position information are nearly equivalent — there are only 8 positions, and every relative distance is also an absolute distance. The real advantage of RoPE would show up if:

1. **Sequence length increased** — at length 512 or 4096, relative positions generalize; absolute sinusoidal positions cannot extrapolate beyond training length
2. **Tasks requiring long-range dependencies** — e.g., coreference resolution where a pronoun must match a noun 50 tokens earlier

At the scale of GPT-4 or LLaMA 3 with 128K context windows, sinusoidal encoding simply cannot work — the positional patterns were only computed up to a fixed `max_len`. RoPE naturally extends to any length.

---

## A Historical Note — Three Generations of Positional Encoding

| Generation | Method | Used in | Limitation |
|------------|--------|---------|------------|
| 1st | Sinusoidal (fixed) | Original Transformer (2017), BERT | Cannot extrapolate beyond training length |
| 2nd | Learned absolute | GPT-2, GPT-3 | Cannot extrapolate; adds parameters per position |
| 3rd | RoPE (relative, formula) | LLaMA 1/2/3, Mistral, Gemma, Falcon | None known at standard scale |

**Learned absolute positions** (used in GPT-2/3) go one step further than sinusoidal — they let the model *learn* what fingerprint to assign to each position through training. This can work better than fixed sinusoidal on short sequences, but it still cannot extrapolate — if you trained on sequences up to length 1024, position 1025 has no learned embedding.

RoPE sidesteps both problems: no learned parameters (no capacity wasted), and relative distance encoding means the model handles any length without needing to have seen that position during training.

---

## Study 3 Key Takeaway

> Sinusoidal encoding injects position *once*, *before* any layer sees the token, as an absolute fingerprint. RoPE injects position *inside* every attention layer as a rotation that encodes *relative distance*. On short sequences both approaches work similarly. At scale and long contexts, RoPE's relative encoding is essential — it is why every modern LLM abandoned the original sinusoidal scheme.

---

## Ablation Study 2: GQA vs Standard Multi-Head Attention (MHA)

> **Script:** `ablation_gqa_vs_mha.py`
> **Question asked:** Does Grouped Query Attention (GQA) match the quality of standard Multi-Head Attention (MHA), and what does it cost?

---

## What Study 2 Is Testing

The Dense vs MoE study explored *which feed-forward network design* works better. This study isolates a completely different question: *does the attention mechanism matter?* Specifically, does reducing the number of Key and Value heads (GQA) hurt model quality compared to giving every Query head its own dedicated K/V pair (MHA)?

Everything else is held constant: same tokenizer, same dataset, same RMSNorm, same SwiGLU FFN, same RoPE, same training loop. Only the attention module changes.

---

## The Two Attention Mechanisms Compared

### Standard Multi-Head Attention (MHA)

**Plain language:**
In standard attention, every "head" is fully independent. If you have 4 heads:

- Head 1 asks its own questions (Q1), searches using its own keys (K1), and retrieves its own values (V1)
- Head 2 does the same with Q2, K2, V2
- And so on for Head 3 and Head 4

Each head learns to focus on something different — one might track grammatical agreement, another long-range topic coherence, another recent local context.

**The memory problem at inference time:**
When a language model is generating text (not training, but actually producing output), it processes tokens one at a time. But attention needs to look back at every past token. To avoid recomputing K and V for all past tokens at every step, models store them in a **KV cache**.

With 4 heads, you store K and V for every head, every past token. In a real large language model (say, 32 layers, 32 heads, long context), this KV cache can consume gigabytes of memory. It becomes the main memory bottleneck at inference time.

**Architecture (this experiment):**

```text
4 Query projections  (Q1, Q2, Q3, Q4)  → 4 × 16 = 64 dims
4 Key projections    (K1, K2, K3, K4)  → 4 × 16 = 64 dims
4 Value projections  (V1, V2, V3, V4)  → 4 × 16 = 64 dims
1 Output projection                    → 64 × 64 dims
```

**In code:**

```python
self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)   # 64 → 64
self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)   # 64 → 64
self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)   # 64 → 64
```

---

### Grouped Query Attention (GQA)

**Plain language:**
GQA asks: do we really need 4 independent K/V pairs? What if Q heads share K and V heads — 2 Q heads per KV pair?

```text
Standard MHA (4Q, 4KV):         GQA (4Q, 2KV):
  Q1 ──► K1, V1                   Q1 ──┐
  Q2 ──► K2, V2                   Q2 ──┼──► K1, V1   (group 1 shares)
  Q3 ──► K3, V3                   Q3 ──┐
  Q4 ──► K4, V4                   Q4 ──┼──► K2, V2   (group 2 shares)
```

Each group of 2 Query heads shares the same Key and Value. The Queries are still different — each Q head is still asking different questions. But the "what tokens are available to attend to" (K) and "what information gets passed" (V) is shared within the group.

**The savings:**
- K and V projections shrink from `64→64` to `64→32` (half the size)
- The KV cache at inference is halved
- Q projections and the output projection are unchanged
- Model quality stays very close to MHA

**The `repeat_interleave` trick:**
After projecting K and V to 2 heads, the code needs to "expand" them back to 4 heads so the matrix multiplication shapes align:

```python
k = k.repeat_interleave(self.groups, dim=1)   # (B, 2, T, D) → (B, 4, T, D)
v = v.repeat_interleave(self.groups, dim=1)
```

This just duplicates each KV head twice — it's not learning anything new, just making the shape match for efficient batch matrix multiplication. The underlying computation is equivalent to Q1 and Q2 sharing K1/V1.

**In code:**
```python
self.q_proj = nn.Linear(embed_dim, num_q_heads * head_dim, bias=False)   # 64 → 64
self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)  # 64 → 32  ← smaller
self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)  # 64 → 32  ← smaller
```

---

## Where the Parameter Difference Comes From

Both models have 4 transformer blocks. In each block, the only difference is the K and V projection matrices:

| Layer | MHA size | GQA size |
|-------|----------|----------|
| Q projection | 64 × 64 = 4,096 | 64 × 64 = 4,096 |
| **K projection** | **64 × 64 = 4,096** | **64 × 32 = 2,048** |
| **V projection** | **64 × 64 = 4,096** | **64 × 32 = 2,048** |
| Output projection | 64 × 64 = 4,096 | 64 × 64 = 4,096 |
| **Attention total** | **16,384** | **12,288** |

Per block, GQA saves 4,096 parameters. Across 4 blocks: **16,384 fewer parameters total**. This matches the observed difference: 197,184 − 180,800 = **16,384**.

---

## Study 2 Results

| Model | Parameters | Final Loss |
|-------|-----------|------------|
| MHA (4Q / 4KV) | 197,184 | 0.2792 |
| GQA (4Q / 2KV) | 180,800 | 0.2815 |

**Difference: 0.0023 loss** — essentially negligible. GQA converges almost identically to MHA on this task.

### Reading the result

The loss curves track almost perfectly for all 500 epochs. GQA finishes 0.0023 higher — a difference so small it would disappear with a different random seed or slightly different hyperparameters.

Meanwhile, GQA uses **8.3% fewer parameters** with a **50% smaller KV cache** at inference.

**Why GQA can match MHA despite having fewer K/V parameters:**
The Q heads are still diverse and independent — each one can learn to ask different questions. The shared K/V acts as a shared "information index" that different Q heads look up in different ways. The loss of K/V diversity is mostly compensated by the diversity in Q, which is retained in full.

### The real-world case for GQA

On this tiny model (64-dim embeddings, 4 heads), the KV cache savings are small in absolute terms. The benefit compounds dramatically at scale:

| Model scale | MHA KV cache (est.) | GQA KV cache (est.) | Savings |
|-------------|---------------------|---------------------|---------|
| Small (64-dim, 4 blocks) | ~0.1 MB | ~0.05 MB | 50% |
| LLaMA 2 7B (4096-dim, 32 layers, 32 heads) | ~2 GB | ~1 GB (16 KV heads) | 50% |
| LLaMA 3 70B (8192-dim, 80 layers, 8 KV heads vs 64 Q heads) | ~40 GB | ~5 GB | 87.5% |

At 70B scale, GQA is not a nice-to-have — it's the difference between running the model on available hardware or not. This is why LLaMA 2 (70B), LLaMA 3, Mistral, and Gemma all adopted GQA.

---

## MQA vs GQA vs MHA — The Full Family

While we're here, it's useful to know there is a whole family of attention variants:

| Variant | Q heads | KV heads | Groups | Used in |
|---------|---------|----------|--------|---------|
| **MHA** (Multi-Head) | N | N | 1:1 | GPT-2, BERT, early transformers |
| **MQA** (Multi-Query) | N | 1 | N:1 | Early fast inference models |
| **GQA** (Grouped Query) | N | N/G | G | LLaMA 2/3, Mistral, Gemma |

MQA (Multi-Query Attention) is the extreme version — all Query heads share a single K/V pair. It gives the most memory savings but more quality degradation than GQA. GQA is the practical sweet spot, which is why it became the standard.

---

## Study 2 Key Takeaway

> GQA is a **free lunch at scale**: by sharing K/V heads among groups of Q heads, you cut the KV cache in half (or more) with near-zero loss in model quality. The attention mechanism's expressiveness comes primarily from diverse Query heads — not from diverse Key/Value heads. This insight, validated at scale in papers like LLaMA 2, is confirmed even in this miniature experiment.

---
