# Mamba: Selective State Space Models

This document explains the Mamba architecture from first principles. It is the final step in a series that builds from Legendre polynomials through HiPPO and S4. By the end of this document you will understand what makes Mamba different from earlier SSMs and why that difference matters.

---

## The three files in this folder

| File | Requires PyTorch? | What it demonstrates |
| --- | --- | --- |
| `selective_ssm_numpy.py` | No (NumPy only) | The bare Mamba mechanism: input-dependent B, C, Delta plus parallel scan, with a proof that both methods give identical results |
| `mamba_train_numpy.py` | No (NumPy only) | A trainable selective SSM with hand-derived gradients, learning a selective recall task from scratch |
| `mamba_block_torch.py` | Yes | The full production-style Mamba block with convolution, gating, and selective scan, ready to replace a Transformer block |

Start with `selective_ssm_numpy.py` to understand the mechanism, then `mamba_train_numpy.py` to see it learn, then `mamba_block_torch.py` for the full architecture.

---

## Running the code

```bash
# These two run on any machine with NumPy installed
python selective_ssm_numpy.py
python mamba_train_numpy.py

# This one requires PyTorch
pip install torch
python mamba_block_torch.py
```

---

## The problem Mamba solves

Everything in this series up to S4 has one limitation: the matrices A, B, C, and the step size Delta are **fixed**. They do not change based on what the current input is. Every token in the sequence is processed with exactly the same parameters. The word "the" and the word "earthquake" go through the same computation.

This is called a **Linear Time-Invariant** (LTI) system. The "time-invariant" part means "the rules do not change over time." LTI systems are mathematically convenient and can be trained efficiently using the convolutional trick. But they have a real limitation: they cannot decide that one token is more important to remember than another.

Consider a sequence like:

```text
"The capital of France is Paris. The capital of Germany is Berlin."
```

If a question asks "What is the capital of France?", an ideal memory system should write "Paris" strongly into the state when it encounters it, and write "Berlin" weakly or not at all (since it is not relevant to the question). But an LTI SSM processes both tokens identically. It cannot selectively write or ignore.

This is the problem Mamba solves with one targeted change: **selectivity**.

---

## The one idea: make B, C, and Delta depend on the input

In a vanilla SSM, the update rule at each step is:

```text
new state = A_bar * old state + B_bar * new input
output    = C * new state
```

Where A_bar and B_bar are computed once from the fixed A, B, and Delta, and then reused at every step. The same A_bar and B_bar apply to every token.

Mamba's change: compute B, C, and Delta fresh from the current input at every step:

```text
Vanilla SSM:    B, C, Delta are fixed parameters (same for every token)

Mamba (S6):     B     = a small linear layer applied to the current input
                C     = a small linear layer applied to the current input
                Delta = softplus( a small linear layer applied to the current input )
```

Now at every step, the model computes new B, C, and Delta based on what the current token is. This means:

- **B** controls how strongly the current input is written into memory. An important token can be written in with full strength. A filler word can be nearly erased.
- **C** controls which aspects of memory are read out to produce the output. The model can attend to different parts of its state for different queries.
- **Delta** controls the step size of the state update. A large Delta means a big update (the current token dominates). A small Delta means a tiny update (the state mostly carries forward unchanged).

Together, these three input-dependent parameters give the model the ability to selectively write, hold, and read from its state based on the content of each token. This is "content-based reasoning."

### A concrete example

Suppose the model is reading: "The **cat** sat on the mat. What animal was there?"

At the token "cat":

- The model sees that this is a content-bearing noun.
- It generates a large Delta for "cat", causing the state to update strongly.
- B is set to write this content into memory with high weight.

At the tokens "sat", "on", "the", "mat":

- These are common function words.
- The model generates a small Delta, causing the state to barely change.
- The memory of "cat" is mostly preserved.

At "What animal":

- C is set to read from the part of the state that holds animal information.

The LTI SSM cannot do any of this. It treats all tokens identically. Mamba, by making B, C, Delta input-dependent, can.

---

## The cost of selectivity: the parallel scan

When B, C, and Delta are fixed (LTI), the sequence of state updates can be computed as a convolution. Convolutions can be parallelized using the Fast Fourier Transform (FFT), giving efficient training over long sequences.

When B, C, and Delta vary per token (Mamba), the convolution interpretation no longer applies. The kernel would be different at every position, so the FFT trick does not work. We are back to a sequential recurrence:

```text
state_1 = A_bar_1 * state_0 + B_bar_1 * input_1
state_2 = A_bar_2 * state_1 + B_bar_2 * input_2
state_3 = A_bar_3 * state_2 + B_bar_3 * input_3
...
```

This looks like it must be computed left-to-right, one step at a time. If you have a sequence of 10,000 tokens, it looks like you need 10,000 sequential steps. That would be very slow on a GPU, where the hardware is designed for massively parallel operations.

Mamba solves this with the **parallel scan algorithm**.

### How the parallel scan works

The key observation is that the recurrence operation is **associative**. Two recurrence pairs `(a2, b2)` and `(a1, b1)` can be combined as:

```text
(a2, b2) combine (a1, b1)  =  (a2 * a1,   a2 * b1 + b2)
```

This means: applying step 1 followed by step 2 is the same as applying this single combined step. Combining two steps into one is legal because the underlying recurrence is linear.

Associativity enables a "divide and conquer" approach called the **Hillis-Steele scan**:

```text
Round 1: combine pairs  (1,2), (3,4), (5,6), ...   in parallel
Round 2: combine pairs  (1-2, 3-4), (5-6, 7-8), ...  in parallel
Round 3: combine pairs  (1-4, 5-8), ...  in parallel
...
```

Each round halves the number of unresolved prefix operations. After `log2(L)` rounds, all L prefix values are computed. A sequence of 1024 tokens needs only 10 rounds of parallel computation instead of 1024 sequential steps.

The `selective_ssm_numpy.py` file implements this scan and includes a direct numerical verification that the parallel scan gives the exact same result as the sequential loop, to machine precision (differences of around 1e-16, which is just floating-point rounding).

Real Mamba goes further: it uses a **hardware-aware** version of the scan that fuses the entire operation into a single GPU kernel, keeping the state in fast on-chip SRAM rather than writing it back to slower GPU HBM memory at each step. The mathematical result is identical, but the speed is significantly better.

---

## What the training demo shows

`mamba_train_numpy.py` trains on a **selective recall task**. The sequence is a random stream of values with one position "marked" by a special flag. The model's job is to output the marked value at the end of the sequence.

This task is specifically designed to require selectivity:

- A model that remembers everything equally well would average over all values and do poorly.
- A model that randomly forgets most things would also do poorly.
- To succeed, the model must learn to recognize the mark, write that value strongly into memory, and hold it while treating all other values as unimportant.

An LTI SSM cannot learn this task effectively because it has no mechanism to treat the marked position differently. Mamba can.

Results from a training run:

```text
Step      Training error
   0        0.3229   (random initialization)
 500        0.0694
1000        0.0101
4000        0.0034

Final error: 0.0036
Baseline error (predict the average): 0.3204
```

That is roughly a 90x improvement over the baseline. The model genuinely learned to use selective memory for content-based recall.

### A subtle but important finding about A

The training demo also makes concrete a key design detail: the values inside the A matrix must be kept small in magnitude.

Recall from the HiPPO and S4 notes that A is discretized as:

```text
A_bar = exp(Delta * A)
```

If A contains large negative values, say `A = -(1, 2, 3, ..., N)` as derived directly from HiPPO, then even with a small Delta the exponential `exp(Delta * A)` decays toward zero very quickly. The state is essentially erased after just a couple of steps. Memory cannot survive a sequence of 20 tokens.

The fix: initialize A with small negative values, around -0.05 to -0.5. Then `exp(Delta * A)` is close to 1, meaning the state changes slowly and information persists across many steps. The input-dependent Delta then controls how much to update at each step. A large Delta (triggered by an important token) causes a meaningful update. A small Delta (for a filler token) allows the state to pass through almost unchanged.

This is why the official Mamba implementation always initializes A with small-magnitude values. The training demo lets you see this concretely by comparing runs with different A initializations.

---

## The full Mamba block

The production-style Mamba block in `mamba_block_torch.py` wraps the selective SSM core in a complete layer structure. Here is the full forward pass:

```text
Input: x with shape (batch, sequence_length, d_model)

Step 1: LayerNorm
        Normalize the input across the d_model dimension.

Step 2: in_proj
        Expand from d_model to 2 * d_inner (roughly 2x wider).
        This creates two paths: the SSM path and the gate path.
        Split into: ssm_input (d_inner) and gate (d_inner)

Step 3: Causal Conv1d  (applied to the SSM path only)
        A depthwise convolution with a small window (typically 4 tokens).
        Each channel gets its own local context before entering the SSM.
        Causal means it only looks at the current and past tokens, never future.

Step 4: SiLU activation
        Applied to the SSM path after the convolution.

Step 5: Selective SSM core
        Compute B, C, Delta from the SSM input (linear projections).
        Discretize to get A_bar, B_bar.
        Run the recurrence via parallel scan.
        Output has shape (batch, sequence_length, d_inner).

Step 6: Gated output
        Multiply the SSM output by SiLU(gate).
        The gate modulates the output, similar to the gating in SwiGLU.

Step 7: out_proj
        Project from d_inner back to d_model.

Step 8: Residual connection
        Add the original input (before LayerNorm).

Output: shape (batch, sequence_length, d_model) -- same as input
```

### Key design choices in the code

**The A_log parameterization.** Instead of storing A directly, the code stores `log(|A|)` and reconstructs `A = -exp(A_log)`. This guarantees A is always negative, which guarantees the state always decays rather than growing. No matter what gradient descent does to `A_log`, the reconstruction `A = -exp(A_log)` can never become positive. Stability is enforced structurally, not by hoping the optimizer stays in a safe region.

**Causal convolution padding.** To make the convolution causal (no peeking at future tokens), the input is padded with `d_conv - 1` zeros at the start, then the output is trimmed back to length L. This is equivalent to a sliding window that never extends past the current position.

**Drop-in shape.** Both the input and output have shape `(batch, seq_len, d_model)`. This is the same shape contract as a Transformer block. The `MambaBlock` can replace a `TransformerBlock` directly in any architecture without changing anything else.

---

## Why Mamba is more efficient than attention for long sequences

Both Transformer attention and Mamba read a sequence and produce an output. But their costs at inference time are very different.

Attention is a **quadratic** operation at inference. When generating token k, attention computes a weighted sum over all k-1 previous tokens. The key-value cache grows with k: at position 1000, you store 1000 vectors; at position 10,000, you store 10,000 vectors. Memory and compute both grow with sequence length.

Mamba uses a **constant** amount of memory at inference. The entire history is compressed into the fixed-size hidden state (N numbers per SSM layer). When token k arrives, the state is updated with one matrix-vector multiply. It does not matter whether k is 100 or 100,000: the computation is the same size.

| Property | Transformer attention | Mamba |
| --- | --- | --- |
| Training compute | O(L^2) per layer | O(L log L) via parallel scan |
| Inference memory | O(L) grows with context | O(N) fixed regardless of context |
| Inference compute per token | O(L) grows with context | O(N) fixed regardless of context |
| Content-based selection | Yes (via attention weights) | Yes (via input-dependent B, C, Delta) |

For short sequences, Transformers and Mamba perform similarly. For very long contexts (books, codebases, audio), Mamba's constant inference cost becomes a significant practical advantage.

---

## How everything in this series connects

The series built up to Mamba in four steps:

```text
Step 1: Legendre polynomials
        Orthogonal reference shapes: flat, slope, U-curve, S-wiggle, ...
        Key property: each captures completely independent information.

Step 2: HiPPO
        Uses Legendre polynomials as the basis for compressing a history.
        Derives the unique correct A and B matrices for this compression.
        A is lower-triangular with negative diagonal: stable by design.

Step 3: S4 and S4D
        Reparameterizes A (diagonal plus low-rank, then just diagonal)
        so the convolution kernel can be computed efficiently.
        Enables O(L log L) training on sequences of any length.

Step 4: Mamba (this document)
        Keeps the stable HiPPO-derived A structure.
        Makes B, C, Delta input-dependent: the "selectivity" upgrade.
        Replaces the convolution trick (which no longer applies)
        with the parallel scan (works even with per-token parameters).
```

The HiPPO-derived A matrix is the backbone throughout. Its lower-triangular structure ensures coarse-to-fine information flow. Its negative diagonal eigenvalues ensure stability. Mamba keeps all of this and adds the ability to reason about which tokens deserve to be remembered.

---

## Where to go next

**Experiment with the architecture:**
Swap `MambaBlock` into the TinyGPT project from earlier in this series. Both have the shape `(batch, seq_len, d_model)` in and out, so the swap is a one-line change. Compare the loss curves and generation speed.

**Improve the scan:**
The NumPy implementation in `selective_ssm_numpy.py` uses a straightforward Hillis-Steele loop. Replace it with `torch.associative_scan` (available in recent PyTorch versions) for a GPU-native parallel scan.

**Read the source implementations:**
The official Mamba implementation at `github.com/state-spaces/mamba` is the authoritative reference. The minimal version at `github.com/johnma2006/mamba-minimal` is much shorter and easier to read. The code in this folder deliberately mirrors their structure so the comparison is direct.

**Go deeper into the theory:**
The paper "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu and Dao, 2023, arXiv:2312.00752) derives the hardware-aware parallel scan algorithm in detail and benchmarks Mamba against Transformer variants on a range of tasks. The companion `HiPPO_SSM_explained.md` in this folder contains the full mathematical derivation of where A and B come from.
