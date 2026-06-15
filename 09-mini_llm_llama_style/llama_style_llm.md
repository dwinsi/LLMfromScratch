# 09 — LLaMA-Style Architecture (RMSNorm + RoPE + SwiGLU)

Three architectural changes that together produce a LLaMA-style Transformer. Each change is small in isolation. Together they represent the core architectural differences between a basic Transformer and a modern open-source language model.

After this project, the model is architecturally equivalent to LLaMA 1.

---

## Change 1: RMSNorm

**Replaces:** `nn.LayerNorm`

**What LayerNorm does:**
```
mean       = mean(x)
variance   = var(x)
normalised = (x - mean) / sqrt(variance + epsilon)
output     = gamma * normalised + beta
```

**What RMSNorm does:**
```
rms        = sqrt(mean(x²) + epsilon)
normalised = x / rms
output     = gamma * normalised
```

RMSNorm removes the mean subtraction entirely. It only divides by the root mean square of the activations. This is simpler, faster, and uses one fewer learned parameter per layer (no `beta` shift term).

The key insight: the mean subtraction in LayerNorm was added to make the normalisation shift-invariant, but in practice the learned scale parameter `gamma` handles this anyway. RMSNorm removes the redundant step without losing quality.

Used in: LLaMA 1/2/3, Gemma 1/2/4, Mistral, Falcon.

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms        = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        normalised = x / rms
        return self.learned_scale * normalised
```

---

## Change 2: RoPE (Rotary Positional Encoding)

**Replaces:** Sinusoidal positional encoding added to embeddings

**What sinusoidal encoding does:**
Adds a fixed position vector to each token embedding before the first attention layer. Position information is injected once at the input and must survive through all subsequent layers.

**What RoPE does:**
Rotates the Query and Key vectors by an angle that depends on their absolute position, directly inside each attention head at every layer.

```
Standard: token_representations = word_embeddings + positional_encoding
RoPE:     Q_rotated = rotate(Q, position)
          K_rotated = rotate(K, position)
          scores    = Q_rotated @ K_rotated.T / sqrt(head_dim)
```

The rotation formula for each dimension pair (i, i+1):

```
Q_rotated[i]   = Q[i]   * cos(position * freq) - Q[i+1] * sin(position * freq)
Q_rotated[i+1] = Q[i]   * sin(position * freq) + Q[i+1] * cos(position * freq)
```

Different frequencies are used for different dimension pairs, similar to sinusoidal encoding. Lower dimensions use high frequencies (change quickly with position). Higher dimensions use low frequencies (change slowly).

**Why RoPE is better:**

Position information is refreshed at every attention layer rather than injected once. The dot product between rotated Q and K naturally encodes relative position: `Q_pos_m @ K_pos_n` depends on `m - n` rather than absolute positions `m` and `n` separately. This makes RoPE better at generalising to longer sequences than those seen during training.

The sinusoidal buffer in `MiniLanguageModel` is completely removed. RoPE frequencies are precomputed once per block and stored as non-learnable buffers.

Used in: LLaMA 1/2/3, Gemma 1/2/4, Mistral, Falcon, GPT-NeoX, PaLM 2.

---

## Change 3: SwiGLU

**Replaces:** `Linear → GELU → Linear` feed forward network

**What the standard FFN does:**
```
hidden = GELU(x @ W_expand + b)
output = hidden @ W_compress + b
```
Expand to a larger dimension, apply non-linearity, compress back.

**What SwiGLU does:**
```
gate   = SiLU(x @ W_gate)     <- learned gate
value  = x @ W_value           <- learned values
hidden = gate * value          <- gating: values filtered by gate
output = hidden @ W_compress   <- compress back
```

SwiGLU introduces a gating mechanism. Two parallel projections are computed from the same input. One is the gate (passed through SiLU activation). The other is the value. Their element-wise product determines what information passes through to the compression layer.

The gate can independently suppress or amplify each dimension. This is more expressive than a simple activation function applied uniformly because different dimensions can be gated differently depending on the input.

**SiLU** (Sigmoid Linear Unit) is `x * sigmoid(x)`. Also called Swish. Smooth, non-monotonic, similar to GELU but with a slightly different curve. PyTorch implements it as `F.silu()` or `nn.SiLU()`.

**Why three weight matrices instead of two?**

SwiGLU uses W_gate, W_value and W_compress instead of W_expand and W_compress. To keep the total parameter count comparable, the hidden dimension is reduced from the standard 4× expansion to approximately 2/3 × the standard hidden dim. This is the ratio used in LLaMA.

Used in: LLaMA 1/2/3, PaLM, Gemma, likely GPT-4.

```python
class SwiGLUFeedForward(nn.Module):
    def forward(self, x):
        gate   = F.silu(self.gate_projection(x))
        value  = self.value_projection(x)
        hidden = gate * value
        return self.compress_projection(hidden)
```

---

## How the three changes work together

```text
Project 7/8 block:
  LayerNorm
  Multi-head attention (sinusoidal position in embeddings)
  Residual
  LayerNorm
  Linear → GELU → Linear
  Residual

Project 9 block (LLaMA-style):
  RMSNorm
  Multi-head attention (RoPE inside Q and K)
  Residual
  RMSNorm
  W_gate × SiLU + W_value → W_compress  (SwiGLU)
  Residual
```

The structure is the same. Every component is replaced with a more efficient or more expressive modern equivalent.

---

## Parameter comparison

```text
Project 9 vs Project 8 (per block):
  LayerNorm -> RMSNorm:  saves beta parameter per layer (minor)
  MHA -> RoPE attention: same Q, K, V, O count (no change)
  GELU FFN -> SwiGLU:    adds W_gate, adjusts hidden dim (minor change)
```

Total parameters are similar to Project 8. The value of these changes is not in reducing parameter count but in improving training dynamics and expressive capacity.

---

## Running

```python
pip install torch tokenizers matplotlib
python mini_llm_llama.py
```

## Files

```text
mini_llm_llama.py        the training script
weather_corpus_v2.txt    shared with Project 8
images/
  loss_curve_llama_style.png
```

## What is next

Project 10 adds Grouped Query Attention. K and V heads are reduced from 4 to 2, with each KV head serving 2 query heads. This brings the model to LLaMA 2 level and introduces the KV cache efficiency that makes large models practical at inference time.