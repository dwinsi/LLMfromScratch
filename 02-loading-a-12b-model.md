# 02 — Loading a 12B Model: What I Learned

*Part 2 of 5 — From Neuron to Agent*
*Series index: [README.md](./README.md)*

---

## Overview

This document covers the complete setup process for loading `google/gemma-4-12B-it` on Kaggle T4 x2 GPU using HuggingFace Transformers 5.x — including every error encountered, the root cause of each, and the correct resolution.

Environment:

- Runtime: Kaggle Notebook, GPU T4 x2 (2 × 15 GB VRAM = ~30 GB combined)
- Python: 3.12
- Transformers: 5.12.0
- bitsandbytes: 0.46.1+

---

## Prerequisites

### 1. Accept the Gemma 4 gated model license

Gemma 4 is a gated model. Your HuggingFace token will return `403 Forbidden` until the license is explicitly accepted.

URL: <https://huggingface.co/google/gemma-4-12B-it>

Navigate to the model page while logged in. Accept the license. Approval is instant.

### 2. Create a HuggingFace access token

Settings → Access Tokens → New Token → Type: **Read**

### 3. Add token to Kaggle Secrets

Add-ons → Secrets → Add new secret

| Field | Value |
| --- | --- |
| Name | `HF_TOKEN` |
| Value | your HuggingFace token |
| Notebook access | **On** (toggle must be enabled) |

> If Notebook access is off, `UserSecretsClient().get_secret()` hangs silently with no error. This is the most common cause of a cell that runs indefinitely.

### 4. Kaggle notebook settings

| Setting | Value |
| --- | --- |
| Accelerator | GPU T4 x2 |
| Internet | On |

---

## Installation

```python
!pip install -q -U transformers bitsandbytes accelerate ddgs
```

> **Restart the kernel after installation.** bitsandbytes registers CUDA extensions at import time. Without a restart, the old version stays loaded in memory and quantization will fail with `ImportError`.

---

## Authentication

The correct pattern for Kaggle is environment variable injection. The `huggingface_hub.login()` call hangs in non-interactive Kaggle kernels even with a token passed directly.

```python
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")

print("Token set:", os.environ["HF_TOKEN"][:8] + "...")
```

HuggingFace Transformers reads `HF_TOKEN` from the environment automatically. No explicit `login()` call is needed.

---

## Errors encountered and resolutions

### Error 1 — Wrong model class

**What Kaggle auto-generates:**

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-12B-it")
```

**Error:**

```text
ImportError: cannot import name 'AutoModelForMultimodalLM' from 'transformers'
```

**Root cause:** `AutoModelForMultimodalLM` does not exist in HuggingFace Transformers. Kaggle's auto-generated snippet hallucinated this class name.

**Correct class:** `AutoModelForImageTextToText`

Gemma 4 12B is `Gemma4UnifiedForConditionalGeneration` — an encoder-free multimodal architecture that handles text, image, video, and audio through direct linear projections. Even for text-only inference the correct loading class is `AutoModelForImageTextToText`.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
```

References:

- <https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified>

---

### Error 2 — Deprecated `torch_dtype` parameter

**Code that triggers the warning:**

```python
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16   # deprecated in Transformers 5.x
)
```

**Warning:**

```text
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

**Root cause:** `torch_dtype` was renamed to `dtype` in Transformers 5.x.

**Resolution:** Remove the parameter entirely. Transformers 5.x defaults to `dtype="auto"`, which reads the correct dtype from the model's `config.json`. For Gemma 4 12B this resolves to `bfloat16`.

```python
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
    # dtype omitted — Transformers 5.x reads from model config automatically
)
```

---

### Error 3 — bitsandbytes not installed or below minimum version

**Error:**

```text
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes:
`pip install -U bitsandbytes>=0.46.1`
```

**Root cause:** bitsandbytes was either missing or below the version required by Transformers 5.x for 4-bit quantization.

**Resolution:**

```python
!pip install -q -U bitsandbytes>=0.46.1
```

Restart the kernel after installation.

---

### Error 4 — `login()` hangs indefinitely

**Code that hangs:**

```python
from huggingface_hub import login
login(token=hf_token, add_to_git_credential=False)  # hangs on Kaggle
```

**Root cause:** `login()` internally waits for an interactive confirmation prompt in some Kaggle kernel versions, even with a token passed directly.

**Resolution:** Use environment variable injection instead (see Authentication section above).

---

## Correct loading code

```python
import os
import torch
from kaggle_secrets import UserSecretsClient
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# --- Authentication ---
user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")

MODEL_ID = "google/gemma-4-12B-it"

# --- Quantization config ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# --- Load processor and model ---
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)

model.eval()

print("Memory footprint:", round(model.get_memory_footprint() / 1e9, 2), "GB")
print("Device map:", model.hf_device_map)
```

**Expected output:**

```text
Memory footprint: 7.5 GB
Device map: {
  'model.language_model.embed_tokens': 0,
  'model.language_model.layers.0': 0,
  ...
  'model.language_model.layers.11': 0,
  'model.language_model.layers.12': 1,
  ...
  'model.language_model.layers.47': 1,
  'model.language_model.norm': 1,
  'model.embed_vision': 1,
  'model.embed_audio': 1
}
```

---

## What the output means

### Memory footprint: 7.5 GB

Without 4-bit NF4 quantization, Gemma 4 12B loads at approximately 24 GB (bfloat16). The `BitsAndBytesConfig` compresses weights from 16 bits to 4 bits per parameter, reducing the footprint to 7.5 GB. This is what makes the model fit on two T4 GPUs.

The quality tradeoff is minimal for instruction-following tasks. NF4 (Normal Float 4) is designed for normally distributed weights and preserves more information per bit than standard INT4.

### Device map: layers split across two GPUs

`device_map="auto"` distributed the 48 Transformer blocks across both GPUs:

| GPU | Layers | Description |
| --- | --- | --- |
| GPU 0 | `layers.0` – `layers.11` | First 12 Transformer blocks + embeddings |
| GPU 1 | `layers.12` – `layers.47` | Remaining 36 Transformer blocks + norm + vision/audio projections |

This is the same structure as `mini_llm.py` from Project 7 — stacked Transformer blocks — distributed across hardware automatically. Each `layers.N` is one complete block: multi-head attention, feed-forward network, RMSNorm pre/post, residual connections.

### `model.embed_vision` and `model.embed_audio`

Gemma 4 12B is a unified multimodal model. These are lightweight linear projection layers that map vision and audio tokens into the language model's embedding space. For text-only inference they are loaded but never activated.

---

## Why `AutoProcessor` instead of `AutoTokenizer`

`AutoProcessor` wraps both the text tokenizer and any image/audio preprocessing in a single object. For `AutoModelForImageTextToText`, using `AutoProcessor` is required — the model expects inputs formatted through the processor's `apply_chat_template()` method, not raw tokenizer output.

For text-only inference, only the tokenizer path inside the processor is used. The interface is the same.

```python
# AutoTokenizer — text-only models (AutoModelForCausalLM)
inputs = tokenizer(prompt, return_tensors="pt")

# AutoProcessor — multimodal models (AutoModelForImageTextToText)
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

---

## Build series connection

In Projects 1–7, loading a model meant instantiating a Python class:

```python
mini_llm_model = MiniLanguageModel(
    vocabulary_size=198,
    embedding_dim=64,
    number_of_heads=4,
    number_of_blocks=4,
    context_length=4
)
# Parameters: 159,558
# Memory: negligible
# Device: single CPU or GPU
```

In Project 12, loading a model means downloading 24 GB of pre-trained weights, compressing them to 7.5 GB at load time, and distributing 48 Transformer blocks across two GPUs:

```python
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-12B-it",
    quantization_config=quantization_config,
    device_map="auto"
)
# Parameters: 12,000,000,000
# Memory: 7.5 GB (4-bit NF4)
# Device: distributed across GPU 0 and GPU 1
```

The blocks are the same structure. The scale is not.

---

## References

- `AutoModelForImageTextToText` API: <https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified>
- `BitsAndBytesConfig` (4-bit quantization): <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes>
- `AutoProcessor` API: <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor>
- `apply_chat_template`: <https://huggingface.co/docs/transformers/main/en/chat_templating>
- Gemma 4 model card: <https://huggingface.co/google/gemma-4-12B-it>
- Google AI Gemma 4 overview: <https://ai.google.dev/gemma/docs/core>
- Agent source: [../Research_Agent_Gemma4/model.py](../Research_Agent_Gemma4/model.py)

---

*[← 01 — From LLM to Agent](./01-from-llm-to-agent.md) | [Series Index](./README.md) | [Next: 03 — Prompt Engineering as Design →](./03-prompt-engineering-as-design.md)*
