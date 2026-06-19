"""
model.py — Gemma 4 Research Agent
Handles model loading and text generation.

Model: google/gemma-4-12B-it (Gemma4UnifiedForConditionalGeneration)
Quantization: 4-bit NF4 via BitsAndBytes
Runtime: Kaggle T4 x2 GPU

References:
    Model card:
        https://huggingface.co/google/gemma-4-12B-it

    Transformers API — Gemma4 Unified (AutoModelForImageTextToText):
        https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified

    Transformers API — AutoProcessor:
        https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor

    Transformers API — BitsAndBytesConfig (4-bit quantization):
        https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes

    Transformers API — GenerationConfig (generate() parameters):
        https://huggingface.co/docs/transformers/main/en/main_classes/text_generation

    Transformers API — apply_chat_template:
        https://huggingface.co/docs/transformers/main/en/chat_templating

    Google AI — Gemma 4 text inference guide:
        https://ai.google.dev/gemma/docs/capabilities/text/basic

    Google AI — Gemma 4 model overview:
        https://ai.google.dev/gemma/docs/core
"""

import json
import os
import pathlib
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_cfg = json.loads((pathlib.Path(__file__).parent / "config.json").read_text())
_model_cfg = _cfg["model"]

MODEL_ID = _model_cfg["model_id"]
MAX_NEW_TOKENS = _model_cfg["max_new_tokens"]
TEMPERATURE = _model_cfg["temperature"]
TOP_P = _model_cfg["top_p"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor():
    """
    Loads Gemma 4 12B in 4-bit NF4 quantization and returns the model
    and processor as a tuple.

    Reads HF_TOKEN automatically from the environment variable set in
    the Kaggle notebook. No token argument needed here.

    Returns:
        tuple: (model, processor)

    References:
        https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified
        https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
        # dtype is intentionally omitted — Transformers 5.x defaults to
        # "auto" which reads the correct dtype from the model's config.json
    )

    model.eval()

    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(
    model,
    processor,
    user_message: str,
    system_message: str = "You are a helpful research assistant."
) -> str:
    """
    Takes a plain text user message, formats it using Gemma 4's chat
    template, runs inference, and returns only the assistant's response
    text with thinking tokens stripped.

    Args:
        model: Loaded Gemma 4 model.
        processor: Loaded Gemma 4 processor.
        user_message: The user's input as a plain string.
        system_message: Optional system prompt. Defaults to research assistant.

    Returns:
        str: Clean assistant response text.

    References:
        apply_chat_template: https://huggingface.co/docs/transformers/main/en/chat_templating
        generate() params:   https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        Thinking mode:       https://ai.google.dev/gemma/docs/capabilities/text/basic
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        }
    ]

    # apply_chat_template handles the full prompt formatting including
    # special tokens — we do not manually construct the prompt string
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_token_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_token_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True
        )

    # Slice off input tokens — output contains input + generated tokens
    response_token_ids = output_token_ids[0][input_token_length:]
    raw_response = processor.decode(response_token_ids, skip_special_tokens=True)

    return _strip_thinking_tokens(raw_response).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_thinking_tokens(text: str) -> str:
    """
    Gemma 4's thinking mode wraps internal reasoning in:
        <|channel>thought\\n ... <channel|>
    This helper removes that block so only the final answer is returned.
    """
    cleaned = re.sub(
        r"<\|channel>thought.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Sanity check — run directly to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model...")
    model, processor = load_model_and_processor()
    print("Memory footprint:", round(model.get_memory_footprint() / 1e9, 2), "GB")
    print("Device map:", model.hf_device_map)

    print("\nRunning sanity check...")
    response = generate_response(
        model,
        processor,
        user_message="What is the capital of France? Answer in one sentence."
    )
    print("Response:", response)