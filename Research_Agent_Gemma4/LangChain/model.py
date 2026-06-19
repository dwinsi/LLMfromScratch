"""
model.py — Gemma 4 Research Agent (Phase 2 — LangChain)
Handles model loading and wraps Gemma 4 in LangChain's ChatHuggingFace.

Model: google/gemma-4-12B-it (Gemma4UnifiedForConditionalGeneration)
Quantization: 4-bit NF4 via BitsAndBytes (~7.5 GB VRAM)
Runtime: Local GPU or Kaggle T4 x2

References:
    Model card:              https://huggingface.co/google/gemma-4-12B-it
    AutoModelForImageTextToText: https://huggingface.co/docs/transformers/main/en/model_doc/gemma4_unified
    BitsAndBytesConfig:      https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
    HuggingFacePipeline:     https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
    ChatHuggingFace:         https://python.langchain.com/docs/integrations/chat/huggingface/
"""

import os
import re
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()

MODEL_ID = "google/gemma-4-12B-it"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor():
    """
    Loads Gemma 4 12B in 4-bit NF4 quantization.
    Reads HF_TOKEN from environment variable (set via .env locally
    or Kaggle Secrets on Kaggle).

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
    )

    model.eval()

    # Override generation config to avoid max_length conflict
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = 2048
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.95
    model.generation_config.do_sample = True

    return model, processor


# ---------------------------------------------------------------------------
# LangChain wrapper
# ---------------------------------------------------------------------------

def build_chat_model(model, processor):
    """
    Wraps the loaded Gemma 4 model in LangChain's ChatHuggingFace.

    ChatHuggingFace applies apply_chat_template() automatically — the
    correct interface for instruction-tuned models. HuggingFacePipeline
    alone does not apply the chat template and will produce token
    repetition with instruction-tuned models.

    Args:
        model: Loaded Gemma 4 model.
        processor: Loaded Gemma 4 processor.

    Returns:
        ChatHuggingFace: LangChain-compatible chat model.

    References:
        https://python.langchain.com/docs/integrations/chat/huggingface/
        https://huggingface.co/docs/transformers/main/en/chat_templating
    """
    generation_pipeline = pipeline(
        task="image-text-to-text",
        model=model,
        processor=processor
    )

    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


# ---------------------------------------------------------------------------
# Output cleaner
# ---------------------------------------------------------------------------

def clean_response(raw: str) -> str:
    """
    Strips Gemma 4 special tokens from generated output.

    Removes:
        - Thinking blocks: <|channel>thought ... <channel|>
        - Input prompt echo: everything before the last <|turn>model marker
        - Turn markers: <|turn>, <turn|>
        - Special tokens: <bos>, <eos>

    Args:
        raw (str): Raw decoded output from the model.

    Returns:
        str: Clean response text.

    References:
        Gemma 4 thinking mode: https://ai.google.dev/gemma/docs/capabilities/text/basic
    """
    # Strip thinking tokens
    raw = re.sub(r"<\|channel>thought.*?<channel\|>", "", raw, flags=re.DOTALL)
    # Strip everything up to and including the last <|turn>model marker
    if "<|turn>model" in raw:
        raw = raw.split("<|turn>model")[-1]
    # Strip all remaining special tokens
    raw = re.sub(r"<\|turn>|<turn\|>|<bos>|<eos>", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Both `max_new_tokens`")

    from langchain_core.messages import SystemMessage, HumanMessage

    print("Loading model...")
    model, processor = load_model_and_processor()
    print("Memory footprint:", round(model.get_memory_footprint() / 1e9, 2), "GB")

    print("Building chat model...")
    chat_model = build_chat_model(model, processor)

    messages = [
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content="What is the capital of France? Answer in one sentence.")
    ]

    response = chat_model.invoke(messages)
    print("Response:", clean_response(response.content))