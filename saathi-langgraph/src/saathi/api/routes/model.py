"""GET /model/info — current LLM configuration."""

from fastapi import APIRouter

from saathi.api.schemas import ModelInfoResponse
from saathi.config import settings

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return the active model configuration drawn from settings / env vars."""
    return ModelInfoResponse(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        context_window=settings.context_window,
        max_tokens=settings.max_tokens,
        max_parallel_tools=settings.max_parallel_tools,
    )
