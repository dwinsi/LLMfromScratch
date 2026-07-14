"""GET /health — Ollama connectivity check."""

import httpx
from fastapi import APIRouter

from saathi.api.schemas import HealthResponse
from saathi.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check whether the API is running and Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        reachable = resp.status_code == 200
        detail = None if reachable else f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        reachable = False
        detail = str(exc)

    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        model=settings.ollama_model,
        detail=detail,
    )
