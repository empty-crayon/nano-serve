from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request

from nanoserve.core.dependencies import get_dispatcher
from nanoserve.relay.dispatcher import Dispatcher

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def health_check(
    request: Request,
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)],
):
    """
    Gateway health check — probes the single vLLM backend.
    """

    backend_status = await _probe_backend(dispatcher._backend_url)

    return {
        "gateway": "up",
        "backend_url": dispatcher._backend_url,
        "backend_status": backend_status,
    }


async def _probe_backend(base_url: str) -> str:
    """
    Probe a backend to check if it's reachable.
    Tries /health first, then /v1/models as fallback.
    Returns "up" or "down".
    """
    async with httpx.AsyncClient(timeout=2.0) as client:
        # Try /health first
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code < 500:
                return "up"
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        # Fallback: try /v1/models (OpenAI-compatible endpoint)
        try:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code < 500:
                return "up"
        except (httpx.RequestError, httpx.TimeoutException):
            pass

    return "down"