from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from nanoserve.core.dependencies import get_dispatcher
from nanoserve.relay.dispatcher import Dispatcher

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models(
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)],
):
    """
    OpenAI-compatible GET /v1/models — proxies to the configured vLLM backend
    and returns whatever models it reports.
    """
    try:
        response = await dispatcher._http_client.get(
            f"{dispatcher._backend_url}/v1/models",
            timeout=5.0,
        )
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except (httpx.RequestError, httpx.HTTPStatusError):
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Backend unavailable", "type": "BackendError"}},
        )
