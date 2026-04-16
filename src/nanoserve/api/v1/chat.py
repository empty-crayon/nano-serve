from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from nanoserve.core.dependencies import get_dispatcher
from nanoserve.core.exceptions import InvalidRequestError
from nanoserve.observability.metrics import ERRORS_TOTAL
from nanoserve.observability.recorder import record_metrics
from nanoserve.relay.dispatcher import Dispatcher
from nanoserve.schemas.chat import ChatCompletionRequest
from nanoserve.schemas.request_context import RequestContext

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)],
):
    # Validate last message is from user
    if body.messages and body.messages[-1].role != "user":
        raise InvalidRequestError("Last message must have role 'user'")

    # Build request context
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    technique = request.headers.get("X-Pipeline-Stage", "") or body.technique

    ctx = RequestContext(
        request_id=request_id,
        model=body.model,
        stream=body.stream,
        messages=list(body.messages),
        technique=technique,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        chat_template_kwargs=body.chat_template_kwargs,
    )

    try:
        result = await dispatcher.dispatch(ctx)
    except Exception:
        record_metrics(ctx)
        ERRORS_TOTAL.labels(technique=ctx.technique or "unknown").inc()
        raise

    if ctx.stream:
        async def _stream_and_record():
            try:
                async for chunk in result:
                    yield chunk
            finally:
                record_metrics(ctx)
                if (ctx.status_code or 0) >= 400:
                    ERRORS_TOTAL.labels(technique=ctx.technique or "unknown").inc()

        return StreamingResponse(
            _stream_and_record(),
            media_type="text/event-stream",
            headers={"X-Request-ID": ctx.request_id},
        )

    # Non-streaming
    record_metrics(ctx)
    return JSONResponse(
        content=result,
        headers={"X-Request-ID": ctx.request_id},
    )