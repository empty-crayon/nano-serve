import time
from typing import Annotated
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from nanorelay.core.config import settings
from nanorelay.core.dependencies import get_dispatcher
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.relay.dispatcher import Dispatcher
from nanorelay.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest, 
    ChatCompletionResponse,  
    MessageRole
)


router = APIRouter(prefix="/chat")


@router.get("/healthz")
async def health_check():
    return {"status": "ok"}


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request, 
    body: ChatCompletionRequest,
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)],
):
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.messages:
        # Extract the last user content as the prompt.
        last_user_msg = body.messages[-1]

        if last_user_msg.role != MessageRole.user:
            raise InvalidRequestError("Last message must have role 'user'")
        
        if not last_user_msg.content.strip():
            raise InvalidRequestError("Prompt cannot be empty")
            
        backend_name, model_output = await dispatcher.dispatch(request_id, body.messages, body.model, body.stream)

        if body.stream:
            return StreamingResponse(model_output, media_type="text/event-stream")
        else:
            return ChatCompletionResponse(**model_output)
