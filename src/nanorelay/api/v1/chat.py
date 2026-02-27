import time
import uuid
from fastapi import APIRouter, Request

from nanorelay.core.config import settings
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.runner.runner import LlamaServerClient, Runner
from nanorelay.schemas.chat import (
    ChatCompletionMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChoice, 
    MessageRole
)


router = APIRouter(prefix="/chat")


@router.get("/healthz")
async def health_check():
    return {"status": "ok"}


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.messages:
        # Extract the last user content as the prompt.
        last_user_msg = body.messages[-1]
        prompt = last_user_msg.content
        role = last_user_msg.role

        if role != MessageRole.user:
            raise InvalidRequestError("Last message must have role 'user'")
        
        if not prompt.strip():
            raise InvalidRequestError("Prompt cannot be empty")

        client = LlamaServerClient(settings.backend_url)
        text_output = await client.chat(prompt)

        if text_output is None:
            runner = Runner()
            text_output = runner.run(prompt)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(
                        role=MessageRole.assistant,
                        content=text_output
                    ),
                    finish_reason="stop"
                ),
            ]
        )