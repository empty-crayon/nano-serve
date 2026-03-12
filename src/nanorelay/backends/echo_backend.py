import json
import time
from typing import AsyncGenerator

from nanorelay.backends.backend_interface import Backend
from nanorelay.schemas.chat import ChatMessage


class EchoBackend(Backend):
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        model: str,
        stream: bool,
    ) -> dict | AsyncGenerator[str, None]:
        content = f"Echo: {messages[-1].content}"

        if stream:
            return self._generate_stream(request_id, model, content)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }

    async def _generate_stream(
        self,
        request_id: str,
        model: str,
        content: str,
    ) -> AsyncGenerator[str, None]:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"