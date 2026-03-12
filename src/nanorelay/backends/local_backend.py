from typing import AsyncGenerator

import httpx

from nanorelay.backends.backend_interface import Backend
from nanorelay.relay.client import OpenAICompatClient
from nanorelay.schemas.chat import ChatMessage


class LocalBackend(Backend):
    def __init__(self, url: str, timeout: float, http_client: httpx.AsyncClient):
        self._client = OpenAICompatClient(http_client)
        self._url = url
        self._timeout = timeout

    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        model: str,
        stream: bool,
    ) -> dict | AsyncGenerator[str, None]:
        if stream:
            return self._client.chat_stream(
                request_id=request_id,
                base_url=self._url,
                messages=messages,
                model=model,
                timeout=self._timeout,
            )

        return await self._client.chat(
            request_id=request_id,
            base_url=self._url,
            messages=messages,
            model=model,
            timeout=self._timeout,
        )