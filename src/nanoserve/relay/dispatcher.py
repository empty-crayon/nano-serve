from __future__ import annotations
from typing import AsyncGenerator

import httpx

from nanoserve.backends.echo_backend import EchoBackend
from nanoserve.core.backend_config import BackendConfig
from nanoserve.relay.client import OpenAICompatClient
from nanoserve.schemas.request_context import RequestContext


class Dispatcher:
    """
    Forwards all requests to a single vLLM backend.
    The only special case is model="echo" which uses an in-process test backend.
    """

    def __init__(self, config: BackendConfig):
        self._http_client = httpx.AsyncClient()
        self._client = OpenAICompatClient(self._http_client)
        self._echo = EchoBackend()
        self._backend_url = config.backend_url
        self._timeout = config.timeout
        self._echo_timeout = config.echo_timeout

    async def close(self):
        await self._http_client.aclose()

    async def dispatch(
        self,
        ctx: RequestContext,
    ) -> dict | AsyncGenerator[str, None]:
        # Echo model — in-process test backend, no HTTP
        if ctx.model == "echo":
            return await self._echo.generate(ctx)

        print(f"[{ctx.request_id}] -> vLLM (model='{ctx.model}', technique='{ctx.technique}')")

        if ctx.stream:
            return self._client.chat_stream(
                ctx=ctx,
                base_url=self._backend_url,
                timeout=self._timeout,
            )

        return await self._client.chat(
            ctx=ctx,
            base_url=self._backend_url,
            timeout=self._timeout,
        )