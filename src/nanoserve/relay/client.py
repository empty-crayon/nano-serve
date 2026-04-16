from __future__ import annotations
import json
import httpx
import time
from typing import TYPE_CHECKING, AsyncGenerator
from nanoserve.core.exceptions import NanoServeException
from nanoserve.schemas.request_context import RequestContext

class OpenAICompatClient:
    """
    A single HTTP client that forwards requests to any backend
    implementing the OpenAI /v1/chat/completions specification.

    Reads request params from RequestContext, writes timing + usage back.
    """

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    # non streaming
    async def chat(
        self,
        ctx: RequestContext,
        base_url: str,
        timeout: float = 60.0,
    ) -> dict:
        payload = {
            "model": ctx.model,
            "messages": [{"role": m.role, "content": m.content} for m in ctx.messages],
            "stream": False,
        }
        if ctx.max_tokens is not None:
            payload["max_tokens"] = ctx.max_tokens
        if ctx.temperature is not None:
            payload["temperature"] = ctx.temperature
        if ctx.top_p is not None:
            payload["top_p"] = ctx.top_p
        if ctx.top_k is not None:
            payload["top_k"] = ctx.top_k
        if ctx.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = ctx.chat_template_kwargs

        start = time.perf_counter()

        try:
            response = await self._client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            ctx.status_code = response.status_code
            response.raise_for_status()
            response_data = response.json()
        except httpx.HTTPStatusError as e:
            ctx.status_code = e.response.status_code if e.response is not None else None
            ctx.e2e_latency_s = time.perf_counter() - start
            detail: str
            try:
                detail = e.response.text
            except Exception:
                detail = str(e)
            raise NanoServeException(
                status_code=int(ctx.status_code or 502),
                message=f"Upstream error {ctx.status_code} from {base_url}: {detail}",
            )

        ctx.e2e_latency_s = time.perf_counter() - start
        response_data["id"] = ctx.request_id

        # Extract usage if the upstream returned it
        usage = response_data.get("usage")
        if usage:
            ctx.prompt_tokens = usage.get("prompt_tokens", 0)
            ctx.completion_tokens = usage.get("completion_tokens", 0)

        return response_data

    async def chat_stream(
        self,
        ctx: RequestContext,
        base_url: str,
        timeout: float = 60.0,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": ctx.model,
            "messages": [{"role": m.role, "content": m.content} for m in ctx.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if ctx.max_tokens is not None:
            payload["max_tokens"] = ctx.max_tokens
        if ctx.temperature is not None:
            payload["temperature"] = ctx.temperature
        if ctx.top_p is not None:
            payload["top_p"] = ctx.top_p
        if ctx.top_k is not None:
            payload["top_k"] = ctx.top_k
        if ctx.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = ctx.chat_template_kwargs

        start = time.perf_counter()
        first_ttft_s: float | None = None
        last_nonempty_s: float | None = None
        inter_chunk_latency: list[float] = []

        try:
            async with self._client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            ) as response:
                ctx.status_code = response.status_code
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    
                    data_part = line[len("data: "):]
                    if data_part == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    
                    try:
                        chunk = json.loads(data_part)
                    except json.JSONDecodeError:
                        continue
                    
                    chunk["id"] = ctx.request_id
                    if "model" not in chunk or not chunk["model"]:
                        chunk["model"] = ctx.backend_name or "unknown"
                    
                    now = time.perf_counter()
                    choices = chunk.get("choices") or []

                    # --- TTFT: first chunk with actual content ---
                    if first_ttft_s is None and choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            first_ttft_s = now - start
                            ctx.ttft_s = first_ttft_s
                    
                    # Inter token latency: time between non-empty content chunks
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if last_nonempty_s is not None:
                                inter_chunk_latency.append(now - last_nonempty_s)
                            last_nonempty_s = now

                    # --- Usage from final chunk (vLLM / OpenAI pattern) ---
                    # Keep overwriting usage until the end, so we capture the final token counts
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        ctx.prompt_tokens = chunk_usage.get("prompt_tokens", 0)
                        ctx.completion_tokens = chunk_usage.get("completion_tokens", 0)
                    
                    yield f"data: {json.dumps(chunk)}\n\n"

        except httpx.HTTPError:
            # Ensure status_code reflects the error for metric recording.
            # HTTP errors (4xx/5xx) already have ctx.status_code set via response.status_code
            # above; network errors (ConnectError, ReadTimeout) never got a response,
            # so status_code is still 0 — default to 502 Bad Gateway.
            if not ctx.status_code or ctx.status_code < 400:
                ctx.status_code = 502
            error_chunk = {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            ctx.e2e_latency_s = time.perf_counter() - start
            if inter_chunk_latency:
                ctx.tpot_avg_s = sum(inter_chunk_latency) / len(inter_chunk_latency)
                ctx.inter_chunk_latency_s = inter_chunk_latency # Store for metrics recorder