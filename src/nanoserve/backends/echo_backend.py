from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from nanoserve.schemas.request_context import RequestContext

class EchoBackend:
    """
    Test backend with simulated latency - streams word-by-word with realistic delays.
    No actual LLM call, useful for GPU-free testing of timing/metrics.
    """

    TTFT_DELAY_MS = 200
    PER_WORD_DELAY_MS = 50

    # Default informative response about the echo backend
    DEFAULT_RESPONSE = (
        "This is the Echo backend responding. I simulate realistic inference timing "
        "with a 200ms time-to-first-token delay and 50ms per word. "
        "I'm useful for testing your gateway's metrics collection, streaming logic, "
        "and observability stack without needing a real GPU. "
        "Your message was: {user_message}"
    )

    async def generate(
        self,
        ctx: RequestContext,
    ) -> dict | AsyncGenerator[str, None]:
        user_message = ctx.messages[-1].content if ctx.messages else "(empty)"
        content = self.DEFAULT_RESPONSE.format(user_message=user_message)

        prompt_tokens = sum(len(m.content.split()) for m in ctx.messages)
        completion_tokens = len(content.split())

        if ctx.stream:
            return self._generate_stream(ctx, content, prompt_tokens, completion_tokens)

        # Non-streaming: simulate some processing delay
        start = time.perf_counter()
        await asyncio.sleep(self.TTFT_DELAY_MS / 1000.0)

        result = {
            "id": ctx.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ctx.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        ctx.e2e_latency_s = time.perf_counter() - start
        ctx.prompt_tokens = prompt_tokens
        ctx.completion_tokens = completion_tokens
        ctx.status_code = 200
        return result

    async def _generate_stream(
        self,
        ctx: RequestContext,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        created = int(time.time())

        # First chunk: role
        role_chunk = {
            "id": ctx.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": ctx.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        }
        
        # Simulate TTFT delay
        await asyncio.sleep(self.TTFT_DELAY_MS / 1000.0)
        ctx.ttft_s = time.perf_counter() - start
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # Stream word by word
        words = content.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self.PER_WORD_DELAY_MS / 1000.0)
            
            # Add space before word (except first)
            word_with_space = (" " + word) if i > 0 else word
            
            word_chunk = {
                "id": ctx.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ctx.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word_with_space},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(word_chunk)}\n\n"

        # Final chunk with finish_reason and usage
        final_chunk = {
            "id": ctx.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": ctx.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"

        ctx.prompt_tokens = prompt_tokens
        ctx.completion_tokens = completion_tokens
        ctx.status_code = 200
        ctx.e2e_latency_s = time.perf_counter() - start

        yield "data: [DONE]\n\n"