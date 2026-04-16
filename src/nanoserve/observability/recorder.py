"""
Prometheus metrics recorder - single function that observes all metrics from RequestContext.

This is called AFTER a request completes (both streaming and non-streaming).
For streaming requests, it's called in a wrapper's finally block after all chunks are consumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import (
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    STREAM_INTER_CHUNK_SECONDS,
    TOKENS_PER_SECOND,
    TPOT_SECONDS,
    TTFT_SECONDS,
)

if TYPE_CHECKING:
    from nanoserve.schemas.request_context import RequestContext

def record_metrics(ctx: RequestContext) -> None:
    """
    Observe all Prometheus metrics from a completed request's context.
    All metrics are labelled by technique (classify / generate / gate).
    """

    labels = {"technique": ctx.technique or "unknown"}

    # --------------------------------------------------------- #
    # Always record                                             #
    # --------------------------------------------------------- #

    REQUESTS_TOTAL.labels(**labels).inc()

    if ctx.e2e_latency_s > 0:
        REQUEST_DURATION.labels(**labels).observe(ctx.e2e_latency_s)

    if ctx.prompt_tokens > 0:
        PROMPT_TOKENS.labels(**labels).inc(ctx.prompt_tokens)

    if ctx.completion_tokens > 0:
        COMPLETION_TOKENS.labels(**labels).inc(ctx.completion_tokens)

    # Throughput: tokens/sec when both e2e and completion are known
    if ctx.e2e_latency_s > 0 and ctx.completion_tokens > 0:
        TOKENS_PER_SECOND.labels(**labels).observe(
            ctx.completion_tokens / ctx.e2e_latency_s
        )

    # --------------------------------------------------------- #
    # Timing - streaming vs non-streaming split                 #
    # --------------------------------------------------------- #

    if ctx.stream:
        # Streaming: TTFT and TPOT are separately measured
        if ctx.ttft_s > 0:
            TTFT_SECONDS.labels(**labels).observe(ctx.ttft_s)

        if ctx.tpot_avg_s > 0:
            TPOT_SECONDS.labels(**labels).observe(ctx.tpot_avg_s)

        # Observe each individual inter-chunk delay for tail latency analysis
        for delay_s in ctx.inter_chunk_latency_s:
            STREAM_INTER_CHUNK_SECONDS.labels(**labels).observe(delay_s)

    else:
        # Non-streaming: TTFT = e2e (first token IS the full response)
        if ctx.e2e_latency_s > 0:
            TTFT_SECONDS.labels(**labels).observe(ctx.e2e_latency_s)

        # TPOT = e2e / completion_tokens (amortized per-token cost)
        if ctx.e2e_latency_s > 0 and ctx.completion_tokens > 0:
            tpot_s = ctx.e2e_latency_s / ctx.completion_tokens
            TPOT_SECONDS.labels(**labels).observe(tpot_s)