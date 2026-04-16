# Observability package - Prometheus metrics and recording
from .metrics import (
    COMPLETION_TOKENS,
    ERRORS_TOTAL,
    GATEWAY_INFO,
    PROMPT_TOKENS,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    STREAM_INTER_CHUNK_SECONDS,
    TOKENS_PER_SECOND,
    TPOT_SECONDS,
    TTFT_SECONDS,
    drop_builtin_collectors,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from .recorder import record_metrics

__all__ = [
    # Metrics
    "COMPLETION_TOKENS",
    "ERRORS_TOTAL",
    "GATEWAY_INFO",
    "PROMPT_TOKENS",
    "REQUEST_DURATION",
    "REQUESTS_TOTAL",
    "STREAM_INTER_CHUNK_SECONDS",
    "TOKENS_PER_SECOND",
    "TPOT_SECONDS",
    "TTFT_SECONDS",
    # Helpers
    "drop_builtin_collectors",
    "generate_latest",
    "CONTENT_TYPE_LATEST",
    # Recorder
    "record_metrics",
]