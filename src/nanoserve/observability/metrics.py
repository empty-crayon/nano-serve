from prometheus_client import Counter, Histogram, Info, REGISTRY, CONTENT_TYPE_LATEST, generate_latest

# Single label: technique (classify / generate / gate / unknown)
METRIC_LABELS = ["technique"]

# ------------------------------------------------------------------------- #
# Core request metrics                                                       #
# ------------------------------------------------------------------------- #

REQUEST_DURATION = Histogram(
    "nanoserve_request_duration_seconds",
    "End-to-end request latency",
    METRIC_LABELS,
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 60),
)

TTFT_SECONDS = Histogram(
    "nanoserve_time_to_first_token_seconds",
    "Time to first token (streaming) or full response (non-streaming)",
    METRIC_LABELS,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 60),
)

TPOT_SECONDS = Histogram(
    "nanoserve_time_per_output_token_seconds",
    "Average inter-token latency",
    METRIC_LABELS,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

STREAM_INTER_CHUNK_SECONDS = Histogram(
    "nanoserve_stream_inter_chunk_delay_seconds",
    "Individual inter-chunk delays (streaming only) - useful for tail latency p95/p99 analysis",
    METRIC_LABELS,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
)

PROMPT_TOKENS = Counter("nanoserve_prompt_tokens_total", "Total prompt tokens", METRIC_LABELS)
COMPLETION_TOKENS = Counter("nanoserve_completion_tokens_total", "Total completion tokens", METRIC_LABELS)
REQUESTS_TOTAL = Counter("nanoserve_requests_total", "Total requests", METRIC_LABELS)
ERRORS_TOTAL = Counter("nanoserve_errors_total", "Total failed requests", METRIC_LABELS)

TOKENS_PER_SECOND = Histogram(
    "nanoserve_completion_tokens_per_second",
    "Completion throughput",
    METRIC_LABELS,
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000),
)

# ------------------------------------------------------------------------- #
# Gateway metadata                                                           #
# ------------------------------------------------------------------------- #

GATEWAY_INFO = Info(
    "nanoserve_gateway_info",
    "Gateway build metadata and configuration",
)
GATEWAY_INFO.info({
    "version": "0.2.0",
    "architecture": "single_backend",
    "extended_timing": "true",
})

# ------------------------------------------------------------------------- #
# Helpers                                                                    #
# ------------------------------------------------------------------------- #

def drop_builtin_collectors() -> None:
    """
    Remove Python's default GC and platform collectors from the registry.
    Keeps /metrics output focused on LLM gateway metrics only.

    Called once at application startup.
    """
    for collector in list(getattr(REGISTRY, "_collector_to_names", {}).keys()):
        if collector.__class__.__name__ in ("GCCollector", "PlatformCollector", "ProcessCollector"):
            try:
                REGISTRY.unregister(collector)
            except (KeyError, ValueError):
                pass