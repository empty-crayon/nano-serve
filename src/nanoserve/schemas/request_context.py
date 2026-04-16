from dataclasses import dataclass, field

from nanoserve.schemas.chat import ChatMessage

@dataclass
class RequestContext:
    """
    Mutable context object that flows through the entire request lifecycle.
    Each layer reads what it needs and writes what it knows.

    Created by: endpoint (api/v1/chat.py)
    Written by: classifier -> dispatcher -> client/backend
    Read by:    metrics, tracing, logging (future phases)
    """

    # --- Set at creation (endpoint) ---
    request_id: str
    model: str
    stream: bool
    messages: list[ChatMessage] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    chat_template_kwargs: dict | None = None

    # --- Set by endpoint from X-Pipeline-Stage header ---
    technique: str = ""                  # "classify" | "generate" | "gate" | ""

    # --- Set after response (by client or backend) ---
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0                  # time to first token (streaming only)
    tpot_avg_s: float = 0.0              # avg time per output token
    e2e_latency_s: float = 0.0           # total wall-clock time
    status_code: int = 0 

    # --- Streaming-only metrics (for detailed tail latency analysis) ---
    inter_chunk_latency_s: list[float] = field(default_factory=list) # individual inter-chunk delays