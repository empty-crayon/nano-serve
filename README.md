# nano-serve

> See [DEEP_DIVE.md](deep-dive.md) for an end-to-end InferenceOps pipeline using nano-serve and [how-fast](https://github.com/empty-crayon/how-fast). It includes custom Agentic workload, controlled ablations on 4 vLLM engine configurations, concurrency sweeps, and SLO analysis on Qwen3.5-4B / A10 GPU on [Lambda Cloud](https://lambda.ai). With detailed justification on various production inferencing decisions.

An LLM inference gateway. OpenAI-compatible `POST /v1/chat/completions`, SSE streaming with per-request TTFT / TPOT / inter-chunk timing, 10 Prometheus metrics on every request, and 4 Grafana dashboards covering both the gateway and the underlying vLLM engine.

Goal of this inference gateway:
- Expose a load balancer and a proxy layer (for policy & observaibility) before any request reaches the inference engine
- Extensibility for multi-backend routing, workload aware routing, and other proxy-layer logic
- Separating gateway level metrics from engine metrics: vLLM exposes great engine-side metrics out of the box, but nothing tells you what your proxy layer is doing on top of that


---

## System architecture

![System Architecture](assets/systemarchitecture-img.png)

Client requests flow through an optional nginx load balancer into multiple possible inference gateway instances, which forwards them to vLLM over HTTP. Both nano-serve (`:8765/metrics`) and vLLM (`:8000/metrics`) get scraped by Prometheus every 15 seconds. Grafana queries Prometheus across 4 pre-provisioned dashboards.

The `X-Pipeline-Stage` header carries a technique label (`classify` / `generate` / `gate`) from the client. Nano-serve stamps this on every Prometheus metric, so per-stage latency breakdown shows up in Grafana with no application-side configuration.

---

## Quick start

**Prerequisites**: Python 3.12+, [uv](https://github.com/astral-sh/uv) (or pip), Docker for the observability stack.

```bash
# 1. Clone and install
git clone <repo>
cd nano-serve
uv sync   # or: pip install -e .

# 2. Configure
cp .env.example .env
cp config.yaml.example config.yaml
# Edit config.yaml: set backend_url to your vLLM instance

# 3a. Start the gateway
python main.py
# Listening on 127.0.0.1:8765

# 3b. Start nginx load balancer in front of the gateway
#     Listens on :8780 and round-robins to nano-serve on :8765
cd nano-serve/monitoring
nginx -t -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"
# Send traffic to :8780 instead of :8765 to go through the load balancer.
# To run two nano-serve workers, start a second process with NANOSERVE_PORT=8766
# and uncomment the second server line in monitoring/nginx-gateway-lb.conf.
# 3c. Stop later:
nginx -s quit -p /tmp -c "nano-serve/monitoring/nginx-gateway-lb.conf"

# 4. Smoke test (no GPU needed — uses the in-process echo backend)
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "hello"}]}'

# 5. Health check
curl http://localhost:8765/health

# 6. Start observability stack
cd monitoring && docker compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:4010  (admin/admin)
```

---

## What it actually solves

Three things broke when I first tried to instrument a vLLM stack the obvious way.

1. **Client disconnects silently dropped error metrics:** The naive version:
    - Under high QPS, clients abort streams constantly. Wrap the generator in `try/finally` instead, so `record_metrics(ctx)` runs even when the client disconnects mid-stream.

```python
async for chunk in result:
    yield chunk
record_metrics(ctx)  # never runs if client aborts
```



2. **Network errors had no status code:** When vLLM times out, `httpx.HTTPError` fires before any HTTP response arrives. `ctx.status_code` stays at 0, and the error counter never increments. The fix is one `except` block defaulting to 502.

3. **Token counts came back zero:** vLLM only returns usage data in the final SSE chunk if you explicitly send `stream_options: {"include_usage": true}` upstream. Without it, `prompt_tokens` and `completion_tokens` stay zero forever and you have no idea what your gateway is actually serving.

---

## Streaming correctness

The streaming path is where most of the design work went:

```python
async def _stream_and_record():
    try:
        async for chunk in result:
            yield chunk
    finally:
        record_metrics(ctx)
        if (ctx.status_code or 0) >= 400:
            ERRORS_TOTAL.labels(technique=ctx.technique).inc()
```

The `finally` block guarantees metrics get recorded even on abort. The status code check inside it handles the network-error case. Every request payload to vLLM includes `stream_options: {"include_usage": true}` so the final chunk carries token counts.

Per-request timing happens inside the streaming loop:

- First content chunk → `ctx.ttft_s = now() - start`
- Each subsequent chunk → append to `inter_chunk_latency` list
- Final usage chunk → `ctx.prompt_tokens`, `ctx.completion_tokens`
- `finally` block → `ctx.e2e_latency_s = now() - start`

`tpot_avg_s` is computed as the mean of `inter_chunk_latency`. The full list also gets observed into a Prometheus histogram one chunk at a time, which lets you compute the true p99 inter-chunk delay across all requests — not just the average per request. A 50ms average TPOT can hide occasional 2-second generation stalls; the per-chunk histogram catches them.

---

## RequestContext

Created once at the endpoint, mutated by each layer as the request flows through. No globals, no thread-local state, no shared dictionaries.

```python
@dataclass
class RequestContext:
    request_id: str
    model: str
    stream: bool
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    chat_template_kwargs: dict | None = None
    technique: str = ""
    # Written by the client during streaming:
    ttft_s: float = 0.0
    tpot_avg_s: float = 0.0
    e2e_latency_s: float = 0.0
    inter_chunk_latency_s: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int = 0
```

Each layer's responsibility is documented in the dataclass: endpoint sets the request fields, client writes the timing and token fields, `record_metrics` reads everything and observes the Prometheus metrics in the finally block.

---

## Pipeline technique labeling

The header takes priority over the body field, so clients can override per-request without changing the payload:

```python
technique = request.headers.get("X-Pipeline-Stage", "") or body.technique
```

Every Prometheus metric carries `technique` as a label. The Grafana dashboards filter and facet by it — TTFT, TPOT, throughput, and error rate all break down per pipeline stage with no extra configuration.

---

## Dashboards

Four dashboards, auto-provisioned by Grafana on startup. All screenshots are from real benchmark runs against Qwen3.5-4B on an A10.

> Note: Below dashboard screenshots are from a later experiment iteration, when I was trying to throttle the vLLM server, firing at a rate of 32 QPS for 120 seconds. These reflect how the dashboards look when your system is under heavy load.

1. **vLLM Config & Health:** vLLM status up/down, KV cache fill (gauge with thresholds at 80% / 95%), prefix cache hit rate, preemption rate, finish reason breakdown.

![vLLM Config & Health](assets/vllmconfig-health.png)

2. **vLLM Engine Performance:** prefill, decode, and queue wait time (these three sum to engine-side TTFT), TTFT and TPOT histograms, generation and prompt token throughput, tokens per engine step as a batch size proxy.

![vLLM Engine Performance](assets/vllm-engine-performance.png)

3. **Gateway Operations:** request rate, error rate, gateway TTFT/TPOT, and the gap between gateway TTFT and engine TTFT (which is your pure proxy overhead). Stacked area chart showing request mix by technique.

![Gateway Operations](assets/gateway-operations.png)

4. **Pipeline Techniques:** every panel broken out by `technique` label, with a `$percentile` dropdown that switches every histogram between p50/p95/p99 simultaneously.

![Pipeline Techniques](assets/pipeline-technqiues.png)

Example PromQL from the technique dashboard:

```promql
histogram_quantile(
    $percentile,
    sum by (le, technique) (
        rate(nanoserve_time_to_first_token_seconds_bucket{technique!="unknown"}[5m])
    )
)
```

The `5m` rate window is intentional. With a 15-second scrape interval, `[1m]` only gives you 4 data points and the noise dominates. `[5m]` gets 20 data points — smooth enough for a stable estimate, fast enough to react to load changes within a couple minutes.

---

## Histogram bucket choices

The bucket boundaries aren't Prometheus defaults. They're chosen for LLM workload shapes:

| Metric | Buckets | Why |
|--------|---------|-----|
| TTFT | `0.01s → 60s` | Prefix cache hits land near 10ms, cold-start with thinking mode can hit 30s+ |
| TPOT | `1ms → 1s` | Healthy serving is 10–50ms inter-token |
| Inter-chunk delay | `1ms → 5s` | Per-chunk gaps; the 5s top bucket catches generation stalls |
| E2E duration | `0.05s → 60s` | Full request wall clock |
| Completion tok/s | `1 → 1000` | Per-request throughput distribution |

If your highest explicit bucket is 5s but your actual p99 is 8s, every observation past 5s ends up in `+Inf` and Grafana reports your p99 as `+Inf`. Bucket boundaries are how `histogram_quantile()` lies to you when you don't pay attention.

---

## Gateway metrics reference

Every metric carries `technique` = `classify` | `generate` | `gate` | `unknown`.

| Metric | Type | What it measures |
|--------|------|-----------------|
| `nanoserve_requests_total` | Counter | Total requests (success + error) |
| `nanoserve_errors_total` | Counter | Failed requests (upstream 4xx/5xx or network failure) |
| `nanoserve_request_duration_seconds` | Histogram | E2E wall-clock latency from gateway perspective |
| `nanoserve_time_to_first_token_seconds` | Histogram | TTFT for streaming; equals E2E for non-streaming |
| `nanoserve_time_per_output_token_seconds` | Histogram | Mean inter-token latency per request |
| `nanoserve_stream_inter_chunk_delay_seconds` | Histogram | Individual inter-chunk gaps (every SSE chunk = one observation) |
| `nanoserve_completion_tokens_per_second` | Histogram | Per-request throughput = `completion_tokens / e2e_latency` |
| `nanoserve_prompt_tokens_total` | Counter | Cumulative prompt tokens |
| `nanoserve_completion_tokens_total` | Counter | Cumulative completion tokens |
| `nanoserve_gateway_info_info` | Info | Static metadata: version, architecture |

---

## API reference

### `POST /v1/chat/completions`

OpenAI-compatible. Standard fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `technique` | `string` | Pipeline stage label. Overridden by `X-Pipeline-Stage` header if present. Stamped on all metrics. |

Headers:

| Header | Description |
|--------|-------------|
| `X-Pipeline-Stage` | Pipeline technique label (priority over body `technique`) |
| `X-Request-ID` | Custom request ID; auto-generated as `chatcmpl-<hex>` if absent |

Streaming and non-streaming both supported. Streaming uses SSE; the final chunk includes usage data via `stream_options: {"include_usage": true}` on the upstream call.

### `GET /health`

```json
{
  "gateway": "up",
  "backend_url": "http://localhost:8000",
  "backend_status": "up"
}
```

Probes `{backend_url}/health` first, falls back to `{backend_url}/v1/models`. Returns `"up"` if either probe returns < 500.

### `GET /metrics`

Prometheus exposition format. Python process and GC metrics are explicitly stripped on startup so the output stays focused on LLM serving.

### `GET /v1/models`

Returns the configured backend in OpenAI-compatible format.

---

## Configuration

### `.env`

How the gateway process runs. All keys use the `NANOSERVE_` prefix.

```
NANOSERVE_HOST=127.0.0.1
NANOSERVE_PORT=8765
NANOSERVE_BACKEND_CONFIG_PATH=config.yaml
```

### `config.yaml`

What the gateway connects to.

```yaml
backend_url: "http://localhost:8000"   # vLLM endpoint
timeout: 120                           # upstream request timeout
echo_timeout: 10                       # timeout for the in-process echo backend
```

---

## Echo backend

`src/nanoserve/backends/echo_backend.py` is an in-process test backend that simulates LLM timing without a GPU: 200ms TTFT delay, 50ms per-word streaming delay. All 10 metrics get populated correctly, so the full observability stack works without any hardware.

Use `model: "echo"` to validate your metrics setup end-to-end before pointing at a real vLLM instance.

---

## Tests

```bash
uv run pytest tests/ -v
```

10 tests across three modules:

| Module | What it tests |
|--------|---------------|
| `tests/e2e/test_chat_e2e.py` | Full app lifecycle via `TestClient`; echo model; request ID propagation |
| `tests/v1/test_chat.py` | Endpoint logic with `MockDispatcher`; error cases (wrong role, invalid role enum) |
| `tests/v1/test_dispatcher.py` | Dispatcher instantiation; technique passthrough |

No live vLLM or GPU needed.

---

## Design decisions

**Q: Why a single mutable `RequestContext` instead of function arguments?**

Streaming requests record timing in 4+ places: endpoint (start), client (TTFT, per-chunk, e2e), finally block. Threading 4-6 parameters through every callsite gets ugly fast. One context object, created once, with each layer writing to its own fields. The dataclass docstring documents who reads what and who writes what.

**Q: Why not middleware for metrics?**

Middleware sees request start and response end. It can't see inside a streaming generator. TTFT happens at the first yield. TPOT happens between chunks. The generator wrapper is the right hook; middleware is the wrong abstraction layer.

**Q: Why drop Python's built-in Prometheus collectors?**

`prometheus-client` registers GC, process, and platform collectors by default. At `/metrics`, 80% of the output would be Python runtime noise. Engineers looking at CPU and GC pressure should use their infra monitoring stack, not the LLM gateway endpoint.

**Why `5m` rate windows everywhere?**

Already covered above, but worth repeating: 15-second scrape interval × `[1m]` window = 4 data points = noise. `[5m]` gives you 20 data points, which is smooth enough for stable estimates and fast enough to react within a couple minutes. Using the same window across all panels also makes cross-panel comparisons valid.

---

## Repository layout

```
nano-serve/
├── main.py                         # FastAPI app entrypoint + exception handlers
├── config.yaml                     # vLLM backend URL + timeout (copy from config.yaml.example)
├── .env                            # Host, port, config path (copy from .env.example)
├── src/nanoserve/
│   ├── api/v1/
│   │   ├── chat.py                 # POST /v1/chat/completions
│   │   ├── health.py               # GET /health with backend probe
│   │   └── models.py               # GET /v1/models
│   ├── relay/
│   │   ├── dispatcher.py           # Routes ctx → vLLM or echo backend
│   │   └── client.py               # Async HTTP client (streaming + non-streaming)
│   ├── observability/
│   │   ├── metrics.py              # 10 Prometheus metrics, custom LLM buckets
│   │   └── recorder.py             # record_metrics(ctx) — called in finally
│   ├── schemas/
│   │   ├── chat.py                 # Request/response Pydantic models
│   │   └── request_context.py      # The mutable context flowing through the request lifecycle
│   ├── backends/
│   │   └── echo_backend.py         # In-process test backend; simulates TTFT + per-word delay
│   └── core/
│       ├── config.py               # Pydantic settings (from .env via NANOSERVE_ prefix)
│       ├── backend_config.py       # YAML → BackendConfig
│       ├── dependencies.py         # FastAPI DI: get_dispatcher
│       └── exceptions.py           # NanoServeException + InvalidRequestError
├── monitoring/
│   ├── docker-compose.yml          # Prometheus + Grafana; one command to start
│   ├── prometheus.yml              # Scrapes :8765/metrics and :8000/metrics
│   ├── grafana/provisioning/       # Auto-provisions datasource + dashboard folder
│   ├── grafana-dashboards/         # 4 JSON dashboards, auto-loaded by Grafana
│   └── MONITORING_GUIDE.md         # Every metric, every panel, PromQL explained
└── tests/
    ├── e2e/test_chat_e2e.py
    ├── v1/test_chat.py
    └── v1/test_dispatcher.py
```