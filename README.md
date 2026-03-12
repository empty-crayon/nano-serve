# nanoRelay

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Run locally

Start the local server:

```bash
uv run python main.py
```

## Request Flow

```
Client
  │
  │  POST /v1/chat/completions
  ▼
chat_completions()          # api/v1/chat.py
  │  validates request
  │  assigns request_id
  ▼
Dispatcher.dispatch()       # relay/dispatcher.py
  │  resolves backend endpoint from MODELS_MAP
  │  falls back to echo if no backend is configured
  ▼
OpenAICompatClient.chat()   # relay/client.py
  │  forwards request to backend's /v1/chat/completions
  ▼
Backend (llama.cpp / vllm / ollama ...)
```

All supported backends expose an OpenAI-compatible `/v1/chat/completions` endpoint,
so a single HTTP client handles all of them without backend-specific logic.

## Configuration

- **NANORELAY_PORT** — Port the server listens on (default: 8080). Set in your environment or `.env` to override.
- **NANORELAY_BACKEND_URL** — Backend URL for proxying chat completion requests. When unset, the server responds with an echo of the last user message.

## Testing

### Unit & Integration tests

No running backend required. The dispatcher is replaced with a mock.

```bash
uv run pytest tests/v1/
```

### E2E tests

Requires a running backend (e.g. llama.cpp server) with `NANORELAY_BACKEND_URL` set in `.env`.

```bash
uv run pytest tests/e2e/
```
