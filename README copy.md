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

## Configuration

- **NANORELAY_PORT** — Port the server listens on (default: 8080). Set in your environment or `.env` to override.
- **NANORELAY_BACKEND_URL** — Backend URL for proxying chat completion requests. When unset, the server responds with an echo of the last user message.
