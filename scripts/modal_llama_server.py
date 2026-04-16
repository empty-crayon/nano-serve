from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import fastapi
import modal

MODEL_REPO = os.environ.get("MODAL_LLAMA_MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
MODEL_FILENAME = os.environ.get("MODAL_LLAMA_MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
LLAMA_SERVER_PORT = 8080

app = modal.App("relay-llama-server")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar", "xz-utils", "libstdc++6", "libgcc-s1", "libcurl4", "ca-certificates")
    .pip_install("fastapi[standard]==0.115.0")
)

_server_proc: Optional[subprocess.Popen] = None
_server_ready = False


def _ensure_server_running() -> None:
    global _server_proc, _server_ready
    if _server_ready and _server_proc is not None and _server_proc.poll() is None:
        return
    import urllib.request
    import tarfile

    work = Path("/tmp/llama_work")
    work.mkdir(parents=True, exist_ok=True)
    bin_dir = work / "bin"
    model_path = work / MODEL_FILENAME

    server_bin = bin_dir / "llama-server"
    if not server_bin.exists():
        try:
            with urllib.request.urlopen(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", timeout=10
            ) as r:
                release = json.loads(r.read().decode())
                assets = release.get("assets", [])
        except Exception:
            release = None
            assets = []
        url = None
        for a in assets:
            name = (a.get("name") or "").lower()
            if "linux" in name and "x64" in name and ("tar" in name or "gz" in name):
                url = a.get("browser_download_url")
                break
        if not url:
            url = "https://github.com/ggml-org/llama.cpp/releases/download/b8121/llama-b8121-bin-ubuntu-x64.tar.gz"
        tar_path = work / "llama.tar.gz"
        if not tar_path.exists():
            with urllib.request.urlopen(url, timeout=120) as r:
                tar_path.write_bytes(r.read())
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(work)
        for p in work.rglob("llama-server"):
            if p.is_file():
                server_bin = p
                bin_dir = p.parent
                break
        else:
            bin_dir = work / "bin" if (work / "bin").exists() else work
            server_bin = bin_dir / "llama-server"

    if not model_path.exists():
        hf_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
        with urllib.request.urlopen(hf_url, timeout=300) as r:
            model_path.write_bytes(r.read())

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "-c", "512",
        "-ngl", "99",
    ]
    _server_proc = subprocess.Popen(
        cmd,
        cwd=str(bin_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    for _ in range(90):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health",
                method="GET",
            )
            urllib.request.urlopen(req, timeout=2)
            break
        except Exception:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{LLAMA_SERVER_PORT}/slots",
                    method="GET",
                )
                urllib.request.urlopen(req, timeout=2)
                break
            except Exception:
                time.sleep(0.5)
    else:
        time.sleep(5)
    _server_ready = True


# FastAPI app defined at MODULE LEVEL — this is the Modal pattern
web_app = fastapi.FastAPI()


@web_app.get("/")
async def root():
    return {"app": "relay-llama-server", "endpoints": ["/health", "/ping", "/completion", "/v1/chat/completions"]}


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.get("/ping")
async def ping():
    return {"pong": True, "service": "relay-llama-server"}


@web_app.put("/config")
async def config():
    return {"ok": True, "message": "Config acknowledged"}


@web_app.post("/completion")
async def completion(request: fastapi.Request) -> fastapi.Response:
    import urllib.request
    import urllib.error

    _ensure_server_running()
    body = await request.body()
    url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/completion"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            return fastapi.Response(content=r.read(), media_type="application/json")
    except urllib.error.HTTPError as e:
        return fastapi.Response(content=e.read(), status_code=e.code, media_type="application/json")
    except Exception as e:
        return fastapi.Response(
            content=json.dumps({"error": str(e), "content": ""}).encode("utf-8"),
            status_code=502,
            media_type="application/json",
        )


@web_app.post("/v1/chat/completions")
async def chat_completions(request: fastapi.Request) -> fastapi.Response:
    import urllib.request
    import urllib.error

    _ensure_server_running()
    body = await request.body()
    url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            return fastapi.Response(content=r.read(), media_type="application/json")
    except urllib.error.HTTPError as e:
        return fastapi.Response(content=e.read(), status_code=e.code, media_type="application/json")
    except Exception as e:
        return fastapi.Response(
            content=json.dumps({"error": str(e), "content": ""}).encode("utf-8"),
            status_code=502,
            media_type="application/json",
        )


@app.function(image=image, gpu="T4", timeout=600)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    return web_app