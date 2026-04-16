from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager

from nanoserve.api.v1 import router as v1_router
from nanoserve.api.v1.health import health_check, router as health_router
from nanoserve.core.config import settings
from nanoserve.core.exceptions import NanoServeException
from nanoserve.observability.metrics import generate_latest, CONTENT_TYPE_LATEST, drop_builtin_collectors
from nanoserve.relay.dispatcher import Dispatcher
from nanoserve.core.backend_config import load_backend_config

# Clean up default Python collectors - keep /metrics focused on LLM gateway metrics
drop_builtin_collectors()

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend_config = load_backend_config(settings.backend_config_path)
    app.state.dispatcher = Dispatcher(config = backend_config)
    yield
    await app.state.dispatcher.close()

app = FastAPI(title="NanoServe Gateway", lifespan=lifespan)
app.include_router(v1_router)
app.include_router(health_router, prefix="")  # /health at root, not under /v1


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint - returns all metrics in exposition format."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(NanoServeException)
async def nanoserve_exception_handler(request: Request, exc: NanoServeException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": type(exc).__name__,
            }
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": type(exc).__name__,
            }
        }
    )


if __name__ == "__main__":
    import logging
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    server = uvicorn.Server(
        uvicorn.Config(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=False,
            log_level="info",
        )
    )
    server.run()