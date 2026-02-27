from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nanorelay.api.v1 import router as v1_router
from nanorelay.core.config import settings
from nanorelay.core.exceptions import NanoRelayException


app = FastAPI()
app.include_router(v1_router)


@app.exception_handler(NanoRelayException)
async def nanorelay_exception_handler(request: Request, exc: NanoRelayException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": type(exc).__name__
            }
        }
    )


if __name__ == "__main__":    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)
