import json
import time
from typing import AsyncGenerator
import httpx
from pydantic import BaseModel
from nanorelay.backends import Backend, EchoBackend, LocalBackend, ModalBackend
from nanorelay.core.backend_config import BackendConfig
from nanorelay.core.config import settings
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.schemas.chat import ChatMessage


class Dispatcher:
    BACKEND_TYPES: dict[str, type[Backend]] = {
        "echo" : EchoBackend,
        "local": LocalBackend,
        "modal": ModalBackend,
    }

    def __init__(self, config: BackendConfig):
        self._http_client = httpx.AsyncClient()
        self._backends = self._build_backends(config, self._http_client)
        self._default = config.default

        if self._default not in self._backends:
            raise ValueError(f"Default backend '{self._default}' is not defined in {list(self._backends.keys())}")

    async def close(self):
        await self._http_client.aclose()

    def _build_backends(self, config, http_client): 
        backends = {}
        backends["echo"] = EchoBackend()  # always include echo backend for testing
        for entry in config.backends:
            cls = self.BACKEND_TYPES.get(entry.type)
            if cls is None:
                raise ValueError(f"Unsupported backend type '{entry.type}'")
            backends[entry.type] = cls(url = entry.url, timeout = entry.timeout, http_client = http_client)
        print(backends)
        return backends
    
    def _resolve_backend(self, model: str) -> tuple [str, Backend]:
        if model and model in self._backends:
            return model, self._backends[model]
        
        return self._default, self._backends[self._default]

    async def dispatch(
        self,
        request_id: str,
        messages: list[ChatMessage],
        model: str,
        stream: bool,
    ) -> tuple[str, dict | AsyncGenerator[str, None]]:
        backend_name, backend = self._resolve_backend(model)
        print(f"[{request_id}] Dispatching to backend '{backend_name}' with model '{model}'")
        result = await backend.generate(request_id, messages, model, stream)
        return backend_name, result