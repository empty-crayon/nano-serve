from nanorelay.core.config import settings
import httpx
from typing import Dict, Any

class Runner:
    def run(self, prompt: str) -> str:
        return f"Echo: {prompt}"

class LlamaServerClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def chat(self, prompt: str) -> str:
        if self.endpoint is None:
            return None
        
        payload = {
            "model": "relay-gguf",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        } 

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=60.0,                   
            )
            response.raise_for_status()
            parsed = response.json()
            choices = parsed.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            return str(message.get("content", "")).strip()
    
    async def stream_chat(self, prompt: str) -> str:
        pass