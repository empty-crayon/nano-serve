from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from nanorelay.schemas.chat import ChatMessage

class Backend(ABC):
    @abstractmethod
    async def generate(
        self,
        request_id: str,
        messages: list[ChatMessage],
        model: str,
        stream: bool,
    ) -> dict | AsyncGenerator[str, None]:
        pass