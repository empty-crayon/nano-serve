from enum import Enum
from typing import Literal

from pydantic import BaseModel


# Request

class MessageRole(str, Enum):
    user = "user"
    system = "system"
    developer = "developer"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


# Response
class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop",]
    message: ChatCompletionMessage


class ChatCompletionResponse(BaseModel):
    """
     Based on 'https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create'
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
