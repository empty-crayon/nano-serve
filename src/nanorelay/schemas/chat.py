from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict


# Request

class MessageRole(str, Enum):
    system = "system"
    developer = "developer"
    user = "user"
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
    model_config = ConfigDict(extra="allow")
    
    """
     Based on 'https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create'
    """
    id: str
    object: Union[Literal["chat.completion"], Literal["chat.completion.chunk"]] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
