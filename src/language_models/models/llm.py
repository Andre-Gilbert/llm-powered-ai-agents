"""OpenAI LLMs."""

import time
from enum import Enum

from pydantic import BaseModel

from language_models.proxy_client import BTPProxyClient
from language_models.settings import settings


class ChatMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str

    class Config:
        use_enum_values = True


class OpenAILanguageModel(BaseModel):
    """Class that implements the OpenAI models."""

    proxy_client: BTPProxyClient
    model: str
    max_tokens: int = 256
    temperature: float = 0.0

    def get_completion(self, messages: list[ChatMessage]) -> str:
        """Creates a model response for the given chat conversation.

        Args:
            messages: A list of messages comprising the conversation so far.

        Returns:
            A chat completion object.
        """
        response = self.proxy_client.request(
            api_endpoint="completions",
            data={
                "deployment_id": self.model,
                "messages": [message.model_dump() for message in messages],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return response["choices"][0]["message"]["content"]
