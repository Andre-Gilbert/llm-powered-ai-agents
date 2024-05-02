"""ReAct agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from language_models.models.llm import OpenAILanguageModel
from language_models.tools.tool import Tool


class ReActAgent(BaseModel):
    """Class that implements a ReAct agent."""

    llm: OpenAILanguageModel
    tools: dict[str, Tool] | None
    tool_names: list[str] | None
    task_prompt: str
    output_format: str
    chat_messages: list[dict[str, str]]
    iterations: int = 20

    def invoke(self, prompt: dict[str, Any]):
        pass

    @classmethod
    def create(cls) -> ReActAgent:
        pass
