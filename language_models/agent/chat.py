"""ReAct agent chat."""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from language_models.models.llm import ChatMessage


class ReasoningStepName(str, Enum):
    PROMPT = "prompt"
    THOUGHT = "thought"
    TOOL = "tool"
    FINAL_ANSWER = "final_answer"


class ReasoningStepTool(BaseModel):
    tool: str
    tool_input: dict[str, Any]
    tool_output: Any


class ReasoningStep(BaseModel):
    name: ReasoningStepName
    content: str | ReasoningStepTool

    class Config:
        use_enum_values = True


class Chat(BaseModel):
    """Class that implements the chat history."""

    messages: list[ChatMessage]
    steps: list[str] = []
    chain_of_thought: list[ReasoningStep] = []

    def update(self, prompt: str) -> None:
        """Modifies the user prompt to include intermediate steps."""
        self.messages[-1].content = "\n\n".join([prompt, f"This was your previous work:\n{'\n\n'.join(self.steps)}"])

    def reset(self) -> None:
        """Resets the chat."""
        self.messages = [self.messages[0]]
        self.steps = []
        self.chain_of_thought = []
