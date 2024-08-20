"""Agent chat."""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from language_models.models.llm import ChatMessage


class StepName(str, Enum):
    """Class that represents a step name."""

    SYSTEM_PROMPT = "system_prompt"
    PROMPTING_STRATEGY = "prompting_strategy"
    PROMPT = "prompt"
    RAW_OUTPUT = "raw_output"
    OBSERVATION = "observation"
    TOOL_USE = "tool_use"
    TOOL_OUTPUT = "tool_output"
    FINAL_ANSWER = "final_answer"
    INPUTS = "inputs"
    OUTPUT = "output"


class StepToolUse(BaseModel):
    thought: str
    used: str
    arguments: dict[str, Any]


class StepFinalAnswer(BaseModel):
    thought: str | None = None
    output: (
        str
        | int
        | float
        | bool
        | date
        | datetime
        | dict[str, Any]
        | BaseModel
        | list[str]
        | list[int]
        | list[float]
        | list[dict[str, Any]]
        | list[BaseModel]
    )


class Step(BaseModel):
    """Class that represents a step of the LLM."""

    name: StepName
    content: Any

    class Config:
        use_enum_values = True


class Chat(BaseModel):
    """Class that implements the chat history."""

    messages: list[ChatMessage]
    previous_steps: list[str] = []
    steps: list[Step]

    def update(self, prompt: str) -> None:
        """Modifies the user prompt to include intermediate steps."""
        self.messages[-1].content = "\n\n".join(
            [prompt, f"These were your previous steps:\n{'\n\n'.join(self.previous_steps)}"]
        )

    def reset(self) -> None:
        """Resets the chat."""
        self.messages = [self.messages[0]]
        self.previous_steps = []
        self.steps = [self.steps[0], self.steps[1]]
