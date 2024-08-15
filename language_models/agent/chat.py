"""Agent chat."""

from pydantic import BaseModel

from language_models.models.llm import ChatMessage


class Chat(BaseModel):
    """Class that implements the chat history."""

    messages: list[ChatMessage]
    previous_steps: list[str] = []

    def update(self, prompt: str) -> None:
        """Modifies the user prompt to include intermediate steps."""
        self.messages[-1].content = "\n\n".join(
            [prompt, f"These were your previous steps:\n{'\n\n'.join(self.previous_steps)}"]
        )

    def reset(self) -> None:
        """Resets the chat."""
        self.messages = [self.messages[0]]
        self.previous_steps = []
