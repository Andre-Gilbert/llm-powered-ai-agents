"""Agent chain."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from language_models.agents.react import ReActAgent


class AgentChain(BaseModel):
    """Class that implements LLM chaining."""

    chain_variables: dict[str, Any]
    chain: list[ReActAgent | Callable[[Any], Any]]

    def invoke(self, prompt: dict[str, Any]):
        for block in self.chain:
            pass

    @classmethod
    def create(cls, chain: list[ReActAgent | Callable[[Any], Any]]) -> AgentChain:
        chain_variables = []
        for block in chain:
            pass
