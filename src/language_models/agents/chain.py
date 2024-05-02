"""Agent chain."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from language_models.agents.react import ReActAgent


class AgentChain(BaseModel):
    agents: list[ReActAgent]

    def invoke(self, prompt: dict[str, Any]):
        for agent in self.agents:
            pass
