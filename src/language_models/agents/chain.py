"""Agent chain."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentChain(BaseModel):
    def invoke(self, prompt: dict[str, Any]):
        pass

    @classmethod
    def create(cls) -> AgentChain:
        pass
