"""Agent chain."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

from language_models.agent.react import ReActAgent
from language_models.tools.tool import Tool


class ChainExecutionStep(str, Enum):
    AGENT = "agent"
    TOOL = "tool"


class ChainExecutionTool(BaseModel):
    name: str
    args: dict | None
    response: Any


class ChainExecution(BaseModel):
    step: ChainExecutionStep
    content: list[dict[str, Any]] | ChainExecutionTool

    class Config:
        use_enum_values = True


class AgentChainResponse(BaseModel):
    prompt: dict[str, Any]
    final_answer: dict[str, Any] | None
    execution_steps: list


class ChainGlobalState(BaseModel):
    state: dict[str, Any]


class ChainStep(BaseModel):
    output_field: str


class Chain(BaseModel):
    steps: list[ChainStep]
    state: ChainGlobalState

    def invoke(self):
        pass

    def as_tool(self) -> Tool:
        pass


class AgentChain(BaseModel):
    """Class that implements LLM chaining."""

    chain: list[ReActAgent | Tool]
    chain_variables: list[str]

    def invoke(self, prompt: dict[str, Any]) -> AgentChainResponse:
        execution_steps: list[ChainExecution] = []
        for i, block in enumerate(self.chain):
            if isinstance(block, ReActAgent):
                response = block.invoke(prompt)
                prompt = {**prompt, **response.final_answer}
                execution_steps.append(
                    ChainExecution(
                        step=ChainExecutionStep.AGENT,
                        content=response.chain_of_thought,
                    )
                )
                if i == len(self.chain) - 1:
                    return AgentChainResponse(
                        prompt=prompt,
                        final_answer=response.final_answer,
                        execution_steps=execution_steps,
                    )
            else:
                logging.info("Running code block:\n%s", block.name)
                response = block.invoke(prompt)
                prompt[block.name] = response
                execution_steps.append(
                    ChainExecution(
                        step=ChainExecutionStep.TOOL,
                        content=ChainExecutionTool(
                            name=block.name,
                            args=(
                                None
                                if block.args is None
                                else {key: prompt.get(key) for key in prompt if key in block.args}
                            ),
                            response=response,
                        ),
                    )
                )
                if i == len(self.chain) - 1:
                    logging.info("Code block output:\n%s", response)
                    return AgentChainResponse(
                        prompt=prompt,
                        final_answer={block.name: response},
                        execution_steps=execution_steps,
                    )
        return AgentChainResponse(
            prompt=prompt,
            final_answer=None,
            execution_steps=execution_steps,
        )
