"""Agent chain."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel

from language_models.agent.agent import Agent
from language_models.agent.chat import (
    ReasoningStep,
    ReasoningStepName,
    ReasoningStepTool,
)
from language_models.tools.tool import Tool


class ChainBlockStepName(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    FILTER = "filter"


class ChainBlockOutput(BaseModel):
    """Class that represents the output of a block."""

    inputs: dict[str, Any]
    output: (
        str
        | int
        | float
        | dict[str, Any]
        | BaseModel
        | list[str]
        | list[int]
        | list[float]
        | list[dict[str, Any]]
        | list[BaseModel]
        | None
    )
    steps: list[ReasoningStep]


class ChainToolBlock(BaseModel):
    """Class that implements a tool or function block."""

    function: Callable[[Any], Any]
    name: str
    inputs: type[BaseModel] | None = None

    def invoke(self, inputs: dict[str, Any]) -> ChainBlockOutput:
        inputs = {key: value for key, value in inputs.items() if key in self.inputs.model_fields}
        inputs = self.inputs.model_validate(inputs)
        output = self.function(**inputs)
        return ChainBlockOutput(
            inputs=inputs.model_dump(),
            output=output,
            steps=ReasoningStep(
                name=ReasoningStepName.TOOL,
                content=ReasoningStepTool(tool=self.name, tool_input=inputs.model_dump(), tool_response=output),
            ),
        )


class ChainAgentBlock(BaseModel):
    """Class that implements an agent block."""

    name: str
    agent: Agent

    def invoke(self, inputs: dict[str, Any]) -> ChainBlockOutput:
        output = self.agent.invoke(inputs)
        return ChainBlockOutput(inputs=inputs, output=output.final_answer, steps=output.chain_of_thought)


class ChainFilterBlock(BaseModel):
    name: str
    inputs: type[BaseModel]

    def invoke(self, inputs: dict[str, Any]) -> ChainBlockOutput:
        return ChainBlockOutput


class ChainBlockStep(BaseModel):
    name: ChainBlockStepName
    content: ChainBlockOutput

    class Config:
        use_enum_values = True


class ChainStateManager(BaseModel):
    """Class that implements a state manager."""

    state: dict[str, Any]
    steps: list[ChainBlockStep] = []

    def update(
        self,
        block: ChainAgentBlock | ChainToolBlock | ChainFilterBlock,
        output: Any,
    ) -> None:
        """Updates the state values."""
        self.state[block.name] = output.output
        if isinstance(block, ChainAgentBlock):
            block_step_name = ChainBlockStepName.AGENT
        elif isinstance(block, ChainToolBlock):
            block_step_name = ChainBlockStepName.TOOL
        else:
            block_step_name = ChainBlockStepName.FILTER
        self.steps.append(ChainBlockStep(name=block_step_name, content=output))


class ChainOutput(BaseModel):
    """Class that represents the chain output."""

    inputs: dict[str, Any]
    output: (
        str
        | int
        | float
        | dict[str, Any]
        | BaseModel
        | list[str]
        | list[int]
        | list[float]
        | list[dict[str, Any]]
        | list[BaseModel]
        | None
    )
    steps: list[ChainBlockStep]


class Chain(BaseModel):
    """Class that implements a chain."""

    name: str
    description: str
    inputs: type[BaseModel]
    output: str
    blocks: list[ChainAgentBlock | ChainToolBlock | ChainFilterBlock]

    def invoke(self, **inputs: dict[str, Any]) -> ChainOutput:
        inputs = self.inputs.model_validate(inputs)
        state_manager = ChainStateManager(state=inputs.model_dump())

        for block in self.blocks:
            logging.info("Running Block: %s", block.name)
            output = block.invoke(state_manager.state)
            state_manager.update(block, output)

        return ChainOutput(
            inputs=inputs.model_dump(),
            output=state_manager.state.get(self.output),
            steps=state_manager.steps,
        )

    def as_tool(self) -> Tool:
        return Tool(
            function=self.invoke,
            name=self.name,
            description=self.description,
            args_schema=self.inputs,
        )
