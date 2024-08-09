"""Agent chain."""

from __future__ import annotations

import sys
from functools import reduce
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel

from language_models.agent.agent import Agent
from language_models.tools.tool import Tool

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


class WorkflowProcess(BaseModel):
    pass


class WorkflowStepOutput(BaseModel):
    """Class that represents the output of a step."""

    inputs: (
        str | int | float | dict | BaseModel | list[str] | list[int] | list[float] | list[dict] | list[BaseModel] | None
    )
    output: (
        str | int | float | dict | BaseModel | list[str] | list[int] | list[float] | list[dict] | list[BaseModel] | None
    )


class WorkflowFunctionStep(BaseModel):
    """Class that implements a function step.

    Attributes:
        name: The name of the step.
        inputs: The Pydantic model that represents the input arguments.
        function: The function that will be invoked when calling this step.
    """

    name: str
    inputs: type[BaseModel]
    function: Callable[[Any], Any]

    def invoke(self, inputs: dict[str, Any], verbose: bool) -> WorkflowStepOutput:
        inputs = {key: value for key, value in inputs.items() if key in self.inputs.model_fields}
        inputs = self.inputs.model_validate(inputs).model_dump()
        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Function Input</fg #EC9A3C></b>: {inputs}")

        output = self.function(**inputs)
        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Function Output</fg #EC9A3C></b>: {output}")

        return WorkflowStepOutput(inputs=inputs, output=output)


class WorkflowAgentStep(BaseModel):
    """Class that implements an agent step.

    Attributes:
        name: The name of the step.
        agent: The agent that will be invoked when calling this step.
    """

    name: str
    agent: Agent

    def invoke(self, inputs: dict[str, Any], verbose: bool) -> WorkflowStepOutput:
        inputs = {variable: inputs.get(variable) for variable in self.agent.prompt_variables}
        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Agent Input</fg #EC9A3C></b>: {inputs}")

        output = self.agent.invoke(inputs)
        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Agent Output</fg #EC9A3C></b>: {output.final_answer}")

        return WorkflowStepOutput(inputs=inputs, output=output.final_answer)


class WorkflowTransformationStep(BaseModel):
    """Class that implements a transformation step.

    Attributes:
        name: The name of the step.
        input_field: The name of the field values to transform.
        transformation: The transformation to apply (can be map, filter, reduce).
        function: The function used for the transformation.
    """

    name: str
    input_field: str
    transformation: Literal["map", "filter", "reduce"]
    function: Callable[[Any], Any]

    def invoke(self, inputs: dict[str, Any], verbose: bool) -> WorkflowStepOutput:
        values = inputs[self.input_field]
        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Transformation Input</fg #EC9A3C></b>: {values}")

        if self.transformation == "map":
            transformed_values = map(self.function, values)
            output = list(transformed_values) if isinstance(values, list) else dict(transformed_values)
        elif self.transformation == "filter":
            transformed_values = filter(self.function, values)
            output = list(transformed_values) if isinstance(values, list) else dict(transformed_values)
        else:
            output = reduce(self.function, values)

        if verbose:
            logger.opt(colors=True).info(f"<b><fg #EC9A3C>Transformation Output</fg #EC9A3C></b>: {output}")

        return WorkflowStepOutput(inputs=values, output=output)


class WorkflowStateManager(BaseModel):
    """Class that implements a state manager."""

    state: dict[str, Any]

    def update(self, name: str, step: WorkflowStepOutput) -> None:
        """Updates the state values."""
        self.state[name] = step.output


class WorkflowOutput(BaseModel):
    """Class that represents the workflow output."""

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


class Workflow(BaseModel):
    """Class that implements a workflow.

    Attributes:
        name: The name of the workflow.
        description: The description of what the workflow does.
        steps: The steps of the workflow.
        inputs: The workflow inputs.
        output: The name of the step value to output.
    """

    name: str
    description: str
    steps: list[WorkflowAgentStep | WorkflowFunctionStep | WorkflowTransformationStep]
    inputs: type[BaseModel]
    output: str
    verbose: bool

    def invoke(self, inputs: dict[str, Any]) -> WorkflowOutput:
        """Runs the workflow."""
        inputs = self.inputs.model_validate(inputs).model_dump()
        state_manager = WorkflowStateManager(state=inputs)
        for step in self.steps:
            if self.verbose:
                logger.opt(colors=True).info(f"<b><fg #2D72D2>Running Step</fg #2D72D2></b>: {step.name}")

            output = step.invoke(state_manager.state, self.verbose)
            state_manager.update(step.name, output)

        output = state_manager.state.get(self.output)
        if self.verbose:
            logger.opt(colors=True).success(f"<b><fg #32A467>Workflow Output</fg #32A467></b>: {output}")

        return WorkflowOutput(inputs=inputs, output=output)

    def as_tool(self) -> Tool:
        """Converts the workflow into an LLM tool."""
        return Tool(
            function=lambda **inputs: self.invoke(inputs).output,
            name=self.name,
            description=self.description,
            args_schema=self.inputs,
        )
