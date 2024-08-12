"""LLM tool."""

from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, ValidationError


class Tool(BaseModel):
    """Class that implements a tool.

    Attributes:
        function: The function that will be invoked when calling this tool.
        name: The name of the tool.
        description: The description of when to use the tool or what the tool does.
        args_schema: The Pydantic model that represents the input arguments.
        requires_approval: Whether the human needs to approve the tool use.
            Defaults to False.
    """

    function: Callable[[Any], Any]
    name: str
    description: str
    args_schema: type[BaseModel] | None = None
    requires_approval: bool = False

    @property
    def args(self) -> dict[str, Any] | None:
        """Gets the tool model JSON schema."""
        if self.args_schema is None:
            return
        return self.args_schema.model_json_schema()["properties"]

    def parse_input(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Converts tool input to pydantic model."""
        input_args = self.args_schema
        if input_args is not None:
            result = input_args.model_validate(tool_input)
            return {key: getattr(result, key) for key, _ in result.model_dump().items() if key in tool_input}
        return tool_input

    def invoke(self, tool_input: dict[str, Any], verbose: bool) -> Any:
        """Invokes a tool given arguments provided by an LLM."""
        if self.requires_approval:
            decision = input(
                "\n\n".join(
                    [
                        "Do you allow the invocation of the tool (Y/y/Yes/yes)?",
                        f"Tool: {self.name}",
                        f"Tool Input: {None if self.args else tool_input}",
                    ]
                )
            )
            if decision not in ("Y", "y", "Yes", "yes"):
                if verbose:
                    logger.opt(colors=True).info("<b><fg #EC9A3C>Tool Use Approved</fg #EC9A3C></b>: No")

                return "\n\n".join(
                    [
                        f"The user did not approve the use of the tool: {self.name}",
                        "Provide the final answer to the user's query",
                    ]
                )

            if verbose:
                logger.opt(colors=True).info("<b><fg #EC9A3C>Tool Use Approved</fg #EC9A3C></b>: Yes")

        if self.args is None:
            output = self.function()
        else:
            try:
                parsed_input = self.parse_input(tool_input)
                output = self.function(**parsed_input)
            except ValidationError as error:
                output = "\n\n".join(
                    [
                        f"Could not run tool {self.name} with input:\n{tool_input}",
                        f"The error was:\n{error}",
                        "You need to correct your response",
                        f"Your <input of the tool to use> must be a JSON format with the keyword arguments of:\n{self.args}",
                    ]
                )
        return output

    def __str__(self) -> str:
        return f"- Tool Name: {self.name}, Tool Description: {self.description}, Tool Input: {self.args}"
