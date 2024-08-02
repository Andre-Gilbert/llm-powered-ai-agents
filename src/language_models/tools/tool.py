"""LLM tool."""

from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class Tool(BaseModel):
    """Class that implements an LLM tool."""

    function: Callable[[Any], Any]
    name: str
    description: str
    args_schema: type[BaseModel] | None = None

    @property
    def args(self) -> dict[str, Any] | None:
        if self.args_schema is None:
            return
        return self.args_schema.model_json_schema()["properties"]

    def __str__(self) -> str:
        return f"- Tool Name: {self.name}, " f"Tool Description: {self.description}, " f"Tool Input: {self.args}"

    def _parse_input(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Converts tool input to pydantic model."""
        input_args = self.args_schema
        if input_args is not None:
            result = input_args.model_validate(tool_input)
            return {key: getattr(result, key) for key, _ in result.model_dump().items() if key in tool_input}
        return tool_input

    def invoke(self, tool_input: dict[str, Any]) -> Any:
        """Invokes a tool given arguments provided by an LLM."""
        try:
            parsed_input = self._parse_input(tool_input)
            observation = self.function(**parsed_input) if parsed_input else self.function()
        except ValidationError:
            observation = (
                f"Could not run tool {self.name} with input: {tool_input}\n\n"
                + "Your goal is to correct your response\n\n"
                + "Your <input of the tool to use> must be a JSON format with the "
                + f"keyword arguments of: {self.args}"
            )
        return observation
