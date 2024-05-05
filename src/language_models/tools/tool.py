"""LLM tool."""

import re
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


class Tool(BaseModel):
    """Class that implements an LLM tool."""

    func: Callable[[Any], Any]
    name: str
    description: str
    args_schema: Type[BaseModel] | None = None

    @property
    def args(self) -> dict | None:
        if self.args_schema is None:
            return
        return self.args_schema.model_json_schema()["properties"]

    def __str__(self) -> str:
        args = self.args
        return (
            f"tool name: {self.name}, "
            f"tool description: {self.description}, "
            f"tool input: {re.sub('}', '}}', re.sub('{', '{{', str(args)))}"
        )

    def _parse_input(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Converts tool input to pydantic model."""
        input_args = self.args_schema
        if input_args is not None:
            result = input_args.model_validate(tool_input)
            return {
                key: getattr(result, key)
                for key, _ in result.model_dump().items()
                if key in tool_input
            }
        return tool_input

    def invoke(self, tool_input: dict[str, Any]) -> Any:
        try:
            parsed_input = self._parse_input(tool_input)
            observation = self.func(**parsed_input) if parsed_input else self.func()
        except ValidationError as e:
            observation = f"{self.name} tool input validation error: {e}"
        return observation
