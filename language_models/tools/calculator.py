"""Calculator tool."""

import numexpr as ne
from pydantic import BaseModel, Field

from language_models.tools.tool import Tool


class Calculator(BaseModel):
    expression: str = Field(description="A math expression")


calculator = Tool(
    function=lambda expression: ne.evaluate(expression).item(),
    name="Calculator",
    description="Use this tool when you want to do calculations",
    args_schema=Calculator,
)
