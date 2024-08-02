"""ReAct agent output parser."""

import re
from datetime import datetime
from enum import Enum
from typing import Any

import dirtyjson as json
from pydantic import BaseModel, ValidationError

from language_models.agent.prompt import (
    OUTPUT_TYPE_ARRAY_FLOAT,
    OUTPUT_TYPE_ARRAY_INTEGER,
    OUTPUT_TYPE_ARRAY_OBJECT_OR_STRUCT,
    OUTPUT_TYPE_ARRAY_STRING,
    OUTPUT_TYPE_BINARY,
    OUTPUT_TYPE_BOOLEAN,
    OUTPUT_TYPE_DATE,
    OUTPUT_TYPE_FLOAT,
    OUTPUT_TYPE_INTEGER,
    OUTPUT_TYPE_OBJECT_OR_STRUCT,
    OUTPUT_TYPE_STRING,
    OUTPUT_TYPE_TIMESTAMP,
)

INSTRUCTIONS_WITH_TOOLS = """You should respond with:
```
Thought: <thought process on how to respond to the prompt>

Tool: <name of the tool to use>

Tool Input: <input of the tool to use>
```

Your <input of the tool to use> must be a JSON format representing the keyword arguments of <name of the tool to use>

When you know the final answer to the user's query you should respond with:
```
Thought: <thought process on how to respond to the prompt>

Final Answer: <response to the prompt>
```"""

INSTRUCTIONS_WITHOUT_TOOLS = """You should respond with:
```
Thought: <thought process on how to respond to the prompt>

Final Answer: <response to the prompt>
```"""


class OutputType(str, Enum):
    """Class that represents the LLM output types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    OBJECT = "object"
    STRUCT = "struct"
    BINARY = "binary"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    ARRAY_STRING = "array of strings"
    ARRAY_INTEGER = "array of integers"
    ARRAY_FLOAT = "array of floats"
    ARRAY_OBJECT = "array of objects"
    ARRAY_STRUCT = "array of structs"


FINAL_ANSWER_INSTRUCTIONS = {
    OutputType.STRING: OUTPUT_TYPE_STRING,
    OutputType.INTEGER: OUTPUT_TYPE_INTEGER,
    OutputType.FLOAT: OUTPUT_TYPE_FLOAT,
    OutputType.OBJECT: OUTPUT_TYPE_OBJECT_OR_STRUCT,
    OutputType.STRUCT: OUTPUT_TYPE_OBJECT_OR_STRUCT,
    OutputType.BINARY: OUTPUT_TYPE_BINARY,
    OutputType.BOOLEAN: OUTPUT_TYPE_BOOLEAN,
    OutputType.DATE: OUTPUT_TYPE_DATE,
    OutputType.TIMESTAMP: OUTPUT_TYPE_TIMESTAMP,
    OutputType.ARRAY_STRING: OUTPUT_TYPE_ARRAY_STRING,
    OutputType.ARRAY_INTEGER: OUTPUT_TYPE_ARRAY_INTEGER,
    OutputType.ARRAY_FLOAT: OUTPUT_TYPE_ARRAY_FLOAT,
    OutputType.ARRAY_OBJECT: OUTPUT_TYPE_ARRAY_OBJECT_OR_STRUCT,
    OutputType.ARRAY_STRUCT: OUTPUT_TYPE_ARRAY_OBJECT_OR_STRUCT,
}


class LLMToolUse(BaseModel):
    thought: str
    tool: str
    tool_input: dict


class LLMFinalAnswer(BaseModel):
    thought: str
    final_answer: (
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
    )


class AgentOutputParser(BaseModel):
    """Class that parses the LLM output."""

    output_type: OutputType
    output_schema: type[BaseModel] | str | None = None
    tool_use: bool

    def _extract_tool_use(self, output: str) -> tuple[str, str, str]:
        pattern = r"\s*Thought: (.*?)\n+Tool: ([a-zA-Z0-9_ ]+).*?\n+Tool Input: .*?(\{.*\})"

        match = re.search(pattern, output, re.DOTALL)
        if not match:
            raise ValueError(
                f"You made a mistake in your response: {output}\n\n"
                + f"Your goal is to correct your response\n\n{INSTRUCTIONS_WITH_TOOLS}"
            )

        thought = match.group(1).strip()
        tool = match.group(2).strip()
        tool_input = match.group(3).strip()
        return thought, tool, tool_input

    def _extract_json_str(self, tool_input: str) -> str:
        match = re.search(r"\{.*\}", tool_input.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(
                f"You made a mistake in your JSON format: {tool_input}\n\n"
                + f"Your goal is to correct your response\n\n{INSTRUCTIONS_WITH_TOOLS}"
            )

        return match.group()

    def _tool_input_parser(self, json_str: str) -> dict[str, Any]:
        processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
        pattern = r'"(\w+)":\s*"([^"]*)"'
        matches = re.findall(pattern, processed_string)
        return dict(matches)

    def _parse_tool(self, output: str) -> tuple[str, str, dict[str, Any]]:
        thought, tool, tool_input = self._extract_tool_use(output)
        json_str = self._extract_json_str(tool_input)
        try:
            tool_input_dict = json.loads(json_str)
        except ValueError:
            tool_input_dict = self._tool_input_parser(json_str)
        return thought, tool, tool_input_dict

    def _validate_final_answer(
        self, final_answer: str
    ) -> (
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
    ):
        if self.output_type == OutputType.STRING:
            return final_answer

        if self.output_type == OutputType.INTEGER:
            try:
                return int(final_answer)
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_INTEGER}"
                ) from error

        if self.output_type == OutputType.FLOAT:
            try:
                return float(final_answer)
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_FLOAT}"
                ) from error

        if self.output_type == OutputType.BINARY:
            if bool(re.fullmatch(r"[01]+", final_answer)):
                return final_answer

            raise ValueError(
                f"You made a mistake in your final answer: {final_answer}\n\n"
                + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_BINARY}"
            )

        if self.output_type == OutputType.BOOLEAN:
            if isinstance(final_answer, bool):
                return final_answer

            raise ValueError(
                f"You made a mistake in your final answer: {final_answer}\n\n"
                + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_BOOLEAN}"
            )

        if self.output_type == OutputType.DATE:
            try:
                return datetime.strptime(final_answer, self.output_schema)
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + "Your goal is to correct your final answer\n\n"
                    + f"{OUTPUT_TYPE_DATE.format(output_schema=self.output_schema)}"
                ) from error

        if self.output_type == OutputType.TIMESTAMP:
            try:
                return datetime.strptime(final_answer, self.output_schema)
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + "Your goal is to correct your final answer\n\n"
                    + f"{OUTPUT_TYPE_TIMESTAMP.format(output_schema=self.output_schema)}"
                ) from error

        if self.output_type in (OutputType.OBJECT, OutputType.STRUCT):
            try:
                json_str = self._extract_json_str(final_answer)

                try:
                    final_answer_dict = json.loads(json_str)
                except ValueError:
                    final_answer_dict = self._tool_input_parser(json_str)

                final_answer_model = self.output_schema.model_validate(final_answer_dict)
                return final_answer_model if self.output_type == OutputType.OBJECT else final_answer_model.model_dump()

            except ValueError as error:
                args = self.output_schema.model_json_schema()["properties"]
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + "Your goal is to correct your final answer\n\n"
                    + f"{OUTPUT_TYPE_OBJECT_OR_STRUCT.format(output_schema=args)}"
                ) from error

        if self.output_type == OutputType.ARRAY_STRING:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)

                if all(isinstance(entry, str) for entry in final_answer):
                    return final_answer

                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_STRING}"
                )

            except TypeError as error:
                raise TypeError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_STRING}"
                ) from error

        if self.output_type == OutputType.ARRAY_INTEGER:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)

                if all(isinstance(entry, int) for entry in final_answer):
                    return final_answer

                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_INTEGER}"
                )

            except TypeError as error:
                raise TypeError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_INTEGER}"
                ) from error

        if self.output_type == OutputType.ARRAY_FLOAT:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)

                if all(isinstance(entry, float) for entry in final_answer):
                    return final_answer

                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_FLOAT}"
                )

            except TypeError as error:
                raise TypeError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + f"Your goal is to correct your final answer\n\n{OUTPUT_TYPE_ARRAY_FLOAT}"
                ) from error

        if self.output_type in (OutputType.ARRAY_OBJECT, OutputType.ARRAY_STRUCT):
            try:
                final_answer_list = json.loads(final_answer)
                final_answer_list = list(final_answer_list)
                final_answer_list_dict = []

                for entry in final_answer_list:
                    json_str = self._extract_json_str(entry)
                    try:
                        final_answer_dict = json.loads(json_str)
                    except ValueError:
                        final_answer_dict = self._tool_input_parser(json_str)
                    final_answer_list_dict.append(final_answer_dict)

                if self.output_type == OutputType.ARRAY_OBJECT:
                    return [self.output_schema.model_validate(entry) for entry in final_answer_list_dict]
                else:
                    return [self.output_schema.model_validate(entry).model_dump() for entry in final_answer_list_dict]

            except ValidationError as error:
                args = self.output_schema.model_json_schema()["properties"]
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\n"
                    + "Your goal is to correct your final answer\n\n"
                    + f"{OUTPUT_TYPE_ARRAY_OBJECT_OR_STRUCT.format(output_schema=args)}"
                ) from error

    def _parse_final_answer(self, output: str) -> tuple[
        str,
        str
        | int
        | float
        | dict[str, Any]
        | BaseModel
        | list[str]
        | list[int]
        | list[float]
        | list[dict[str, Any]]
        | list[BaseModel],
    ]:
        pattern = r"\s*Thought: (.*?)\n+Final Answer:([\s\S]*.*?)(?:$)"

        match = re.search(pattern, output, re.DOTALL)
        if not match:
            instructions = INSTRUCTIONS_WITH_TOOLS if self.tool_use else INSTRUCTIONS_WITHOUT_TOOLS
            if self.output_type in (OutputType.OBJECT, OutputType.ARRAY_OBJECT):
                args = self.output_schema.model_json_schema()["properties"]
                final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_type].format(output_schema=args)
            else:
                final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_type]

            raise ValueError(
                f"You made a mistake in your response: {output}\n\nYour goal is to correct your response\n\n{instructions}\n\n{final_answer_instructions}"
            )

        thought = match.group(1).strip()
        final_answer = match.group(2).strip()
        final_answer = self._validate_final_answer(final_answer)
        return thought, final_answer

    def parse(self, output: str) -> LLMToolUse | LLMFinalAnswer:
        if "Tool:" in output:
            thought, tool, tool_input = self._parse_tool(output)
            return LLMToolUse(thought=thought, tool=tool, tool_input=tool_input)

        if "Final Answer:" in output:
            thought, final_answer = self._parse_final_answer(output)
            return LLMFinalAnswer(thought=thought, final_answer=final_answer)

        instructions = INSTRUCTIONS_WITH_TOOLS if self.tool_use else INSTRUCTIONS_WITHOUT_TOOLS
        if self.output_type in (OutputType.OBJECT, OutputType.ARRAY_OBJECT):
            args = self.output_schema.model_json_schema()["properties"]
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_type].format(output_schema=args)
        else:
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_type]

        raise ValueError(
            f"You made a mistake in your response: {output}\n\nYour goal is to correct your response\n\n{instructions}\n\n{final_answer_instructions}"
        )
