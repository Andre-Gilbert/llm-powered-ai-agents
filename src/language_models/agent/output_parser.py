"""ReAct agent output parser."""

import re
from enum import Enum
from typing import Any

import dirtyjson as json
from pydantic import BaseModel, ValidationError

from language_models.agent.prompt import (
    FINAL_ANSWER_FLOAT,
    FINAL_ANSWER_INTEGER,
    FINAL_ANSWER_LIST_FLOAT,
    FINAL_ANSWER_LIST_INTEGER,
    FINAL_ANSWER_LIST_OBJECT,
    FINAL_ANSWER_LIST_STRING,
    FINAL_ANSWER_OBJECT,
    FINAL_ANSWER_STRING,
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


class OutputFormat(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    LIST = "list"
    LIST_STRING = "list of strings"
    LIST_INTEGER = "list of integers"
    LIST_FLOAT = "list of floats"
    OBJECT = "object"
    LIST_OBJECT = "list of objects"


FINAL_ANSWER_INSTRUCTIONS = {
    OutputFormat.STRING: FINAL_ANSWER_STRING,
    OutputFormat.INTEGER: FINAL_ANSWER_INTEGER,
    OutputFormat.FLOAT: FINAL_ANSWER_FLOAT,
    OutputFormat.OBJECT: FINAL_ANSWER_OBJECT,
    OutputFormat.LIST_STRING: FINAL_ANSWER_LIST_STRING,
    OutputFormat.LIST_INTEGER: FINAL_ANSWER_LIST_INTEGER,
    OutputFormat.LIST_FLOAT: FINAL_ANSWER_LIST_FLOAT,
    OutputFormat.LIST_OBJECT: FINAL_ANSWER_LIST_OBJECT,
}


class LLMToolUse(BaseModel):
    thought: str
    tool: str
    tool_input: dict[str, Any]


class LLMFinalAnswer(BaseModel):
    thought: str
    final_answer: str | int | float | dict | list[str] | list[int] | list[float] | list[dict]


class ReActOutputParser(BaseModel):
    """Class that parses the LLM output."""

    output_format: OutputFormat
    object_schema: type[BaseModel] | None = None
    tool_use: bool

    def _extract_tool_use(self, output: str) -> tuple[str, str, str]:
        pattern = r"\s*Thought: (.*?)\n+Tool: ([a-zA-Z0-9_ ]+).*?\n+Tool Input: .*?(\{.*\})"

        match = re.search(pattern, output, re.DOTALL)
        if not match:
            raise ValueError(
                f"You made a mistake in your response: {output}\n\nYour goal is to correct your response\n\n{INSTRUCTIONS_WITH_TOOLS}"
            )

        thought = match.group(1).strip()
        tool = match.group(2).strip()
        tool_input = match.group(3).strip()
        return thought, tool, tool_input

    def _extract_json_str(self, tool_input: str):
        match = re.search(r"\{.*\}", tool_input.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(
                f"You made a mistake in your JSON format: {tool_input}\n\nYour goal is to correct your response\n\n{INSTRUCTIONS_WITH_TOOLS}"
            )

        return match.group()

    def _tool_input_parser(self, json_str: str) -> tuple[str, str, dict]:
        processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
        pattern = r'"(\w+)":\s*"([^"]*)"'
        matches = re.findall(pattern, processed_string)
        return dict(matches)

    def _parse_tool(self, output: str) -> tuple[str, str, dict]:
        thought, tool, tool_input = self._extract_tool_use(output)
        json_str = self._extract_json_str(tool_input)
        try:
            tool_input_dict = json.loads(json_str)
        except ValueError:
            tool_input_dict = self._tool_input_parser(json_str)
        return thought, tool, tool_input_dict

    def _validate_final_answer(self, final_answer: str) -> Any:
        if self.output_format == OutputFormat.STRING:
            return final_answer

        if self.output_format == OutputFormat.INTEGER:
            try:
                final_answer = int(final_answer)
                return final_answer
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_INTEGER}"
                ) from error

        if self.output_format == OutputFormat.FLOAT:
            try:
                final_answer = float(final_answer)
                return final_answer
            except ValueError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_FLOAT}"
                ) from error

        if self.output_format == OutputFormat.OBJECT:
            try:
                json_str = self._extract_json_str(final_answer)
                try:
                    final_answer_dict = json.loads(json_str)
                except ValueError:
                    final_answer_dict = self._tool_input_parser(json_str)
                final_answer_model = self.object_schema.model_validate(final_answer_dict)
                return final_answer_model.model_dump()
            except ValueError as error:
                args = self.object_schema.model_json_schema()["properties"]
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n"
                    + f"{FINAL_ANSWER_OBJECT.format(object_schema=args)}"
                ) from error

        if self.output_format == OutputFormat.LIST_STRING:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)
                if all(isinstance(entry, str) for entry in final_answer):
                    return final_answer
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_STRING}"
                )
            except TypeError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_STRING}"
                ) from error

        if self.output_format == OutputFormat.LIST_INTEGER:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)
                if all(isinstance(entry, int) for entry in final_answer):
                    return final_answer
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_INTEGER}"
                )
            except TypeError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_INTEGER}"
                ) from error

        if self.output_format == OutputFormat.LIST_FLOAT:
            try:
                final_answer = json.loads(final_answer)
                final_answer = list(final_answer)
                if all(isinstance(entry, float) for entry in final_answer):
                    return final_answer
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_FLOAT}"
                )
            except TypeError as error:
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n{FINAL_ANSWER_LIST_FLOAT}"
                ) from error

        if self.output_format == OutputFormat.LIST_OBJECT:
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
                return [self.object_schema.model_validate(entry).model_dump() for entry in final_answer_list_dict]
            except ValidationError as error:
                args = self.object_schema.model_json_schema()["properties"]
                raise ValueError(
                    f"You made a mistake in your final answer: {final_answer}\n\nYour goal is to correct your final answer\n\n"
                    + f"{FINAL_ANSWER_LIST_OBJECT.format(object_schema=args)}"
                ) from error

    def _parse_final_answer(self, output: str) -> tuple[str, Any]:
        pattern = r"\s*Thought: (.*?)\n+Final Answer:([\s\S]*.*?)(?:$)"

        match = re.search(pattern, output, re.DOTALL)
        if not match:
            instructions = INSTRUCTIONS_WITH_TOOLS if self.tool_use else INSTRUCTIONS_WITHOUT_TOOLS
            if self.output_format in (OutputFormat.OBJECT, OutputFormat.LIST_OBJECT):
                args = self.object_schema.model_json_schema()["properties"]
                final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_format].format(object_schema=args)
            else:
                final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_format]

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
        if self.output_format in (OutputFormat.OBJECT, OutputFormat.LIST_OBJECT):
            args = self.object_schema.model_json_schema()["properties"]
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_format].format(object_schema=args)
        else:
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[self.output_format]

        raise ValueError(
            f"You made a mistake in your response: {output}\n\nYour goal is to correct your response\n\n{instructions}\n\n{final_answer_instructions}"
        )
