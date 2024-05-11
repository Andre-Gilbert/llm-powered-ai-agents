"""ReAct agent."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Type

import tiktoken
from pydantic import BaseModel, ValidationError

from language_models.models.llm import ChatMessage, ChatMessageRole, OpenAILanguageModel
from language_models.tools.tool import Tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

_MODEL_TOKEN_LIMIT = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-16k": 16385,
}

_FORMAT_INSTRUCTIONS = """Respond to the user as helpfully and accurately as possible.

You have access to the following tools: {tools}

Use a JSON blob to specify a thought, a tool by providing an tool key (tool name) and a tool_input key (tool input).

Valid "tool" values: {tool_names}

Always use the following JSON format:
{{
  "thought": "You should always think about what to do consider previous and subsequent steps",
  "tool": "The tool to use",
  "tool_input": "Valid key value pairs",
}}

Observation: tool result
... (this Thought/Tool/Observation can repeat N times)

When you know the answer, use the following JSON format:
{{
  "thought": "I now know what to respond",
  "tool": "Final Answer",
  "tool_input": "Valid key value pairs",
}}"""


def extract(response: dict, key: str) -> str | None:
    """Gets the value to a key from a dict if it is not none and of length larger than 0."""
    value = response.get(key)
    return value if value is not None and len(value) > 0 else None


def num_tokens_from_messages(messages: list[ChatMessage]) -> int:
    """Counts the number of tokens in the conversation history."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.model_dump().items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


class LLMCoTStep(str, Enum):
    PROMPT = "prompt"
    THOUGHT = "thought"
    TOOL = "tool"
    FINAL_ANSWER = "final_answer"


class LLMCoTTool(BaseModel):
    name: str
    args: dict[str, Any]
    response: Any


class LLMCoT(BaseModel):
    step: LLMCoTStep
    content: str | LLMCoTTool | dict[str, Any]

    class Config:
        use_enum_values = True


class LLMResponse(BaseModel):
    thought: str
    tool: str
    tool_input: dict[str, Any]


class AgentResponse(BaseModel):
    prompt: str
    final_answer: dict[str, Any]
    chain_of_thought: list[dict[str, Any]]


class ReActAgent(BaseModel):
    """Class that implements a ReAct agent."""

    llm: OpenAILanguageModel
    tools: dict[str, Tool] | None
    task_prompt: str
    task_prompt_variables: list[str]
    output_format: Type[BaseModel]
    chat_messages: list[ChatMessage]
    iterations: int = 20

    def reset(self) -> None:
        """Resets the ReAct agent."""
        self.chat_messages = [self.chat_messages[0]]

    def _trim_conversation(self) -> None:
        """Trims the chat messages to fit the LLM context length."""
        num_tokens = num_tokens_from_messages(self.chat_messages)
        while num_tokens + self.llm.max_tokens >= _MODEL_TOKEN_LIMIT[self.llm.model]:
            del self.chat_messages[1]
            num_tokens = num_tokens_from_messages(self.chat_messages)

    def _parse_response(self, response: str) -> tuple[LLMResponse | None, str]:
        """Parses the LLM response."""
        try:
            response = json.loads(response, strict=False)
            response = LLMResponse.model_validate(response)
            observation = None
        except json.decoder.JSONDecodeError as e:
            response = None
            observation = f"Your response format was incorrect. The error was: {e}"
        except ValidationError as e:
            response = None
            observation = f"Your response failed validation. The error was: {e}"
        return response, observation

    def invoke(self, prompt: dict[str, Any]) -> AgentResponse:
        """Runs the AI agent."""
        prompt = self.task_prompt.format(
            **{
                variable: prompt.get(variable)
                for variable in self.task_prompt_variables
            }
        )
        logging.info("Prompt:\n%s", prompt)
        self.chat_messages.append(
            ChatMessage(role=ChatMessageRole.USER, content=prompt)
        )
        chain_of_thought: list[LLMCoT] = [
            LLMCoT(step=LLMCoTStep.PROMPT, content=prompt)
        ]
        iterations = 0
        while iterations <= self.iterations:
            self._trim_conversation()
            response = self.llm.get_completion(self.chat_messages)
            logging.info("Raw response:\n%s", response)
            response, observation = self._parse_response(response)
            if response is not None:
                logging.info("Thought:\n%s", response.thought)
                chain_of_thought.append(
                    LLMCoT(step=LLMCoTStep.THOUGHT, content=response.thought)
                )
                if response.tool == "Final Answer":
                    try:
                        logging.info("Final answer:\n%s", response.tool_input)
                        result = self.output_format.model_validate(response.tool_input)
                        final_answer = result.model_dump()
                        chain_of_thought.append(
                            LLMCoT(step=LLMCoTStep.FINAL_ANSWER, content=final_answer)
                        )
                        return AgentResponse(
                            prompt=prompt,
                            final_answer=final_answer,
                            chain_of_thought=[
                                step.model_dump() for step in chain_of_thought
                            ],
                        )
                    except ValidationError as e:
                        observation = (
                            f"Your final answer failed validation. The error was: {e}"
                        )
                else:
                    if self.tools is not None:
                        logging.info("Tool:\n%s", response.tool)
                        logging.info("Tool input:\n%s", response.tool_input)
                        tool = self.tools.get(response.tool)
                        if tool is not None:
                            tool_response = tool.invoke(response.tool_input)
                            observation = f"Tool response:\n{tool_response}"
                            logging.info(observation)
                            chain_of_thought.append(
                                LLMCoT(
                                    step=LLMCoTStep.TOOL,
                                    content=LLMCoTTool(
                                        name=response.tool,
                                        args=response.tool_input,
                                        response=tool_response,
                                    ),
                                )
                            )
                        else:
                            observation = (
                                f"{response.tool} tool doesn't exist."
                                f" Try one of these tools: {list(self.tools.keys())}"
                            )
            self.chat_messages.append(
                ChatMessage(role=ChatMessageRole.ASSISTANT, content=observation)
            )
            iterations += 1
        return AgentResponse(
            prompt=prompt,
            final_answer={
                key: None
                for key in self.output_format.model_json_schema()["properties"]
            },
            chain_of_thought=[step.model_dump() for step in chain_of_thought],
        )

    @classmethod
    def create(
        cls,
        llm: OpenAILanguageModel,
        system_prompt: str,
        task_prompt: str,
        task_prompt_variables: list[str],
        output_format: Type[BaseModel],
        tools: list[Tool] | None = None,
        iterations: int = 20,
    ) -> ReActAgent:
        """Creates a instance of the ReAct agent."""
        output_tool = Tool(
            func=lambda _: None,
            name="Final Answer",
            description="Use this tool to answer the question.",
            args_schema=output_format,
        )
        tools = [output_tool] if tools is None else tools + [output_tool]
        format_instructions = _FORMAT_INSTRUCTIONS.format(
            tools=[str(tool) for tool in tools],
            tool_names=[tool.name for tool in tools],
        )
        chat_messages = [
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="\n\n".join([system_prompt, format_instructions]),
            )
        ]
        return ReActAgent(
            llm=llm,
            tools={tool.name: tool for tool in tools},
            task_prompt=task_prompt,
            task_prompt_variables=task_prompt_variables,
            output_format=output_format,
            chat_messages=chat_messages,
            iterations=iterations,
        )
