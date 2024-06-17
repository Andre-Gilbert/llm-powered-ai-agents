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

You have access to the following tools:
{tools}

Please ALWAYS use the following JSON format:
{{
  "thought": "Explain your thought. Consider previous and subsequent steps",
  "tool": "The tool to use. Must be on of {tool_names}",
  "tool_input": "Valid keyword arguments (e.g. {{"key": value}})",
}}

Observation: tool result
... (this Thought/Tool/Tool input/Observation can repeat N times)

When you know the answer, you MUST use the following JSON format:
{{
  "thought": "Explain the reason of your final answer when you know what to respond",
  "tool": "Final Answer",
  "tool_input": "Valid keyword arguments (e.g. {{"key": value}})",
}}"""


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

    def _parse_response(self, response: str) -> tuple:
        """Parses the LLM response."""
        try:
            response = json.loads(response, strict=False)
            response = LLMResponse.model_validate(response)
            observation = None
        except json.decoder.JSONDecodeError:
            response = None
            tool_names = ", ".join(list(self.tools.keys()))
            observation = (
                "Your response format was incorrect."
                + " Look at the JSON format below and correct your answer."
                + "\n\nPlease ALWAYS use the following JSON format:"
                + '\n{\n  "thought": "Explain your thought. Consider previous and subsequent steps",'
                + f'\n  "tool": "The tool to use. Must be one of {tool_names}",'
                + '\n  "tool_input": "Valid keyword arguments (e.g. {"key": value})"\n}'
                + "\n\nWhen you know the answer, you MUST use the following JSON format:"
                + '\n{\n  "thought": "Explain the reason of your final answer when you know what to respond",'
                + '\n  "tool": "Final Answer",'
                + '\n  "tool_input": "Valid keyword arguments (e.g. {"key": value})"\n}'
            )
        except ValidationError as e:
            response = None
            observation = f"Your response failed validation. The error was: {e}"
        return response, observation

    def invoke(self, prompt: dict[str, Any]) -> AgentResponse:
        """Runs the AI agent."""
        previous_work = []
        prompt = self.task_prompt.format(**{variable: prompt.get(variable) for variable in self.task_prompt_variables})
        logging.info("Prompt:\n%s", prompt)
        self.chat_messages.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))
        chain_of_thought: list[LLMCoT] = [LLMCoT(step=LLMCoTStep.PROMPT, content=prompt)]
        iterations = 0
        while iterations <= self.iterations:
            self._trim_conversation()
            response = self.llm.get_completion(self.chat_messages)
            logging.info("Raw response:\n%s", response)
            response, observation = self._parse_response(response)
            if response is not None:
                logging.info("Thought:\n%s", response.thought)
                previous_work.append(f"Thought: {response.thought}")
                chain_of_thought.append(LLMCoT(step=LLMCoTStep.THOUGHT, content=response.thought))
                if response.tool == "Final Answer":
                    try:
                        logging.info("Final answer:\n%s", response.tool_input)
                        result = self.output_format.model_validate(response.tool_input)
                        final_answer = result.model_dump()
                        chain_of_thought.append(LLMCoT(step=LLMCoTStep.FINAL_ANSWER, content=final_answer))
                        self.chat_messages.append(
                            ChatMessage(role=ChatMessageRole.ASSISTANT, content=str(final_answer))
                        )
                        return AgentResponse(
                            prompt=prompt,
                            final_answer=final_answer,
                            chain_of_thought=[step.model_dump() for step in chain_of_thought],
                        )
                    except ValidationError as e:
                        observation = f"Your final answer failed validation. The error was: {e}"
                else:
                    if self.tools is not None:
                        logging.info("Tool:\n%s", response.tool)
                        logging.info("Tool input:\n%s", response.tool_input)
                        tool = self.tools.get(response.tool)
                        if tool is not None:
                            tool_response = tool.invoke(response.tool_input)
                            observation = f"Tool response:\n{tool_response}"
                            logging.info(observation)
                            previous_work.append(f"Tool: {tool.name}")
                            previous_work.append(f"Tool input: {response.tool_input}")
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
                            tool_names = ", ".join(list(self.tools.keys()))
                            observation = f"{response.tool} tool doesn't exist. Try one of these tools: {tool_names}"
            previous_work.append(f"Observation: {observation}")
            self.chat_messages[-1].content = prompt + "\n\nThis was your previous work:\n\n" + "\n".join(previous_work)
            iterations += 1
        return AgentResponse(
            prompt=prompt,
            final_answer={key: None for key in self.output_format.model_json_schema()["properties"]},
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
            description="Use this tool when you know the final answer.",
            args_schema=output_format,
        )
        tools = [output_tool] if tools is None else tools + [output_tool]
        format_instructions = _FORMAT_INSTRUCTIONS.format(
            tools="\n".join([str(tool) for tool in tools]),
            tool_names=", ".join([tool.name for tool in tools]),
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
