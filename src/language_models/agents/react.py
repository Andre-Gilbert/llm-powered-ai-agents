"""ReAct agent."""

from __future__ import annotations

import json
from typing import Any, Type

import tiktoken
from pydantic import BaseModel

from language_models.models.llm import ChatMessage, ChatMessageRole, OpenAILanguageModel
from language_models.tools.tool import Tool

_MODEL_TOKEN_LIMIT = {"gpt-4": 8192, "gpt-4-32k": 32768}

_FORMAT_INSTRUCTIONS = """Respond to the user as helpfully and accurately as possible.

You have access to the following tools:
{tools}

Use a json blob to specify a tool by providing an action (tool name) and an action_input (tool input).

Always use the following JSON response format:
{{
    "thought": you should always think about what to do consider previous and subsequent steps,
    "tool": $TOOL_NAME,
    "tool_input": a valid dictionary in this format {{"<key>": <value>, ...}},
}}
... (this Thought/Action/Observation can repeat N times)

When you know the final answer, structure your final answer in the following way:
{output_format}

Always use the following JSON response format for final answers:
{{
    "thought": I now know the final answer,
    "final_answer": a valid dictionary in this format {{"<key>": <value>, ...}},
}}
"""


def extract(response: dict, key: str) -> str | None:
    """Gets the value to a key from a dict if it is not none and of length larger than 0."""
    value = response.get(key)
    return value if value is not None and len(value) > 0 else None


def num_tokens_from_messages(messages: list[dict]) -> int:
    """Counts the number of tokens in the conversation history."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


class LLMResponse(BaseModel):
    thought: str
    tool: str
    tool_input: dict
    final_answer: Type[BaseModel]


class ReActAgent(BaseModel):
    """Class that implements a ReAct agent."""

    llm: OpenAILanguageModel
    tools: dict[str, Tool] | None
    task_prompt: str
    output_format: str
    chat_messages: list[dict[str, str]]
    iterations: int = 20

    def _trim_conversation(self) -> None:
        num_tokens = num_tokens_from_messages(self.chat_messages)
        while num_tokens + self.llm.max_tokens >= _MODEL_TOKEN_LIMIT[self.llm.model]:
            del self.chat_messages[1]
            num_tokens = num_tokens_from_messages(self.chat_messages)

    def _parse_response(self, response: str) -> None | dict[str, Any]:
        try:
            response = json.loads(response, strict=False)
            thought = extract(response, "thought")
            tool = extract(response, "tool")
            tool_input = extract(response, "tool_input")
            answer = extract(response, "final_answer")
            parsed = True
            observation = "Your response format was correct."
        except json.decoder.JSONDecodeError as e:
            parsed = False
            thought = None
            tool = None
            tool_input = None
            answer = None
            observation = (
                "Your response format was incorrect. Please correct as specified in the first message. "
                f"The error was: {e}"
            )
        return thought, tool, tool_input, answer, observation, parsed

    def invoke(self, prompt: dict[str, Any]):
        prompt = self.task_prompt.format(**prompt)
        self.chat_messages.append({"role": "user", "content": prompt})
        steps = []
        iterations = 0
        while iterations <= self.iterations:
            iterations += 1
            self._trim_conversation()
            response = self.llm.get_completion(self.chat_messages)
            thought, action, action_input, parsed, observation = self._parse_response(
                response
            )
            if parsed:
                steps.append(
                    {"thought": thought, "action": action, "action_input": action_input}
                )
                if action == "Final Answer":
                    return {
                        "prompt": prompt,
                        "answer": answer,
                        "intermediate_steps": steps,
                    }
                else:
                    tool = self.tools.get(action, None)
                    if tool is None:
                        observation = f"{action} tool doesn't exist. Try one of these tools: {self.tool_names}"
                    else:
                        observation = tool.invoke(action_input)
            self.chat_messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
        return {"prompt": prompt, "answer": "", "intermediate_steps": steps}

    @classmethod
    def create(
        cls,
        llm: OpenAILanguageModel,
        system_prompt: str,
        task_prompt: str,
        output_format: Type[BaseModel],
        tools: list[Tool] | None = None,
        iterations: int = 20,
    ) -> ReActAgent:
        format_instructions = _FORMAT_INSTRUCTIONS.format(
            tools=[str(tool) for tool in tools] or None,
            output_format=output_format,
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
            output_format=output_format,
            chat_messages=chat_messages,
            iterations=iterations,
        )
