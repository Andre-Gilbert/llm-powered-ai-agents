"""ReAct agent."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken
from pydantic import BaseModel

from language_models.agent.chat import (
    Chat,
    ReasoningStep,
    ReasoningStepName,
    ReasoningStepTool,
)
from language_models.agent.output_parser import (
    FINAL_ANSWER_INSTRUCTIONS,
    LLMFinalAnswer,
    LLMToolUse,
    OutputFormat,
    ReActOutputParser,
)
from language_models.agent.prompt import (
    INSTRUCTIONS_WITH_TOOLS,
    INSTRUCTIONS_WITHOUT_TOOLS,
)
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


class AgentResponse(BaseModel):
    prompt: str
    final_answer: str | int | float | dict | list[str] | list[int] | list[float] | list[dict] | None
    chain_of_thought: list[ReasoningStep]


class ReActAgent(BaseModel):
    """Class that implements a ReAct agent."""

    llm: OpenAILanguageModel
    tools: dict[str, Tool] | None
    prompt: str
    prompt_variables: list[str]
    output_parser: ReActOutputParser
    chat: Chat
    iterations: int = 10

    def _trim_conversation(self) -> None:
        """Trims the chat messages to fit the LLM context length."""
        num_tokens = num_tokens_from_messages(self.chat.messages)
        while num_tokens + self.llm.max_tokens >= _MODEL_TOKEN_LIMIT[self.llm.model]:
            del self.chat.messages[1]
            num_tokens = num_tokens_from_messages(self.chat.messages)

    def _parse_output(self, output: str) -> LLMToolUse | LLMFinalAnswer:
        """Parses the LLM output."""
        try:
            output = self.output_parser.parse(output)
            observation = None
        except ValueError as error:
            output = None
            observation = error
        except TypeError as error:
            output = None
            observation = error
        return output, observation

    def invoke(self, prompt: dict[str, Any]) -> AgentResponse:
        """Runs the agent given a prompt."""
        prompt = self.prompt.format(**{variable: prompt.get(variable) for variable in self.prompt_variables})

        logging.info("Prompt:\n%s", prompt)
        self.chat.messages.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))
        self.chat.chain_of_thought = [ReasoningStep(name=ReasoningStepName.PROMPT, content=prompt)]
        self.chat.steps = []

        iteration = 0
        while iteration <= self.iterations:
            self._trim_conversation()
            output = self.llm.get_completion(self.chat.messages)
            logging.info("Raw Output:\n%s", output)
            output, observation = self._parse_output(output)

            if output is not None:
                logging.info("Thought:\n%s", output.thought)
                self.chat.steps.append(f"Thought: {output.thought}")
                self.chat.chain_of_thought.append(ReasoningStep(name=ReasoningStepName.THOUGHT, content=output.thought))

                if isinstance(output, LLMFinalAnswer):
                    logging.info("Final Answer:\n%s", output.final_answer)
                    self.chat.chain_of_thought.append(
                        ReasoningStep(name=ReasoningStepName.FINAL_ANSWER, content=str(output.final_answer))
                    )
                    self.chat.messages.append(
                        ChatMessage(role=ChatMessageRole.ASSISTANT, content=str(output.final_answer))
                    )
                    return AgentResponse(
                        prompt=prompt,
                        final_answer=output.final_answer,
                        chain_of_thought=self.chat.chain_of_thought,
                    )

                else:
                    if self.tools is not None:
                        logging.info("Tool:\n%s", output.tool)
                        logging.info("Tool Input:\n%s", output.tool_input)
                        tool = self.tools.get(output.tool)

                        if tool is not None:
                            tool_response = tool.invoke(output.tool_input)
                            observation = f"Tool Response:\n{tool_response}"
                            logging.info(observation)
                            self.chat.steps.append(f"Tool: {tool.name}")
                            self.chat.steps.append(f"Tool Input: {output.tool_input}")
                            self.chat.chain_of_thought.append(
                                ReasoningStep(
                                    name=ReasoningStepName.TOOL,
                                    content=ReasoningStepTool(
                                        tool=output.tool, tool_input=output.tool_input, tool_response=tool_response
                                    ),
                                )
                            )

                        else:
                            tool_names = ", ".join(list(self.tools.keys()))
                            observation = f"{output.tool} tool doesn't exist. Try one of these tools: {tool_names}"

            self.chat.steps.append(f"Observation: {observation}")
            self.chat.update(prompt)
            iteration += 1

        if self.output_parser.output_format == OutputFormat.OBJECT:
            final_answer = {key: None for key in self.output_parser.object_schema.model_json_schema()["properties"]}

        elif self.output_parser.output_format == OutputFormat.LIST_OBJECT:
            final_answer = [{key: None for key in self.output_parser.object_schema.model_json_schema()["properties"]}]
        else:
            final_answer = None

        return AgentResponse(
            prompt=prompt,
            final_answer=final_answer,
            chain_of_thought=self.chat.chain_of_thought,
        )

    @classmethod
    def create(
        cls,
        llm: OpenAILanguageModel,
        system_prompt: str,
        prompt: str,
        prompt_variables: list[str],
        output_format: OutputFormat,
        object_schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        iterations: int = 10,
    ) -> ReActAgent:
        """Creates an instance of the ReAct agent."""
        if tools is None:
            instructions = INSTRUCTIONS_WITHOUT_TOOLS
            tool_use = False
            tools = None
        else:
            instructions = INSTRUCTIONS_WITH_TOOLS.format(tools="\n\n".join([str(tool) for tool in tools]))
            tool_use = True
            tools = {tool.name: tool for tool in tools}

        if output_format in (OutputFormat.OBJECT, OutputFormat.LIST_OBJECT):
            if object_schema is None:
                raise ValueError(
                    "When using object or list object as the output format a schema of the object must be provided."
                )

            args = object_schema.model_json_schema()
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[output_format].format(object_schema=args)
        else:
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[output_format]

        chat = Chat(
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="\n\n".join([system_prompt, instructions, final_answer_instructions]),
                )
            ]
        )

        output_parser = ReActOutputParser(output_format=output_format, object_schema=object_schema, tool_use=tool_use)

        return ReActAgent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            prompt_variables=prompt_variables,
            output_parser=output_parser,
            chat=chat,
            iterations=iterations,
        )
