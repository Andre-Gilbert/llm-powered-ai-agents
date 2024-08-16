"""Agent."""

from __future__ import annotations

import sys
from typing import Any

import tiktoken
from loguru import logger
from pydantic import BaseModel, ValidationError, create_model

from language_models.agent.chat import (
    Chat,
    Step,
    StepFinalAnswer,
    StepName,
    StepToolUse,
)
from language_models.agent.output_parser import (
    CHAIN_OF_THOUGHT_FINAL_ANSWER_INSTRUCTIONS,
    FINAL_ANSWER_INSTRUCTIONS,
    AgentOutputParser,
    LLMFinalAnswer,
    LLMToolUse,
    OutputType,
    PromptingStrategy,
    get_schema_from_args,
)
from language_models.agent.prompt import (
    CHAIN_OF_THOUGHT_INSTRUCTIONS_WITH_TOOLS,
    CHAIN_OF_THOUGHT_INSTRUCTIONS_WITHOUT_TOOLS,
    SINGLE_COMPLETION_INSTRUCTIONS,
)
from language_models.models.llm import ChatMessage, ChatMessageRole, OpenAILanguageModel
from language_models.tools.tool import Tool

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


_MODEL_TOKEN_LIMIT = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-16k": 16385,
}


def num_tokens_from_messages(messages: list[ChatMessage]) -> int:
    """Counts the number of tokens in the conversation history."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.model_dump().items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


class AgentOutput(BaseModel):
    """Class that represents the agent output."""

    prompt: str
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
        | None
    )
    steps: list[Step]


class Agent(BaseModel):
    """Class that implements a ReAct agent."""

    llm: OpenAILanguageModel
    tools: dict[str, Tool] | None
    prompt: str
    prompt_variables: list[str]
    output_parser: AgentOutputParser
    chat: Chat
    prompting_strategy: PromptingStrategy = PromptingStrategy.CHAIN_OF_THOUGHT
    iterations: int = 5
    verbose: bool = True

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
        except (ValueError, ValidationError) as error:
            output = None
            observation = error
        return output, observation

    def _tool_use_approved(self, tool: Tool, tool_input: dict[str, Any]) -> bool:
        """Approved the use of a tool."""
        if tool.requires_approval:
            decision = input(
                "\n\n".join(
                    [
                        "Do you allow the invocation of the tool (Y/y/Yes/yes)?",
                        f"Tool: {tool.name}",
                        f"Tool Input: {None if tool.args else tool_input}",
                    ]
                )
            )
            if decision not in ("Y", "y", "Yes", "yes"):
                if self.verbose:
                    logger.opt(colors=True).info("<b><fg #EC9A3C>Tool Use Approved</fg #EC9A3C></b>: No")

                return False

            if self.verbose:
                logger.opt(colors=True).info("<b><fg #EC9A3C>Tool Use Approved</fg #EC9A3C></b>: Yes")

        return True

    def invoke(self, prompt: dict[str, Any]) -> AgentOutput:
        """Runs the agent given a prompt."""
        prompt = self.prompt.format(**{variable: prompt.get(variable) for variable in self.prompt_variables})

        self.chat.steps = [self.chat.steps[0], self.chat.steps[1], Step(name=StepName.PROMPT, content=prompt)]
        self.chat.messages.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))
        self.chat.previous_steps = []

        iteration = 0
        while iteration <= self.iterations:
            self._trim_conversation()
            raw_output = self.llm.get_completion(self.chat.messages)
            output, observation = self._parse_output(raw_output)
            if self.prompting_strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
                self.chat.steps.append(Step(name=StepName.RAW_OUTPUT, content=raw_output))
                if output is not None:
                    if self.verbose:
                        logger.opt(colors=True).info(f"<b><fg #2D72D2>Thought</fg #2D72D2></b>: {output.thought}")

                    self.chat.previous_steps.append(f"Thought: {output.thought}")
                    if isinstance(output, LLMFinalAnswer):
                        if self.verbose:
                            logger.opt(colors=True).success(
                                f"<b><fg #32A467>Final Answer</fg #32A467></b>: {output.final_answer}"
                            )

                        self.chat.steps.append(
                            Step(
                                name=StepName.FINAL_ANSWER,
                                content=StepFinalAnswer(thought=output.thought, output=output.final_answer),
                            )
                        )
                        self.chat.messages.append(
                            ChatMessage(role=ChatMessageRole.ASSISTANT, content=str(output.final_answer))
                        )
                        return AgentOutput(prompt=prompt, final_answer=output.final_answer, steps=self.chat.steps)
                    else:
                        if self.tools is not None:
                            if self.verbose:
                                logger.opt(colors=True).info(f"<b><fg #EC9A3C>Tool</fg #EC9A3C></b>: {output.tool}")
                                logger.opt(colors=True).info(
                                    f"<b><fg #EC9A3C>Tool Input</fg #EC9A3C></b>: {output.tool_input}"
                                )

                            tool = self.tools.get(output.tool)
                            if tool is not None:
                                self.chat.previous_steps.append(f"Tool: {output.tool}")
                                self.chat.previous_steps.append(f"Tool Input: {output.tool_input}")
                                self.chat.steps.append(
                                    Step(
                                        name=StepName.TOOL_USE,
                                        content=StepToolUse(
                                            thought=output.thought,
                                            used=output.tool,
                                            arguments=output.tool_input,
                                        ),
                                    )
                                )
                                if self._tool_use_approved(tool, output.tool_input):
                                    tool_output = tool.invoke(output.tool_input)
                                    self.chat.steps.append(Step(name=StepName.TOOL_OUTPUT, content=tool_output))
                                    observation = f"Tool Output: {tool_output}"
                                    if self.verbose:
                                        logger.opt(colors=True).info(
                                            f"<b><fg #EC9A3C>Tool Output</fg #EC9A3C></b>: {tool_output}"
                                        )
                                else:
                                    observation = "\n\n".join(
                                        [
                                            f"The user did not approve the use of the tool: {output.tool}",
                                            "Do not use the tool again and provide the user with the final answer",
                                            CHAIN_OF_THOUGHT_FINAL_ANSWER_INSTRUCTIONS,
                                        ]
                                    )
                            else:
                                tool_names = ", ".join(list(self.tools.keys()))
                                observation = f"{output.tool} tool doesn't exist. Try one of these tools: {tool_names}"

                self.chat.previous_steps.append(f"Observation: {observation}")
                if self.chat.steps[-1].name != StepName.TOOL_OUTPUT:
                    self.chat.steps.append(Step(name=StepName.OBSERVATION, content=observation))

                self.chat.update(prompt)
            else:
                if isinstance(output, LLMFinalAnswer):
                    if self.verbose:
                        logger.opt(colors=True).success(
                            f"<b><fg #32A467>Final Answer</fg #32A467></b>: {output.final_answer}"
                        )

                    self.chat.steps.append(
                        Step(name=StepName.FINAL_ANSWER, content=StepFinalAnswer(output=output.final_answer))
                    )
                    self.chat.messages.append(
                        ChatMessage(role=ChatMessageRole.ASSISTANT, content=str(output.final_answer))
                    )
                    return AgentOutput(prompt=prompt, final_answer=output.final_answer, steps=self.chat.steps)

            iteration += 1

        if self.output_parser.output_type == OutputType.STRUCT:
            final_answer = {key: None for key in self.output_parser.output_schema.model_json_schema()["properties"]}
        elif self.output_parser.output_type == OutputType.ARRAY_STRUCT:
            final_answer = [{key: None for key in self.output_parser.output_schema.model_json_schema()["properties"]}]
        elif self.output_parser.output_type in (OutputType.OBJECT, OutputType.ARRAY_OBJECT):
            fields = self.output_parser.output_schema.__annotations__
            optional_fields = {field: (data_type | None, None) for field, data_type in fields.items()}
            model = create_model(self.output_parser.output_schema.__name__, **optional_fields)
            final_answer = model() if self.output_parser.output_type == OutputType.OBJECT else [model()]
        else:
            final_answer = None

        if self.verbose:
            logger.opt(colors=True).warning(f"<b><fg #CD4246>Final Answer</fg #CD4246></b>: {final_answer}")

        if self.prompting_strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            return AgentOutput(prompt=prompt, final_answer=final_answer, steps=self.chat.steps)
        else:
            return AgentOutput(prompt=prompt, final_answer=final_answer, steps=self.chat.steps)

    @classmethod
    def create(
        cls,
        llm: OpenAILanguageModel,
        system_prompt: str,
        prompt: str,
        prompt_variables: list[str],
        output_type: OutputType,
        output_schema: type[BaseModel] | str | None = None,
        tools: list[Tool] | None = None,
        prompting_strategy: PromptingStrategy = PromptingStrategy.CHAIN_OF_THOUGHT,
        verbose: bool = True,
    ) -> Agent:
        """Creates an instance of the ReAct agent."""
        if prompting_strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            if tools is None:
                instructions = CHAIN_OF_THOUGHT_INSTRUCTIONS_WITHOUT_TOOLS
                tool_use = False
                tools = None
                iterations = 5
            else:
                instructions = CHAIN_OF_THOUGHT_INSTRUCTIONS_WITH_TOOLS.format(
                    tools="\n\n".join([str(tool) for tool in tools])
                )
                tool_use = True
                tools = {tool.name: tool for tool in tools}
                iterations = max(5, len(tools) * 2)
        else:
            instructions = SINGLE_COMPLETION_INSTRUCTIONS
            tool_use = False
            tools = None
            iterations = 1

        if output_type in (OutputType.OBJECT, OutputType.ARRAY_OBJECT, OutputType.STRUCT, OutputType.ARRAY_STRUCT):
            if output_schema is None:
                raise ValueError(f"When using {output_type} as the output type a schema must be provided.")

            schema = output_schema.model_json_schema()
            if "$defs" not in schema:
                schema = get_schema_from_args(schema["properties"])
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[output_type].format(output_schema=schema)
        elif output_type in (OutputType.DATE, OutputType.TIMESTAMP):
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[output_type].format(output_schema=output_schema)
        else:
            final_answer_instructions = FINAL_ANSWER_INSTRUCTIONS[output_type]

        chat = Chat(
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="\n\n".join([system_prompt, instructions, final_answer_instructions]),
                )
            ],
            steps=[
                Step(name=StepName.SYSTEM_PROMPT, content=system_prompt),
                Step(name=StepName.PROMPTING_STRATEGY, content=prompting_strategy.value),
            ],
        )

        output_parser = AgentOutputParser(
            output_type=output_type,
            output_schema=output_schema,
            prompting_strategy=prompting_strategy,
            tool_use=tool_use,
        )

        return Agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            prompt_variables=prompt_variables,
            output_parser=output_parser,
            chat=chat,
            prompting_strategy=prompting_strategy,
            iterations=iterations,
            verbose=verbose,
        )
