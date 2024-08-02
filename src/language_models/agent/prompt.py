"""ReAct agent prompt."""

INSTRUCTIONS_WITH_TOOLS = """### Tools ###

You have access to the following tools:
{tools}

### Instructions ###

Your goal is to solve the problem you will be provided with

You should respond with:
```
Thought: <thought process on how to respond to the prompt>

Tool: <name of the tool to use>

Tool Input: <input of the tool to use>
```

Your <input of the tool to use> must be a JSON format with the keyword arguments of <name of the tool to use>

When you know the final answer to the user's query you should respond with:
```
Thought: <thought process on how to respond to the prompt>

Final Answer: <response to the prompt>
```"""


INSTRUCTIONS_WITHOUT_TOOLS = """### Instructions ###

Your goal is to solve the problem you will be provided with

You should respond with:
```
Thought: <thought process on how to respond to the prompt>

Final Answer: <response to the prompt>
```"""


OUTPUT_TYPE_STRING = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a string"""
)


OUTPUT_TYPE_INTEGER = (
    """Your <response to the prompt> should be the final answer to the user's query and must be an integer"""
)


OUTPUT_TYPE_FLOAT = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a float"""
)


OUTPUT_TYPE_OBJECT_OR_STRUCT = """Your <response to the prompt> should be the final answer to the user's query and must be a JSON format

Here are the properties of the Pydantic model JSON schema:
{output_schema}

Here is an example:
```
Thought: Now that I have the information to answer the user's query, I will provide it in the specified format.

Final Answer: {example}
```"""


OUTPUT_TYPE_ARRAY_STRING = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of strings"""
)


OUTPUT_TYPE_ARRAY_INTEGER = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of integers"""
)


OUTPUT_TYPE_ARRAY_FLOAT = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of floats"""
)


OUTPUT_TYPE_ARRAY_OBJECT_OR_STRUCT = """Your <response to the prompt> should be the final answer to the user's query and must be a list of JSON format

Here are the properties of the Pydantic model JSON schema:
{output_schema}

Here is an example:
```
Thought: Now that I have the information to answer the user's query, I will provide it in the specified format.

Final Answer: {example}"""


OUTPUT_TYPE_BINARY = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a binary"""
)


OUTPUT_TYPE_BOOLEAN = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a boolean"""
)


OUTPUT_TYPE_DATE = """Your <response to the prompt> should be the final answer to the user's query and must be a date with the format: {output_schema}"""


OUTPUT_TYPE_TIMESTAMP = """Your <response to the prompt> should be the final answer to the user's query and must be a timestamp with the format: {output_schema}"""
