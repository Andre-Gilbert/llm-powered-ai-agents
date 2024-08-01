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


FINAL_ANSWER_STRING = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a string"""
)


FINAL_ANSWER_INTEGER = (
    """Your <response to the prompt> should be the final answer to the user's query and must be an integer"""
)


FINAL_ANSWER_FLOAT = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a float"""
)


FINAL_ANSWER_LIST_STRING = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of strings"""
)


FINAL_ANSWER_OBJECT = """Your <response to the prompt> should be the final answer to the user's query and must be a JSON format

Here are the properties of the Pydantic model JSON schema:
{object_schema}

Here is an example:
```
Thought: Now that I have the information to answer the user's query, I will provide it in the specified format.

Final Answer: {{'key': <value>}}
```"""


FINAL_ANSWER_LIST_INTEGER = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of integers"""
)


FINAL_ANSWER_LIST_FLOAT = (
    """Your <response to the prompt> should be the final answer to the user's query and must be a list of floats"""
)


FINAL_ANSWER_LIST_OBJECT = """Your <response to the prompt> should be the final answer to the user's query and must be a list of JSON format

Here are the properties of the Pydantic model JSON schema:
{object_schema}

Here is an example:
```
Thought: Now that I have the information to answer the user's query, I will provide it in the specified format.

Final Answer: {{'key': <value>}}
```"""
