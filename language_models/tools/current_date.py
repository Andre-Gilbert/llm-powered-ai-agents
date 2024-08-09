"""Current date tool."""

from datetime import datetime

from language_models.tools.tool import Tool

current_date = Tool(
    function=lambda _: datetime.now(),
    name="Current Date",
    description="Use this tool to access the current local date and time",
)
