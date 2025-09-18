import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from utilities.common_agent_library import create_agent

# 1. Load environment variables
load_dotenv()

# 2. Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Define the Agent's Specific Data and Tools
STORE_HOURS_DATA = """
Monday: 9:00 AM - 5:00 PM
Tuesday: 9:00 AM - 5:00 PM
Wednesday: 9:00 AM - 5:00 PM
Thursday: 9:00 AM - 5:00 PM
Friday: 9:00 AM - 7:00 PM
Saturday: 10:00 AM - 4:00 PM
Sunday: Closed
"""

@tool
def get_current_datetime_tool() -> str:
    """
    Returns the current date and time in a human-readable format.
    This tool is useful for answering questions about the current time or day.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

tools = [get_current_datetime_tool]

# 4. Define the Agent's Specific System Message
system_message_content = f"""
You are a store hours agent. Your task is to determine if the store is open 
based on the user's question and the provided store hours data.

Store Hours:
{STORE_HOURS_DATA}

You must use the tool `get_current_datetime_tool` if the user asks for "now", "today", or "current time".
If the user mentions a specific date or time, use that instead.

Respond concisely and in a friendly manner.
"""

# 5. Create the Agent using the common library
store_hours_agent = create_agent(
    llm=llm,
    system_message_content=system_message_content,
    tools=tools
)

'''
# Example Usage
if __name__ == "__main__":
    print("Store Hours Agent created!")
    for s in store_hours_agent.stream({"messages": [("user", "Is the store open now?")]}):
        print(s)
    print("\n---")
    for s in store_hours_agent.stream({"messages": [("user", "What are the hours on Sunday?")]}):
        print(s)
'''