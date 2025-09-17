import os
import json
import operator
from typing import TypedDict, Annotated, List, Union
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_tool

# 1. Load environment variables
load_dotenv()

# 2. Set up the LLM with tool-calling capabilities
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define the State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 4. Define the Store Hours Data
STORE_HOURS_DATA = """
Monday: 9:00 AM - 5:00 PM
Tuesday: 9:00 AM - 5:00 PM
Wednesday: 9:00 AM - 5:00 PM
Thursday: 9:00 AM - 5:00 PM
Friday: 9:00 AM - 7:00 PM
Saturday: 10:00 AM - 4:00 PM
Sunday: Closed
"""

# 5. Define Tools and bind them to the LLM
@tool
def get_current_datetime_tool() -> str:
    """
    Returns the current date and time in a human-readable format.
    This tool is useful for answering questions about the current time or day.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

tools = [get_current_datetime_tool]
llm_with_tools = llm.bind_tools(tools)

# 6. Define the Agent's System Message
# This will be prepended to every conversation.
system_message_content = f"""
You are a store hours agent. Your task is to determine if the store is open 
based on the user's question and the provided store hours data.

Store Hours:
{STORE_HOURS_DATA}

You must use the tool `get_current_datetime_tool` if the user asks for "now", "today", or "current time".
If the user mentions a specific date or time, use that instead.

Respond concisely and in a friendly manner.
"""
system_message = SystemMessage(content=system_message_content)

# 7. Define the Nodes of the Graph
def agent_node(state: AgentState):
    """
    This node invokes the LLM with the current conversation history and system message.
    """
    # Prepend the system message to the user's message for the LLM to process.
    messages = [system_message] + state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}

def tool_node(state: AgentState):
    """
    This node executes the tool call based on the LLM's decision.
    """
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_input = tool_call["args"]

    if tool_name == "get_current_datetime_tool":
        tool_output = get_current_datetime_tool.invoke(tool_input)
    else:
        tool_output = f"Unknown tool: {tool_name}"
    
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=tool_call["id"])]}

def should_continue(state: AgentState):
    """
    A conditional edge to decide if the agent should continue or end.
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    
    return "end"


# 8. Define the Graph Building Function
def get_graph():
    """
    Builds and returns a LangGraph graph for the store hours agent.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "call_tool": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")

    return workflow.compile()