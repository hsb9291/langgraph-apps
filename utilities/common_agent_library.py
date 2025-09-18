import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage

# 1. Define the Generic Agent State
class AgentState(TypedDict):
    """Represents the state of the agent's conversation."""
    messages: Annotated[List[BaseMessage], operator.add]

# 2. Define the Generic Graph Building Function
def create_agent(llm, system_message_content: str, tools: list):
    """
    Creates and compiles a generic LangGraph agent.

    Args:
        llm: The Language Model instance.
        system_message_content (str): The system prompt for the agent.
        tools (list): A list of LangChain tools to bind to the LLM.

    Returns:
        A compiled LangGraph object.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}
    system_message = SystemMessage(content=system_message_content)

    # Generic Graph Nodes
    def agent_node(state: AgentState):
        """Invokes the LLM with the current conversation history."""
        messages = [system_message] + state["messages"]
        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}

    def tool_node(state: AgentState):
        """Executes the tool call based on the LLM's decision."""
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        
        if tool_name in tool_map:
            tool_output = tool_map[tool_name].invoke(tool_input)
        else:
            tool_output = f"Unknown tool: {tool_name}"
        
        return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=tool_call["id"])]}
    
    # Generic Conditional Edge
    def should_continue(state: AgentState):
        """Decides if the agent should continue (call a tool) or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        return "end"

    # Build and compile the graph
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