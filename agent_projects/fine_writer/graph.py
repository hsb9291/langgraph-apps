from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def writer(state: State) -> State:
    resp = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=resp.content)]}

def enhancer(state: State) -> State:
    last_msg = state["messages"][-1].content
    resp = llm.invoke([HumanMessage(content=f"Make this more fun: {last_msg}")])
    return {"messages": [AIMessage(content=resp.content)]}

def get_graph():
    graph = StateGraph(State)
    graph.add_node("writer", writer)
    graph.add_node("enhancer", enhancer)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", "enhancer")
    graph.add_edge("enhancer", END)
    return graph.compile()
