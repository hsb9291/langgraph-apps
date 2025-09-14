import os
from typing import Annotated, List, TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# --- Define State ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Define Nodes ---
def writer(state: State) -> State:
    resp = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=resp.content)]}

def enhancer(state: State) -> State:
    last_msg = state["messages"][-1].content
    resp = llm.invoke([HumanMessage(content=f"Make this sound more exciting: {last_msg}")])
    return {"messages": [AIMessage(content=resp.content)]}

# --- Build Graph ---
graph = StateGraph(State)
graph.add_node("writer", writer)
graph.add_node("enhancer", enhancer)
graph.add_edge(START, "writer")
graph.add_edge("writer", "enhancer")
graph.add_edge("enhancer", END)
app_graph = graph.compile()

# --- FastAPI App ---
app = FastAPI()

# Allow client requests (adjust origins if you want stricter security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if serving HTML via VSCode Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: parse incoming JSON messages
def parse_messages(data: List[dict]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for m in data:
        role = m.get("role")
        content = m.get("content", "")
        if role == "human":
            msgs.append(HumanMessage(content=content))
        elif role == "ai":
            msgs.append(AIMessage(content=content))
    return msgs

@app.post("/chat")
async def chat(request: Request):
    """Return final graph result given chat history."""
    body = await request.json()
    messages = parse_messages(body.get("messages", []))
    final_state = app_graph.invoke({"messages": messages})
    return {
        "messages": [{"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                     for m in final_state["messages"]]
    }

@app.post("/stream")
async def stream(request: Request):
    """Stream graph execution step by step as Server-Sent Events (SSE)."""
    body = await request.json()
    messages = parse_messages(body.get("messages", []))

    def event_generator():
        for event in app_graph.stream({"messages": messages}, stream_mode="updates"):
            for node, value in event.items():
                msg = value["messages"][-1]
                role = "human" if isinstance(msg, HumanMessage) else "ai"
                yield f"data: [{node}] {role}: {msg.content}\n\n"

        final_state = app_graph.invoke({"messages": messages})
        yield "data: [final] done\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
