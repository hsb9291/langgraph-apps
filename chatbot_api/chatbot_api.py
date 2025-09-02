import os
import uuid
from typing import TypedDict, Annotated, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LangGraph Setup ---
# Define the state of our graph.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the nodes of our graph.
def chat_node(state: State):
    """A node that invokes the OpenAI model with the current conversation history."""
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Build the graph and compile it once at startup
def create_chat_graph():
    """Creates and compiles the LangGraph-based chat application."""
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chat_node)
    workflow.add_edge(START, "chatbot")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create the LangGraph app instance
app_graph = create_chat_graph()

# --- FastAPI Setup ---
# Create the FastAPI app
api = FastAPI(
    title="LangGraph Chatbot API",
    description="An API to serve a conversational chatbot powered by LangGraph."
)

# Pydantic model for the request body
class ChatRequest(BaseModel):
    message: str
    thread_id: str

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    API endpoint to handle a single turn of the conversation.
    It takes a user message and a thread_id to maintain state.
    """
    # Create the configurable thread ID from the request
    config = {"configurable": {"thread_id": request.thread_id}}

    # Create a HumanMessage from the user's input
    input_message = HumanMessage(content=request.message)

    # Define a generator function to stream the response
    async def event_generator():
        # Stream the response from the LangGraph app
        async for event in app_graph.astream(
            {"messages": [input_message]},
            config,
            stream_mode="values"
        ):
            # We are only interested in the last message, which is the AI's response
            response_message = event["messages"][-1]
            yield f"data: {response_message.content}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# You can run this file with 'uvicorn api:api --reload'
