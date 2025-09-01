import os
import uuid
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state of our graph.
# This holds the list of messages in the conversation.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the nodes of our graph.
def chat_node(state: State):
    """
    A node that invokes the OpenAI model with the current conversation history.
    """
    # Initialize the OpenAI chat model
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Get the messages from the current state
    messages = state["messages"]
    
    # Invoke the model and get the response
    response = model.invoke(messages)
    
    # Return the response as a new message to update the state
    return {"messages": [response]}

# Build the graph
def create_chat_graph():
    """
    Creates and compiles the LangGraph-based chat application.
    """
    # Create the graph with the defined state schema
    workflow = StateGraph(State)
    
    # Add the single node for the chat model
    workflow.add_node("chatbot", chat_node)
    
    # Set the starting point and connect it to our node
    workflow.add_edge(START, "chatbot")
    
    # After the chatbot node, we transition back to the START,
    # effectively creating a loop for continuous conversation
    workflow.add_edge("chatbot", END)

    # Use an in-memory checkpointer for simple state persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app

# The main chat loop
def chat():
    """
    Runs the command-line chat interface.
    """
    # Check for the API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Create the chat application
    app = create_chat_graph()
    
    # Generate a unique thread ID for the conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Welcome to the LangGraph Chatbot! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        # Create a HumanMessage from the user input
        input_message = HumanMessage(content=user_input)
        
        # Stream the response from the app
        # We use stream_mode="values" to get the full state updates
        try:
            for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
                # We are only interested in the last message, which is the AI's response
                response_message = event["messages"][-1]
                print(f"AI: {response_message.content}")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
            
if __name__ == "__main__":
    chat()