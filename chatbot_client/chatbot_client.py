import requests
import json
import uuid
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Client Program ---
def run_chat_client():
    """
    Runs the command-line chat interface that communicates with the API.
    """
    # Base URL for the API
    api_url = "http://127.0.0.1:8000/chat"
    
    # Generate a unique thread ID for the entire conversation
    thread_id = str(uuid.uuid4())
    
    print("Welcome to the LangGraph Chatbot Client! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Prepare the request payload
        payload = {
            "message": user_input,
            "thread_id": thread_id
        }
        
        try:
            # Make a POST request to the API with a stream
            with requests.post(api_url, json=payload, stream=True) as response:
                response.raise_for_status() # Raise an exception for bad status codes
                
                print("AI: ", end="", flush=True)
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    # Each chunk is a Server-Sent Event data block
                    if chunk.startswith("data:"):
                        content = chunk[len("data:"):].strip()
                        print(content, end="", flush=True)
                print() # Print a newline at the end of the AI's response
                
        except requests.exceptions.RequestException as e:
            print(f"\nError connecting to the API: {e}")
            print("Please ensure the server is running.")
            
if __name__ == "__main__":
    run_chat_client()
