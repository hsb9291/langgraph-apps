import os
import importlib.util
import glob
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from chatbot_multi_project_api.utils import parse_messages, format_messages # Assuming this utility exists

# Load environment variables
load_dotenv()

# --- Load all agents from project modules ---
agents = {}

cwd = Path.cwd()
AGENTS_DIR = os.path.join(cwd, "agent_projects") # A new, more logical directory name

# Cross-platform glob pattern to find all agent files
pattern = os.path.join(AGENTS_DIR, "*_agent\*_agent.py")

print(f"AGENTS_DIR: {AGENTS_DIR}")
print(f"Agent loading pattern: {pattern}")

for path in glob.glob(pattern):
    module_name = Path(path).stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check for the compiled agent variable (e.g., store_hours_agent)
    # The convention is that the agent variable name is the module name
    # without the '_agent' suffix.
    agent_name = module_name.replace('_agent', '')
    if hasattr(module, f"{agent_name}_agent"):
        agents[agent_name] = getattr(module, f"{agent_name}_agent")

print(f"Loaded agents: {list(agents.keys())}")

# --- FastAPI App ---
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_agent_or_404(name: str):
    """Retrieve the agent or raise a 404 error."""
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agents[name]

@app.post("/chat")
async def chat(request: Request):
    """Synchronous chat endpoint for a specific agent."""
    body = await request.json()
    agent_name = body.get("agent")
    if not agent_name:
        raise HTTPException(status_code=400, detail="Missing 'agent' in request")
    agent = get_agent_or_404(agent_name)

    messages = parse_messages(body.get("messages", []))
    final_state = agent.invoke({"messages": messages})
    return {"messages": format_messages(final_state["messages"])}

@app.post("/stream")
async def stream(request: Request):
    """Streaming chat endpoint for a specific agent."""
    body = await request.json()
    agent_name = body.get("agent")
    if not agent_name:
        raise HTTPException(status_code=400, detail="Missing 'agent' in request")
    agent = get_agent_or_404(agent_name)

    messages = parse_messages(body.get("messages", []))

    def event_generator():
        """Generates Server-Sent Events from the agent's stream."""
        for event in agent.stream({"messages": messages}, stream_mode="updates"):
            for node, value in event.items():
                if isinstance(value, dict) and "messages" in value:
                    msg = value["messages"][-1]
                    role = "human" if msg.type == "human" else "ai"
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    yield f"data: [{node}] {role}: {content}\n\n"
        yield "data: [final] done\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/agents")
def list_agents():
    """List all available agents by name."""
    return {"available_agents": list(agents.keys())}