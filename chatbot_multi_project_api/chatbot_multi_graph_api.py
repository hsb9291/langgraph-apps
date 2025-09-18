import os, importlib.util, glob
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from chatbot_multi_project_api.utils import parse_messages, format_messages

# Load environment
load_dotenv()

# --- Load all graphs from projects/ ---
graphs = {}

cwd = Path.cwd()
PROJECTS_DIR = os.path.join(cwd, "agent_projects")

# Cross-platform glob pattern
pattern = os.path.join(PROJECTS_DIR, "*", "graph.py")

print(f"PROJECTS_DIR: {PROJECTS_DIR}")
print(f"PROJECTS PATH: {pattern}")

for path in glob.glob(pattern):
    project_name = os.path.basename(os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(project_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_graph"):
        graphs[project_name] = module.get_graph()

print(f"Loaded graphs: {graphs}")
print(f"Loaded graphs: {list(graphs.keys())}")

# --- FastAPI App ---
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_graph_or_404(name: str):
    if name not in graphs:
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
    return graphs[name]

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    graph_name = body.get("graph")
    if not graph_name:
        raise HTTPException(status_code=400, detail="Missing 'graph' in request")
    graph = get_graph_or_404(graph_name)

    messages = parse_messages(body.get("messages", []))
    final_state = graph.invoke({"messages": messages})
    return {"messages": format_messages(final_state["messages"])}

@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    graph_name = body.get("graph")
    if not graph_name:
        raise HTTPException(status_code=400, detail="Missing 'graph' in request")
    graph = get_graph_or_404(graph_name)

    messages = parse_messages(body.get("messages", []))

    def event_generator():
        for event in graph.stream({"messages": messages}, stream_mode="updates"):
            for node, value in event.items():
                msg = value["messages"][-1]
                role = "human" if msg.type == "human" else "ai"
                yield f"data: [{node}] {role}: {msg.content}\n\n"
        yield "data: [final] done\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/graphs")
def list_graphs():
    """List all available graphs by name."""
    return {"available_graphs": list(graphs.keys())}
