from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

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

def format_messages(msgs: List[BaseMessage]) -> List[dict]:
    out = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            out.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "ai", "content": m.content})
    return out
