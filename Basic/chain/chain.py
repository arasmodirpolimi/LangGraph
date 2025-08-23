# -*- coding: utf-8 -*-
"""
Adapted "chain" example in the same pattern as the simple StateGraph demo:
- explicit nodes
- START/END edges
- builder.compile()
- __main__ guard with optional visualization + demo invocations
"""

from typing import Annotated
from typing_extensions import TypedDict
import os, getpass

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

# -----------------------------
# Config (optional helper)
# -----------------------------
def _set_env(var: str):
    if not os.environ.get(var):
        # Prompts only when missing (safe for scripts/Colab)
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# -----------------------------
# MessagesState (prebuilt)
# -----------------------------
# MessagesState already includes:
#   messages: Annotated[list[AnyMessage], add_messages]
# If you want to extend it with extra keys, you can subclass:
class ChainState(MessagesState, total=False):
    # Add any extra keys you might want to carry around
    pass

# -----------------------------
# Model + Tool(s)
# -----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

llm_with_tools = llm.bind_tools([multiply])

# -----------------------------
# Nodes
# -----------------------------
def tool_calling_llm(state: ChainState) -> dict:
    """Single-node chain that lets the LLM (optionally) call tools.
    Because we're using MessagesState, we must return a *list* of new messages.
    The add_messages reducer will append them to state['messages'].
    """
    response_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [response_msg]}

# (Optional) You can add more nodes later, e.g., a router for tool execution,
# a post-processor, or a condition to branch on tool calls.

# -----------------------------
# Build & compile the graph
# -----------------------------
builder = StateGraph(ChainState)
builder.add_node("tool_calling_llm", tool_calling_llm)

builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()

# -----------------------------
# Run (only when executing this file directly)
# -----------------------------
if __name__ == "__main__":
    # Optional visualization (won't error out if graphviz isn't available)
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    # --- Demo 1: small talk (no tool call expected)
    print("\n--- Demo 1: No tool call ---")
    out1 = graph.invoke({"messages": [HumanMessage(content="Hello there!")]})
    for m in out1["messages"]:
        m.pretty_print()

    # --- Demo 2: triggers the tool call
    print("\n--- Demo 2: Tool call (multiply) ---")
    out2 = graph.invoke({"messages": [HumanMessage(content="Multiply 2 and 3")]})
    for m in out2["messages"]:
        m.pretty_print()
