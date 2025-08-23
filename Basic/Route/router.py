# -*- coding: utf-8 -*-
"""
Router-style graph in the same pattern as your simple StateGraph demo:
- explicit nodes
- START/END edges
- builder.compile()
- __main__ guard with optional visualization + demo invocations

Behavior:
- The LLM can either answer directly OR emit a tool call.
- If it emits a tool call, control routes to the ToolNode to execute it, then END.
"""

from typing_extensions import TypedDict
import os, getpass

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

# -----------------------------
# Config (optional helper)
# -----------------------------
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# -----------------------------
# MessagesState (prebuilt)
# -----------------------------
class ChainState(MessagesState, total=False):
    # Extend with extra keys if needed later
    pass

# -----------------------------
# Model + Tool(s)
# -----------------------------
# Feel free to switch models; this one is strong and widely available
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# Bind the tool so the LLM can decide to call it
llm_with_tools = llm.bind_tools([multiply])

# -----------------------------
# Nodes
# -----------------------------
def tool_calling_llm(state: ChainState) -> dict:
    """LLM that may return a direct response or a tool call.
    Because we're using MessagesState, we must return a list of new messages.
    """
    response_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [response_msg]}

# Tool execution node: uses built-in ToolNode to execute any pending tool calls
tools_node = ToolNode([multiply])

# -----------------------------
# Build & compile the graph
# -----------------------------
builder = StateGraph(ChainState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tools_node)

builder.add_edge(START, "tool_calling_llm")

# Conditional edge:
# - If latest assistant message contains a tool call -> route to "tools"
# - Otherwise -> END
builder.add_conditional_edges("tool_calling_llm", tools_condition)

# After running tools, finish. (You could add another LLM node to "observe" tool results.)
builder.add_edge("tools", END)

graph = builder.compile()

# -----------------------------
# Run (only when executing this file directly)
# -----------------------------
if __name__ == "__main__":
    # Optional visualization (won't error if graphviz isn't available)
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
    out2 = graph.invoke({"messages": [HumanMessage(content="Multiply 2 and 3")]} )
    for m in out2["messages"]:
        m.pretty_print()
