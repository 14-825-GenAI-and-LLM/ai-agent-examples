"""
  This is a Tool-calling agent implemented using LangGraph. 
  It uses a Google Gemini model to answer user queries about math and finance by calling a Python execution tool when needed.
  pip install -U langgraph langchain langchain-google-genai langchain-community langchain-experimental
"""
from __future__ import annotations

from typing import Any, Dict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Tooling / graph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

# Built-in Python execution tool (commonly used with LangGraph tool calling)
from langchain_experimental.tools import PythonREPLTool


os.environ["GOOGLE_API_KEY"] = "" # Update this with your Google AI Studio API Key

# ----------------------------
# 1) State: messages (LangGraph standard)
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ----------------------------
# 2) Built-in tool: execute_python
# ----------------------------
# PythonREPLTool exposes a tool callable by the LLM. We'll present it as "execute_python".
python_tool = PythonREPLTool()
python_tool.name = "execute_python"
python_tool.description = (
    "Execute Python code for calculations. "
    "Input should be a Python expression or small snippet that prints or returns a numeric result."
)

tools = [python_tool]


# ----------------------------
# 3) LLM (tool-calling capable)
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.0,
).bind_tools(tools)

parser = StrOutputParser()


# ----------------------------
# 4) Prompt with required #Key: value structure
# ----------------------------
SYSTEM_PROMPT = """#Role: You are a precise calculator and interpreter agent.
#Task: Answer user math/finance questions by calling tools when needed.
#Topic: Numeric reasoning and financial calculations (e.g., compound interest).
#Format: Provide a short final answer in plain English, including the final numeric result.
#Tone / Style: Clear, accurate, and professional.
#Context: You can call tools to compute exact results; do not do mental math if a tool is available.
#Goal: Produce correct calculations and explain results briefly.
#Requirements / Constraints:
If computation is needed, call the tool execute_python with correct Python code.
Use the observation (tool output) as the source of truth.
Round money to 2 decimal places unless the user asks otherwise.
Do not show raw code in the final answer unless the user asks.
"""


# ----------------------------
# 5) Nodes
# ----------------------------
def agent_llm_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM node that either:
    - returns a tool call, or
    - returns a final natural-language answer
    """
    # Ensure system prompt is present at the beginning of the conversation.
    msgs = state["messages"]
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

    resp = llm.invoke(msgs)
    return {"messages": [resp]}


tool_node = ToolNode(tools)


# ----------------------------
# 6) Build graph (matches the diagram)
#    User -> LLM -> (maybe tool) -> LLM -> Response
# ----------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("llm", agent_llm_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("llm")

    # If LLM requests a tool call, route to tools; otherwise end.
    g.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})

    # After tools run, send the observation back to the LLM to generate final answer.
    g.add_edge("tools", "llm")

    return g.compile()


graph = build_graph()

# ----------------------------
# 7) Public API
# ----------------------------
def run_calculator_agent(user_query: str) -> str:
    result = graph.invoke({"messages": [{"role": "user", "content": user_query}]})

    # The last AI message should be the final answer (after tool calls, if any).
    final_msg = result["messages"][-1]
    return final_msg.content


if __name__ == "__main__":
    # Example message prompt in langsmith
    '''
    [{"role": "user", "content": "Calculate the compound interest on $5,000 at 7% for 12 years."}]
    '''
    # Example if you wanna run the code as a python file
    q = 'Calculate the compound interest on $5,000 at 7% for 12 years.'
    print(run_calculator_agent(q))
