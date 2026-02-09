'''
 This is a ReAct 'Recursive Researcher' agent in LangGraph.
 For an example query like "Compare the 2025 GDP of the country with the most Olympic gold medals in 2024 to the GDP of the runner-up."
 
 It will follow a reasoning process like:
-- Reason it needs the 2024 medal table (gold ranking)
-- Act (search web) to find winner + runner-up
-- Reason it now needs 2025 GDP for those countries
-- Act again (search web) to fetch GDP values
-- Generate a final comparison answer with citations from tool observations
 
  Project dependencies (install with pip):
    pip install -U langgraph langchain langchain-google-genai langchain-community langchain-experimental duckduckgo-search requests beautifulsoup4 ddgs

'''
from __future__ import annotations

import re
import json
import textwrap
from typing import Any, Dict, List, TypedDict, Annotated, Optional
import os
import requests
from bs4 import BeautifulSoup

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchResults

os.environ["GOOGLE_API_KEY"] = "" # Update this with your Google AI Studio API Key


# =============================================================================
# 1) Tools (Action -> Environment -> Result)
#    Free / no API keys: DuckDuckGo search + URL fetch
# =============================================================================

ddg = DuckDuckGoSearchResults()

@tool("search_web")
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo). Returns a short JSON string of results."""
    results = ddg.invoke({"query": query, "max_results": max_results})
    # results is typically a list[dict] with title/snippet/link
    return json.dumps(results, ensure_ascii=False)

@tool("fetch_url_text")
def fetch_url_text(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return readable page text (best-effort)."""
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return txt[:max_chars]


TOOLS = [search_web, fetch_url_text]
tool_node = ToolNode(TOOLS)


# =============================================================================
# 2) State (messages-driven ReAct loop)
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# 3) Prompt template structure (#Key: value)
# =============================================================================

REASON_SYSTEM_PROMPT = """#Role: You are a ReAct recursive researcher (multi-hop).
#Task: Answer the user's question by reasoning and using tools iteratively.
#Topic: Multi-step web research and comparison.
#Format: You MUST follow this loop:
- Decide what you still need
- Call the appropriate tool(s)
- Use observations to refine the next step
Stop only when you have enough evidence to answer.
#Tone / Style: Analytical, careful, and explicit about sources.
#Context: Tools available:
1) search_web(query, max_results)
2) fetch_url_text(url, max_chars)
#Goal: Resolve multi-hop questions by decomposing them into sub-questions.
#Requirements / Constraints:
Do not guess facts that should be looked up.
Use tools for any factual claim that depends on current data.
If search results conflict, call fetch_url_text on high-quality sources to verify.
When ready to answer, do NOT answer here; instead, output a final plan-free response in the "Generate" step.
"""

GENERATE_SYSTEM_PROMPT = """#Role: You are a professional research writer.
#Task: Produce the final answer using ONLY the tool observations in the conversation.
#Topic: Comparing two countries’ 2025 GDP given 2024 Olympic gold medal ranking.
#Format: A concise comparison with:
- Winner country and runner-up country (from medal table)
- 2025 GDP figures for both (with source snippets)
- Clear numeric comparison (difference and/or ratio if possible)
#Tone / Style: Clear, grounded, and citation-forward.
#Context: The conversation contains tool outputs (observations) from search_web/fetch_url_text.
#Goal: Provide a trustworthy final response.
#Requirements / Constraints:
Do not introduce new facts without tool evidence.
If 2025 GDP is a projection/estimate, say so explicitly.
Include inline source pointers like [Obs1], [Obs2] referencing which observation supports the claim.
"""


# =============================================================================
# 4) Two LLM nodes to mirror the image:
#    - LLM (Reason): tool-calling enabled, loops with tools
#    - LLM (Generate): no tool-calling, final response
# =============================================================================

llm_reason = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1,
).bind_tools(TOOLS)

llm_generate = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

def reason_node(state: AgentState) -> Dict[str, Any]:
    msgs = state["messages"]
    # Ensure system prompt is present
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=REASON_SYSTEM_PROMPT)] + msgs
    resp = llm_reason.invoke(msgs)
    return {"messages": [resp]}

def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Final generation step: convert the tool observations into a final answer.
    """
    msgs = state["messages"]
    # Prepend generate-system prompt (separate role)
    gen_msgs = [SystemMessage(content=GENERATE_SYSTEM_PROMPT)] + msgs
    resp = llm_generate.invoke(gen_msgs)
    return {"messages": [resp]}


# =============================================================================
# 5) Graph (ReAct loop + final generate)
#    Pattern:
#      User -> LLM(Reason) -> Tools -> LLM(Reason) -> ... -> LLM(Generate) -> END
# =============================================================================

def should_finish(state: AgentState) -> str:
    """
    If the last AI message contains tool calls, go to tools.
    If not, we assume reasoning is done and we go to generate.
    (A common simple ReAct stop rule.)
    """
    return tools_condition(state)  # returns "tools" or END

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("reason", reason_node)      # LLM (Reason)
    g.add_node("tools", tool_node)         # Tools (Action)
    g.add_node("generate", generate_node)  # LLM (Generate)

    g.set_entry_point("reason")

    # Route based on whether the reason-LLM requested tools
    g.add_conditional_edges(
        "reason",
        should_finish,
        {
            "tools": "tools",
            END: "generate",  # if no tool calls, go generate final response
        },
    )

    # After tools run, loop back to reasoning
    g.add_edge("tools", "reason")

    # Generate -> END
    g.add_edge("generate", END)

    return g.compile()


graph = build_graph()

# =============================================================================
# 6) Runner
# =============================================================================


def run_recursive_researcher(query: str) -> str:
    result = graph.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    q = "Compare the 2025 GDP of the country with the most Olympic gold medals in 2024 to the GDP of the runner-up."
    print(run_recursive_researcher(q))
