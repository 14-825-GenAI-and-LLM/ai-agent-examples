"""
  This is a Custom Tool-calling agent implemented using LangGraph. 
  The custom tool:
  -- takes a raw bullet + target role
  -- rewrites it using strong action verbs + impact

  returns a quick quality score + suggested variants  
  
  pip install -U langgraph langchain langchain-google-genai langchain-community langchain-experimental
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, TypedDict, Annotated, List

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

os.environ["GOOGLE_API_KEY"] = "" # Update this with your Google AI Studio API Key


# =============================================================================
# 1) Custom Tool 
# =============================================================================

ACTION_VERBS = [
    "Led", "Built", "Designed", "Implemented", "Optimized", "Automated",
    "Improved", "Reduced", "Accelerated", "Delivered", "Scaled", "Streamlined",
    "Developed", "Launched", "Spearheaded", "Refactored", "Integrated",
]

WEAK_PHRASES = [
    "responsible for", "worked on", "helped with", "did", "tasked with", "involved in"
]

def _score_bullet(bullet: str) -> Dict[str, Any]:
    """
    Simple heuristic scoring (0-10):
    + action verb start
    + numbers/metrics
    + clarity (short-ish)
    - weak phrases
    """
    b = bullet.strip()
    score = 5

    # starts with action verb?
    if any(b.startswith(v + " ") for v in ACTION_VERBS):
        score += 2

    # contains a number/percent?
    if re.search(r"\b\d+(\.\d+)?%?\b", b):
        score += 2

    # too long?
    if len(b) > 180:
        score -= 1

    # weak phrases penalty
    low = b.lower()
    if any(p in low for p in WEAK_PHRASES):
        score -= 2

    score = max(0, min(10, score))
    issues = []
    if score <= 5:
        if not any(b.startswith(v + " ") for v in ACTION_VERBS):
            issues.append("Doesn't start with a strong action verb.")
        if not re.search(r"\b\d+(\.\d+)?%?\b", b):
            issues.append("No measurable impact (numbers/metrics).")
        if any(p in low for p in WEAK_PHRASES):
            issues.append("Contains weak phrasing (e.g., 'worked on', 'helped with').")
    return {"score": score, "issues": issues}


@tool("resume_bullet_tool")
def resume_bullet_tool(bullet: str, target_role: str = "Software Engineer") -> Dict[str, Any]:
    """
    Rewrite a resume bullet to be stronger and more impact-oriented.
    Returns rewritten variants + a heuristic score for the best variant.
    """
    original = bullet.strip()

    # lightweight, deterministic rewrite logic
    verb = "Improved"
    for v in ACTION_VERBS:
        # pick a verb that semantically fits a bit by keyword
        if "deploy" in original.lower() and v in ["Implemented", "Delivered", "Scaled"]:
            verb = v
            break
        if "optimiz" in original.lower() and v == "Optimized":
            verb = v
            break
        if "automat" in original.lower() and v == "Automated":
            verb = v
            break

    # remove weak phrases
    cleaned = original
    for p in WEAK_PHRASES:
        cleaned = re.sub(re.escape(p), "", cleaned, flags=re.IGNORECASE).strip()

    # If no metric, add a placeholder prompt
    metric_hint = ""
    if not re.search(r"\b\d+(\.\d+)?%?\b", cleaned):
        metric_hint = " (add metric: e.g., reduced latency by X% / saved Y hrs/week)"

    variants = [
        f"{verb} {cleaned}{metric_hint}",
        f"{verb} {cleaned} for {target_role}{metric_hint}",
        f"{verb} {cleaned}; contributed to measurable outcomes{metric_hint}",
    ]

    # choose best by heuristic scoring
    scored = [(v, _score_bullet(v)["score"]) for v in variants]
    best_variant = max(scored, key=lambda x: x[1])[0]
    best_score_info = _score_bullet(best_variant)

    return {
        "original": original,
        "best_variant": best_variant,
        "variants": variants,
        "score": best_score_info["score"],
        "issues": best_score_info["issues"],
    }


# =============================================================================
# 2) LangGraph Agent (tool-calling pattern)
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """#Role: You are a resume writing assistant.
#Task: Improve resume bullet points using the resume_bullet_tool when rewriting is needed.
#Topic: Resume bullets (impact, clarity, metrics).
#Format: Provide:
1) Improved bullet (best)
2) 2 alternative variants
3) Quick score (0-10) and what to improve next
#Tone / Style: Professional, confident, and concise.
#Context: You have access to resume_bullet_tool(bullet, target_role) which returns variants and a score.
#Goal: Produce stronger, metric-oriented bullets aligned with the target role.
#Requirements / Constraints:
Use resume_bullet_tool for rewriting or scoring bullets.
Do not fabricate metrics; if missing, ask the user to supply numbers or include placeholders.
Keep bullets to ~1-2 lines.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.2,
).bind_tools([resume_bullet_tool])

tool_node = ToolNode([resume_bullet_tool])


def llm_node(state: AgentState) -> Dict[str, Any]:
    msgs = state["messages"]
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

    resp = llm.invoke(msgs)
    return {"messages": [resp]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("llm")
    g.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()


graph = build_graph()

def run_agent(user_query: str) -> str:
    out = graph.invoke({"messages": [{"role": "user", "content": user_query}]})
    return out["messages"][-1].content


if __name__ == "__main__":
    # Example message prompt in langsmith
    '''
    [{"role": "user", "content": "Rewrite this resume bullet for a Data Engineer role: 'worked on pipelines and data stuff for the team'"}]
    '''
    # Example if you wanna run the code as a python file
    q = "Rewrite this resume bullet for a Data Engineer role: 'worked on pipelines and data stuff for the team'"
    print(run_agent(q))
