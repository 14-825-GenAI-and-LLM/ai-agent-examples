"""
This is a "Self-Correction Coder" using LangGraph. This agent uses the reflection pattern:
- It generates code + brief explanation + tests (optional)
- It reflects strictly on correctness, edge cases, runtime, and adherence to requirements
- It revises until PASS or max_iterations reached

Uses TWO different Gemini models:
- Generator: gemini-3-flash-preview
- Reflector: gemini-2.5-flash

Before using, install the required packages in your virtual environment:
  pip install -U langgraph langchain langchain-google-genai
"""
# The below annotation will help you use pipes in the code
from __future__ import annotations

from typing import TypedDict, Optional
import re
import os
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["GOOGLE_API_KEY"] = "" # Update this with your Google AI Studio API Key


# ----------------------------
# 1) State
# ----------------------------
class AgentState(TypedDict, total=False):
    query: str
    initial_output: str
    reflection: str
    revised_output: str
    iteration: int
    max_iterations: int
    status: str  # "PASS" or "FAIL"


# ----------------------------
# 2) Two different Gemini models
# ----------------------------
# Generator model (fast code drafting)
llm_generate = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.2,
)

# Reflector model (stricter critique)
llm_reflect = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
)

parser = StrOutputParser()


# ----------------------------
# 3) Prompts using the required #Key: value structure
# ----------------------------

# LLM (Generate)
generate_prompt = PromptTemplate.from_template(
"""#Role: You are a senior software engineer and careful code writer.
#Task: Produce a correct, runnable solution (code) to the user's request.
#Topic: Self-correction coding with iterative refinement.
#Format: Return a single markdown answer with these sections in order:
1) "Assumptions" (bullet list, short)
2) "Solution" (a single fenced code block with the full code)
3) "How to run" (2-5 bullets)
4) "Notes" (brief; include complexity and edge cases)
#Tone / Style: Clear, practical, and professional.
#Context: This output is the initial draft and may be reviewed by a strict reflection agent.
#Goal: Provide the best first-pass implementation that satisfies the requirements.
#Requirements / Constraints:
Write code that is complete and runnable.
Prefer Python unless the user explicitly requests another language.
Include reasonable error handling and input validation.
Do not invent external files or dependencies unless necessary; if needed, list them explicitly.
User request:
{query}
"""
)

# LLM (Reflect)
reflection_prompt = PromptTemplate.from_template(
"""#Role: You are a strict code reviewer and reflection agent.
#Task: Critique the draft output for correctness and completeness, then decide PASS/FAIL.
#Topic: Self-correction of generated code.
#Format: Return exactly the following structure (no extra sections):
#Issues:
- (bullets)
#Fixes:
- (bullets)
#Verdict: PASS or FAIL
#Tone / Style: Direct, rigorous, specific.
#Context: You must verify the draft matches the user request and is runnable.
#Goal: Detect bugs, missing requirements, unclear steps, or unsafe assumptions.
#Requirements / Constraints:
If any requirement is unmet, Verdict must be FAIL.
Focus on: correctness, edge cases, missing imports, broken logic, unclear run steps, dependency issues.
Do not propose vague fixes; each fix should be actionable.
User request:
{query}

Draft output:
{initial_output}
"""
)

# LLM (Revise)
revise_prompt = PromptTemplate.from_template(
"""#Role: You are a senior software engineer performing self-correction.
#Task: Revise the draft to address the reflection issues and produce a corrected final solution.
#Topic: Iterative improvement of code based on review feedback.
#Format: Return a single markdown answer with these sections in order:
1) "Assumptions" (bullet list, short)
2) "Solution" (a single fenced code block with the full code)
3) "How to run" (2-5 bullets)
4) "Notes" (brief; include complexity and edge cases)
#Tone / Style: Clear, practical, and professional.
#Context: You are revising based on a strict code review.
#Goal: Produce a corrected, runnable solution that passes all stated requirements.
#Requirements / Constraints:
Fix every issue listed in #Issues if applicable, or explain why it is not actually an issue.
Do not add unrelated features.
Ensure the code is runnable and the run steps are accurate.
User request:
{query}

Reflection:
{reflection}

Previous draft:
{initial_output}
"""
)


# ----------------------------
# 4) Nodes
# ----------------------------
def generate_node(state: AgentState) -> AgentState:
    out = (generate_prompt | llm_generate | parser).invoke({"query": state["query"]})
    state["initial_output"] = out
    return state


def reflect_node(state: AgentState) -> AgentState:
    refl = (reflection_prompt | llm_reflect | parser).invoke(
        {"query": state["query"], "initial_output": state["initial_output"]}
    )
    state["reflection"] = refl

    # Extract PASS/FAIL robustly
    verdict_match = re.search(r"#Verdict:\s*(PASS|FAIL)", refl, re.IGNORECASE)
    state["status"] = verdict_match.group(1).upper() if verdict_match else "FAIL"
    return state


def revise_node(state: AgentState) -> AgentState:
    revised = (revise_prompt | llm_generate | parser).invoke(
        {
            "query": state["query"],
            "reflection": state["reflection"],
            "initial_output": state["initial_output"],
        }
    )
    state["revised_output"] = revised

    # Move revised -> initial for next loop (so reflect reviews the latest)
    state["initial_output"] = revised
    return state


def iterate_node(state: AgentState) -> AgentState:
    state["iteration"] = int(state.get("iteration", 0)) + 1
    return state


def should_continue(state: AgentState) -> str:
    # Stop if PASS or max iterations reached; otherwise continue looping
    if state.get("status") == "PASS":
        return "stop"
    if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 2)):
        return "stop"
    return "continue"


# ----------------------------
# 5) Build the LangGraph workflow (matches the image loop)
# ----------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("generate", generate_node)   # LLM (Generate) -> Initial output
    g.add_node("reflect", reflect_node)     # LLM (Reflect) -> Reflection
    g.add_node("revise", revise_node)       # Apply fixes -> new draft
    g.add_node("iterate", iterate_node)     # Increment iteration

    g.set_entry_point("generate")
    g.add_edge("generate", "reflect")
    g.add_edge("reflect", "revise")
    g.add_edge("revise", "iterate")

    g.add_conditional_edges(
        "iterate",
        should_continue,
        {"continue": "reflect", "stop": END},
    )

    return g.compile()


graph = build_graph()

# ----------------------------
# 6) Convenience runner
# ----------------------------
def self_correction_coder(query: str, max_iterations: int = 2) -> str:
    result = graph.invoke(
        {"query": query, "iteration": 0, "max_iterations": max_iterations}
    )
    # The latest draft sits in initial_output by design
    return result.get("initial_output", "")


if __name__ == "__main__":
    # Example coding prompt
    demo_query = (
        "Write a Python function `top_k_frequent(nums, k)` that returns the k most frequent "
        "integers from a list. Requirements: O(n log k) or better, handle ties deterministically "
        "by smaller number first, and include a short usage example."
    )
    final = self_correction_coder(demo_query, max_iterations=2)
    print(final)