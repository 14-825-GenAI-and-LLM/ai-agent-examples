"""
Graph does:
-- call tool to fetch/parse tweets
-- answer question from tweets
-- reflect
-- revise
  
  pip install -U langgraph langchain langchain-google-genai langchain-community langchain-experimental beautifulsoup4
"""

from __future__ import annotations

import re
from typing import List, TypedDict, Dict, Any
import os
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

os.environ["GOOGLE_API_KEY"] = "" # Update this with your Google AI Studio API Key

# ----------------------------
# 1) Tool: tweet_parser
# ----------------------------
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TweetQA/1.0)"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_profile_html(viewer_base_url: str, handle: str) -> str:
    candidates = [
        f"{viewer_base_url.rstrip('/')}/{handle.lstrip('@')}",
        f"{viewer_base_url.rstrip('/')}/profile/{handle.lstrip('@')}",
        f"{viewer_base_url.rstrip('/')}/twitter-profile-viewer?user={handle.lstrip('@')}",
    ]
    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fetch profile page for @{handle}. Last error: {last_err}")

def _extract_tweets(html: str, limit: int = 50) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and len(ln) >= 20]

    blacklist = ["Privacy", "Terms", "Cookie", "Log in", "Sign up", "Download", "Search"]
    def boilerplate(s: str) -> bool:
        low = s.lower()
        return any(b.lower() in low for b in blacklist)

    candidates = [ln for ln in lines if not boilerplate(ln)]
    tweetish = []
    for ln in candidates:
        if len(ln) > 400:
            continue
        if sum(ch.isalpha() for ch in ln) < 10:
            continue
        tweetish.append(ln)

    # De-dupe, preserve order
    seen, out = set(), []
    for t in tweetish:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:limit]


@tool("tweet_parser")
def tweet_parser(handle: str, viewer_base_url: str = "https://twitter-viewer.com", max_tweets: int = 50) -> Dict[str, Any]:
    """
    Fetch and parse recent public tweet text for a given handle from a viewer site.
    Returns {"handle": ..., "viewer_base_url": ..., "tweets": [...]}.
    """
    html = _fetch_profile_html(viewer_base_url, handle)
    tweets = _extract_tweets(html, limit=max_tweets)
    return {"handle": handle, "viewer_base_url": viewer_base_url, "tweets": tweets}


# ----------------------------
# 2) State
# ----------------------------
class AgentState(TypedDict, total=False):
    handle: str
    viewer_base_url: str
    question: str
    tweets: List[str]
    selected_tweets: List[str]
    draft_answer: str
    reflection: str
    final_answer: str


# ----------------------------
# 3) Gemini model
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
parser = StrOutputParser()


# ----------------------------
# 4) Prompts (#Key format)
# ----------------------------
select_prompt = PromptTemplate.from_template(
"""#Role: You are a relevance filtering assistant for social media content.
#Task: Select the best tweet snippets to answer a question.
#Topic: Answering a user question using recent tweets from a public profile feed.
#Format: Return ONLY a comma-separated list of tweet indices (0-based), up to {k} indices. No extra text.
#Tone / Style: Precise and minimal.
#Context: You are given a question and a numbered list of tweets.
#Goal: Pick the tweets that contain direct evidence for answering.
#Requirements / Constraints:
Use only the provided tweets.
Prefer tweets that directly mention the asked details.
If nothing is relevant, return an empty string.
Question: {question}

Tweets:
{tweets_block}
"""
)

answer_prompt = PromptTemplate.from_template(
"""#Role: You are an evidence-grounded analyst.
#Task: Answer the question using only the provided tweets.
#Topic: Q&A over a public tweet feed.
#Format: A concise answer with evidence citations as [T#]. Include short quoted snippets when helpful.
#Tone / Style: Clear, factual, and cautious.
#Context: The tweets may be incomplete or noisy due to viewer-site extraction.
#Goal: Provide the best possible answer grounded in tweet evidence.
#Requirements / Constraints:
Do not invent facts not present in the tweets.
If evidence is insufficient, explicitly say what is missing.
Cite the tweets you used as [T#].
Question: {question}

Relevant tweets:
{tweets_block}
"""
)

reflection_prompt = PromptTemplate.from_template(
"""#Role: You are a strict reflection agent and quality reviewer.
#Task: Critique the draft answer and propose concrete improvements.
#Topic: Improving an evidence-grounded answer based on tweet evidence.
#Format: Bullet points under two sections: "Issues" and "Fixes".
#Tone / Style: Direct, rigorous, and specific.
#Context: The draft may have weak grounding, unclear structure, or missing caveats.
#Goal: Improve correctness, grounding, clarity, and usefulness.
#Requirements / Constraints:
Check grounding: every claim must be supported by a tweet or clearly marked as unknown.
Flag any hallucinations or overreach.
Suggest how to improve citations [T#] and phrasing.
Question: {question}

Tweets used:
{tweets_block}

Draft answer:
{draft_answer}
"""
)

revise_prompt = PromptTemplate.from_template(
"""#Role: You are an expert editor for evidence-grounded answers.
#Task: Rewrite the answer applying the reflection feedback.
#Topic: Producing a higher-quality tweet-grounded answer.
#Format: Final answer only, with citations [T#] and short quoted snippets where useful.
#Tone / Style: Polished, confident, and accurate.
#Context: The reflection lists issues and fixes.
#Goal: Produce the best possible grounded answer.
#Requirements / Constraints:
Remain strictly grounded in the provided tweets.
If evidence is insufficient, state limitations explicitly.
Keep the answer concise.
Question: {question}

Tweets used:
{tweets_block}

Reflection:
{reflection}

Draft answer:
{draft_answer}
"""
)


# ----------------------------
# 5) Graph nodes
# ----------------------------
def node_get_tweets(state: AgentState) -> AgentState:
    # Call the custom tool (direct python call here; you can also run it via ToolNode)
    result = tweet_parser.invoke({
        "handle": state["handle"],
        "viewer_base_url": state.get("viewer_base_url", "https://twitter-viewer.com"),
        "max_tweets": 50,
    })
    state["tweets"] = result.get("tweets", [])
    return state

def node_select(state: AgentState) -> AgentState:
    tweets = state.get("tweets", [])
    if not tweets:
        state["selected_tweets"] = []
        return state

    tweets_block = "\n".join([f"[T{i}] {t}" for i, t in enumerate(tweets[:60])])
    raw = (select_prompt | llm | parser).invoke({
        "question": state["question"],
        "tweets_block": tweets_block,
        "k": 10,
    })
    idxs = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if part.isdigit():
            idxs.append(int(part))
    idxs = [i for i in idxs if 0 <= i < len(tweets)]
    uniq, seen = [], set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    state["selected_tweets"] = [tweets[i] for i in uniq[:10]]
    return state

def node_draft(state: AgentState) -> AgentState:
    sel = state.get("selected_tweets", [])
    tweets_block = "\n".join([f"[T{i}] {t}" for i, t in enumerate(sel)])
    state["draft_answer"] = (answer_prompt | llm | parser).invoke({
        "question": state["question"],
        "tweets_block": tweets_block,
    })
    return state

def node_reflect(state: AgentState) -> AgentState:
    sel = state.get("selected_tweets", [])
    tweets_block = "\n".join([f"[T{i}] {t}" for i, t in enumerate(sel)])
    state["reflection"] = (reflection_prompt | llm | parser).invoke({
        "question": state["question"],
        "tweets_block": tweets_block,
        "draft_answer": state["draft_answer"],
    })
    return state

def node_revise(state: AgentState) -> AgentState:
    sel = state.get("selected_tweets", [])
    tweets_block = "\n".join([f"[T{i}] {t}" for i, t in enumerate(sel)])
    state["final_answer"] = (revise_prompt | llm | parser).invoke({
        "question": state["question"],
        "tweets_block": tweets_block,
        "reflection": state["reflection"],
        "draft_answer": state["draft_answer"],
    })
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("get_tweets", node_get_tweets)
    g.add_node("select", node_select)
    g.add_node("draft", node_draft)
    g.add_node("reflect", node_reflect)
    g.add_node("revise", node_revise)

    g.set_entry_point("get_tweets")
    g.add_edge("get_tweets", "select")
    g.add_edge("select", "draft")
    g.add_edge("draft", "reflect")
    g.add_edge("reflect", "revise")
    g.add_edge("revise", END)
    return g.compile()


graph = build_graph()

def answer_from_tweets(handle: str, question: str, viewer_base_url: str = "https://twitter-viewer.com") -> str:
    out = graph.invoke({"handle": handle, "question": question, "viewer_base_url": viewer_base_url})
    return out.get("final_answer", out.get("draft_answer", "No answer produced."))


if __name__ == "__main__":
    print(answer_from_tweets("NASA", "What are they posting about lately?"))
