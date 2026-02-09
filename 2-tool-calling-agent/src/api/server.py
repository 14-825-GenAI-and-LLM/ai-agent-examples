from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from agent.graph import build_graph  # <-- your function

app = FastAPI(title="LangGraph Stateless API", version="0.1.0")

# Build once at startup (shared across requests)
GRAPH = build_graph()


@app.get("/ok")
def ok() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(payload: Dict[str, Any]) -> JSONResponse:
    """
    Stateless invoke.
    payload example:
      {
        "input": {"question": "What is the status of the project?"},
        "config": {"configurable": {"thread_id": "optional"}}
      }
    """
    try:
        user_input = payload.get("input", payload)  # allow passing raw input
        config = payload.get("config")

        if config is None:
            result = await GRAPH.ainvoke(user_input)
        else:
            result = await GRAPH.ainvoke(user_input, config=config)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream(payload: Dict[str, Any]):
    """
    SSE streaming (optional).
    """
    user_input = payload.get("input", payload)
    config = payload.get("config")

    async def event_gen():
        try:
            # Use astream if available on your compiled graph
            if config is None:
                async for chunk in GRAPH.astream(user_input):
                    yield {"event": "chunk", "data": chunk}
            else:
                async for chunk in GRAPH.astream(user_input, config=config):
                    yield {"event": "chunk", "data": chunk}

            yield {"event": "done", "data": {"status": "done"}}
        except Exception as e:
            yield {"event": "error", "data": {"error": str(e)}}

    return EventSourceResponse(event_gen())
