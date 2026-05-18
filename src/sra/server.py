from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openai import AuthenticationError, NotFoundError, RateLimitError
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .graph import build_workflow
from .schemas import FinalReport
from .state import AgentState

app = FastAPI(title="Sentinel Research Agent UI")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research query to investigate")
    max_iters: int = Field(4, ge=1, le=20, description="Maximum planner/search loops")


class RunResponse(BaseModel):
    topic: str
    executive_summary: str
    sections: list[dict]
    sources: list[dict]


@app.get("/", response_class=HTMLResponse)
def read_index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.post("/api/run", response_model=RunResponse)
def run_research(request: QueryRequest):
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        workflow = build_workflow(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build workflow: {e}")

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.query)],
        "research_query": "",
        "search_results": [],
        "num_results": 5,
        "freshness": None,
        "iterations": 0,
        "max_iters": request.max_iters,
        "status": "CONTINUE",
        "final_report": None,
    }

    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": max(8, request.max_iters * 4),
    }

    try:
        final_state = workflow.invoke(initial_state, config=config)
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Provider authentication failed. Verify LLM_API_KEY is valid.",
        )
    except NotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Configured model is unavailable. Set LLM_MODEL to a valid model id.",
        )
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Provider rate limit hit. Retry shortly or switch to a non-free model.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}")

    report = final_state.get("final_report")
    if report is None:
        raise HTTPException(status_code=500, detail="Run completed without a report.")

    return report.model_dump()