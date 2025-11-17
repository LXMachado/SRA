from __future__ import annotations

import json
from uuid import uuid4

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .config import Settings
from .graph import build_workflow
from .state import AgentState

app = typer.Typer(help="Sentinel Research Agent CLI")


@app.command()
def run(query: str, max_iters: int = typer.Option(4, "--max-iters", min=1, help="Maximum planner/search loops.")):
    """Execute the Sentinel Research Agent for the provided query."""

    load_dotenv()
    settings = Settings.from_env()
    workflow = build_workflow(settings)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "research_query": "",
        "search_results": [],
        "status": "CONTINUE",
        "final_report": None,
    }
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": max_iters * 2}
    final_state = workflow.invoke(initial_state, config=config)
    report = final_state.get("final_report")
    if report is None:
        typer.echo("Run completed without a report. Check logs.")
        raise typer.Exit(code=1)

    typer.echo(json.dumps(report.model_dump(), indent=2))
