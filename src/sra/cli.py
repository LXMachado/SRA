from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4
from typing import Annotated

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openai import AuthenticationError, NotFoundError, RateLimitError

from .config import Settings
from .graph import build_workflow
from .state import AgentState

app = typer.Typer(help="Sentinel Research Agent CLI")


def _execute(query: str, max_iters: int) -> None:
    """Execute the Sentinel Research Agent for the provided query."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    settings = Settings.from_env()
    workflow = build_workflow(settings)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "research_query": "",
        "search_results": [],
        "num_results": 5,
        "freshness": None,
        "iterations": 0,
        "max_iters": max_iters,
        "status": "CONTINUE",
        "final_report": None,
    }
    # Planner -> search -> analyzer -> reporter can exceed max_iters*2 for small values.
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": max(8, max_iters * 4),
    }
    try:
        final_state = workflow.invoke(initial_state, config=config)
    except AuthenticationError as exc:
        typer.echo(
            "OpenRouter authentication failed (401). "
            "Verify OPENROUTER_API_KEY is active and belongs to your OpenRouter account."
        )
        raise typer.Exit(code=1) from exc
    except NotFoundError as exc:
        typer.echo(
            "Configured model is unavailable on OpenRouter (404). "
            "Set OPENROUTER_MODEL to a currently available concrete model id "
            "(example: google/gemini-2.5-flash)."
        )
        raise typer.Exit(code=1) from exc
    except RateLimitError as exc:
        typer.echo(
            "OpenRouter/provider rate limit hit (429). "
            "Retry shortly, switch to a non-free model, or use BYOK in OpenRouter."
        )
        raise typer.Exit(code=1) from exc
    report = final_state.get("final_report")
    if report is None:
        typer.echo("Run completed without a report. Check logs.")
        raise typer.Exit(code=1)

    typer.echo(json.dumps(report.model_dump(), indent=2))


@app.command()
def run(
    query_parts: Annotated[
        list[str], typer.Argument(help='Use either: sra "query" or sra run "query".')
    ],
    max_iters: int = typer.Option(
        4, "--max-iters", min=1, help="Maximum planner/search loops."
    ),
) -> None:
    """Execute the Sentinel Research Agent for the provided query."""

    if not query_parts:
        raise typer.BadParameter("QUERY is required.")

    if query_parts[0].lower() == "run":
        query_parts = query_parts[1:]
        if not query_parts:
            raise typer.BadParameter("QUERY is required after 'run'.")

    query = " ".join(query_parts).strip()
    if not query:
        raise typer.BadParameter("QUERY is required.")
    _execute(query, max_iters)
