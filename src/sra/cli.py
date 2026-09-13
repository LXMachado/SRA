from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openai import AuthenticationError, NotFoundError, RateLimitError

from .config import Settings
from .graph import build_workflow
from .state import AgentState


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _query_from_parts(query_parts: list[str]) -> str:
    if not query_parts:
        raise ValueError("QUERY is required.")
    query = " ".join(query_parts).strip()
    if not query:
        raise ValueError("QUERY is required.")
    return query


def _run_ui(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sra ui",
        description="Start the web UI server for the Sentinel Research Agent.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the UI server.")
    parser.add_argument("--port", default=8000, type=int, help="Port for the UI server.")
    args = parser.parse_args(argv)

    import uvicorn

    from .server import app as fastapi_app

    print(f"Starting Sentinel Research Agent UI at http://{args.host}:{args.port}")
    uvicorn.run(fastapi_app, host=args.host, port=args.port)
    return 0


def _run_research_command(argv: list[str], prog: str) -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Execute the Sentinel Research Agent for the provided query.",
    )
    parser.add_argument(
        "--max-iters",
        default=4,
        type=_positive_int,
        help="Maximum planner/search loops.",
    )
    parser.add_argument("query_parts", nargs="+", metavar="QUERY")
    args = parser.parse_args(argv)

    try:
        query = _query_from_parts(args.query_parts)
    except ValueError as exc:
        parser.error(str(exc))
    return _execute(query, args.max_iters)


def _execute(query: str, max_iters: int) -> int:
    """Execute the Sentinel Research Agent for the provided query."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"LLM config: base_url={settings.llm_base_url} model={settings.llm_model}")
    print(f"Search provider: {settings.search_provider}")
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
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": max(8, max_iters * 4),
    }
    try:
        final_state = workflow.invoke(initial_state, config=config)
    except AuthenticationError:
        print(
            "Provider authentication failed (401). "
            "Verify LLM_API_KEY (or provider-specific key fallback) is valid for the configured LLM_BASE_URL.",
            file=sys.stderr,
        )
        return 1
    except NotFoundError:
        print(
            "Configured model is unavailable on the current provider (404). "
            "Set LLM_MODEL to a currently available concrete model id "
            "(example: google/gemini-2.5-flash).",
            file=sys.stderr,
        )
        return 1
    except RateLimitError:
        print(
            "Provider rate limit hit (429). "
            "Retry shortly, switch to a non-free model, or use BYOK where supported.",
            file=sys.stderr,
        )
        return 1

    report = final_state.get("final_report")
    if report is None:
        print("Run completed without a report. Check logs.", file=sys.stderr)
        return 1

    print(json.dumps(report.model_dump(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: sra [--max-iters MAX_ITERS] QUERY...\n"
            "       sra run [--max-iters MAX_ITERS] QUERY...\n"
            "       sra ui [--host HOST] [--port PORT]\n\n"
            "Sentinel Research Agent CLI"
        )
        return 0

    command = args[0].lower()
    if command == "ui":
        return _run_ui(args[1:])
    if command == "run":
        return _run_research_command(args[1:], "sra run")
    return _run_research_command(args, "sra")


app = main


if __name__ == "__main__":
    raise SystemExit(main())
