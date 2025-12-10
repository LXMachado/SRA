from __future__ import annotations

import atexit

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .config import Settings
from .llm import build_llm
from .schemas import FinalReport, SearchInput
from .state import AgentState, SearchHit
from .tools import GoogleSearchTool


_MAX_CONTEXT_HITS = 12
_MAX_SNIPPET_LEN = 500


def _dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Keep the most recent hit per URL/title key."""

    seen: dict[str, SearchHit] = {}
    for hit in reversed(hits):
        key = (hit.get("url") or hit.get("title", "")).lower()
        if key and key not in seen:
            seen[key] = hit
    return list(reversed(list(seen.values())))


def _format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return ""

    formatted = []
    for hit in _dedupe_hits(hits)[:_MAX_CONTEXT_HITS]:
        snippet = hit.get("snippet", "")
        if len(snippet) > _MAX_SNIPPET_LEN:
            snippet = snippet[: _MAX_SNIPPET_LEN - 3] + "..."
        source_id = hit.get("source_id")
        prefix = f"[{source_id}]" if source_id else "-"
        formatted.append(f"{prefix} {hit['title']} ({hit['url']}): {snippet}")
    return "\n".join(formatted)


def build_workflow(settings: Settings):
    """Create the LangGraph workflow for the Sentinel Research Agent."""

    planner_llm = build_llm(settings, temperature=0.2)
    analyzer_llm = build_llm(settings, temperature=0.1)
    reporter_llm = build_llm(settings, temperature=0.0, timeout=90)
    search_tool = GoogleSearchTool(settings)
    atexit.register(search_tool.close)

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You plan research. Produce a single focused search query that would "
                "retrieve authoritative, up-to-date information. Also choose "
                "num_results (1-10) and optional freshness (e.g., d7) to control "
                "breadth and recency. Capture your reasoning.",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    analyzer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You determine whether more searching is required. "
                "If gaps remain, propose the next search query. You may also adjust "
                "num_results (1-10) and freshness (e.g., d7) if more breadth/recency "
                "is needed.",
            ),
            ("placeholder", "{messages}"),
            (
                "human",
                "Current search evidence:\n{search_context}\n\n"
                "Should we finish or continue searching?",
            ),
        ]
    )

    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the final reporting module. Using the accumulated discussion "
                "and evidence, output a FinalReport that would satisfy analysts. "
                "Cite every factual statement with the provided source_id tokens "
                "like [S1]. Do not invent new citations or sources.",
            ),
            ("placeholder", "{messages}"),
            (
                "human",
                "Evidence you may cite:\n{search_context}\n\n"
                "Allowed source_ids: {source_ids}\n"
                "Produce the FinalReport schema now. Every claim must include at least one [Sx] marker.",
            ),
        ]
    )

    from pydantic import BaseModel, Field
    from typing import Literal, Optional

    class PlannerDecision(BaseModel):
        research_query: str = Field(
            description="The exact search query to issue next."
        )
        rationale: str = Field(description="Why this query is a good next step.")
        num_results: int = Field(
            default=5, ge=1, le=10, description="How many results to request."
        )
        freshness: Optional[str] = Field(
            default=None,
            description="Optional freshness token (e.g., d7 for last 7 days).",
        )

    class AnalyzerDecision(BaseModel):
        status: Literal["CONTINUE", "FINISH"]
        summary: str = Field(description="Summary of what is known so far.")
        gaps: list[str] = Field(
            default_factory=list,
            description="Outstanding gaps that justify more searching.",
        )
        research_query: Optional[str] = Field(
            default=None,
            description="If continuing, the next search query to issue.",
        )
        num_results: Optional[int] = Field(
            default=None, ge=1, le=10, description="Override result count if continuing."
        )
        freshness: Optional[str] = Field(
            default=None, description="Override freshness token if continuing."
        )

    planner_chain = planner_prompt | planner_llm.with_structured_output(
        PlannerDecision
    )
    analyzer_chain = analyzer_prompt | analyzer_llm.with_structured_output(
        AnalyzerDecision
    )
    reporter_chain = reporter_prompt | reporter_llm.with_structured_output(FinalReport)

    def planner_node(state: AgentState):
        decision = planner_chain.invoke({"messages": state["messages"]})
        planner_message = AIMessage(
            content=(
                f"[Planner] Query: {decision.research_query}\n"
                f"Rationale: {decision.rationale}\n"
                f"Num results: {decision.num_results}\n"
                f"Freshness: {decision.freshness or 'none'}"
            )
        )
        return {
            "messages": [planner_message],
            "research_query": decision.research_query,
            "num_results": decision.num_results,
            "freshness": decision.freshness,
            "status": "CONTINUE",
        }

    def search_node(state: AgentState):
        search_input = SearchInput(
            query=state["research_query"],
            num_results=state.get("num_results", 5),
            freshness=state.get("freshness"),
        )
        hits, error = search_tool.run(search_input)
        base = len(_dedupe_hits(state.get("search_results", [])))
        for idx, hit in enumerate(hits, start=1):
            hit["source_id"] = f"S{base + idx}"
        path = "google_search"
        content_lines = [f"[Search Results]{' (error)' if error else ''}"]
        if error:
            content_lines.append(f"Error: {error}")
        if hits:
            content_lines.append(_format_hits(hits))
        else:
            content_lines.append("No results.")
        tool_message = ToolMessage(
            content="\n".join(content_lines),
            tool_call_id=path,
            name=path,
        )
        return {
            "search_results": hits,
            "messages": [tool_message],
            "search_error": error,
            "iterations": state.get("iterations", 0) + 1,
        }

    def analyzer_node(state: AgentState):
        decision = analyzer_chain.invoke(
            {
                "messages": state["messages"],
                "search_context": _format_hits(state.get("search_results", [])),
            }
        )
        next_query = decision.research_query or state["research_query"]
        next_num_results = decision.num_results or state.get("num_results", 5)
        next_freshness = (
            decision.freshness
            if decision.freshness is not None
            else state.get("freshness")
        )
        iterations = state.get("iterations", 0)
        max_iters = state.get("max_iters", 0)
        force_finish = iterations >= max_iters > 0 and decision.status == "CONTINUE"
        analyzer_message = AIMessage(
            content=(
                f"[Analyzer] Status: {decision.status}{' (max iters reached)' if force_finish else ''}\n"
                f"Summary: {decision.summary}\n"
                f"Gaps: {', '.join(decision.gaps) if decision.gaps else 'None'}\n"
                f"Next query: {next_query if decision.status == 'CONTINUE' else 'N/A'}\n"
                f"Num results: {next_num_results}\n"
                f"Freshness: {next_freshness or 'none'}"
            )
        )
        return {
            "messages": [analyzer_message],
            "status": "FINISH" if force_finish else decision.status,
            "research_query": next_query,
            "num_results": next_num_results,
            "freshness": next_freshness,
        }

    def reporter_node(state: AgentState):
        allowed_sources = [hit.get("source_id") for hit in _dedupe_hits(state.get("search_results", [])) if hit.get("source_id")]
        report = reporter_chain.invoke(
            {
                "messages": state["messages"],
                "search_context": _format_hits(state.get("search_results", [])),
                "source_ids": ", ".join(allowed_sources) if allowed_sources else "None",
            }
        )
        reporter_message = AIMessage(
            content="[Reporter] Final report validated and ready for export."
        )
        return {"messages": [reporter_message], "final_report": report, "status": "FINISH"}

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("search_tool", search_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("reporter", reporter_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "search_tool")
    workflow.add_edge("search_tool", "analyzer")
    workflow.add_conditional_edges(
        "analyzer", lambda state: state["status"], {"CONTINUE": "search_tool", "FINISH": "reporter"}
    )
    workflow.add_edge("reporter", END)
    return workflow.compile()
