from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .config import Settings
from .llm import build_llm
from .schemas import FinalReport, SearchInput
from .state import AgentState, SearchHit
from .tools import GoogleSearchTool


def _format_hits(hits: list[SearchHit]) -> str:
    return "\n".join(
        f"- {hit['title']} ({hit['url']}): {hit['snippet']}" for hit in hits
    )


def build_workflow(settings: Settings):
    """Create the LangGraph workflow for the Sentinel Research Agent."""

    planner_llm = build_llm(settings, temperature=0.2)
    analyzer_llm = build_llm(settings, temperature=0.1)
    reporter_llm = build_llm(settings, temperature=0.0, timeout=90)
    search_tool = GoogleSearchTool(settings)

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You plan research. Produce a single focused search query that would "
                "retrieve authoritative, up-to-date information. Capture your reasoning.",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    analyzer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You determine whether more searching is required. "
                "If gaps remain, propose the next search query.",
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
                "Cite every factual statement and do not hallucinate sources.",
            ),
            ("placeholder", "{messages}"),
            (
                "human",
                "Evidence you may cite:\n{search_context}\n\n"
                "Produce the FinalReport schema now.",
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
                f"Rationale: {decision.rationale}"
            )
        )
        return {
            "messages": [planner_message],
            "research_query": decision.research_query,
            "status": "CONTINUE",
        }

    def search_node(state: AgentState):
        search_input = SearchInput(query=state["research_query"])
        hits = search_tool.run(search_input)
        path = "google_search"
        tool_message = ToolMessage(
            content=f"[Search Results]\n{_format_hits(hits) or 'No results.'}",
            tool_call_id=path,
            name=path,
        )
        return {"search_results": hits, "messages": [tool_message]}

    def analyzer_node(state: AgentState):
        decision = analyzer_chain.invoke(
            {
                "messages": state["messages"],
                "search_context": _format_hits(state.get("search_results", [])),
            }
        )
        next_query = decision.research_query or state["research_query"]
        analyzer_message = AIMessage(
            content=(
                f"[Analyzer] Status: {decision.status}\n"
                f"Summary: {decision.summary}\n"
                f"Gaps: {', '.join(decision.gaps) if decision.gaps else 'None'}\n"
                f"Next query: {next_query if decision.status == 'CONTINUE' else 'N/A'}"
            )
        )
        return {
            "messages": [analyzer_message],
            "status": decision.status,
            "research_query": next_query,
        }

    def reporter_node(state: AgentState):
        report = reporter_chain.invoke(
            {
                "messages": state["messages"],
                "search_context": _format_hits(state.get("search_results", [])),
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
