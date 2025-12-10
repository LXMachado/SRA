from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage

from .schemas import FinalReport


class SearchHit(TypedDict):
    """Normalized search result stored in agent state."""

    title: str
    snippet: str
    url: str
    source_id: str | None


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the Sentinel Research Agent."""

    messages: Annotated[List[BaseMessage], operator.add]
    research_query: str
    search_results: Annotated[List[SearchHit], operator.add]
    search_error: str | None
    num_results: int
    freshness: str | None
    iterations: int
    max_iters: int
    status: Literal["CONTINUE", "FINISH"]
    final_report: FinalReport | None
