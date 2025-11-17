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


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the Sentinel Research Agent."""

    messages: Annotated[List[BaseMessage], operator.add]
    research_query: str
    search_results: Annotated[List[SearchHit], operator.add]
    status: Literal["CONTINUE", "FINISH"]
    final_report: FinalReport | None
