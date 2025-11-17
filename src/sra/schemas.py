from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Arguments passed into the Google Search API tool."""

    query: str = Field(description="The specific query string to send to Google Custom Search.")
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many organic results to return for the query.",
    )
    freshness: str | None = Field(
        default=None,
        description="Optional freshness filter (e.g., 'd7' for last 7 days). Leave null for no filter.",
    )


class Source(BaseModel):
    """A citation source used in the final report."""

    title: str = Field(description="Title of the source document or webpage.")
    url: str = Field(description="The URL link to the source document.")


class ReportSection(BaseModel):
    """A single section of the final research report."""

    section_title: str = Field(
        description="The title for this section (e.g., 'Key Findings', 'Market Impact')."
    )
    content: str = Field(
        description="Summarized content for this section, backed by search results."
    )


class FinalReport(BaseModel):
    """The complete, structured output report."""

    topic: str = Field(description="The main topic of the original user query.")
    executive_summary: str = Field(
        description="High-level summary of the entire report."
    )
    sections: List[ReportSection] = Field(
        description="Structured body sections for the report."
    )
    sources: List[Source] = Field(description="Unique external sources referenced.")
