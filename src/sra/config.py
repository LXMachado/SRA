from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    google_api_key: str
    google_cx: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")

        google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if not google_key:
            raise RuntimeError("GOOGLE_SEARCH_API_KEY is required.")

        google_cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not google_cx:
            raise RuntimeError("GOOGLE_SEARCH_ENGINE_ID is required.")

        return cls(
            openrouter_api_key=api_key,
            openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-pro-1.5"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            google_api_key=google_key,
            google_cx=google_cx,
        )
