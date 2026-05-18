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
        if not api_key.startswith("sk-or-"):
            raise RuntimeError(
                "OPENROUTER_API_KEY appears invalid. It should start with 'sk-or-'."
            )

        model = os.getenv("OPENROUTER_MODEL")
        if not model:
            raise RuntimeError(
                "OPENROUTER_MODEL is required (example: google/gemini-2.5-flash)."
            )
        if model.strip().lower() in {"openrouter/free", "free"}:
            raise RuntimeError(
                "OPENROUTER_MODEL='openrouter/free' is not a concrete model id. "
                "Set an explicit model, e.g. google/gemini-2.5-flash."
            )

        google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if not google_key:
            raise RuntimeError("GOOGLE_SEARCH_API_KEY is required.")

        google_cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not google_cx:
            raise RuntimeError("GOOGLE_SEARCH_ENGINE_ID is required.")

        return cls(
            openrouter_api_key=api_key,
            openrouter_model=model,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            google_api_key=google_key,
            google_cx=google_cx,
        )
