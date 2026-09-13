from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str
    search_provider: str
    tavily_api_key: str | None
    google_api_key: str | None
    google_cx: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("VENICE_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        base_url_lower = base_url.lower()

        # Provider-agnostic key takes precedence.
        generic_key = os.getenv("LLM_API_KEY")

        if "venice.ai" in base_url_lower:
            api_key = (
                generic_key
                or os.getenv("VENICE_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            )
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY (recommended) or VENICE_API_KEY is required when LLM_BASE_URL targets Venice."
                )
        else:
            api_key = generic_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY (recommended) or OPENROUTER_API_KEY is required."
                )

        if "openrouter.ai" in base_url and not api_key.startswith("sk-or-"):
            raise RuntimeError(
                "API key appears invalid for OpenRouter. It should start with 'sk-or-'."
            )

        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("VENICE_MODEL")
        )
        if not model:
            raise RuntimeError(
                "LLM_MODEL is required (legacy fallback: OPENROUTER_MODEL)."
            )
        if model.strip().lower() in {"openrouter/free", "free"}:
            raise RuntimeError(
                "LLM_MODEL='openrouter/free' is not a concrete model id. "
                "Set an explicit model, e.g. google/gemini-2.5-flash."
            )

        tavily_key = os.getenv("TAVILY_API_KEY")
        google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        google_cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        search_provider = (
            os.getenv("SEARCH_PROVIDER")
            or ("tavily" if tavily_key else "google")
        ).strip().lower()
        if search_provider not in {"tavily", "google"}:
            raise RuntimeError(
                "SEARCH_PROVIDER must be either 'tavily' or 'google'."
            )
        if search_provider == "tavily" and not tavily_key:
            raise RuntimeError("TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily.")
        if search_provider == "google":
            if not google_key:
                raise RuntimeError("GOOGLE_SEARCH_API_KEY is required when SEARCH_PROVIDER=google.")
            if not google_cx:
                raise RuntimeError("GOOGLE_SEARCH_ENGINE_ID is required when SEARCH_PROVIDER=google.")

        return cls(
            llm_api_key=api_key,
            llm_model=model,
            llm_base_url=base_url,
            search_provider=search_provider,
            tavily_api_key=tavily_key,
            google_api_key=google_key,
            google_cx=google_cx,
        )
