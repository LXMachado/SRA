from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Settings


def build_llm(
    settings: Settings, temperature: float = 0.2, timeout: int = 60
) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client configured for the selected provider."""

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
