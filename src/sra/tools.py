from __future__ import annotations

from typing import List

import httpx

from .config import Settings
from .schemas import SearchInput
from .state import SearchHit


class GoogleSearchTool:
    """Thin wrapper around Google Custom Search API."""

    _BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=30)

    def run(self, search_input: SearchInput) -> List[SearchHit]:
        params = {
            "key": self._settings.google_api_key,
            "cx": self._settings.google_cx,
            "q": search_input.query,
            "num": search_input.num_results,
        }
        if search_input.freshness:
            params["sort"] = f"date:r:{search_input.freshness}"

        response = self._client.get(self._BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        hits: List[SearchHit] = []
        for item in items:
            hits.append(
                SearchHit(
                    title=item.get("title", "Unknown Title"),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                )
            )
        return hits

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        self.close()
