from __future__ import annotations

from typing import List, Tuple
import time

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

    def __enter__(self) -> "GoogleSearchTool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run(self, search_input: SearchInput) -> Tuple[List[SearchHit], str | None]:
        """Execute the search with basic retry/backoff for transient failures."""

        params = {
            "key": self._settings.google_api_key,
            "cx": self._settings.google_cx,
            "q": search_input.query,
            "num": search_input.num_results,
        }
        if search_input.freshness:
            params["sort"] = f"date:r:{search_input.freshness}"

        attempts = 3
        backoff = 0.5
        last_error: str | None = None

        for attempt in range(1, attempts + 1):
            try:
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
                return hits, None
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_error = self._format_http_error(exc.response)
                if status in {429, 500, 502, 503, 504} and attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break
            except httpx.HTTPError as exc:
                last_error = f"HTTP error: {exc}"
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"Unexpected error: {exc}"
                break

        return [], last_error

    @staticmethod
    def _format_http_error(response: httpx.Response) -> str:
        try:
            error = response.json().get("error", {})
        except ValueError:
            return f"HTTP {response.status_code}: {response.text[:200]}"

        message = str(error.get("message") or response.text[:200])
        reason = ""
        errors = error.get("errors")
        if isinstance(errors, list) and errors:
            reason = str(errors[0].get("reason") or "")

        if response.status_code == 400 and "API key not valid" in message:
            return (
                "Google Custom Search API key is invalid. "
                "Update GOOGLE_SEARCH_API_KEY with a valid Google Cloud API key "
                "that has the Custom Search API enabled."
            )

        if response.status_code in {400, 403} and reason:
            return f"Google Custom Search API error ({reason}): {message}"

        return f"HTTP {response.status_code}: {message[:200]}"

    def close(self) -> None:
        self._client.close()
