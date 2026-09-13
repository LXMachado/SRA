from __future__ import annotations

import time

import httpx

from .config import Settings
from .schemas import SearchInput
from .state import SearchHit


class WebSearchTool:
    """Provider-aware web search wrapper."""

    _GOOGLE_BASE_URL = "https://www.googleapis.com/customsearch/v1"
    _TAVILY_BASE_URL = "https://api.tavily.com/search"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, search_input: SearchInput) -> tuple[list[SearchHit], str | None]:
        """Execute the search with basic retry/backoff for transient failures."""

        if self._settings.search_provider == "tavily":
            return self._run_tavily(search_input)
        return self._run_google(search_input)

    def _run_tavily(
        self, search_input: SearchInput
    ) -> tuple[list[SearchHit], str | None]:
        payload: dict[str, object] = {
            "query": search_input.query,
            "max_results": search_input.num_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        time_range = self._tavily_time_range(search_input.freshness)
        if time_range:
            payload["time_range"] = time_range

        headers = {
            "Authorization": f"Bearer {self._settings.tavily_api_key}",
            "Content-Type": "application/json",
        }
        data, error = self._request_json(
            "POST", self._TAVILY_BASE_URL, headers=headers, json=payload
        )
        if error:
            return [], error

        hits: list[SearchHit] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=item.get("title", "Unknown Title"),
                    snippet=item.get("content", ""),
                    url=url,
                )
            )
        return hits, None

    def _run_google(
        self, search_input: SearchInput
    ) -> tuple[list[SearchHit], str | None]:
        params = {
            "key": self._settings.google_api_key,
            "cx": self._settings.google_cx,
            "q": search_input.query,
            "num": search_input.num_results,
        }
        if search_input.freshness:
            params["sort"] = f"date:r:{search_input.freshness}"

        data, error = self._request_json("GET", self._GOOGLE_BASE_URL, params=params)
        if error:
            return [], error

        hits: list[SearchHit] = []
        for item in data.get("items", []):
            hits.append(
                SearchHit(
                    title=item.get("title", "Unknown Title"),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                )
            )
        return hits, None

    def _request_json(self, method: str, url: str, **kwargs) -> tuple[dict, str | None]:
        attempts = 3
        backoff = 0.5
        last_error: str | None = None

        with httpx.Client(timeout=30) as client:
            for attempt in range(1, attempts + 1):
                try:
                    response = client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response.json(), None
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

        return {}, last_error

    @staticmethod
    def _tavily_time_range(freshness: str | None) -> str | None:
        if not freshness:
            return None
        prefix = freshness.strip().lower()[:1]
        return {
            "d": "day",
            "w": "week",
            "m": "month",
            "y": "year",
        }.get(prefix)

    @staticmethod
    def _format_http_error(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return f"HTTP {response.status_code}: {response.text[:200]}"

        response_message = data.get("message") if isinstance(data, dict) else None
        error = data.get("error", {}) if isinstance(data, dict) else {}
        if isinstance(error, dict):
            message = str(error.get("message") or response_message or response.text[:200])
        else:
            message = str(error or response_message or response.text[:200])
        reason = ""
        if isinstance(error, dict):
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

        if response.url.host == "api.tavily.com":
            return f"Tavily Search API error ({response.status_code}): {message[:200]}"

        return f"HTTP {response.status_code}: {message[:200]}"
