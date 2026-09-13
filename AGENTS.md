# AGENTS.md

This file is an operational guide for humans and coding agents working in this repository.

## Mission

Maintain and improve **Sentinel Research Agent (SRA)**, a LangGraph-based research workflow that:
- accepts a natural-language query,
- performs iterative web retrieval,
- and emits a structured, validated JSON report.

Primary goals:
- keep output schema stable,
- keep CLI behavior predictable,
- keep provider configuration flexible without breaking OpenRouter defaults.

## Repo Overview

- `src/sra/cli.py`:
  - CLI entrypoint.
  - Loads `.env` from repo root with `override=True`.
  - Builds workflow and executes graph.
  - Handles auth/model/rate-limit errors with user-friendly messages.
- `src/sra/config.py`:
  - Environment parsing and validation.
  - Supports provider-agnostic `LLM_API_KEY` and provider-specific fallbacks.
- `src/sra/graph.py`:
  - LangGraph workflow and node logic.
  - Planner/Search/Analyzer/Reporter nodes.
  - Structured-output fallback parsing for models that ignore strict schemas.
- `src/sra/tools.py`:
  - Provider-aware web search wrapper.
  - Supports Tavily by default and Google Custom Search as a legacy option.
  - Retries transient failures with backoff.
- `src/sra/schemas.py`:
  - Pydantic models: `SearchInput`, `FinalReport`, `Source`, `ReportSection`.
- `src/sra/state.py`:
  - Typed graph state (`AgentState`, `SearchHit`).

## Runtime Flow (Canonical)

1. `planner` proposes:
   - `research_query`
   - `num_results`
   - optional `freshness`
2. `search_tool` calls the configured web search provider and appends normalized hits.
3. `analyzer` decides:
   - `CONTINUE` (loop to search), or
   - `FINISH` (route to reporter)
4. `reporter` produces `FinalReport`.
5. If max iterations are reached while analyzer says continue, workflow forces finish.

## CLI Contract

Current CLI supports both forms:

```bash
sra "your query"
sra run "your query"
```

Important option:

```bash
--max-iters <int>
```

Notes:
- In code, `src/sra/cli.py` uses an `argparse` dispatcher.
- `sra "query"` and `sra run "query"` route to the same execution path.

## Configuration Contract

### Search provider configuration

- `SEARCH_PROVIDER` may be `tavily` or `google`.
- If unset, config chooses `tavily` when `TAVILY_API_KEY` is present; otherwise it falls back to `google`.
- Tavily path:
  - `TAVILY_API_KEY`
- Google legacy path:
  - `GOOGLE_SEARCH_API_KEY`
  - `GOOGLE_SEARCH_ENGINE_ID`

Notes:
- Tavily is the documented default for search retrieval.
- Google Custom Search is retained for existing projects with Custom Search JSON API access.
- Newer Google Cloud projects may return `403` entitlement errors for Custom Search JSON API even with a valid key and correct restrictions.

### LLM/provider configuration

- `LLM_BASE_URL` (default fallback: `https://openrouter.ai/api/v1`)
- `LLM_MODEL` (must be a concrete model ID)
- API key resolution order:
  1. `LLM_API_KEY` (recommended)
  2. Venice path (`venice.ai` base URL): `VENICE_API_KEY`, then `OPENROUTER_API_KEY`
  3. OpenRouter path: `OPENROUTER_API_KEY`

Legacy compatibility:
- `OPENROUTER_BASE_URL` is used if `LLM_BASE_URL` is unset.
- `OPENROUTER_MODEL` is used if `LLM_MODEL` is unset.

Validation rules:
- If base URL contains `openrouter.ai`, key must look like `sk-or-...`.
- Model cannot be `openrouter/free` or `free`.

## Provider Profiles

### Default profile (GitHub docs default)

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.5-flash
LLM_API_KEY=...
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
```

### Local Venice profile

```bash
LLM_BASE_URL=https://api.venice.ai/api/v1
LLM_MODEL=venice:uncensored
LLM_API_KEY=...
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
```

## Development Commands

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run:

```bash
sra "How is the EU regulating frontier AI safety tests?"
sra run "Find english translations for Mundus Subterraneus" --max-iters 2
```

Help:

```bash
sra --help
```

## Error Handling Expectations

The CLI should surface these cases clearly:
- `401` auth failure (`AuthenticationError`)
- `404` model not found (`NotFoundError`)
- `429` provider throttling (`RateLimitError`)
- missing final report (graph completed without output)

When changing error handling, preserve concise actionable messages.

## Structured Output Fallbacks

Some free/experimental models ignore strict JSON schema output.

Current behavior in `graph.py`:
- planner/analyzer:
  - try `with_structured_output(...)`
  - on failure, re-invoke raw and parse JSON/text heuristically
- reporter:
  - try strict output
  - then parse embedded JSON
  - finally synthesize minimal valid `FinalReport` from collected hits

If you modify these fallbacks:
- keep FinalReport validity guaranteed,
- avoid removing strict path,
- keep fallback deterministic and debuggable.

## Non-Obvious Implementation Details

- `.env` loading is pinned to repo root in CLI and uses `override=True`.
  - This intentionally overrides inherited shell values.
- Graph recursion limit is set to `max(8, max_iters * 4)`.
  - Prevents premature `GraphRecursionError` at low `--max-iters`.
- Search hits are deduplicated by URL/title key and truncated for context safety.

## Guardrails for Future Changes

1. Preserve schema compatibility:
   - `FinalReport` keys should not change without migration note.
2. Preserve CLI compatibility:
   - both `sra "query"` and `sra run "query"` should continue to work.
3. Keep OpenRouter as documented default in README.
4. Keep provider-agnostic key path (`LLM_API_KEY`) working.
5. Do not hardcode a single provider in config.

## Recommended Test Checklist (Manual)

Run these after touching CLI/config/graph:

1. OpenRouter happy path:
```bash
LLM_BASE_URL=https://openrouter.ai/api/v1 \
LLM_MODEL=google/gemini-2.5-flash \
LLM_API_KEY=... \
SEARCH_PROVIDER=tavily \
TAVILY_API_KEY=... \
sra "test query" --max-iters 1
```

2. Venice path key resolution:
```bash
LLM_BASE_URL=https://api.venice.ai/api/v1 \
LLM_MODEL=venice:uncensored \
LLM_API_KEY=... \
SEARCH_PROVIDER=tavily \
TAVILY_API_KEY=... \
sra "test query" --max-iters 1
```

3. Google legacy path, only when the project has Custom Search JSON API access:
```bash
LLM_BASE_URL=https://openrouter.ai/api/v1 \
LLM_MODEL=google/gemini-2.5-flash \
LLM_API_KEY=... \
SEARCH_PROVIDER=google \
GOOGLE_SEARCH_API_KEY=... \
GOOGLE_SEARCH_ENGINE_ID=... \
sra "test query" --max-iters 1
```

4. CLI compatibility:
```bash
sra "test query"
sra run "test query"
```

5. Fallback resilience:
- Use a model known to ignore strict JSON and confirm output is still valid JSON.

## Security and Secrets

- Never commit real API keys.
- Keep `.env` local.
- Avoid logging full keys or raw sensitive headers.

## When Updating README

- Keep OpenRouter defaults in main setup instructions.
- Keep Tavily as the documented default search provider.
- Mention alternatives (like Venice) as optional/local.
- Ensure command examples match actual CLI behavior.
- Ensure diagrams match actual node transitions and forced-finish logic.
