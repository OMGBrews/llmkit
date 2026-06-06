# Contributing

Thanks for your interest. This is a small, opinionated, best-effort project — see
the scope notes in the [README](README.md). Bug reports and focused pull requests
are welcome; large feature proposals may not be a fit for the library's
deliberately-thin design, so please open an issue to discuss before investing in
a big change.

## Development setup

```bash
uv sync --extra dev
```

## Checks must pass

CI runs the same four gates on every push and pull request, with **no baseline**:

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright          # 0 errors, 0 warnings
uv run pytest
```

Please run them locally before opening a PR. New behavior needs a test.

### Live provider tests

The suite is split into two explicit, non-overlapping halves, and which half a
test belongs to never changes with your environment:

- **Offline tests** (the default) never touch the network or read a credential.
  Plain `uv run pytest` runs *only* these, so it's green everywhere with no keys
  — on your laptop and in CI alike.
- **Live tests** (`tests/integration/`, marked `@pytest.mark.live`) make a real
  structured call against each provider to prove the model strings,
  `instructor.Mode` pinning, and credential kwargs are accepted by the live
  APIs. They run **only** when you pass `--run-live` — never merely because a key
  happens to be in your environment. Under `--run-live` every one **must pass**:
  a missing key or unreachable server is a hard failure, not a skip.

Run the full live suite (export every key first):

```bash
OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai/keys
GEMINI_API_KEY=...             # https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/settings/keys
OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
DEEPSEEK_API_KEY=...           # https://platform.deepseek.com/api_keys
uv run pytest tests/integration --run-live -v
```

To exercise just one provider you hold a key for, select it explicitly instead
of relying on which keys are set: `uv run pytest tests/integration --run-live -k openai`.

Ollama needs no key — just a reachable server. Run `ollama serve` and pull the
smoke model (`ollama pull llama3.2`). By default the test talks to
`http://localhost:11434`; set `OLLAMA_HOST` to point elsewhere. In the maintainer
devcontainer Ollama runs on your **host** and the container reaches it at
`http://host.docker.internal:11434` (already wired as the container's default
`OLLAMA_HOST`), so just run `ollama serve` on the host.

## Conventions

- Keep the public surface small — `llmkit` owns the call ergonomics, not transport.
- No `dict[str, Any]` / bare `Any`; use precise types (basedpyright enforces this).
- Hard cuts over deprecation shims for internal changes.
