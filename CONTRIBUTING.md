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

### Live provider tests (optional)

`tests/integration/` makes a real structured call against each supported
provider to prove the model strings, `instructor.Mode` pinning, and credential
kwargs are accepted by the live APIs. They cost money and need credentials, so
each test **skips** unless its key is set — `uv run pytest` stays green with no
keys. To exercise them, export the keys for the providers you have and run:

```bash
OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai/keys
GEMINI_API_KEY=...             # https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/settings/keys
uv run pytest tests/integration -v
```

Ollama needs no key — just a local server (`ollama serve`) on `:11434`.

## Conventions

- Keep the public surface small — `llmkit` owns the call ergonomics, not transport.
- No `dict[str, Any]` / bare `Any`; use precise types (basedpyright enforces this).
- Hard cuts over deprecation shims for internal changes.
