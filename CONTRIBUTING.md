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

CI runs the same four gates on every push and pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright          # recommended tier, clean with no baseline
uv run pytest
```

basedpyright runs in its `recommended` tier (stricter than the `standard`
default, and at least as strict as the editor extension's defaults) and is clean
at **0 errors, 0 warnings with no baseline** — there is no `.basedpyright/baseline.json`,
so **any new finding fails CI**. The untyped LiteLLM/instructor surface is given
precise types at the boundary; the few genuinely-unavoidable suppressions are
inline `# pyright: ignore[...]` (or, for the package's deferred-import test seam,
a single file-level rule disable), each tagged with a `raw-*` reason. Prefer real
types; reach for a tagged suppression only at the provider-SDK boundary where a
precise type isn't reachable, and never to silence a finding in new code.

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
`http://localhost:11434`; set `OLLAMA_HOST` to point at your server if it runs
elsewhere (for example, when Ollama runs on the Docker host, point it at
`http://host.docker.internal:11434`).

AWS Bedrock is the one deliberate exception to "every live test must pass under
`--run-live`". Unlike the others it has no API key: it authenticates through the
ambient **AWS credential chain** and needs the `boto3`-bearing extra, so its live
test is allowed to skip when that isn't set up. To run it, install the extra and
provide AWS credentials plus a region:

```bash
uv sync --extra dev --extra bedrock
export AWS_REGION_NAME=us-east-1   # plus credentials via the standard AWS chain
uv run pytest tests/integration --run-live -k bedrock
```

## Conventions

- Keep the public surface small — `llmkit` owns the call ergonomics, not transport.
- No `dict[str, Any]` / bare `Any`; use precise types (basedpyright enforces this).
- Hard cuts over deprecation shims for internal changes.

## See also

- [`README.md`](README.md) — what `llmkit` is, how to use it, and its scope.
- [`PRINCIPLES.md`](PRINCIPLES.md) — the design principles behind the library.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability.
- [`CHANGELOG.md`](CHANGELOG.md) — release history.
