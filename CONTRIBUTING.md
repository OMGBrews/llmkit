# Contributing

Thanks for your interest. This is a small, opinionated, best-effort project — see
the scope notes in the [README](README.md). Bug reports and focused pull requests
are welcome; large feature proposals may not be a fit for the library's
deliberately-thin design, so please open an issue to discuss before investing in
a big change.

## Development setup

```bash
uv sync
```

## Checks must pass

CI runs the same four gates on every push and pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright          # recommended tier, clean with no baseline
uv run pytest
```

CI additionally runs a lowest-versions resolution job (`--resolution
lowest-direct`), a wheel smoke test, and a weekly unlocked-resolution cron —
those rarely concern a PR author, so the four gates above are what to run before
opening a PR.

basedpyright runs in its `recommended` tier (stricter than the `standard`
default, and at least as strict as the editor extension's defaults) and is clean
at **0 errors, 0 warnings with no baseline** — there is no `.basedpyright/baseline.json`,
so **any new finding fails CI**. The untyped LiteLLM/instructor surface is given
precise types at the boundary; the suppressions that remain are inline
`# pyright: ignore[<rule>]` comments (plus one file-level rule disable in
`__init__.py` for the package's deferred-import test seam), each naming the
specific rule it silences. Two conventions govern them: a suppression forced by
an untyped or loosely-typed **third-party surface** carries a `raw-*` tag naming
the culprit (`raw-llm` for LiteLLM/instructor/yaml, `raw-pydantic` for pydantic's
`Any`-typed seams) — these cluster in `_litellm.py`, `logging.py`, and
`json_schema.py`; the rest (pytest fixtures invoked by name, test helpers,
runtime guards at a public boundary) carry a short trailing reason comment
instead. Prefer real types; when you must suppress, name the rule and say why
(with a `raw-*` tag when a third-party type forces it), and never suppress to
quiet a finding in new code that could be typed precisely.

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
`--run-live`" — and the exception is narrow. Unlike the others it has no API
key: it authenticates through the ambient **AWS credential chain** and needs the
`boto3`-bearing extra. The *only* allowed skip is structural —
`pytest.importorskip("boto3")` when the extra isn't installed. Once `boto3` is
present, the usual hard-fail rule applies: a missing region
(`AWS_REGION_NAME`/`AWS_REGION`) or an unresolvable credential chain **fails**
the test under `--run-live`, exactly like a missing key for any other provider.
To run it, provide AWS credentials plus a region (`uv sync` already installs the
`[bedrock]` extra via the `dev` dependency group):

```bash
uv sync
export AWS_REGION_NAME=us-east-1   # plus credentials via the standard AWS chain
uv run pytest tests/integration --run-live -k bedrock
```

**Google Vertex AI** is the same kind of structural exception. Like Bedrock it
has no API key — it authenticates through Google **Application Default
Credentials** and needs the `google-auth`-bearing `[vertex]` extra — so its only
allowed skip is `pytest.importorskip("google.auth")` when the extra isn't
installed. Once present, a missing `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` or an
unresolvable ADC chain **fails** the test under `--run-live`. `vertex_location`
selects the data-residency region, and this test is where the Vertex
`Mode.JSON_SCHEMA` pin is *measured*:

```bash
uv sync                      # the dev group already pulls [vertex]
gcloud auth application-default login          # resolvable ADC
export VERTEXAI_PROJECT=my-gcp-project VERTEXAI_LOCATION=europe-west4
uv run pytest tests/integration --run-live -k vertex
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
