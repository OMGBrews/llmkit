# llmkit

A thin, opinionated, **local-first** layer over [LiteLLM](https://github.com/BerriAI/litellm) (with [instructor](https://github.com/567-labs/instructor) for structured output). It gives an application one provider-agnostic call surface across **OpenRouter, Google, Anthropic, OpenAI, and local Ollama**, with validated structured output, a global async rate limiter, transient-error retries, and **agent-readable per-call logging** out of the box.

LiteLLM is the implementation of the HTTP providers; llmkit owns the ergonomic call surface, the structured-output mode pinning, the rate-limit policy, and the logging convention. It is **not** a gateway and does not reimplement transport — that is solved, and reimplementing it is the thing this library deliberately does not do.

## Why llmkit

- **Structured output that actually validates.** Each provider is pinned to its *native* JSON-schema mode (never instructor's auto-`Mode.TOOLS`, which silently regresses Gemini to empty shapes), and instructor's in-call validation-retry repairs truncated JSON. You pass a Pydantic model; you get a validated instance back.
- **Provider switching is config, not code.** OpenRouter / Google / Anthropic / OpenAI / Ollama behind one `Provider` enum and one `LLMClientConfig`. Call sites never change when you switch.
- **Logging tuned for coding agents.** Every call is logged verdict-first (see below) — the design assumption is that the reader is usually an LLM coding agent debugging a run, not a dashboard.
- **Local-first, zero infra.** The default sink writes plain files to a directory. No collector, no account, no network. A pluggable `LogSink` lets you ship records anywhere later without touching call sites.

## Install

```bash
uv add omg-llmkit          # or: pip install omg-llmkit
```

The distribution is published as **`omg-llmkit`** (the bare `llmkit` name was already
taken on PyPI), but the import name is just `llmkit`:

```python
import llmkit
```

Requires Python ≥ 3.13.

## Quick start

```python
from pydantic import BaseModel
from llmkit import (
    LLMClientConfig,
    Provider,
    configure_llm_client,
    structured_llm_call,
)

# Point the library at a provider once, at startup.
configure_llm_client(lambda: LLMClientConfig(
    provider=Provider.OPENROUTER,
    model="google/gemini-2.5-flash",
    api_key="sk-or-...",
))

class Summary(BaseModel):
    title: str
    bullets: list[str]

result: Summary = await structured_llm_call(
    prompt="Summarize the attached report.",
    output_schema=Summary,
    feature="reports",      # groups calls in the logs
    label="exec_summary",   # names this specific call in the logs
)
```

The public call surface:

| Function | Use |
|----------|-----|
| `structured_llm_call(prompt, output_schema, feature, label, ...)` | Async, returns a validated Pydantic instance |
| `structured_llm_call_sync(...)` | Synchronous wrapper around the above |
| `text_llm_call(prompt, feature, label, ...)` | Async, returns plain text (coerces provider list-content blocks) |
| `stream_text_with_log(prompt, feature, label, ...)` | Async generator yielding text chunks, logged on completion |

`configure_rate_limit(...)` sets the process-global concurrency cap; `configure_llm_logging(sink)` swaps the log sink (below).

## Logging: agent-readable by default

`LocalYamlLogSink` (the default) writes **two** things to `data/llm-logs/`:

1. **One YAML file per call, laid out verdict-first.** The file opens with a one-line `#` header — `ok`/`ERROR`, feature/label, resolved model, schema, duration, approximate cost — so `head -1 *.yaml` triages a whole run. Small metadata is next; the large `response` and `prompt` blobs are last, so the *head* of the file is the whole story for most reads.
2. **A compact append-only `index.jsonl`** — one JSON line per call (file, timestamp, feature, label, model, provider, schema, duration, cost, error). Cross-call questions — "which calls errored / were slowest / most expensive / the last call for feature X" — are a single small scan instead of globbing and parsing every YAML.

```
# ok | reports/exec_summary | google/gemini-2.5-flash | Summary | 1840ms | $0.0007
# 2026-06-05T14:22:31.004512

timestamp: '2026-06-05T14:22:31.004512'
feature: reports
label: exec_summary
model: google/gemini-2.5-flash
provider: openrouter
schema: Summary
temperature: 0.0
duration_ms: 1840.2
approximate_cost: 0.0007
error: null
response: ...
prompt: ...
```

`approximate_cost` is LiteLLM's per-response estimate for budget visibility — **not** a billing figure (and `None` when the provider does not report it, e.g. streamed calls).

### Write your own `LogSink`

`LogSink` is a one-method `Protocol`. Records (`LLMCallRecord`, a frozen dataclass) are handed to your sink for every call; failures are swallowed so logging can never break a call. To send records somewhere other than local YAML — a database, an HTTP collector, structured stdout — implement `write` and register it:

```python
import logging
from pathlib import Path
from llmkit import LLMCallRecord, configure_llm_logging

logger = logging.getLogger("llm-calls")

class StructuredStdoutSink:
    def write(self, record: LLMCallRecord) -> Path | None:
        logger.info(
            "llm_call",
            extra={
                "feature": record.feature,
                "label": record.label,
                "model": record.model,
                "provider": record.provider,
                "schema": record.schema,
                "duration_ms": record.duration_ms,
                "approximate_cost": record.approximate_cost,
                "error": record.error,
            },
        )
        return None  # nothing persisted to a path

configure_llm_logging(StructuredStdoutSink())   # pass None to disable logging entirely
```

An OpenTelemetry exporter (e.g. to Langfuse/Phoenix) is a natural future `llmkit[otel]` extra; the pluggable seam makes it a non-breaking addition.

## Configuration

`LLMClientConfig` is flat and carries only what a call needs:

```python
@dataclass(frozen=True)
class LLMClientConfig:
    provider: Provider               # OPENROUTER | OLLAMA | GOOGLE | ANTHROPIC | OPENAI
    model: str                       # the provider's default model
    api_key: str | None = None
    base_url: str | None = None      # OpenRouter / OpenAI-compatible endpoints; unused by Google/Anthropic
    reasoning_effort: str | None = None  # "disable" | "low" | "medium" | "high"
```

Per-call `model=` overrides the default, so "strong/small/current" model roles are the host's concern — resolve them to a model string and pass it at the call site. The library has no opinion about roles.

`reasoning_effort` controls provider "thinking"/reasoning tokens, forwarded to LiteLLM. Leave it `None` (the default) for the provider's own behaviour — the outbound request is byte-identical to omitting it. Set it once (e.g. `"disable"`) and every call inherits it; the call functions also take a `reasoning_effort=` override for a single call. This matters most for Gemini, whose thinking is **on by default** and spends reasoning tokens against `max_tokens` — `reasoning_effort="disable"` turns it off so a small `max_tokens` cap doesn't truncate structured output.

Register the config with `configure_llm_client(source)`, where `source` is a zero-arg callable returning an `LLMClientConfig` (re-read on each provider construction, so it tracks live settings changes).

## Retries

Two retry layers, kept deliberately separate:

- **`with_retries()`** ([`retry.py`](src/llmkit/retry.py)) handles *transient provider* errors (429 / 503 / 5xx; the recoverable set is `LLM_RECOVERABLE_ERRORS`).
- **instructor's own low `max_retries`** handles *schema-validation* repair (re-ask the model to fix malformed JSON).

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run basedpyright          # 0 errors, 0 warnings — no baseline
uv run pytest
```

## Status & support

`llmkit` is a small, opinionated, **best-effort** project, extracted from a real
application and maintained in the open. It is used in production by its author
but carries no support SLA. Bug reports and focused pull requests are welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md). For security issues, see
[SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
