# llmkit

A thin, opinionated, **local-first** layer over [LiteLLM](https://github.com/BerriAI/litellm) (with [instructor](https://github.com/567-labs/instructor) for structured output). It gives an application one provider-agnostic call surface across **OpenRouter, Google, Anthropic, OpenAI, DeepSeek, AWS Bedrock, and local Ollama**, with validated structured output, a global async rate limiter, **agent-readable per-call logging**, and **transient-error retries on by default** out of the box.

LiteLLM is the implementation of the HTTP providers; llmkit owns the ergonomic call surface, the structured-output mode pinning, the rate-limit policy, and the logging convention. It is **not** a gateway and does not reimplement transport — that is solved, and reimplementing it is the thing this library deliberately does not do.

## Why llmkit

- **Structured output that actually validates.** Each provider is pinned to its *native* JSON-schema mode (never instructor's auto-`Mode.TOOLS`, which silently regresses Gemini to empty shapes), and instructor's in-call validation-retry repairs truncated JSON. You pass a Pydantic model; you get a validated instance back.
- **Provider switching is config, not code.** OpenRouter / Google / Anthropic / OpenAI / DeepSeek / AWS Bedrock / Ollama behind one `Provider` enum and one `LLMClientConfig`. Call sites never change when you switch.
- **Logging tuned for coding agents.** Every call is logged verdict-first (see below) — the design assumption is that the reader is usually an LLM coding agent debugging a run, not a dashboard.
- **Local-first, zero infra.** The default sink writes plain files to a directory. No collector, no account, no network. A pluggable `LogSink` lets you ship records anywhere later without touching call sites.

These four are the headline; [`PRINCIPLES.md`](PRINCIPLES.md) states the full set of design principles behind the library.

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
    provider: Provider               # OPENROUTER | OLLAMA | GOOGLE | ANTHROPIC | OPENAI | DEEPSEEK | BEDROCK
    model: str                       # the provider's default model
    api_key: str | None = None
    base_url: str | None = None      # OpenRouter / OpenAI-compatible endpoints; unused by Google/Anthropic
    reasoning_effort: str | None = None  # "disable" | "low" | "medium" | "high"
    aws_region_name: str | None = None   # AWS Bedrock region; unused by every other provider
```

`aws_region_name` is the only AWS-shaped field, and it carries **only** the region. AWS Bedrock authenticates through the standard **AWS credential chain** (environment, shared config, or instance/role), so Bedrock secrets never pass through `LLMClientConfig`; leave the region `None` too and it resolves from the chain (`AWS_REGION_NAME` / `AWS_REGION`). Bedrock routing needs `boto3` for request signing — install it with the opt-in extra:

```bash
pip install "omg-llmkit[bedrock]"
```

The first cut targets plain **on-demand** Claude-on-Bedrock models (default `anthropic.claude-3-5-sonnet-20240620-v1:0`). Newer Claude 4.x models on Bedrock are typically reached through a cross-region inference profile — pass the profile-prefixed id as `model` (e.g. `us.anthropic.claude-sonnet-4-...`).

Per-call `model=` overrides the default, so "strong/small/current" model roles are the host's concern — resolve them to a model string and pass it at the call site. The library has no opinion about roles.

`reasoning_effort` controls provider "thinking"/reasoning tokens, forwarded to LiteLLM. Leave it `None` (the default) for the provider's own behaviour — the outbound request is byte-identical to omitting it. Set it once (e.g. `"disable"`) and every call inherits it; the call functions also take a `reasoning_effort=` override for a single call. This matters most for Gemini, whose thinking is **on by default** and spends reasoning tokens against `max_tokens` — `reasoning_effort="disable"` turns it off so a small `max_tokens` cap doesn't truncate structured output.

Register the config with `configure_llm_client(source)`, where `source` is a zero-arg callable returning an `LLMClientConfig` (re-read on each provider construction, so it tracks live settings changes).

## Retries

Two retry layers, kept deliberately separate:

- **Transient-provider retries, on by default.** Every call function (`structured_llm_call`, `text_llm_call`, `structured_llm_call_sync`, `stream_text_with_log`) retries *transient* provider errors (429 / 503 / 5xx; the recoverable set is `LLM_RECOVERABLE_ERRORS`) on its own — you don't wrap anything. The default `RetryPolicy` is three attempts with bounded **full-jitter** backoff. Programming errors (e.g. `TypeError`) are outside the recoverable set and propagate immediately, never retried. Each attempt is its own logged call, so `data/llm-logs/` shows one record per attempt.

  Tune or opt out per call with the `retry=` argument:

  ```python
  from llmkit import structured_llm_call, RetryPolicy, NO_RETRY

  # Opt this one call out of automatic retries (e.g. latency-sensitive):
  result = await structured_llm_call(
      prompt="Summarize the attached report.",
      output_schema=Summary,
      feature="reports",
      label="exec_summary",
      retry=NO_RETRY,
  )

  # Or tune the budget / backoff for this call:
  result = await structured_llm_call(
      prompt="Summarize the attached report.",
      output_schema=Summary,
      feature="reports",
      label="exec_summary",
      retry=RetryPolicy(max_attempts=5, backoff_base_seconds=1.0),
  )
  ```

  **Streaming caveat:** `stream_text_with_log` can only retry a transient failure that happens *before the first chunk reaches the caller*. Once any chunk has been yielded, a mid-stream error propagates unretried — a partially-consumed stream can't be safely restarted.

  **`with_retries()`** (exported from `llmkit`; see [`retry.py`](src/llmkit/retry.py)) remains the explicit, composable advanced path for wrapping *any* awaitable — useful when you want to retry a unit of work that isn't a single call function. Wrap a `retry_progress_callback(...)` scope around the work to observe per-attempt failures (e.g. for a progress UI):

  ```python
  from llmkit import with_retries, LLM_RECOVERABLE_ERRORS

  result = await with_retries(
      lambda: do_some_work(),
      max_retries=3,
      backoff_base_seconds=0.5,
      retry_on=LLM_RECOVERABLE_ERRORS,
  )
  ```

- **instructor's own low `max_retries`** handles *schema-validation* repair (re-ask the model to fix malformed JSON). This stays **separate** from the transient-retry layer above — the two budgets are never conflated, so attempts aren't double-counted.

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
