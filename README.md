# llmkit

A thin, opinionated, **local-first** layer over [LiteLLM](https://github.com/BerriAI/litellm) (with [instructor](https://github.com/567-labs/instructor) for structured output). It gives an application one provider-agnostic call surface across **OpenRouter, Google, Anthropic, OpenAI, DeepSeek, AWS Bedrock, and local Ollama**, with validated structured output, per-provider rate limiting (concurrency on by default; optional requests-/tokens-per-minute), **agent-readable per-call logging**, and **transient-error retries on by default** — all out of the box.

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

You **install** `omg-llmkit` but **import** `llmkit` — that split trips a natural
post-install smoke test. A mistaken `import omg_llmkit` (the install name) raises
a clear one-line redirect to `import llmkit`, not a bare
`ModuleNotFoundError` that leaves you guessing.

Requires Python ≥ 3.13.

The core install routes OpenRouter, Google, OpenAI, DeepSeek, and Ollama with no
extra dependencies. Two providers gate their dependencies behind opt-in extras so
hosts pay only for what they call:

```bash
pip install "omg-llmkit[anthropic]"  # direct Anthropic (Claude) routing
pip install "omg-llmkit[bedrock]"    # Claude-on-Bedrock (pulls in [anthropic] too)
```

The Anthropic SDK is opt-in because `instructor` reaches it only at *call time*,
on its `ANTHROPIC_JSON` usage-accounting path — plain `import llmkit` and a
Google-only flow never touch it. Constructing the `AnthropicProvider` or
`BedrockProvider` without the SDK raises a clear `install omg-llmkit[anthropic]`
error at construction, not a cryptic failure on the first call.

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
| `text_llm_call_sync(...)` | Synchronous wrapper around the above |
| `stream_text_with_log(prompt, feature, label, ...)` | Async generator yielding text chunks, logged on completion |

> **Two defaults worth knowing up front.**
> - **`temperature` defaults to `0.2`** — biased toward deterministic output. A *creative* caller must override it explicitly (e.g. `temperature=1.0`); it is otherwise quietly conservative.
> - **Any call takes a per-call `provider=` override** — route a single call through a different provider family, model, or credential without touching the global `configure_llm_client(...)` registration. See [Constructing a provider for a per-call override](#constructing-a-provider-for-a-per-call-override).

### Reusing call options

The call functions (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and `stream_text_with_log`) take up to nine keyword arguments. When a feature module makes many calls with the same settings, repeating that block at every site is noise. Build an `LLMCallOptions` once and pass it as `options=`:

```python
from llmkit import LLMCallOptions, structured_llm_call

# Built once per feature module.
RISK_OPTS = LLMCallOptions(
    temperature=0.0,
    model="gemini-2.5-flash",
    reasoning_effort="high",
    max_tokens=2048,
)

async def extract(prompt: str) -> RiskRegister:
    return await structured_llm_call(
        prompt, RiskRegister, feature="extraction", options=RISK_OPTS
    )
```

`LLMCallOptions` is **frozen** and carries any subset of `temperature` / `model` / `max_tokens` / `reasoning_effort` / `retry` / `provider`. Every field is optional and *unset* by default — an unset field defers to the call's keyword (and through it to the configured client), so a partially-filled `LLMCallOptions` only supplies the fields you set.

`feature` is intentionally **not** part of `LLMCallOptions`. It stays a required per-call keyword as a telemetry forcing function: it scopes the per-call log filename and the `index.jsonl` grouping operators grep, so it must be a conscious choice at each call site rather than something defaulted-away into a shared object.

The flat-keyword path is unchanged — pass no `options` and nothing about existing calls changes.

#### Call-vs-config precedence

`model` and `reasoning_effort` are *dual-homed*: they can be set both on `LLMClientConfig` (the app-wide default) and on the call surface. The precedence, lowest to highest, is:

**config < `options` < explicit per-call keyword**

So a value passed directly as a keyword wins; an `LLMCallOptions` field sits between the keyword and the config; and when neither the keyword nor `options` supplies a value, the configured `LLMClientConfig` default applies (e.g. `model=None` defers to the provider/config default). An *unset* `LLMCallOptions` field never overrides config — only a field you explicitly set on the options participates.

### Contracts as JSON-schema dicts

If your structured-output contract is a **JSON-schema dict** — typically because the same schema is shared with a Node backend or a frontend — `model_from_json_schema(schema)` converts it to a Pydantic model at runtime, so you don't hand-write the converter (and re-discover its footguns). Build the model **once and reuse it**; `structured_llm_call` stays Pydantic-model-only and takes the result as `output_schema`.

```python
from llmkit import model_from_json_schema, structured_llm_call

INVOICE_SCHEMA = {                       # shared with Node / the frontend
    "title": "Invoice",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "total": {"type": "number"},
        "status": {"enum": ["open", "closed", "void"]},
        "note": {"type": ["string", "null"]},          # optional, nullable
        "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
    },
    "required": ["id", "total", "status", "lines"],
    "$defs": {
        "Line": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku"],
        }
    },
}

Invoice = model_from_json_schema(INVOICE_SCHEMA)   # build once, at import

result = await structured_llm_call(
    prompt="Extract the invoice.",
    output_schema=Invoice,                         # reuse on every call
    feature="billing",
)
```

**Supported subset** (anything outside it raises a clear `ValueError` naming the construct): `object` with `properties` and a `required` array; scalars (`string` / `integer` / `number` / `boolean`, plus `null` / nullable); `array` with `items` (including arrays of objects); `enum` (string or integer members); nested objects inline or via local `$ref` (`#/$defs/...`); and `additionalProperties` as `true` / `false` / absent (a *typed* `additionalProperties` map is rejected). A non-required field becomes an optional defaulting to `None`, and the generated model's `model_dump` / `model_dump_json` **drop a `None` left in an optional field by default** — so an omitted optional is *absent*, not `"field": null` (which would fail downstream re-validation against the same schema). The drop is scoped to optionals: a *required*-but-nullable field explicitly set to `None` is kept. Pass `exclude_none=False` to keep every null, or `exclude_none=True` to drop them all. A title-less schema still gets a valid default class name (`JsonSchemaModel`); pass `name=` to set it explicitly. Generated models default to **`extra="forbid"`**, so a response carrying a key not in the schema is *rejected* rather than silently kept — for an LLM output contract you want a hallucinated extra field to fail loudly (stricter than JSON Schema's permissive `additionalProperties` default); `"additionalProperties": true` opts an object into `extra="allow"` (extra keys accepted and kept), while `false` or absent stays strict. An explicit `"type": "object"` with **no `properties`** raises rather than silently building a zero-field model that rejects every real response — set `"additionalProperties": true` for an intentionally free-form object.

**Want plain data back, not a model instance?** Call `.model_dump()` on the result — it inherits the optional-`None` drop above, so the dict matches the schema:

```python
Person = model_from_json_schema(person_schema)   # build once, at import

result = await structured_llm_call(prompt, Person, feature="extraction")
data = result.model_dump()                        # {"name": "Ada", "age": 36}
```

#### Schema constraints

`model_from_json_schema` carries a small, fixed set of per-field JSON-schema
constraints through to the generated Pydantic `Field`, so the model validates
*value bounds*, not just shape. The supported set is **exactly**:

| JSON schema | Pydantic `Field` | Applies to |
|-------------|------------------|------------|
| `minimum` | `ge` | numbers / integers |
| `maximum` | `le` | numbers / integers |
| `exclusiveMinimum` | `gt` | numbers / integers |
| `exclusiveMaximum` | `lt` | numbers / integers |
| `minLength` | `min_length` | strings |
| `maxLength` | `max_length` | strings |
| `minItems` | `min_length` | arrays |
| `maxItems` | `max_length` | arrays |
| `description` | `Field(description=...)` | any field (surfaced to the model by `instructor`) |

```python
Score = model_from_json_schema(
    {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["score"],
    }
)
Score(score=3)   # ok
Score(score=6)   # raises pydantic.ValidationError
```

Bounds are resolved through `$ref` chains of any depth and through nullable
wrappers, so a constraint declared inside a `$def` (even several `$ref` hops
deep) or on the non-null branch of a nullable field is still enforced (and
`null` itself still passes for a nullable field).

One form caveat: `exclusiveMinimum` / `exclusiveMaximum` are recognised in
their **numeric** (Draft 2020-12) form only. The Draft-4 / OpenAPI-3.0
*boolean* form (`"exclusiveMinimum": true` qualifying a sibling `"minimum"`)
is not recognised and is dropped — the bound is enforced as the sibling's
*inclusive* `minimum`/`maximum`. If your schema comes from an OpenAPI 3.0
document, rewrite exclusive bounds in the numeric form.

**Anything outside the table above is silently dropped** — `pattern`, `format`,
`multipleOf`, `uniqueItems`, `const`, and the rest are *not* enforced. This is
deliberate: partial enforcement that looks complete is worse than none. If a
schema relies on one of those, validate it elsewhere.

### Rate limiting

Rate limiting is **on by default**, scoped **per provider** (keyed by the effective provider name, matching how logging records it), across three independent dimensions:

- **Concurrency** — **on by default**, default cap **8 concurrent calls per provider**: enough headroom for the fan-out workloads consumers actually run, while still bounding a self-inflicted burst; lower it for a tightly-metered account, raise it for a local Ollama server. This cap is also **adaptive by default** — it backs off when a provider signals overload and recovers automatically (see [Adaptive concurrency](#adaptive-concurrency) below). The cap binds async callers and the `*_sync` wrappers alike — and because every `*_sync` call runs on llmkit's single persistent event loop, a thread-pool fan-out of sync calls shares one per-provider cap. One caveat follows from the concurrency gate being bound to a single loop: a process that drives calls on more than one loop (e.g. running the async call functions on its own event loop *and* the `*_sync` wrappers on the persistent loop) caps each loop's population independently, so it can momentarily hold up to *loops* × the cap per provider; the adaptive limit and the RPM/TPM budgets are shared across all of them.
- **Requests per minute (RPM)** — **opt-in**, off by default. A per-provider request-rate ceiling.
- **Tokens per minute (TPM)** — **opt-in**, off by default. A per-provider token-rate ceiling, debited by each call's measured token usage.

`configure_rate_limit(max_concurrent=..., enabled=..., rpm=..., tpm=..., adaptive=..., breaker=...)` sets them; `get_rate_limit_config()` reads back the effective `enabled` / `max_concurrent` / `rpm` / `tpm` / `adaptive` / `breaker` (handy to log or assert at startup); `configure_llm_logging(sink)` swaps the log sink (below). The `breaker` switch arms the opt-in per-provider [circuit breaker](#circuit-breaker).

```python
from llmkit import configure_rate_limit

# Stay under a metered account's published per-minute limits:
configure_rate_limit(rpm=3_500, tpm=2_000_000)
```

RPM and TPM are **opt-in** because — unlike concurrency, which has a universally sane default of 8 — the right per-minute number is the metered limit of *your* account, with no safe default to assume. Leaving them unset sends a request **byte-identical** to the pre-feature behaviour (no throttle on those dimensions). The binding limit on a metered cloud account is usually RPM/TPM rather than concurrency, so a migrator coming from a requests-per-minute knob should set `rpm=` here — **the concurrency cap does not stand in for an RPM limit** (the two limit different things, and an old RPM tuning otherwise goes inert). Both use a per-provider **token bucket**, which tolerates a small burst above the configured ceiling and then smooths to the sustained rate. That burst is deliberately small — `min(max_concurrent, rpm)` requests for RPM, roughly one second of tokens for TPM — *not* a full minute's quota. Against a provider that enforces a strict fixed minute window, the burst is the worst-case overshoot, so its *relative* size scales with your limits: with the default `max_concurrent=8` it is negligible at `rpm=3_500` (~0.2%) but a meaningful fraction of a small limit (8 extra requests on `rpm=50` is 16%). A tightly-metered account should lower `max_concurrent` (which shrinks the RPM burst with it) or set `rpm=` a little below the published number to leave headroom. When the RPM ceiling does make calls wait, they are admitted in **arrival order** (FIFO): a late arrival cannot jump the queue ahead of a caller already waiting *on the same event loop*, so no caller on that loop is starved under sustained saturation. The queue is **per loop**, exactly like the per-loop concurrency caveat above — a process that drives the same provider across more than one loop (the async call functions on its own loop *and* the `*_sync` wrappers on the persistent loop) gets per-loop FIFO rather than one global arrival order across them. (Host code that [joins the limiter directly](#joining-the-global-rate-limit-directly) through the synchronous `rate_limit_acquire_sync` orders on its own independent ticket queue as well.) The aggregate rate stays exact regardless of how the queues interleave. (A streamed call usually reports no token usage, so it does not debit TPM — consistent with cost being `None` for streamed calls.)

#### Adaptive concurrency

The per-provider concurrency limit is **adaptive by default**: when a provider pushes back with an overload signal (HTTP **429 / 503 / 529**), llmkit lowers that provider's in-flight limit, and raises it back toward `max_concurrent` once the provider stops pushing back — a TCP-style additive-increase / multiplicative-decrease (AIMD) loop, with **zero per-account tuning**. It *discovers* a safe concurrency under a sustained-overload window that a fixed cap cannot ride out, and is the library-side generalization of a hand-tuned RPM ceiling.

It only ever lowers the limit *below* `max_concurrent`, never above, so a provider that never throttles behaves **identically to a fixed cap** — adaptive concurrency is a no-op until the provider actually pushes back. The decrease only fires when you were genuinely *saturated* (running at the limit when the throttle arrived), so an isolated throttle received while running well under the cap doesn't penalise a healthy provider; recovery is paced by wall-clock time, so it is bounded regardless of how slowly your workload trickles. It is, honestly, a **trade**, not a free win: during a real overload it deliberately runs narrower (which is correct), and the residual cost is a brief reduced-width window after the provider recovers. Turn it off with `configure_rate_limit(adaptive=False)` to pin the limit at `max_concurrent`. (Adaptive concurrency is also a strong complement to `Retry-After`, [below](#retries) — the retry waits as the provider directs, while the fleet as a whole backs off.)

```python
from llmkit import configure_rate_limit

configure_rate_limit(max_concurrent=8)             # adaptive (the default)
configure_rate_limit(max_concurrent=8, adaptive=False)  # fixed cap, pre-feature behaviour
```

#### Observing backpressure

Install a `backpressure_callback` to *see* the adaptive limiter move in real time — for metrics, a budget-visibility dashboard, or just logging. The callback receives a `BackpressureEvent(provider, old_limit, new_limit, reason)` each time a provider's limit changes (`reason` is `"throttle"` or `"recover"`) or — when the circuit breaker is armed, [below](#circuit-breaker) — the breaker changes state (`"breaker_open"`, `"breaker_half_open"`, `"breaker_closed"`). It is read from a context variable — like `retry_progress_callback` — so it propagates across the `run_sync` boundary; install it once around your fan-out.

```python
from llmkit import backpressure_callback, BackpressureEvent

def on_backpressure(event: BackpressureEvent) -> None:
    print(f"{event.provider}: {event.old_limit} -> {event.new_limit} ({event.reason})")

with backpressure_callback(on_backpressure):
    ...  # calls here report every adaptive limit change to on_backpressure
```

A callback that raises is swallowed and logged — observability can never break a call.

#### Circuit breaker

Adaptive concurrency drives a struggling provider's limit *toward 1*; the **circuit breaker** is the "limit is effectively 0 while the provider is down" case it cannot express. It is **opt-in and off by default** — arm it with `configure_rate_limit(breaker=True)`. Once a provider's throttle rate over a rolling window (the last 20 real outcomes, at least half of them throttled) trips it, llmkit **fails fast** for that provider: every call raises `CircuitOpenError` *immediately* — holding no concurrency slot and deducting no RPM token — for a cooldown, instead of letting each call burn its retry budget into the storm (the load-*adding* pathology of retries under sustained overload). After the cooldown a single probe tests recovery: a clean success closes the breaker, any failure re-opens it for another cooldown.

```python
from llmkit import configure_rate_limit, CircuitOpenError, structured_llm_call

configure_rate_limit(breaker=True)  # opt in; off by default

try:
    result = await structured_llm_call(prompt, MySchema, feature="extract")
except CircuitOpenError as exc:
    ...  # the breaker is open for exc.provider — fall back fast, don't hammer it
```

It is **off by default** on purpose. Adaptive concurrency is a safe default because it only ever *reduces* load below your cap; the breaker changes the contract — it flips "eventually succeeds" into "fails fast" — so the host decides. `CircuitOpenError` is in `LLM_RECOVERABLE_ERRORS` (a host that already writes `except LLM_RECOVERABLE_ERRORS` to degrade on a 503 keeps catching it, and falls back fast) but the library **never retries it** — retrying a circuit you already know is open would defeat the point. Each provider has its own breaker, and the internal thresholds (window 20, trip fraction 0.5, cooldown 30s) are library-owned mechanism, not knobs — the public surface stays "three numbers and three switches" (`enabled` / `adaptive` / `breaker`).

#### Joining the global rate limit directly

llmkit's own call functions already pass every provider call through the
global, per-provider limit (concurrency on by default; RPM/TPM when
configured). If your app issues provider calls **outside** those functions —
for example a LangChain chat-model wrapper that calls the provider itself — you
can join the same per-provider budget by hand with the module-level acquire
functions:

```python
from llmkit.rate_limiting import (
    rate_limit_acquire_async,
    rate_limit_acquire_sync,
)

# Async path (e.g. an async _agenerate):
async with rate_limit_acquire_async("openai") as slot:
    response = ...  # one slot held against openai's budget
    slot.record_tokens(response.usage.total_tokens)  # debits TPM (no-op when off)

# Sync path (e.g. a synchronous _generate / _stream):
with rate_limit_acquire_sync("openai") as slot:
    response = ...  # one slot held against openai's budget
    slot.record_tokens(response.usage.total_tokens)
```

The argument is the **provider name** (`provider.name`, e.g. `"openai"`,
`"ollama"`); each provider has an independent budget on every dimension. Each
context manager yields a `RateLimitSlot`; call its `record_tokens(...)` once you
know the call's token usage to debit the TPM budget (a no-op when TPM is off).
Both are no-ops when rate limiting is disabled, and they share the exact
throttle llmkit's own call paths use, so a hand-joined slot counts against the
same budgets.

To check whether limiting is currently active, read the effective config rather
than reaching into the limiter:

```python
from llmkit.rate_limiting import get_rate_limit_config

if get_rate_limit_config().enabled:
    ...
```

`get_rate_limit_config().enabled` is the public replacement for the old
`GlobalRateLimiter.is_enabled()` check; `GlobalRateLimiter` itself is no longer
part of the headline surface (it remains importable from `llmkit.rate_limiting`
for internal use).

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

### Capturing call records

Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and `stream_text_with_log`) builds an `LLMCallRecord` and hands it to the configured log sink. A higher-level orchestrator that needs to cross-reference those calls — to total approximate cost, attribute spend per feature, or weave per-call traces — has two additive capture primitives, neither of which requires authoring a sink.

**`capture_llm_records()` — records (cost / metadata).** Wrap a scope to receive the `LLMCallRecord` for every call made inside it. Each record carries `approximate_cost` (a best-effort USD estimate, `None` when the provider doesn't report it), the resolved `model`/`provider`, `duration_ms`, `error`, and the rest — so a host gets cost and metadata without writing a custom sink. Capture is sink-independent: it works even with logging disabled (`configure_llm_logging(None)`), and crosses the `run_sync` sync bridge, so `structured_llm_call_sync` is captured exactly like the async path. One record is appended per attempt (retries each produce their own).

```python
from llmkit import capture_llm_records, structured_llm_call

with capture_llm_records() as records:
    result = await structured_llm_call(prompt, MySchema, feature="extraction")

total_cost = sum(r.approximate_cost or 0.0 for r in records)
```

**`capture_llm_log_paths()` — file paths.** Wrap a scope to receive the per-call log-file path written by the configured file sink. Only a file sink (the default `LocalYamlLogSink`) yields a path; with a third-party sink, or with logging disabled, the list stays empty — reach for `capture_llm_records()` when you want cost/metadata regardless of the sink.

```python
from llmkit import capture_llm_log_paths, structured_llm_call

with capture_llm_log_paths() as paths:
    result = await structured_llm_call(prompt, MySchema, feature="extraction")
# paths -> [PosixPath("data/llm-logs/...yaml"), ...]
```

### Write your own `LogSink`

`LogSink` is a `Protocol` with a single, file-agnostic method:

```python
class LogSink(Protocol):
    def write(self, record: LLMCallRecord) -> None: ...
```

A custom sink (a database, a metrics pipe, an in-memory buffer) is a one-method object that returns nothing; records (`LLMCallRecord`, a frozen dataclass) are handed to it for every call, and failures are swallowed so logging can never break a call. To send records somewhere other than local YAML — a database, an HTTP collector, structured stdout — implement `write` and register it:

```python
import logging
from llmkit import LLMCallRecord, configure_llm_logging

logger = logging.getLogger("llm-calls")

class StructuredStdoutSink:
    def write(self, record: LLMCallRecord) -> None:
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

configure_llm_logging(StructuredStdoutSink())   # pass None to disable logging entirely
```

The shipped `LocalYamlLogSink` additionally exposes the path it wrote via its own `write_returning_path(record) -> Path | None` method — that file detail stays off the shared `LogSink` contract, and it is what powers `capture_llm_log_paths()` internally.

An OpenTelemetry exporter (e.g. to Langfuse/Phoenix) is a natural future `llmkit[otel]` extra; the pluggable seam makes it a non-breaking addition.

## Configuration

`LLMClientConfig` is flat and carries only what a call needs:

```python
@dataclass(frozen=True)
class LLMClientConfig:
    provider: Provider               # OPENROUTER | OLLAMA | GOOGLE | ANTHROPIC | OPENAI | DEEPSEEK | BEDROCK
    model: str | None = None         # None -> the provider's own default model
    api_key: str | None = None
    base_url: str | None = None      # OpenRouter / OpenAI-compatible endpoints; unused by Google/Anthropic
    reasoning_effort: str | None = None  # "disable" | "low" | "medium" | "high"
    aws_region_name: str | None = None   # AWS Bedrock region; unused by every other provider
```

`aws_region_name` is the only AWS-shaped field, and it carries **only** the region. AWS Bedrock authenticates through the standard **AWS credential chain** (environment, shared config, or instance/role), so Bedrock secrets never pass through `LLMClientConfig`; leave the region `None` too and it resolves from the chain (`AWS_REGION_NAME` / `AWS_REGION`). Bedrock routing needs `boto3` for request signing — install it with the opt-in extra:

```bash
pip install "omg-llmkit[bedrock]"
```

The default model is Claude Haiku 4.5 via its **cross-region inference profile** id (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) — current Claude models on Bedrock are typically reached through inference profiles rather than plain on-demand ids. Pass a different profile- or partition-prefixed id as `model` (e.g. `eu.anthropic.claude-...`) when your account routes elsewhere.

Per-call `model=` overrides the default, so "strong/small/current" model roles are the host's concern — resolve them to a model string and pass it at the call site. The library has no opinion about roles.

`reasoning_effort` controls provider "thinking"/reasoning tokens, forwarded to LiteLLM. Leave it `None` (the default) for the provider's own behaviour — the outbound request is byte-identical to omitting it. Set it once (e.g. `"disable"`) and every call inherits it; the call functions also take a `reasoning_effort=` override for a single call. This matters most for Gemini, whose thinking is **on by default** and spends reasoning tokens against `max_tokens` — `reasoning_effort="disable"` turns it off so a small `max_tokens` cap doesn't truncate structured output.

Register the config with `configure_llm_client(source)`, where `source` is a zero-arg callable returning an `LLMClientConfig` (re-read on each provider construction, so it tracks live settings changes).

### Constructing a provider for a per-call override

Most callers configure one provider once via `configure_llm_client(...)` and let
every call pick it up. To send a *single* call through a different provider
family, model, or credential, build a provider on the fly and pass it as the
per-call `provider=` override. `make_provider` is the one-liner for that — it
builds straight from raw credentials, with no `LLMClientConfig` and no
module-level config source:

```python
from llmkit import make_provider, structured_llm_call_sync, Provider

provider = make_provider(Provider.ANTHROPIC, api_key=anthropic_key)
result = structured_llm_call_sync(
    prompt,
    output_schema=MyModel,
    feature="summarize",
    provider=provider,
)
```

`make_provider` accepts the knobs each provider actually reads —
`api_key`, `model`, `base_url`, `reasoning_effort`, `aws_region_name` — and
ignores the ones a given provider doesn't use (e.g. `base_url` for Anthropic,
`api_key` for Ollama or Bedrock, which signs via the ambient AWS credential
chain). Leave `model` unset to inherit the provider's own default; the assembled
LiteLLM id is always well-formed (e.g. `anthropic/claude-sonnet-4-6`).

**A fully per-call host needs no global config at all.** If you pass `provider=`
on *every* call, you don't have to call `configure_llm_client(...)` — there is no
global source to register, the call runs on the per-call provider alone, and the
log records that provider as the effective one. The "configure once globally" and
"provide per call" models are independent: use either, or mix them (a global
default with per-call overrides where needed). A call that passes *neither* a
per-call `provider=` nor a registered global source raises a clear
`RuntimeError` telling you to configure one.

To build from a full config instead, use `build_provider(config)`:

```python
from llmkit import build_provider, LLMClientConfig, Provider

provider = build_provider(LLMClientConfig(provider=Provider.OPENAI, api_key=key))
```

`LLMClientConfig.model` is optional. When it is `None` (or empty), the provider
falls back to its own built-in default model rather than emitting a broken
`"<prefix>/"` id.

#### Naming: `get_*` reads, `build_*` / `make_*` construct

The accessor verbs are split by intent:

- `build_provider(config)` / `make_provider(...)` **construct** a provider —
  from a config, or from raw credentials.
- `describe_llm(config)` (importable from `llmkit.providers`) and
  `get_rate_limit_config()` **read** effective state — a snapshot for
  display/telemetry; they construct nothing you keep.

`describe_llm` replaces the old `get_llm_config`, and `build_provider` replaces
`get_provider`; both old names are gone from the public surface.

### OpenRouter: schema-honoring routing

OpenRouter is a *router* — it forwards your request to one of several **serving
providers** behind each model. There's a sharp edge for structured output:
`structured_outputs` is a **model-level** capability, but the strict
`response_format` is actually enforced by the *serving* endpoint the request
lands on. A model can advertise the capability while one of its endpoints quietly
ignores the schema and returns free-form JSON — which then surfaces only as a
confusing downstream validation failure, with nothing pointing at routing as the
cause.

`OpenRouterProvider` defends against this **by default**: it sets OpenRouter's
[`provider.require_parameters`](https://openrouter.ai/docs/features/provider-routing#requiring-providers-to-support-all-parameters)
routing preference, so a request only lands on a serving endpoint that honors
*every* parameter sent — including the structured `response_format`. The trade-off
is that restricting routing to capable endpoints can in principle reduce
availability or shift cost. To opt out (and accept the silent-free-form-JSON
risk), construct the provider directly:

```python
from llmkit import structured_llm_call
from llmkit.providers import OpenRouterProvider

provider = OpenRouterProvider(api_key="sk-or-...", require_parameters=False)
result = await structured_llm_call(prompt, MySchema, feature="x", provider=provider)
```

Routing stays on for the config-driven path (`configure_llm_client` /
`build_provider`); the direct constructor above is the way to turn it off.

## Retries

Two retry layers, kept deliberately separate:

- **Transient-provider retries, on by default.** Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, `stream_text_with_log`) retries *transient* provider errors on its own — you don't wrap anything. The recoverable set splits into two budgets the policy counts **separately**:
  - **Transport errors** (`LLM_TRANSPORT_ERRORS`: 429 / 503 / 5xx, network/timeout) get the full `max_attempts` budget — **three attempts** by default — since a retry on a fresh connection routinely succeeds.
  - **Schema-validation errors** (`LLM_SCHEMA_ERRORS`: pydantic `ValidationError`, instructor `InstructorRetryException`) get the lower `validation_max_attempts` budget — **two attempts (one retry)** by default — so a transiently-malformed JSON response is still recovered, but a *deterministically-wrong* schema can't burn the full transport budget on doomed re-asks. (instructor wraps *transport* failures in `InstructorRetryException` too; the retry layer unwraps it, so a wrapped 429/5xx/network error still gets the full transport budget, not this lower one — and a wrapped *permanent* error such as a 401/400/403 fails fast after a single attempt, never charged to either budget.)

  `LLM_RECOVERABLE_ERRORS` remains the documented single catch-set — now the **union of three** subsets: the two above plus `LLM_BACKPRESSURE_ERRORS` (llmkit's own fail-fast `CircuitOpenError`, raised by the opt-in [circuit breaker](#circuit-breaker)). Keep using `LLM_RECOVERABLE_ERRORS` in `except` clauses; the split only changes how the *retry layer* budgets them — and the breaker subset is deliberately **never retried** (it sits outside the transport set the retry loop keys off, so re-asking a known-open circuit is impossible). One footnote: so that `import llmkit` doesn't pay LiteLLM's multi-second import cost, the litellm-native 503 entry (`litellm.exceptions.ServiceUnavailableError`) is a lazy stand-in resolved at `isinstance` time. `isinstance` checks — what the retry layer uses — behave identically, but a bare `except LLM_TRANSPORT_ERRORS:` / `except LLM_RECOVERABLE_ERRORS:` clause cannot catch that one litellm-native class (Python's `except` matching bypasses the lazy check); every other member still catches as usual, and an openai-SDK 503 arrives as `openai.InternalServerError`, which matches. Both budgets use bounded **full-jitter** backoff: the sleep before retry *n* is a random delay in `[0, min(backoff_base_seconds * 2**(n-1), max_backoff_seconds)]`, with the per-sleep cap (`max_backoff_seconds`) defaulting to 30s so a large attempt budget can't grow the worst-case sleep unboundedly. **`Retry-After` is honoured first:** when a retried provider error carries a `Retry-After` (a header — delta-seconds, `retry-after-ms`, or an HTTP-date — or the SDK's numeric attribute), the backoff waits *that* duration instead of the computed exponential, capped at `RetryPolicy.retry_after_cap` (default 60s) so a hostile value can't wedge a call. It is read from the *unwrapped* provider error, so a structured call honours it too, and it is honoured even when `backoff_base_seconds` is 0 (a server directive, not opt-in backoff); absent a header, the exponential is used unchanged. Programming errors (e.g. `TypeError`) are outside the recoverable set and propagate immediately, never retried. Each attempt is its own logged call, so `data/llm-logs/` shows one record per attempt.

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

  **`with_retries()`** (imported from `llmkit.retry`; see [`retry.py`](src/llmkit/retry.py)) remains the explicit, composable advanced path for wrapping *any* awaitable — useful when you want to retry a unit of work that isn't a single call function. The attempt count is `max_attempts` (total attempts including the first, **N not 1+N**); the previously-deprecated `max_retries` alias has been removed outright, so passing it now raises `TypeError`. Wrap a `retry_progress_callback(...)` scope around the work to observe per-attempt failures (e.g. for a progress UI):

  ```python
  from llmkit.retry import with_retries
  from llmkit import LLM_TRANSPORT_ERRORS

  result = await with_retries(
      lambda: do_some_work(),
      max_attempts=3,
      backoff_base_seconds=0.5,
      retry_on=LLM_TRANSPORT_ERRORS,
  )
  ```

  A `RetryProgressCallback` is invoked once per non-final failed attempt with keyword arguments `label`, `attempt`, `max_attempts`, and `error` — the callback keyword is `max_attempts` (it was previously `max_retries`; rename it):

  ```python
  def on_retry(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
      print(f"{label}: attempt {attempt}/{max_attempts} failed: {error}")
  ```

  > **Don't double-wrap the call functions.** They already retry internally, so `with_retries(structured_llm_call, ...)` would otherwise multiply the budgets (the `3 × 3 = 9` trap). `with_retries` guards against this — it detects an active inner llmkit retry loop and collapses the inner layer to a single pass (warning once), so the budgets don't multiply. To drive retries entirely from your own wrapper instead, opt the inner call out with `retry=NO_RETRY`.

- **instructor's own in-call schema repair** re-asks the model to fix malformed JSON *within a single call*, before any `ValidationError`/`InstructorRetryException` reaches the retry layer. llmkit pins instructor's `max_retries` to **2** — instructor counts *total attempts*, so that is two in-call attempts, i.e. exactly one repair re-ask — and it is not a caller-facing knob. This stays **separate** from the cross-call retry layer above: instructor repairs within one attempt; the policy's `validation_max_attempts` (default 2) governs how many *fresh* attempts a persistent schema failure earns. The two budgets are never conflated, so attempts aren't double-counted.

### Re-rolling on a semantically-bad result

A response can pass the schema and still be *wrong* — an empty register, a citation that doesn't resolve, a total that doesn't reconcile. Rather than hand-rolling an LLM-then-validate-then-re-roll loop around the call, pass an `on_result` hook: it's called with each attempt's result, and raising `ResultValidationError` from it **rejects** that result and re-rolls the call.

```python
from llmkit import structured_llm_call, ResultValidationError

def _must_have_findings(report: Report) -> None:
    if not report.findings:
        raise ResultValidationError("empty report — re-roll")

result = await structured_llm_call(
    prompt, Report, feature="reports", on_result=_must_have_findings,
)
```

The re-roll is charged against the **validation budget** (`RetryPolicy.validation_max_attempts`, default 2) — the same budget a schema failure uses, and for the same reason: a deterministically-bad result shouldn't burn the full transport budget on doomed re-asks. When the budget is exhausted the last `ResultValidationError` propagates. Each attempt — including a rejected one — is its own logged call, so `data/llm-logs/` shows the rejected response alongside the error.

`on_result` is available on `structured_llm_call` and `text_llm_call`, and on both sync wrappers (`structured_llm_call_sync`, `text_llm_call_sync`); the text-path hooks receive the response *text*. It is *not* part of `LLMCallOptions` — like `feature`, it stays a conscious per-call choice.

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run basedpyright          # recommended tier; clean with no baseline
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
