# llmkit

A thin, opinionated, **local-first** layer over [LiteLLM](https://github.com/BerriAI/litellm) (with [instructor](https://github.com/567-labs/instructor) for structured output). It gives an application one provider-agnostic call surface across **OpenRouter, Google AI Studio, Google Vertex AI, Anthropic, OpenAI, DeepSeek, AWS Bedrock, and local Ollama**, with validated structured output, per-provider rate limiting (concurrency on by default; optional requests-/tokens-per-minute), **agent-readable per-call logging**, and **transient-error retries on by default** — all out of the box.

LiteLLM is the implementation of the HTTP providers; llmkit owns the ergonomic call surface, the structured-output mode pinning, the rate-limit policy, and the logging convention. It is **not** a gateway and does not reimplement transport — that is solved, and reimplementing it is the thing this library deliberately does not do.

## Why llmkit

- **Structured output that actually validates.** Each provider is pinned to an explicit, live-measured structured-output mode (never instructor's auto-`Mode.TOOLS`, which silently regresses Gemini to empty shapes), and instructor's in-call validation-retry repairs malformed JSON. You pass a Pydantic model; you get a validated instance back. (A completion *truncated by the output-token limit* is the one thing never re-asked — it fails fast as `OutputLimitError`; see [Retries](#retries).)
- **Provider switching is config, not code.** OpenRouter / Google AI Studio / Google Vertex AI / Anthropic / OpenAI / DeepSeek / AWS Bedrock / Ollama behind one `Provider` enum and one `LLMClientConfig`. Call sites never change when you switch. The same Gemini models are reachable two ways — AI Studio (a bearer key) or Vertex AI (Google Cloud, with a `vertex_location` data-residency control) — just like Claude is reachable direct or via Bedrock.
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

`llmkit.__version__` reports the installed version. It is read from the
`omg-llmkit` distribution's metadata rather than hardcoded, so it cannot drift
from what you installed; a source tree with no installed metadata reports
`"0.0.0+unknown"` instead of raising at import.

Requires Python ≥ 3.12.

The core install routes OpenRouter, Google AI Studio, Anthropic, OpenAI,
DeepSeek, and Ollama with no extra dependencies. Two providers gate their
dependencies behind opt-in extras so hosts pay only for what they call:

```bash
pip install "omg-llmkit[bedrock]"    # Claude-on-Bedrock (boto3 for SigV4 signing)
pip install "omg-llmkit[vertex]"     # Gemini via Google Cloud Vertex AI
pip install "omg-llmkit[all]"        # both of the above
```

Bedrock's `[bedrock]` extra pulls only `boto3` (LiteLLM signs Bedrock requests
with AWS SigV4 through it), and constructing the `BedrockProvider` without it
raises a clear `install omg-llmkit[bedrock]` error at construction, not a
cryptic failure on the first call. Vertex is the same shape: its `[vertex]`
extra pulls only `google-auth` (LiteLLM mints the Vertex OAuth token through
it), and constructing the `VertexProvider` without it raises a clear `install
omg-llmkit[vertex]` error — a non-Vertex host takes on no Google dependency.
Direct Anthropic routing needs no SDK at all: LiteLLM speaks the Anthropic
HTTP API itself, so the `[anthropic]` extra that existed through 0.6.x is
gone.

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
| `text_llm_call_stream(prompt, feature, label, ...)` | Async generator yielding text chunks, logged on completion |
| `tool_llm_call(prompt, tools, feature, ..., output_schema=...)` | Async tool turn; with a schema, returns tool calls or a validated final answer |
| `tool_llm_call_sync(...)` | Synchronous wrapper around the tool turn |

`prompt` is typed `str | list[Message]` on all call functions. A plain string is sent as-is; the list form is a list of `llmkit.Message` — a `TypedDict` whose `role` is `"system"`, `"user"`, or `"assistant"` and whose `content` is either a string or a list of content-part dicts, the multimodal shape LiteLLM accepts (`{"type": "text", ...}`, `{"type": "image_url", ...}`), forwarded verbatim. `Message` is exported so your own prompt builders can be annotated against it rather than against the transport's wire shape: an unknown key (`{"roel": ...}`) or a mistyped role is a type error, and multimodal content type-checks instead of being rejected.

### Tool calls with a structured final answer

The portable pattern is still **two steps**: run a tool loop with `tool_llm_call`, then make one `structured_llm_call` for the final answer. That works across every provider, including Gemini.

For a measured compose-capable route, pass `output_schema=` to the tool call instead. Each turn is either a normal `ToolComposeResult` with `tool_calls` and `parsed is None`, or a final answer with `parsed` set to your validated Pydantic instance. Anthropic and OpenAI support the combined request; unsupported providers and unsafe legacy Claude models raise `ComposeUnsupportedError` *before any request* with a steer to the portable pattern.

```python
from pydantic import BaseModel
from llmkit import ToolDefinition, tool_llm_call

class Answer(BaseModel):
    total: int

result = await tool_llm_call(
    "Calculate 2 + 3; use the tool if needed.",
    [ToolDefinition("add", "Add values", {"type": "object"})],
    output_schema=Answer,
    feature="calculator",
)
if result.tool_calls:
    # Execute calls and make the next tool turn with the updated history.
    ...
else:
    assert result.parsed is not None
    assert result.parsed.total == 5
```

Use `describe_llm(config).compose_tools_schema` (from `llmkit.providers`) to choose the optimization without using exceptions. Compose calls preserve the tool lane's retry, rate-limit, usage, and log behavior; their log schema is `tools+<ModelName>`. The compose lane validates the final text locally and retries a malformed final answer within `validation_max_attempts`; unlike the instructor-backed structured lane it has no repair prompt, so the complete tool history is re-sent on a retry. Gemini's compose feature remains preview-only, so llmkit deliberately keeps it on the portable path; its `gemini_structured_output="json"` escape hatch is an instructor-mode setting and does not apply here.

**Type-check migration.** `prompt` was previously `str | list[dict[str, str]]`, so a call site passing a *variable* annotated `list[dict[str, str]]` now fails the type check — re-annotate it `list[Message]`. Inline message-dict literals are unaffected, and runtime behaviour is identical either way (a `TypedDict` is a plain `dict`).

> **Deprecated alias.** `stream_text_with_log` is the old name for
> `text_llm_call_stream`; it still works (same signature and behaviour) but
> warns `DeprecationWarning` and is removed in 1.0. Switch the call.

> **Two defaults worth knowing up front.**
> - **`temperature` defaults to `0.2`** — biased toward deterministic output. A *creative* caller must override it explicitly (e.g. `temperature=1.0`); it is otherwise quietly conservative. Pass `temperature=None` to send **no `temperature` field at all**, using the provider's own default sampling — see [Provider-default sampling (`temperature=None`)](#provider-default-sampling-temperaturenone).
> - **Any call takes a per-call `provider=` override** — route a single call through a different provider family, model, or credential without touching the global `configure_llm_client(...)` registration. See [Constructing a provider for a per-call override](#constructing-a-provider-for-a-per-call-override).

### Reusing call options

The call functions (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and `text_llm_call_stream`) take a block of per-call keyword arguments. When a feature module makes many calls with the same settings, repeating that block at every site is noise. Build an `LLMCallOptions` once and pass it as `options=`:

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

`LLMCallOptions` is **frozen** and carries any subset of `temperature` / `model` / `max_tokens` / `reasoning_effort` / `retry` / `provider`. Every field is optional and *unset* by default — an unset field defers to the call's keyword (and through it to the configured client), so a partially-filled `LLMCallOptions` only supplies the fields you set. Like the call keywords, `temperature` accepts `None` (`LLMCallOptions(temperature=None)`) to request the provider's default sampling instead of llmkit's `0.2`.

`feature` is intentionally **not** part of `LLMCallOptions`. It stays a required per-call keyword as a telemetry forcing function: it scopes the per-call log filename and the `index.jsonl` grouping operators grep, so it must be a conscious choice at each call site rather than something defaulted-away into a shared object.

The flat-keyword path is unchanged — pass no `options` and nothing about existing calls changes.

#### Call-vs-config precedence

`model` and `reasoning_effort` are *dual-homed*: they can be set both on `LLMClientConfig` (the app-wide default) and on the call surface. The precedence, lowest to highest, is:

**config < `options` < explicit per-call keyword**

So a value passed directly as a keyword wins — *any* passed value, including `None` or one equal to the documented default. The call functions' mergeable keywords default to the `UNSET` sentinel rather than to real values, so "was it passed" is a structural fact, never inferred by comparing values: `structured_llm_call(..., temperature=0.2, options=LLMCallOptions(temperature=0.9))` runs at `0.2`, and an explicit `model=None` forces the provider/config default even when `options` carries a model. The same holds for `temperature=None`: it overrides a numeric `options` value (`structured_llm_call(..., temperature=None, options=LLMCallOptions(temperature=0.9))` sends no `temperature` field), and `LLMCallOptions(temperature=None)` overrides the `0.2` default when the keyword is unset. An `LLMCallOptions` field sits between the keyword and the config; when neither the keyword nor `options` supplies a value, the true default applies (`DEFAULT_TEMPERATURE`, `DEFAULT_RETRY_POLICY`, or the configured `LLMClientConfig` resolution for `model`/`reasoning_effort`). An *unset* `LLMCallOptions` field never overrides config — only a field you explicitly set on the options participates.

`Unset` (the type) and `UNSET` (the value) are exported for one idiom: your own typed wrapper can declare `temperature: float | None | Unset = UNSET` and forward it unconditionally, letting llmkit resolve "not passed" vs "provider default" (`None`) vs a number instead of re-inventing the sentinel. Compare with `is`/`is not` only — never truthiness.

#### Provider-default sampling (`temperature=None`)

The three states are distinct, and each means something different on the wire:

| You pass | Meaning | Wire |
|---|---|---|
| *(nothing)* | use llmkit's default | `temperature: 0.2` |
| `temperature=0.5` (any number, incl. `0.0`) | your value | `temperature: 0.5` |
| `temperature=None` | provider's built-in default sampling | **no `temperature` key at all** |

`None` is accepted on every call surface — structured, plain-text, streaming, the sync wrappers, and the deprecated `stream_text_with_log` alias — both as a direct keyword and through `LLMCallOptions`. When the resolved value is `None`, llmkit omits the `temperature` kwarg from the provider request entirely (an identity check, so `0.0` is never mistaken for unset). Log records reflect the omission: `LLMCallRecord.temperature` is `None` and the YAML sink writes `temperature: null`, distinct from the `0.2` a default call records.

**Gemini 3.x caveat.** Google's Gemini 3 guidance deprecates `temperature`/`top_p`/`top_k` and recommends removing them from every request; LiteLLM currently emits a `DeprecationWarning` when any of them is present. Omitting the field via `temperature=None` removes that warning — but with every *released* LiteLLM, `VertexGeminiConfig.map_openai_params` re-inserts `temperature = 1.0` (Google's recommended value) for Gemini 3 models after the kwarg is dropped, so the deprecated field is still sent on the wire for Gemini 3.x specifically. Removing that injection is an upstream LiteLLM change; until its release is consumed by llmkit (tracked in the `contribute-and-consume-litellm-gemini-3-temperature-fix` task), `temperature=None` on Gemini 3.x means "no warning, wire value `1.0`".

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

**Supported subset** (anything outside it raises a clear `ValueError` naming the construct): `object` with `properties` and a `required` array; scalars (`string` / `integer` / `number` / `boolean`, plus `null` / nullable); `array` with `items` (including arrays of objects); `enum` (string or integer members); nested objects inline or via local `$ref` (`#/$defs/...`); and `additionalProperties` as `true` / `false` / absent (a *typed* `additionalProperties` map is rejected). A non-required field becomes an optional defaulting to `None`, and the generated model's `model_dump` / `model_dump_json` **drop a `None` left in an optional field by default** — so an omitted optional is *absent*, not `"field": null` (which would fail downstream re-validation against the same schema). The drop is scoped to optionals: a *required*-but-nullable field explicitly set to `None` is kept. Pass `exclude_none=False` to keep every null, or `exclude_none=True` to drop them all. A title-less schema still gets a valid default class name (`JsonSchemaModel`); pass `name=` to set it explicitly. Generated models default to **`extra="forbid"`**, so a response carrying a key not in the schema is *rejected* rather than silently kept — for an LLM output contract you want a hallucinated extra field to fail loudly (stricter than JSON Schema's permissive `additionalProperties` default); `"additionalProperties": true` opts an object into `extra="allow"` (extra keys accepted and kept), while `false` or absent stays strict. An `"type": "object"` with **no properties** — `properties` absent *or* an explicit empty `{}` — raises rather than silently building a zero-field model that rejects every real response; set `"additionalProperties": true` for an intentionally free-form object.

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
`null` itself still passes for a nullable field). The same resolution carries a
`$def`'s **`description`** through, including behind a nullable wrapper — a
nullable `$ref` field surfaces the target's model-facing guidance just like a
bare `$ref` does.

Keywords that sit **beside** a `$ref` are handled by kind. Metadata and the
bounds above merge with the referenced schema, the outer (property-level) value
winning on conflict — so `{"$ref": "#/$defs/Count", "minimum": 5}` keeps the
bound. A keyword that would redefine the reference's *structure* (`type`,
`enum`, `items`, `properties`, `required`, `additionalProperties`, `anyOf`,
`oneOf`) is a JSON-Schema conjunction a merge cannot express, so it is
**rejected** unless it restates the target's own value — an `enum` beside a
`$ref` raises a clear `ValueError` rather than silently widening the field to an
unconstrained scalar. Move such a keyword into the referenced `$def`, or inline
the schema.

**Subschema applicators are rejected everywhere, not only beside a `$ref`.**
`allOf`, `not`, `if` / `then` / `else`, `dependentSchemas`,
`dependentRequired`, `propertyNames`, `patternProperties`, `prefixItems`,
`contains`, `unevaluatedProperties`, and `unevaluatedItems` constrain an
instance by *composition*, and a generated field is one annotation plus the
bounds in the table above — so an applicator has nowhere to land. Each raises a
`ValueError` naming the keyword and its path, at every site: a property, an
array's `items`, a nested object, the body of a `$def` reached through a
`$ref`, the non-null branch of a nullable `anyOf`, and the root object.

```python
model_from_json_schema(
    {
        "type": "object",
        "properties": {"score": {"type": "integer", "allOf": [{"maximum": 10}]}},
        "required": ["score"],
    }
)
# ValueError: Unsupported keyword 'allOf' at '$.score': subschema applicators ...
```

Rejecting is not pedantry — dropping one is wrong in *both* directions. A
dropped `allOf` bound makes the model accept a value the schema forbids; a
dropped `prefixItems` changes what the sibling `items` means (Draft 2020-12
reads it as "every element *after* the prefix"), so the model rejects a
response the schema permits and the spurious `ValidationError` is paid for in
retries. Either express the constraint with a supported keyword, or validate it
outside the generated model. The nullable spellings — `anyOf` / `oneOf` with a
single non-null branch plus a `null` branch — are unaffected and still build.

A propertyless object is rejected the same way: `{"type": "object"}` with no
`properties` — or an explicit empty `{"properties": {}}` — raises at translation
time (it would otherwise build a zero-field model that rejects every real
response), unless it opts into open-ended keys with `"additionalProperties":
true`.

One form caveat: `exclusiveMinimum` / `exclusiveMaximum` are recognised in
their **numeric** (Draft 2020-12) form only. The Draft-4 / OpenAPI-3.0
*boolean* form (`"exclusiveMinimum": true` qualifying a sibling `"minimum"`)
is not recognised and is dropped — the bound is enforced as the sibling's
*inclusive* `minimum`/`maximum`. If your schema comes from an OpenAPI 3.0
document, rewrite exclusive bounds in the numeric form.

**Leaf constraints outside the table above are silently dropped** — `pattern`,
`format`, `multipleOf`, `uniqueItems`, `const`, and the rest are *not*
enforced. This is deliberate: partial enforcement that looks complete is worse
than none. If a schema relies on one of those, validate it elsewhere. The drop
is scoped to per-value keywords like these; a *structural* construct outside
the supported subset — a subschema applicator, a multi-variant union, a typed
`additionalProperties` map — raises instead of vanishing, because losing one
silently changes the shape the model validates rather than merely leaving a
value unchecked.

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

`LocalYamlLogSink` (the default) writes **two** things to the log directory:

1. **One YAML file per call, laid out verdict-first.** The file opens with a one-line `#` header — `ok`/`ERROR`, feature/label, resolved model, schema, duration, approximate cost — so `head -1 *.yaml` triages a whole run (the second header line carries the timestamp plus `call=<id> attempt=<n>`, so retries of one logical call are joinable from the file heads alone). Small metadata is next; the large `response` and `prompt` blobs are last, so the *head* of the file is the whole story for most reads.
2. **A compact append-only `index.jsonl`** — one JSON line per call (file, timestamp, feature, label, model, provider, schema, run_id, call_id, attempt, duration, queue wait, cost, error). Cross-call questions — "which calls errored / were slowest / most expensive / the last call for feature X" — are a single small scan instead of globbing and parsing every YAML.

```
# ok | reports/exec_summary | google/gemini-2.5-flash | Summary | 1840ms | $0.0007
# 2026-06-05T14:22:31.004512 | call=9f3c21ab attempt=1

timestamp: '2026-06-05T14:22:31.004512'
feature: reports
label: exec_summary
model: google/gemini-2.5-flash
provider: openrouter
schema: Summary
run_id: nightly-eval-2026-06-05
call_id: 9f3c21ab54d64f1f8f2c14febc03a7d1
attempt: 1
temperature: 0.0
max_tokens: null
reasoning_effort: null
duration_ms: 1840.2
queue_wait_ms: 0.4
approximate_cost: 0.0007
error: null
response: ...
prompt: ...
```

`approximate_cost` is LiteLLM's per-response estimate for budget visibility — **not** a billing figure (and `None` when the provider does not report it, e.g. streamed calls). `call_id` is one id per *logical* call and `attempt` the 1-based attempt within it, so the N records a retried call produces join on `call_id`. `duration_ms` measures the whole attempt **including** `queue_wait_ms` — the time spent queued behind llmkit's own rate limiter — so provider latency is approximately `duration_ms - queue_wait_ms` (hook time and in-call schema-repair re-asks are also inside `duration_ms`). `queue_wait_ms` is `float | None`, not always a float: it is `0.0` when the limiter is disabled and `None` when the attempt failed *before* acquiring a slot, so a custom sink or `index.jsonl` parser has to handle the null rather than subtract it blindly. `temperature` is the same kind of field now: an omitted temperature (`temperature=None` on the call) records as `null`, distinct from the `0.2` a default call records, so a typed custom sink reading `record.temperature` must handle `None` (a `float` format spec, for instance, will raise). `run_id` is the *outer* scope `call_id` does not give you — see below — and is `null` unless you set one.

### Grouping calls by run

`call_id` joins the attempts of one logical call. It cannot tell you which calls belonged to one **run** — an eval sweep, a rehearsal, an incident replay — and neither can the timestamp, because a time window stops discriminating the moment two runs overlap. Set a `run_id` and every record, YAML body and `index.jsonl` line carries it, so one shared log directory stays filterable:

```bash
jq 'select(.run_id == "nightly-eval-2026-06-05")' index.jsonl
```

Three ways to set it, most specific first:

```python
import llmkit

llmkit.set_run_id("nightly-eval-2026-06-05")   # process-wide, until cleared with None

with llmkit.run_scope("tuner-session-7"):      # scoped to the block
    ...

llmkit.get_run_id()                            # what's in force right now
```

…and `LLMKIT_RUN_ID` in the environment, which needs no code change at all:

```bash
LLMKIT_RUN_ID=nightly-eval-2026-06-05 python -m your_eval_sweep
```

The environment variable is the lowest layer, so an explicit `set_run_id` or `run_scope` overrides it; a blank one counts as unset. A blank *programmatic* value raises instead — pass `None` to mean "no run id". With nothing set, `run_id` is `null` and records are otherwise exactly what llmkit wrote before this existed.

Which of the two programmatic setters you want depends on your concurrency, because they fail in opposite directions. `run_scope` is context-scoped: it survives the sync bridge (a `*_sync` call runs on llmkit's persistent loop inside a copy of your context), but a `threading.Thread` you start yourself gets a *fresh* context and will not see it. `set_run_id` is a process global — visible from every thread and every loop, but a single value, so it cannot express two runs overlapping in one process. Use `set_run_id` for "this process is one run" (including a thread-pool fan-out), `run_scope` for a host driving several runs at once.

`run_id` and `LLMKIT_LOG_DIR` are independent and compose freely — but tagging is what lets you keep *one* greppable history instead of walking N per-run directories.

### Where the logs go

The default directory is resolved **lazily at the first write** and then frozen for the sink's lifetime (a mid-run `chdir` can't split logs):

1. `LLMKIT_LOG_DIR`, when set;
2. `data/llm-logs/` under the nearest ancestor directory carrying a `pyproject.toml` or `.git` (nearest wins) — and when the sink creates that directory it also seeds a `.gitignore`, so prompt logs never land in your repository's history;
3. otherwise a per-user state directory (`$XDG_STATE_HOME/llmkit/llm-logs` on Linux, `~/Library/Logs/llmkit` on macOS, `%LOCALAPPDATA%\llmkit\logs` on Windows) — never a CWD-relative path.

The first successful write logs one INFO naming the absolute directory and the retention policy. `default_log_dir()` returns the currently-resolved answer; an explicit `LocalYamlLogSink(log_dir=...)` is used as given, made absolute at construction so that a *relative* path (or a relative `LLMKIT_LOG_DIR`) names one directory for the sink's lifetime instead of following the process around. Pass `configure_llm_logging(None)` to disable logging entirely.

On POSIX, a sink-created directory is `0o700` and log files are `0o600` — prompt data is owner-only by default. A pre-existing directory is never re-chmodded: pre-create the directory yourself to share logs with other readers.

### Retention: bounded by default

Long-running services no longer accumulate unbounded prompt data. By default the sink prunes per-call YAML files older than **30 days** and rotates `index.jsonl` past **50 MiB** to a date-stamped sibling (which ages out under the same policy). Housekeeping runs on the write path, throttled to once per hour, off the event loop.

```python
LocalYamlLogSink(retention_days=None)                  # keep everything forever
LocalYamlLogSink(retention_days=7, max_index_bytes=None)  # tighter age bound, no rotation
LocalYamlLogSink(retention_days=0)                     # ValueError — see below
```

⚠️ **The sink owns its directory's `*.yaml` and `index-*.jsonl` namespace.** Pruning is a glob over `log_dir`, not a list of files the sink remembers writing: **any** `*.yaml` or rotated-index `index-*.jsonl` file in that directory whose mtime is older than `retention_days` is deleted, whoever wrote it. That is deliberate — a consumer who parks cross-reference YAML beside the per-call logs usually wants the two to rot on the same clock — but it makes `log_dir` llmkit's directory rather than shared storage. If you keep your own files there and want them to outlive the policy, give llmkit a directory of its own (`LocalYamlLogSink(log_dir=...)` or `LLMKIT_LOG_DIR`), or turn pruning off with `retention_days=None`. The active `index.jsonl` is never age-pruned — only its rotated `index-<timestamp>.jsonl` generations are.

Each bound takes a **positive** integer or `None`; `0` and negatives raise `ValueError` at construction. `0` is rejected rather than quietly interpreted because it has two opposite plausible meanings — "keep nothing" (`logrotate`'s `rotate 0`) and "no limit" (many SaaS retention settings) — and the destructive reading is what the mechanics would actually do: the prune cutoff would be *now*, so the first write would delete every log in the directory, including the file it had just written and whose path it returns. `None` is the only opt-out, for both bounds.

Sink I/O never runs on the event loop: writes are offloaded to a worker thread (the one deliberate exception is a stream abandoned mid-flight, whose record is written synchronously so it can't be lost while the generator unwinds).

### Capturing call records

Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and `text_llm_call_stream`) builds an `LLMCallRecord` and hands it to the configured log sink. A higher-level orchestrator that needs to cross-reference those calls — to total approximate cost, attribute spend per feature, or weave per-call traces — has two additive capture primitives, neither of which requires authoring a sink.

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

`configure_llm_logging` checks the object you hand it: anything that is neither `None` nor a `LogSink` — including the sink *class* where you meant an instance — raises `TypeError` there and then, rather than being installed and silently swallowing every subsequent call's log. The check is structural, so it tests that `write` exists, not that its signature is right (a type checker catches the rest at the call site). One consequence in your own tests: a bare `MagicMock()` doesn't register as a sink, because structural checks use static attribute lookup and see through no `__getattr__` — stub with `Mock(spec=LogSink)`, a small fake class, or `configure_llm_logging(None)`.

The shipped `LocalYamlLogSink` additionally exposes the path it wrote via its own `write_returning_path(record) -> Path | None` method — that file detail stays off the shared `LogSink` contract, and it is what powers `capture_llm_log_paths()` internally. A sink that defines `write_returning_path` opts into path capture and must honor that return type: anything else is treated as a failed write (one warning, no captured path), so `capture_llm_log_paths()` only ever hands you real `Path`s.

An OpenTelemetry exporter (e.g. to Langfuse/Phoenix) is a natural future `llmkit[otel]` extra; the pluggable seam makes it a non-breaking addition.

## Configuration

`LLMClientConfig` is flat and carries only what a call needs:

```python
@dataclass(frozen=True)
class LLMClientConfig:
    provider: Provider               # OPENROUTER | OLLAMA | GOOGLE | ANTHROPIC | OPENAI | DEEPSEEK | BEDROCK | VERTEX
    model: str | None = None         # None -> the provider's own default model
    api_key: str | None = None       # bearer providers only; masked in repr; else the provider env var, else raises
    base_url: str | None = None      # endpoint: config, then env var, then a default (sent as api_base; see below); not accepted by Bedrock/Vertex
    reasoning_effort: ReasoningEffort | None = None  # "disable" | "low" | "medium" | "high", or any provider value
    aws_region_name: str | None = None   # AWS Bedrock region; not accepted by any other provider
    vertex_project: str | None = None    # Vertex AI GCP project; not accepted by any other provider
    vertex_location: str | None = None   # Vertex AI region (data residency); not accepted by any other provider
    gemini_structured_output: Literal["schema", "json"] = "schema"  # Gemini strategy; a non-default value is not accepted by any other provider
```

**Populating a knob the selected provider does not read is rejected, not
ignored.** Each provider declares which provider-shaped fields it honours, and
`build_provider` / `make_provider` raise a clear `ValueError` if the config
carries any other populated one — an `api_key` for Bedrock/Vertex, a `base_url`
for a fixed-endpoint provider, a non-default `gemini_structured_output` for a
non-Gemini provider. Populate only the fields the active provider uses. (This
closes a silent-misconfiguration footgun: a config generically filled from a
settings object no longer *looks* like it pinned a credential or endpoint that
was in fact dropped.)

**`api_key` is masked in the config's `repr`.** A set key renders as
`api_key=<redacted>`, so a stray `print(config)`, log line, or traceback never
leaks the credential; its presence still shows for debugging.

**Bearer-key resolution is explicit.** The five key-authenticated providers
resolve their key from `api_key` if set, else the provider's own environment
variable, else they raise at construction naming the variable — no silent
fallback to an ambient key:

| Provider | Environment variable |
|----------|----------------------|
| `OPENROUTER` | `OPENROUTER_API_KEY` |
| `ANTHROPIC` | `ANTHROPIC_API_KEY` |
| `OPENAI` | `OPENAI_API_KEY` |
| `DEEPSEEK` | `DEEPSEEK_API_KEY` |
| `GOOGLE` (AI Studio) | `GEMINI_API_KEY` |

`OLLAMA` needs no key; `BEDROCK` and `VERTEX` authenticate through their ambient
AWS / Google credential chains (below), so none of the three accepts an
`api_key`.

**Endpoint resolution is explicit too.** Every provider that accepts a
`base_url` resolves its endpoint the same way — the configured `base_url` if
set, else the first non-empty environment variable in the order below, else
llmkit's own default — and sends the result on the wire as `api_base`. (`GOOGLE`
is the one exception, deliberately: it owns no default, so with nothing
configured it sends no `api_base` at all — see below.) So the endpoint is
llmkit's decision, readable up front from the provider's
`completion_kwargs()`, rather than something LiteLLM's internal chain resolves
later. The resolution runs **per call**, not when the provider is constructed,
so it reflects the environment the call actually goes out in: `import llmkit`
does not import LiteLLM (that is deferred to the first call) and importing
LiteLLM is what runs `load_dotenv()`, so a provider built once at startup with
`make_provider(...)` and used later would otherwise answer from an environment
the host's `.env` had not been read into yet.

| Provider | Environment variable(s), in precedence order | Default endpoint |
|----------|----------------------------------------------|------------------|
| `OPENROUTER` | *(none)* | `https://openrouter.ai/api/v1` |
| `ANTHROPIC` | `ANTHROPIC_API_BASE`, then `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` |
| `OPENAI` | `OPENAI_BASE_URL`, then `OPENAI_API_BASE` | `https://api.openai.com/v1` |
| `DEEPSEEK` | `DEEPSEEK_API_BASE` | `https://api.deepseek.com/beta` |
| `GOOGLE` (AI Studio) | `GEMINI_API_BASE` | *(none — LiteLLM picks, see below)* |
| `OLLAMA` | *(none)* | `http://localhost:11434` |
| `BEDROCK` | `AWS_BEDROCK_RUNTIME_ENDPOINT` — read by **LiteLLM**, not llmkit | derived from `aws_region_name` by LiteLLM |
| `VERTEX` | `VERTEXAI_API_BASE`, outranked by the `litellm.api_base` global — both read by **LiteLLM**, not llmkit | derived from `vertex_location` by LiteLLM |

The last two rows are governed differently from the six above them, and the
[section below](#bedrock-and-vertex-do-not-own-their-endpoints) says why and what
it costs.

**`GOOGLE` is the one provider with no llmkit default**, and the absence is
deliberate: its base is the one that cannot be a constant. The AI Studio base
carries an API *version* that LiteLLM derives **from the model** — `v1alpha` for
Gemini 3 and newer, `v1beta` otherwise — and it applies that only when no
`api_base` is given; a base that *is* given is used verbatim. Measured against
litellm 1.92.0 (2026-07-21): pinning
`https://generativelanguage.googleapis.com/v1beta` sent `gemini-3-pro-preview`
to `/v1beta` where it had gone to `/v1alpha`, and pinning the bare host dropped
the version segment entirely — so any static default would silently change the
wire shape for some models, which is the one thing this resolution promises not
to do. Deriving the version inside llmkit would be endpoint *computation*, the
gateway-shaped work the library deliberately does not do. So for Google: a
configured `base_url` wins, else `GEMINI_API_BASE`, else **no `api_base` is sent
at all** and LiteLLM picks the endpoint, and with it the model's API version.
That last case has a real, scoped cost: with neither configured, Google's
endpoint can still come from a source llmkit does not read — notably the
`litellm.api_base` module global. Naming a `base_url` or setting
`GEMINI_API_BASE` closes it.

**The listed variables are still honoured** — llmkit now reads them itself, in
the order shown (LiteLLM's own measured precedence), so a host that points its
endpoint with one of them is unaffected. What changed is the *closure*, and it
covers `OPENAI`, `ANTHROPIC`, `DEEPSEEK`, and a `GOOGLE` that has a `base_url`
or `GEMINI_API_BASE`: for those, sources llmkit does not read can no longer
choose the endpoint — the `litellm.api_base` module global, a LiteLLM
key-management backend serving one of these names, or an alias some future
LiteLLM release adds. That is a closed fix rather than a blocklist chasing an
unversioned dependency, and it is the same bargain `api_key` already strikes:
make the ambient fallback explicit, documented, and tested rather than delete
it. With nothing configured, the outbound request is byte-identical to before.

Two cases sit **outside** that closure. `GOOGLE` with neither `base_url` nor
`GEMINI_API_BASE` set is one, per the paragraph above. `OLLAMA` is the other,
and structurally so: both of its LiteLLM dispatch arms order the chain
`litellm.api_base or api_base or …` — inverted relative to every other route —
so for Ollama alone the module global outranks even the `api_base`
llmkit sends (measured against litellm 1.92.0, 2026-07-21). That is accepted
rather than fixed: it is a global a host has to set deliberately inside its own
process, not something the ambient environment or a stray `.env` can reach, and
closing it would mean reaching into a dependency's globals.

`OPENROUTER` and `OLLAMA` deliberately read **no** endpoint variable: both
already named their endpoint in every configuration (a default that is a real
URL, overridable via `base_url`), so honouring extra variables would widen the
ambient surface for no gain.

### `BEDROCK` and `VERTEX` do not own their endpoints

These two reject `base_url` outright, and llmkit sends them no `api_base`:
their endpoints are *derived* rather than configured — Bedrock's from
`aws_region_name`, Vertex's from `vertex_location`. llmkit names an endpoint
where the endpoint is a constant it can measure once; where it is computed from
an input llmkit does not own, llmkit declines to compute it and LiteLLM's own
resolution stays in charge. (Google AI Studio with neither `base_url` nor
`GEMINI_API_BASE` set is the third case in that same family, for the API-version
reason above.) The alternative would be llmkit constructing regional URLs — the
gateway-shaped work this library [deliberately does not do](PRINCIPLES.md), and
for Vertex it would mean running Google credential resolution just to build a
path, since the project id comes from ADC.

**The cost is scoped, and it is sharper here than elsewhere, because these are
the two providers whose region knob is a residency control.** An ambient
endpoint value overrides the region you pinned. Measured against litellm 1.92.0
(2026-07-23), with the region/location explicitly pinned in *both* arms:

```
BEDROCK, aws_region_name="eu-central-1"
  nothing set   -> https://bedrock-runtime.eu-central-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse
  with the var  -> https://hijacked.invalid/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse

VERTEX, vertex_location="europe-west4"
  nothing set   -> https://europe-west4-aiplatform.googleapis.com/v1/projects/<p>/locations/europe-west4/publishers/google/models/gemini-2.5-flash-lite:generateContent
  with the var  -> https://hijacked.invalid/v1:generateContent
```

Four things that matter if you pin residency:

- **What travels.** Vertex sends the ADC-minted `Authorization: Bearer ya29.…`
  to the overridden host; Bedrock hands it a live SigV4 signature over the real
  payload. This is credential exposure as much as region drift. Bedrock's SigV4
  scope keeps following the region you pinned, so redirecting to a *different
  AWS region* fails loudly on signature; redirecting anywhere else succeeds
  silently.
- **Vertex's surface is two sources, Bedrock's is one.** For Vertex, LiteLLM
  resolves `api_base or litellm.api_base or VERTEXAI_API_BASE` on one line, and
  the in-process global **outranks** the variable — no environment hygiene
  reaches a global that any dependency or notebook cell can set. Bedrock's route
  never consults that global. (`VERTEX_API_BASE`, without the `AI`, is
  embeddings-only and does not affect completions.)
- **Vertex replaces the whole URL rather than swapping the host**, composing
  `"{value}:{action}"` — so a value carrying a path or trailing slash redirects
  silently, while a *bare host* fails loudly with `Invalid port:
  'generateContent'` and sends nothing. Do not read a crash as the only failure
  mode.
- **`printenv` is not a valid check.** Importing LiteLLM runs `load_dotenv()`,
  and its search walks up from the installed package directory — so a `.env`
  sitting above your virtualenv is in force from any working directory. Both
  names are read through LiteLLM's `get_secret`, which also consults a
  configured key-management backend. An empty string is safe; a stray value is
  not.

**Vertex has a second residency override with no environment variable
involved.** LiteLLM reroutes a pinned `vertex_location` to a model's first
`supported_regions` entry when its shipped cost map lists regions that exclude
yours, warning only on the verbose logger. llmkit's `gemini-2.5-flash-lite`
default carries no such restriction today, but that is data in a dependency, not
a guarantee — so treat `vertex_location` as pinning residency *for models the
installed LiteLLM does not constrain*, and check a region-sensitive workload
against the endpoint it actually reaches.

`aws_region_name` is the only AWS-shaped field, and it carries **only** the region. AWS Bedrock authenticates through the standard **AWS credential chain** (environment, shared config, or instance/role), so Bedrock secrets never pass through `LLMClientConfig`; leave the region `None` too and it resolves from the chain (`AWS_REGION_NAME` / `AWS_REGION`). Bedrock routing needs `boto3` for request signing — install it with the opt-in extra:

```bash
pip install "omg-llmkit[bedrock]"
```

The default model is Claude Haiku 4.5 via its **cross-region inference profile** id (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) — current Claude models on Bedrock are typically reached through inference profiles rather than plain on-demand ids. Pass a different profile- or partition-prefixed id as `model` (e.g. `eu.anthropic.claude-...`) when your account routes elsewhere.

`vertex_project` and `vertex_location` are the Vertex AI analog and are not accepted by any other provider. Vertex reaches the same Gemini models as the `GOOGLE` (AI Studio) provider, but through Google Cloud — and like Bedrock it carries **no secret** here: Google credentials resolve from **Application Default Credentials** (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or a workload-identity / metadata-server token), never through `LLMClientConfig`. Leave `vertex_project` / `vertex_location` `None` to resolve them from the environment (`VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`), with the location otherwise falling back to Google's default region. Vertex routing needs `google-auth` to mint its OAuth token — install it with the opt-in extra:

```bash
pip install "omg-llmkit[vertex]"
```

**`vertex_location` is the data-processing residency control.** It selects the endpoint where the request is processed, so a regional value (e.g. `vertex_location="europe-west4"`) pins in-region processing; the `"global"` endpoint gives no residency guarantee. LiteLLM builds three endpoint shapes from it, not one: `"global"` yields `https://aiplatform.googleapis.com`, a multi-region geography (`"us"`, `"eu"`) yields `https://aiplatform.<geo>.rep.googleapis.com`, and any other value yields `https://<location>-aiplatform.googleapis.com`. llmkit does not construct that URL and does not send an `api_base` — see [`BEDROCK` and `VERTEX` do not own their endpoints](#bedrock-and-vertex-do-not-own-their-endpoints), which also covers the two ways a pinned location can be overridden. The default model is Gemini 2.5 Flash-Lite (parity with the AI Studio provider). As with AI Studio, Gemini 2.5 thinks by default — set `reasoning_effort="disable"` so a small `max_tokens` cap doesn't truncate structured output.

> **A residency region can constrain which model you may use.** Gemini model availability is region-specific, so a region you pick for residency may not host every model — including the `gemini-2.5-flash-lite` default. A model that isn't deployed in your region fails with a Vertex `400 FAILED_PRECONDITION` ("Precondition check failed."), which is an *availability* error, not an auth one. Pin a `model` the region actually serves (e.g. some regions offer `gemini-2.5-flash` but not `-flash-lite`). Check the [Gemini-on-Vertex locations table](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) for your region.

**`gemini_structured_output` picks Gemini's structured-output strategy** and is read only by the two Gemini providers (`VERTEX`, `GOOGLE`); a non-default value on any other provider is rejected (the default `"schema"` is always accepted). `"schema"` (the default) keeps Gemini's native JSON-schema constrained decoding (`instructor.Mode.JSON_SCHEMA`) with server-side schema enforcement — the pre-existing wire behavior, unchanged. `"json"` switches to `Mode.JSON`: the response is still server-side guaranteed to be JSON *syntax*, but the schema moves into the system prompt and is validated client-side (with instructor's single repair re-ask on a mismatch). Reach for `"json"` when a non-trivial schema drives **runaway output loops**: Gemini's constrained-decoding grammar mask is a repetition-loop trap — once the model starts looping, the mask blocks exactly the tokens that would break the pattern, so the call spins until `max_tokens` kills it (measured 67-83% first-attempt runaway under `"schema"` vs 0% under `"json"` on real prompts). The trade is giving up server-side *schema* enforcement for an occasional repair round-trip. An unrecognized value raises at provider construction rather than silently falling back.

```python
configure_llm_client(lambda: LLMClientConfig(
    provider=Provider.VERTEX,
    gemini_structured_output="json",   # escape the constrained-decoding loop trap
))
```

Per-call `model=` overrides the default, so "strong/small/current" model roles are the host's concern — resolve them to a model string and pass it at the call site. The library has no opinion about roles.

`reasoning_effort` controls provider "thinking"/reasoning tokens. Leave it `None` (the default) for the provider's own behaviour — the outbound request is byte-identical to omitting it. Set it once (e.g. `"disable"`) and every call inherits it; the call functions also take a `reasoning_effort=` override for a single call. This matters most for Gemini, whose thinking is **on by default** and spends reasoning tokens against `max_tokens` — `reasoning_effort="disable"` turns it off so a small `max_tokens` cap doesn't truncate structured output. On OpenRouter, llmkit translates the portable setting to its native `reasoning.effort` object: Gemini 3.x receives `"minimal"` because it requires thinking, while other models receive `"none"`. With OpenRouter's default `require_parameters` routing, an effort-carrying request is routed only to endpoints that support reasoning.

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

`make_provider` takes the same knobs as `LLMClientConfig`, and — like
`build_provider` — **rejects** any that the selected provider does not read
rather than silently dropping them: `base_url` for Bedrock/Vertex (whose
endpoints derive from region/location), `api_key` for Ollama or Bedrock/Vertex
(which authenticate via their ambient AWS/Google chains), `aws_region_name` for
anything but Bedrock, and so on. Pass only the fields the provider uses. Leave
`model` unset to inherit the provider's own default; the assembled LiteLLM id is
always well-formed (e.g. `anthropic/claude-sonnet-4-6`).

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
*every* parameter sent — including the structured `response_format` and native
`reasoning` control when configured. The trade-off
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

- **Transient-provider retries, on by default.** Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, `text_llm_call_stream`) retries *transient* provider errors on its own — you don't wrap anything. The recoverable set splits into two budgets the policy counts **separately**:
  - **Transport errors** (`LLM_TRANSPORT_ERRORS`: 429 / 503 / 5xx, network/timeout) get the full `max_attempts` budget — **three attempts** by default — since a retry on a fresh connection routinely succeeds.
  - **Schema-validation errors** (`LLM_SCHEMA_ERRORS`: pydantic `ValidationError`, instructor `InstructorRetryException`) get the lower `validation_max_attempts` budget — **two attempts (one retry)** by default — so a transiently-malformed JSON response is still recovered, but a *deterministically-wrong* schema can't burn the full transport budget on doomed re-asks. (instructor wraps *transport* failures in `InstructorRetryException` too; the retry layer unwraps it, so a wrapped 429/5xx/network error still gets the full transport budget, not this lower one — and a wrapped *permanent* error such as a 401/400/403 fails fast after a single attempt, never charged to either budget.)
  - **Output-limit truncations** (`LLM_OUTPUT_LIMIT_ERRORS`: llmkit's own `OutputLimitError`, raised when a structured completion is cut off by the output-token limit, `finish_reason='length'`) get **zero budget — never retried**: a re-ask with an identical token budget can only truncate again (the motivating production failure was a degenerate repetition loop that burned to the provider's 65k-token ceiling on the original ask *and* on every blind re-ask, turning a seconds-long call into minutes of doomed generation). The error carries `model` / `max_tokens` / `completion_tokens`, so the fix is legible from the error alone: `completion_tokens` at a cap you set means *raise the cap*; a huge count under no cap means *the prompt induces runaway output*. A caller that genuinely wants the resample can opt back in by listing `OutputLimitError` explicitly in `retry_on`.

  `LLM_RECOVERABLE_ERRORS` remains the documented single catch-set — now the **union of four** subsets: the three above plus `LLM_BACKPRESSURE_ERRORS` (llmkit's own fail-fast `CircuitOpenError`, raised by the opt-in [circuit breaker](#circuit-breaker)). Keep using `LLM_RECOVERABLE_ERRORS` in `except` clauses; the split only changes how the *retry layer* budgets them — and the backpressure/output-limit subsets are deliberately **never retried** (re-asking a known-open circuit or a same-budget truncation is doomed by construction). One footnote on the 503 case: so that `import llmkit` doesn't pay LiteLLM's multi-second import cost, litellm's own 503 class (`litellm.exceptions.ServiceUnavailableError`) is never imported eagerly — instead, llmkit re-raises every litellm-native 503 at its transport boundary as llmkit's own **`ServiceUnavailableError`**, a plain member of `LLM_TRANSPORT_ERRORS`, so `except LLM_RECOVERABLE_ERRORS:` genuinely catches it. It carries `provider`, `model`, `status_code` (always 503), and the original `response` (so a server `Retry-After` stays honoured), with the original litellm error on `__cause__`. If you previously caught `litellm.exceptions.ServiceUnavailableError` directly around llmkit calls, catch `llmkit.ServiceUnavailableError` (or the documented tuple) instead; the raw litellm class still *classifies* as transport in `isinstance` checks — e.g. your own litellm call wrapped in `with_retries` — via a lazy stand-in resolved once litellm is loaded. Both budgets use bounded **full-jitter** backoff: the sleep before retry *n* is a random delay in `[0, min(backoff_base_seconds * 2**(n-1), max_backoff_seconds)]`, with the per-sleep cap (`max_backoff_seconds`) defaulting to 30s so a large attempt budget can't grow the worst-case sleep unboundedly. **`Retry-After` is honoured first:** when a retried provider error carries a `Retry-After` (a header — delta-seconds, `retry-after-ms`, or an HTTP-date — or the SDK's numeric attribute), the backoff waits *that* duration instead of the computed exponential, capped at `RetryPolicy.retry_after_cap` (default 60s) so a hostile value can't wedge a call. It is read from the *unwrapped* provider error, so a structured call honours it too, and it is honoured even when `backoff_base_seconds` is 0 (a server directive, not opt-in backoff); absent a header, the exponential is used unchanged. Programming errors (e.g. `TypeError`) are outside the recoverable set and propagate immediately, never retried. Each attempt is its own logged call, so `data/llm-logs/` shows one record per attempt.

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

  **Streaming caveat:** `text_llm_call_stream` can only retry a transient failure that happens *before the first chunk reaches the caller*. Once any chunk has been yielded, a mid-stream error propagates unretried — a partially-consumed stream can't be safely restarted.

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

  > **Don't double-wrap the call functions.** They already retry internally, so `with_retries(structured_llm_call, ...)` would otherwise multiply the budgets (the `3 × 3 = 9` trap). `with_retries` guards against this — it detects an active llmkit retry loop **owned by the current `asyncio` task** and collapses the inner layer to a single pass (warning once), so the budgets don't multiply. The task scoping bounds the guard to the pattern it warns about: a nested loop awaited inline in the same task, whose failure really does propagate out to the outer loop. A *distinct* llmkit call that merely inherited the scope across a task boundary keeps its full retry budget — an `on_result` hook calling `structured_llm_call_sync` / `text_llm_call_sync` (the sync bridge drives the coroutine as a new task on llmkit's persistent loop), or a call spawned with `asyncio.create_task` from inside an attempt. To drive retries entirely from your own wrapper instead, opt the inner call out with `retry=NO_RETRY`.

- **instructor's own in-call schema repair** re-asks the model to fix malformed JSON *within a single call*, before any `ValidationError`/`InstructorRetryException` reaches the retry layer. llmkit pins instructor's budget to **two in-call attempts** (a per-call tenacity `AsyncRetrying` stopping after 2 — instructor counts *total attempts*, so that is exactly one repair re-ask) — and it is not a caller-facing knob. The re-ask fires for a **genuine parse failure only**: malformed JSON, a Pydantic `ValidationError` (a failing async field validator included), or instructor's own `ResponseParsingError` (e.g. a blocked Gemini `Mode.JSON` response). Every other failure is declined by the in-call loop, so it costs exactly **one** provider request per attempt: a transport failure (429/5xx/network) leaves the rate-limiter slot immediately and is retried by the cross-call layer above, with backoff and `Retry-After` honoured, rather than being re-sent inside the same slot with neither; a permanent failure (401/400/403) fails fast with no in-call duplicate; and a completion **truncated by the output-token limit** is never re-asked (the re-ask would run on the identical budget and can only truncate again) and surfaces immediately as `OutputLimitError`. This stays **separate** from the cross-call retry layer above: instructor repairs within one attempt; the policy's `validation_max_attempts` (default 2) governs how many *fresh* attempts a persistent schema failure earns. The two budgets are never conflated, so attempts aren't double-counted.

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
uv sync
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
