# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The accumulated work below is the next release (planned `0.1.3`) and has not yet
been published to PyPI — the last published version is `0.1.2`. It moves to a
dated `## [0.1.3]` section when the release is cut.

### Added

- `model_from_json_schema(schema, *, name=None)` — converts a **JSON-schema
  dict** into a Pydantic model class at runtime, so consumers who declare their
  structured-output contracts as JSON-schema dicts (shared across Node /
  frontend / Python) no longer hand-write a converter before calling
  `structured_llm_call`. The intended pattern is **build-once-reuse**
  (`Invoice = model_from_json_schema(schema)` at import, then pass `Invoice` as
  `output_schema` on every call); `structured_llm_call`'s signature is
  unchanged (still `output_schema: type[T] -> T`, no `dict` overload). Built on
  `pydantic.create_model` (no new third-party dependency). Supported subset:
  `object` with `properties` and a `required` array; scalars (`string` /
  `integer` / `number` / `boolean`, plus `null` / nullable via
  `["string", "null"]` or an `anyOf` null branch); `array` with `items`
  (including arrays of objects); `enum` (string or integer members); and nested
  objects inline or via local `$ref` (`#/$defs/...` / `#/definitions/...`).
  Anything outside the subset raises a clear `ValueError` naming the construct
  and its path, rather than silently producing a wrong model. Two footguns the
  CaCL dogfood hit are handled and tested: (1) a non-required field maps to an
  *optional* Pydantic field defaulting to `None`, and the generated model's
  `model_dump` / `model_dump_json` default to `exclude_none=True`, so an
  omitted optional is **absent** rather than `"field": null` (which would fail
  downstream re-validation against the same schema); pass `exclude_none=False`
  to keep the nulls. (2) A title-less or empty-titled schema still yields a
  validly-named class (default `JsonSchemaModel`), which `create_model` and
  `instructor` both require. Exported from the package root.
- A `py.typed` marker so consumers' type checkers honor llmkit's type hints
  (the package is basedpyright-clean and already declares the `Typing :: Typed`
  classifier, but without the PEP 561 marker downstream tools treated it as
  untyped and required an `ignore_missing_imports` override).
- `get_rate_limit_config()` (returning a frozen `RateLimitConfig`) — the
  symmetric read for `configure_rate_limit`, so a host can log or assert its
  effective `enabled` / `max_concurrent` at startup without reaching into
  limiter internals. Both are exported from the package root.
- **AWS Bedrock** provider (`Provider.BEDROCK` / `BedrockProvider`, `bedrock/`
  LiteLLM prefix), giving first-class access to Claude-on-Bedrock. Unlike every
  other provider, Bedrock authenticates through the standard **AWS credential
  chain** rather than a bearer `api_key`: the new optional
  `LLMClientConfig.aws_region_name` carries only the region (the single
  AWS-shaped field; unused by all other providers), while secrets resolve from
  the ambient chain (environment / shared config / instance role) and never
  pass through the config. Structured output is pinned to
  `instructor.Mode.ANTHROPIC_JSON` — the same Claude-native mode the direct
  Anthropic provider uses (the model is Claude). `Mode.BEDROCK_JSON` is
  deliberately avoided: it targets instructor's `from_bedrock` (boto3) client
  and drops `model` when driven through `from_litellm`, this library's call
  seam. `reasoning_effort` is forwarded where the underlying model supports it. The
  first cut targets plain on-demand models (default
  `anthropic.claude-3-5-sonnet-20240620-v1:0`); 4.x models reachable only via a
  cross-region inference profile are supported by passing the profile-prefixed
  id as `model`. `boto3` (for SigV4 signing) ships via the opt-in
  `omg-llmkit[bedrock]` extra, so non-Bedrock installs gain no AWS dependency.
  `BedrockProvider` is exported from the package root.
- Direct **DeepSeek** provider (`Provider.DEEPSEEK` / `DeepSeekProvider`,
  `deepseek/` LiteLLM prefix), giving first-class access to `deepseek-chat` (V3)
  and `deepseek-reasoner` (R1) on a first-party key rather than the indirect
  `openrouter/deepseek/...` hop (which adds a gateway markup). Structured output
  is pinned to DeepSeek's native JSON mode (`instructor.Mode.JSON`); the strict
  `Mode.JSON_SCHEMA` the other direct providers use is rejected by DeepSeek's
  API, while `Mode.JSON` is measured to validate on both models (live smoke
  test). `reasoning_effort` is forwarded for `deepseek-reasoner` and is harmless
  on `deepseek-chat`. `DeepSeekProvider` is exported from the package root.
- Direct **OpenAI** provider (`Provider.OPENAI` / `OpenAIProvider`, `openai/`
  LiteLLM prefix), giving first-class access to GPT / o-series / GPT-5 models
  without the indirect `openrouter/openai/...` hop (which adds a markup and a
  different structured-output mode). Structured output is pinned to OpenAI's
  native structured-outputs mode (`instructor.Mode.JSON_SCHEMA`), and
  `reasoning_effort`
  is forwarded for reasoning models on the same seam as the other providers. An
  optional `base_url` points the provider at OpenAI-compatible gateways; left
  unset, LiteLLM uses OpenAI's default endpoint. `OpenAIProvider` is exported
  from the package root.
- The retry helpers `retry_progress_callback` and `RetryProgressCallback` are
  now exported from the package root, so callers no longer reach into
  `llmkit.retry` directly. (`with_retries` is **not** part of the headline
  surface — it stays importable from `llmkit.retry`; see _Removed_ below.)
- `RetryPolicy` (plus `DEFAULT_RETRY_POLICY` and `NO_RETRY`) is exported from
  the package root — the per-call knob for the now default-on transient-error
  retry layer (see _Changed_ below). `RetryPolicy` is a frozen dataclass
  (`max_attempts`, `backoff_base_seconds`, `retry_on`); pass `retry=NO_RETRY`
  to opt a call out, or a custom instance to tune the budget.
- Reasoning/thinking control via `LLMClientConfig.reasoning_effort` (and a
  per-call `reasoning_effort` override on `structured_llm_call` /
  `structured_llm_call_sync` / `text_llm_call`), forwarded to LiteLLM. Lets
  callers disable Gemini thinking (`reasoning_effort="disable"`) so it
  doesn't consume the `max_tokens` budget and truncate structured output.
  Set once on the config and every call inherits it; the per-call value
  overrides it. Backward-compatible; defaults to provider behaviour — when
  unset (`None`) no `reasoning_effort` kwarg is sent, so the outbound
  request is byte-identical to before. `LLMCallRecord` also records the
  per-call setting for log completeness.
- `make_provider(provider, *, api_key=None, model=None, base_url=None,
  reasoning_effort=None, aws_region_name=None)` — build a provider from raw
  credentials in one line for the per-call `provider=` override, without an
  `LLMClientConfig`. Exported from the package root.
- `build_provider(config)` as the canonical construct-from-config function
  (replaces `get_provider`). Exported from the package root.
- `describe_llm(config)` as the read-snapshot accessor (replaces
  `get_llm_config`), importable from `llmkit.providers`.
- `model_from_json_schema` now carries per-field JSON-schema constraints into
  the generated Pydantic model: numeric `minimum`/`maximum`/`exclusiveMinimum`/
  `exclusiveMaximum` map to `ge`/`le`/`gt`/`lt`, and `minLength`/`maxLength`
  plus `minItems`/`maxItems` map to `min_length`/`max_length`, so generated
  models validate value bounds, not just shape. Bounds are resolved through
  `$ref` and nullable wrappers; constraints outside the supported set (e.g.
  `pattern`, `format`, `multipleOf`) remain silently dropped with no partial
  enforcement, and per-field `description` passthrough is unchanged.
- `capture_llm_records()` context manager — captures the per-call
  `LLMCallRecord` objects (approximate cost, resolved model/provider, duration,
  error) for calls in its scope, with no custom sink, across both the async
  call functions and the `run_sync` sync bridge. Exported from the package
  root.
- `LLMCallOptions`, an opt-in **frozen** bundle of the per-call keyword
  arguments (`temperature` / `model` / `max_tokens` / `reasoning_effort` /
  `retry` / `provider`) accepted as `options=` on `structured_llm_call`,
  `structured_llm_call_sync`, `text_llm_call`, `stream_text_with_log`, and
  `structured_data_call`, to cut the repeated per-call keyword block; the
  flat-keyword path is unchanged. Precedence is config < `options` < explicit
  per-call keyword. `feature` is intentionally **not** part of `LLMCallOptions`
  — it stays a required per-call keyword as a deliberate telemetry forcing
  function. Exported from the package root.
- `structured_data_call` and `structured_data_call_sync` — a dict-in /
  data-out structured call (JSON-schema dict in, plain `dict` out) built on
  `model_from_json_schema`, for consumers who never want the intermediate
  Pydantic instance. Inherits the full `structured_llm_call` surface (logging,
  default-on retries, per-call `provider=` override, `model` / `temperature` /
  `max_tokens` / `reasoning_effort`, and `LLMCallOptions`). An omitted optional
  field is absent from the result, not present-as-null. Exported from the
  package root.
- `rate_limit_acquire_async(provider_key)` and
  `rate_limit_acquire_sync(provider_key)` context managers in
  `llmkit.rate_limiting` — the public way to join the global per-provider
  concurrency limit by hand (e.g. from a LangChain chat-model wrapper) without
  referencing `GlobalRateLimiter`.

### Changed

- **The Anthropic SDK moved from a core dependency to the opt-in
  `omg-llmkit[anthropic]` extra.** It was core through 0.1.2 on the belief that
  `instructor` forced it at import; measured against `instructor>=1.15.1`, that
  is false — `instructor` reaches the SDK only at *call time*, on its
  `ANTHROPIC_*` usage-accounting path (`from anthropic.types import Usage` inside
  `instructor/core/retry.py`, guarded by the mode), so plain `import llmkit` and
  a Google-only flow never touch it. Non-Anthropic hosts now take on no Anthropic
  dependency. The `AnthropicProvider` and `BedrockProvider` (both pin
  `ANTHROPIC_JSON`, so both need the SDK at call time) raise a clear
  `install omg-llmkit[anthropic]` error *at construction* when the SDK is absent,
  rather than failing cryptically on the first completion. The `[bedrock]` extra
  pulls in `[anthropic]` (it routes Claude), and a new convenience `[all]` extra
  installs every provider's optional dependencies at once. **Migration:** hosts
  that route Anthropic or Bedrock should install `omg-llmkit[anthropic]` (or
  `omg-llmkit[bedrock]`, or `omg-llmkit[all]`); hosts on other providers need no
  change.
- **Transient-error retries are now on by default.** `structured_llm_call`,
  `text_llm_call`, `stream_text_with_log`, and `structured_llm_call_sync` retry
  the curated recoverable set (429 / 503 / 5xx, timeouts, transient
  provider errors) on their own — with full-jitter backoff — so
  reliability no longer depends on every caller wrapping each call. The transient
  set names the specific transient `openai` subclasses (`RateLimitError` /
  `InternalServerError` / `APIConnectionError`) rather than their broad
  `openai.APIError` base, so **permanent 4xx errors — authentication (401),
  bad-request (400), permission (403) — fail fast** instead of burning the retry
  budget. Programming errors (e.g. `TypeError`) still propagate immediately. The budget is the new
  per-call `retry: RetryPolicy` argument: pass `retry=NO_RETRY` to opt out or a
  custom `RetryPolicy` to tune it. This layer stays **separate** from
  instructor's in-call schema-repair budget (`validation_retries`, default 1) —
  no double-counting — and each attempt remains its own logged call (so
  `capture_llm_log_paths` sees one path per attempt). Streaming retries only a
  failure that occurs **before the first chunk** is yielded; a partially
  consumed stream cannot be transparently restarted. `with_retries()` remains
  exported as the explicit, composable path for wrapping any awaitable, and now
  takes a `retry_on` filter.
- **Transport and schema-validation retries now have separate budgets.** The
  recoverable set is split into `LLM_TRANSPORT_ERRORS` (rate limits, transient
  5xx, network/timeout) and `LLM_SCHEMA_ERRORS` (pydantic `ValidationError`,
  instructor `InstructorRetryException`), and `RetryPolicy` counts each against
  its own budget: transport keeps `max_attempts` (default **3**), while the new
  `validation_max_attempts` (default **2** = one retry) governs schema failures.
  A deterministically-wrong schema or impossible constraint can no longer burn
  the full transport budget on doomed re-asks, while a *transiently*-malformed
  JSON response still earns one cross-call retry. `LLM_RECOVERABLE_ERRORS`
  remains the **union** of the two subsets, so its documented `except`-clause
  catch-set contract is unchanged. `NO_RETRY` disables **both** budgets (a single
  attempt). `RetryPolicy` gains `validation_max_attempts` and
  `validation_retry_on` (defaulting to `LLM_SCHEMA_ERRORS`); this cross-call
  layer remains separate from instructor's in-call `validation_retries`
  (default 1, which repairs malformed JSON *within* one attempt before any
  `ValidationError` reaches the retry layer). Flagged by the PIA Maker and FiW
  dogfoods of the 0.1.3 RC.
- **Unified the public retry attempt-count on `max_attempts`.** `RetryPolicy`
  already used `max_attempts`; `with_retries(...)` now uses it too, with the same
  semantics everywhere — *total attempts including the first* (`N`, not `1 + N`).
  The old `with_retries(..., max_retries=...)` keyword keeps working as a
  **deprecated alias** that emits a `DeprecationWarning` and maps to
  `max_attempts` (`max_attempts` wins if both are given), so existing consumers
  (e.g. pia-maker) don't break. The progress-callback keyword stays `max_retries`
  for backward compatibility. `LLM_TRANSPORT_ERRORS` and `LLM_SCHEMA_ERRORS` are
  exported from the package root alongside `LLM_RECOVERABLE_ERRORS`.
- **`with_retries` now guards against nested retry-budget multiplication.**
  Because the call functions retry internally by default, wrapping one in
  `with_retries` previously multiplied the budgets silently (the `3 × 3 = 9`
  trap the PIA Maker dogfood hit). `with_retries` now detects (via a context
  variable) when it runs *inside* an already-active llmkit retry loop and
  collapses that inner layer to a **single pass** — so the budgets no longer
  multiply, and per-attempt logging is preserved. An *accidental* double-wrap
  additionally emits a filterable `RuntimeWarning` (so consumers running under
  `-W error` can filter it). The guard does not affect `with_retries` wrapping a
  plain (non-llmkit) awaitable, which still retries normally. The clean way to
  drive retries from an outer wrapper is to opt the inner call out with
  `retry=NO_RETRY` — which collapses the inner pass **without** warning; this is
  documented in `with_retries`' docstring.
- **Concurrency rate limiting is now on by default, scoped per provider.**
  Previously the limiter was off until a host opted in, and a single process-wide
  budget fronted every provider. It is now active out of the box and bounds
  concurrent calls **per provider** (keyed by the effective provider name, the
  same value logging records), so fan-out to one provider can't overrun its rate
  limits or eat another provider's budget. The default cap is **8 concurrent
  calls per provider** — headroom for the fan-out workloads consumers actually
  run, while still bounding a self-inflicted burst; a tightly-metered account can
  lower it. `configure_rate_limit(max_concurrent=..., enabled=...)` changes the
  cap or turns it off, and `get_rate_limit_config()` reads back the effective
  values.
- The internal LiteLLM call layer now forwards a provider's **full**
  `completion_kwargs()` dict (splatting it into the call) instead of
  cherry-picking `api_key` / `api_base`. This lets a provider carry whatever
  credential kwargs LiteLLM needs — e.g. Bedrock's `aws_region_name` — without
  the shared call layer growing per-provider knowledge. No change for the
  existing providers (their kwarg dicts are unchanged); the request shape is
  byte-identical for `api_key` / `api_base` providers.
- `get_provider` now **fails loud** on an unknown provider instead of silently
  constructing an `OllamaProvider`. The previous `else` catch-all meant a
  newly-added `Provider` enum member routed to a confusing local-Ollama failure;
  dispatch is now an exhaustive `match` whose fall-through calls
  `typing.assert_never`, so an unwired member is caught statically by
  basedpyright, raises `AssertionError` at runtime, and fails a dedicated
  exhaustiveness test. No behaviour change for the existing providers.
- The provider layer is reorganized from a single `providers.py` module into a
  `llmkit.providers` **package** with one module per provider over a
  provider-agnostic `base` module, so adding a provider is a self-contained new
  file plus one wiring line. Purely internal: the public API is unchanged — every
  symbol (`Provider`, `LLMClientConfig`, the `*Provider` classes, `get_provider`,
  `get_llm_config`, …) imports from `llmkit` and `llmkit.providers` exactly as
  before.
- README now describes `with_retries()` as a composable helper the caller wraps
  a call in — the public call functions do not retry on their own — instead of
  implying transient-error retries happen automatically "out of the box".
- `anthropic` is now a required runtime dependency. `instructor` imports the
  Anthropic SDK to account usage for its native `ANTHROPIC_JSON` mode, so the
  Anthropic provider needs it present at call time.
- `LLMClientConfig.model` is now optional (`str | None`, default `None`); a
  falsy model resolves to the selected provider's own default model instead of
  producing a broken `"<prefix>/"` LiteLLM id.
- Reserved `get_*` for reads and `build_*` / `make_*` for construction:
  `build_provider` replaces `get_provider`, and `describe_llm` replaces
  `get_llm_config`. (Both old names are removed from the public surface — see
  _Removed_ below.)
- The `LogSink` protocol method is now `write(record) -> None` (was
  `-> Path | None`); the file detail no longer leaks into the shared contract.
  `LocalYamlLogSink` exposes its written path via a new `write_returning_path()`
  method, and `capture_llm_log_paths()` continues to work unchanged.
- Documented the dual-homed `model` / `reasoning_effort` precedence explicitly
  — config < `options` < explicit per-call keyword — across the call-function
  docstrings and README, and documented `feature` as a deliberate required
  telemetry forcing function (intentionally excluded from `LLMCallOptions`).

### Removed

- **Breaking:** `with_retries` no longer accepts the deprecated `max_retries`
  keyword (hard-cut, no shim) — use `max_attempts` (same meaning: total attempts
  including the first). Passing `max_retries` now raises `TypeError`. The
  retry-progress callback keyword was likewise renamed `max_retries` ->
  `max_attempts` (the `RetryProgressCallback` protocol and the
  `handle_retry_failure` invocation), so callbacks now receive `max_attempts`.
- **Breaking:** `get_provider` and `get_llm_config` are removed from the public
  surface; use `build_provider` and `describe_llm` (importable from
  `llmkit.providers`) respectively.
- **Breaking:** `GlobalRateLimiter` removed from `llmkit.__all__` (still
  importable from `llmkit.rate_limiting` for internal use); replace
  `GlobalRateLimiter.acquire_async` / `acquire_sync` with `rate_limit_acquire_async`
  / `rate_limit_acquire_sync`, and `GlobalRateLimiter.is_enabled()` with
  `get_rate_limit_config().enabled`.
- **Breaking:** third-party `LogSink` implementations must drop the
  `Path | None` return from `write` and return `None`. The shipped
  `LocalYamlLogSink` is updated; consumers tracking file paths use
  `capture_llm_log_paths()` or `LocalYamlLogSink.write_returning_path()`.
- **Headline-surface trim (Breaking for `from llmkit import` of these names).**
  The seven concrete `*Provider` classes
  (`OpenRouterProvider` / `OllamaProvider` / `GoogleProvider` /
  `AnthropicProvider` / `OpenAIProvider` / `DeepSeekProvider` /
  `BedrockProvider`), `LLMInfo`, `describe_llm`, `with_retries`, and
  `GlobalRateLimiter` are demoted out of `llmkit.__all__`. They remain
  importable from their submodules (`llmkit.providers`, `llmkit.retry`,
  `llmkit.rate_limiting`); only the top-level re-export is gone.

### Fixed

- **`model_from_json_schema` no longer crashes on signed / colliding integer
  enums, and enum fields now dump as raw scalars.** A non-contiguous integer
  enum such as `[-1, 1, 2, 3, 4, 5]` (FiW's eval-judge schema, where `-1` is an
  "Unknown" sentinel) raised `ValueError: _sunder_ names ... are reserved`: the
  member-name builder stripped the sign, so `-1` and `1` collided on key `"1"`
  and the collision suffix bumped one to the reserved `_1_`. Member names are
  now letter-led and sign-preserving (`NEG_1`, `N_1`), so they can never form a
  reserved `_sunder_`/`__dunder__` name. Generated models also set
  `use_enum_values=True`, so a validated instance stores the raw scalar —
  `model_dump()` yields `2`, not `<Enum._2: 2>` (this also makes
  `structured_data_call` hand dict consumers raw values). Surfaced by the FiW
  0.2.0 RC dogfood.
- **Synchronous calls no longer leak a `coroutine 'Logging.async_success_handler'
  was never awaited` `RuntimeWarning` to stderr.** LiteLLM logs successes
  asynchronously without awaiting inline — it queues the
  `async_success_handler` coroutine on a background worker — so the sync bridge
  (`structured_llm_call_sync` and the sync text path, via `run_sync`) could
  close its event loop with that coroutine still un-awaited, surfacing the
  warning on an otherwise clean call (flagged by the FiW dogfood of the 0.1.3 RC
  in both a CLI and a Lambda). `run_sync` now **drains** LiteLLM's pending async
  logging before tearing the loop down: it flushes the logging worker's queue
  (awaiting the queued handlers) and then cancels any remaining background tasks
  — bounded by the same `timeout` budget so a hung callback can't wedge the
  bridge, and applied to **both** `run_sync` branches (the `asyncio.run` path
  and the worker-thread path used when a loop is already running). The drain is
  best-effort and never converts a successful call into a failure; genuine call
  errors still propagate.
- `text_llm_call` now coerces provider **list-content** responses to a single
  string. Some providers return `message.content` as a list of content blocks
  rather than a string; the call previously returned that list verbatim,
  violating its `str` return annotation (and the README's documented coercion).
  Text blocks are now joined and non-text blocks skipped, so the return value
  is always a string.
- The Anthropic provider raised `ModuleNotFoundError: No module named
  'anthropic'` on the first call, because the SDK `instructor` requires for
  `ANTHROPIC_JSON` usage accounting was never declared as a dependency. A clean
  `pip install omg-llmkit` can now call the Anthropic provider out of the box.

## [0.1.2] — 2026-06-05

### Added

- `max_tokens` parameter on `structured_llm_call` /
  `structured_llm_call_sync`, forwarded to the provider (parity with
  `text_llm_call`). Backward-compatible; defaults to no cap. When unset
  (`None`) the outbound request is byte-identical to before — no
  `max_tokens` key is sent. `LLMCallRecord` also records the cap for log
  completeness.

## [0.1.1] — 2026-06-05

### Added

- Per-call `provider` override on `structured_llm_call`,
  `structured_llm_call_sync`, `text_llm_call`, and `stream_text_with_log`.
  Passing an explicit `LLMProviderInterface` (e.g. an `OpenRouterProvider`
  built from credentials) routes that single call through a different
  provider family without changing the app-wide `configure_llm_client`
  registration; `None` (the default) preserves the existing behaviour, so
  every current caller is unchanged. The invocation log records the
  provider that actually ran.

## [0.1.0] — 2026-06-05

Initial public release.

### Added

- Provider-agnostic call surface over LiteLLM (with `instructor` for structured
  output) across OpenRouter, Google, Anthropic, and local Ollama.
- `structured_llm_call` / `structured_llm_call_sync` — validated Pydantic output,
  with each provider pinned to its native JSON-schema mode (never auto-`Mode.TOOLS`).
- `text_llm_call` and `stream_text_with_log` for plain-text and streamed calls.
- Process-global async rate limiter (`GlobalRateLimiter`, `configure_rate_limit`).
- Transient-error retries (`with_retries`, `LLM_RECOVERABLE_ERRORS`), kept
  separate from instructor's schema-repair retries.
- Agent-readable logging: `LocalYamlLogSink` writes verdict-first per-call YAML
  plus an append-only `index.jsonl`; pluggable `LogSink` protocol for custom sinks.
- Approximate per-call cost (`approximate_cost`) sourced from LiteLLM's response
  estimate, for budget visibility.

[Unreleased]: https://github.com/OMGBrews/llmkit/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.2
[0.1.1]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.1
[0.1.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.0
