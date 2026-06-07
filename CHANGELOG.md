# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The accumulated work below is the next release (planned `0.1.3`) and has not yet
been published to PyPI — the last published version is `0.1.2`. It moves to a
dated `## [0.1.3]` section when the release is cut.

### Added

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
- The retry helpers (`with_retries`, `retry_progress_callback`,
  `RetryProgressCallback`) are now exported from the package root, so callers
  no longer reach into `llmkit.retry` directly.
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

### Changed

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

### Fixed

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
