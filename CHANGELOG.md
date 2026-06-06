# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- `get_provider` now **fails loud** on an unknown provider instead of silently
  constructing an `OllamaProvider`. The previous `else` catch-all meant a
  newly-added `Provider` enum member routed to a confusing local-Ollama failure;
  dispatch is now an exhaustive `match` whose fall-through calls
  `typing.assert_never`, so an unwired member is caught statically by
  basedpyright, raises `AssertionError` at runtime, and fails a dedicated
  exhaustiveness test. No behaviour change for the five existing providers.
- The provider layer is reorganized from a single `providers.py` module into a
  `llmkit.providers` **package** with one module per provider (`openrouter`,
  `ollama`, `google`, `anthropic`, `openai`) over a provider-agnostic
  `base` module, so adding a provider is a self-contained new file plus one
  wiring line. Purely internal: the public API is unchanged — every symbol
  (`Provider`, `LLMClientConfig`, the `*Provider` classes, `get_provider`,
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

## [0.1.3] — 2026-06-06

### Added

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

[Unreleased]: https://github.com/OMGBrews/llmkit/compare/v0.1.3...HEAD
[0.1.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.0
