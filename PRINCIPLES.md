# Principles

The design principles behind `llmkit` — the durable promises the library makes to the people who call it. Read this to understand *why* llmkit behaves the way it does; for the API itself, see the [README](README.md). This describes the library as it ships today.

`llmkit` is opinionated on purpose: it decides the boring-but-easy-to-get-wrong things — which structured-output mode each provider needs, where logs go, a sane default temperature — so you don't have to. These principles are what that buys you.

## Validated structured output is the contract

The core promise: you pass a Pydantic model and get back a validated instance of it, or an exception — never a plausible-looking object with empty or wrong fields. Across every provider, structured output is pinned to the mode that *actually* validates, verified against the live APIs (instructor's automatic mode silently regresses some providers to empty results — llmkit refuses it). So `structured_llm_call` gives you the same reliable, typed result whether you're on OpenAI, Gemini, Claude, DeepSeek, Bedrock, or a local model.

## Opinionated where it counts, out of your way everywhere else

llmkit holds firm opinions about reliability, correctness, and observability — and deliberately none about *your* domain. It takes no position on model "roles" (strong/small/current are yours to define), on how you build prompts, on where your logs ultimately live, or on your retry budgets. The opinions exist to remove footguns, not to box you in; every default has an escape hatch.

## Calling an LLM is one function

One call does it: pass a prompt — plus a Pydantic model when you want structured output — and get a result back. `structured_llm_call` returns a validated instance; `text_llm_call` returns plain text. The only bookkeeping you add is a `feature`/`label` pair, and it shows up in the logs.

## Providers switch by config; models switch per call

OpenRouter, Google, Anthropic, OpenAI, DeepSeek, AWS Bedrock, and local Ollama sit behind one `Provider` enum and one `LLMClientConfig`, registered once at startup — so switching vendors is a config edit, not a refactor, and call sites never name a provider. Switching models is even lighter: every call takes an optional `model=` that overrides the default just for that call.

## Local-first, zero infrastructure

Getting full value from llmkit requires nothing but a provider key — no collector, no account, no service to stand up. The only network call is to the provider you chose; nothing phones home. Logs are plain local files, and Ollama gives you a fully-local option where no data leaves the host. You can plug in external systems later (a custom `LogSink` ships records anywhere), but you never have to.

## Logging is on by default — and written for an AI agent

Every call is logged with zero setup; opt out with `configure_llm_logging(None)`. Each per-call log is laid out verdict-first: a one-line header (`ok`/`ERROR`, model, schema, duration, approximate cost) on top, the prompt and response below, plus an append-only `index.jsonl` for cross-call scans. The assumed reader is a coding agent like Claude Code debugging a run — so it can see at a glance what was sent, what came back, and what failed, and use that to diagnose problems and improve prompts.

## Async-first; sync is one call away

The call surface is async-first, because LLM calls are I/O-bound and usually fanned out. Callers that can't `await` use `structured_llm_call_sync`, which drives the same async path — and inherits the same rate limiting and logging.

## Built for reliable calls

Reliability is the point, not an add-on. Structured output is validated before you ever see it (above); a curated `LLM_RECOVERABLE_ERRORS` set names exactly what's worth retrying; instructor repairs malformed JSON in-call; and `with_retries()` is a composable helper you wrap a call in to recover transient provider errors with backoff. A built-in concurrency limiter — enabled and sized with `configure_rate_limit(...)` — keeps a burst of parallel calls from overrunning a provider's rate limits.

## A thin layer, not a gateway

llmkit owns the call ergonomics, the structured-output mode pinning, the rate-limit policy, and the logging convention — and stops there. It does **not** reimplement HTTP transport or run a proxy: that is LiteLLM's job, and reimplementing it is the one thing this library deliberately won't do. Staying thin is what keeps it small, predictable, and easy to reason about.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
