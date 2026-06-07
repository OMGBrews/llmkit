# Principles

The design principles behind `llmkit` — the durable promises the library makes to the people who call it. Read this to understand *why* llmkit behaves the way it does; for the API itself, see the [README](README.md). This describes the library as it ships today.

`llmkit` is opinionated on purpose: it decides the boring-but-easy-to-get-wrong things — which structured-output mode each provider needs, where logs go, a sane default temperature — so you don't have to. These eight principles are what that buys you.

## Calling an LLM is one function

One call does it: pass a prompt — plus a Pydantic model when you want structured output — and get a result back. `structured_llm_call` returns a validated instance of your model (or raises); `text_llm_call` returns plain text. The only bookkeeping you add is a `feature`/`label` pair, and it shows up in the logs.

## Switching providers is configuration, not code

OpenRouter, Google, Anthropic, OpenAI, DeepSeek, AWS Bedrock, and local Ollama sit behind one `Provider` enum and one `LLMClientConfig`. You register the config once at startup; call sites never name a provider, so switching vendors is a config edit, not a refactor.

## Switching models is a per-call argument

Every call takes an optional `model=` that overrides the configured default just for that call. llmkit imposes no "strong/small/current" role system — resolve your own roles to a model string and pass it. Changing the default model is a one-line config change.

## Concurrency is bounded so fan-out can't hammer a provider

llmkit ships a global concurrency limiter that caps how many calls run at once, so a burst of parallel requests doesn't overrun a provider's rate limits. Enable it and set the cap with `configure_rate_limit(...)`.

## Logging is on by default

Every call is logged with zero setup — the default sink writes plain files to a local directory. No collector, no account, no network. Opt out with `configure_llm_logging(None)`, or swap in your own `LogSink` to send records elsewhere.

## The logs are written for an AI agent to read

Each per-call log is laid out verdict-first: a one-line header (`ok`/`ERROR`, model, schema, duration, approximate cost) on top, the prompt and response below — plus an append-only `index.jsonl` for cross-call scans. The assumed reader is a coding agent like Claude Code debugging a run, so it can see at a glance what was sent, what came back, and what failed, and use that to diagnose problems and improve prompts.

## Async is the default; sync is one call away

The call surface is async-first, because LLM calls are I/O-bound and usually fanned out. Callers that can't `await` use `structured_llm_call_sync`, which drives the same async path — and inherits the same rate limiting and logging.

## Built for reliable calls

Reliability is the point, not an add-on. Structured output is validated before you ever see it; instructor repairs malformed JSON in-call; and every call function retries *transient* provider errors on its own by default, with bounded full-jitter backoff over a curated `LLM_RECOVERABLE_ERRORS` set — so reliability doesn't depend on each caller remembering to wrap every call. Programming errors stay out of that set and propagate immediately. The default budget is tunable (or opt out) per call via `retry=`, and `with_retries()` remains the explicit advanced path for wrapping any awaitable. This layer is kept separate from instructor's schema-repair budget, so the two are never conflated.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
