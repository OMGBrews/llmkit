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

Reliability is the point, not an add-on. Structured output is validated before you ever see it (above); instructor repairs malformed JSON in-call; and every call function retries *transient* provider errors on its own by default, with bounded full-jitter backoff over a curated recoverable set — so reliability doesn't depend on each caller remembering to wrap every call. That set is budgeted in two parts the policy counts separately: *transport* failures (rate limits, transient 5xx, network/timeout — `LLM_TRANSPORT_ERRORS`) get the full `max_attempts` budget (three by default), while *schema-validation* failures (`LLM_SCHEMA_ERRORS`) get the lower `validation_max_attempts` (two by default), so a deterministically-wrong schema can't burn the full transport budget on doomed re-asks while transiently-malformed JSON is still recovered. (instructor reports an exhausted attempt — transport failure or schema failure alike — as a single `InstructorRetryException`, so the layer unwraps it and charges a wrapped transport cause to the transport budget rather than the lower one.) `LLM_RECOVERABLE_ERRORS` stays the union of the two for `except` clauses. Programming errors stay out of the set entirely and propagate immediately. The attempt counts are named `max_attempts`/`validation_max_attempts` (total attempts including the first), tunable (or opt out) per call via `retry=`; `with_retries()` remains the explicit advanced path for wrapping any awaitable, and guards against double-wrapping the already-retrying call functions (the `3 × 3 = 9` trap) by collapsing a nested inner loop to a single pass. A built-in rate limiter is on by default too, bounded per provider (keyed by the effective provider name the logs record) so fan-out to one provider can't overrun its limits or eat another's budget: a concurrency cap (8 concurrent calls each by default), plus opt-in requests-per-minute and tokens-per-minute ceilings for metered accounts whose binding limit is per-minute rather than concurrency. Raise the cap, set `rpm=`/`tpm=`, or disable it with `configure_rate_limit(...)`; RPM/TPM are off unless set, so an unset request stays byte-identical. This cross-call retry layer stays separate from instructor's in-call schema-repair budget (llmkit pins instructor's `max_retries` to 2 — two in-call attempts, i.e. exactly one repair re-ask; it is not a caller knob), so the two are never conflated.

## A thin layer, not a gateway

llmkit owns the call ergonomics, the structured-output mode pinning, the rate-limit policy, and the logging convention — and stops there. It does **not** reimplement HTTP transport or run a proxy: that is LiteLLM's job, and reimplementing it is the one thing this library deliberately won't do. Staying thin is what keeps it small, predictable, and easy to reason about.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
