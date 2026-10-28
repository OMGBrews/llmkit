# Principles

The design principles behind `llmkit` — the durable promises the library makes to the people who call it. Read this to understand *why* llmkit behaves the way it does; for the API itself, see the [README](README.md). This describes the library as it ships today.

`llmkit` is opinionated on purpose: it decides the boring-but-easy-to-get-wrong things — which structured-output mode each provider needs, where logs go, how retries and rate limits behave — so you don't have to. These principles are what that buys you.

## Validated structured output is the contract

The core promise: you pass a Pydantic model and get back a validated instance of it, or an exception. Validation enforces your schema and validators; your application owns domain correctness. A schema-valid answer can still be empty or wrong for your purpose, so use constraints or an [on_result hook](README.md#re-rolling-on-a-semantically-bad-result) to reject it. Across every provider, structured output is pinned to the mode that *actually* validates, verified against the live APIs (instructor's automatic mode silently regresses some providers to empty results — llmkit refuses it). So `structured_llm_call` gives you the same reliable, typed result whether you're on OpenAI, Gemini, Claude, DeepSeek, Cerebras, Bedrock, or a local model.

On [measured compose-capable routes](README.md#tool-calls-with-a-structured-final-answer), `tool_llm_call(..., output_schema=...)` makes a tool turn return either requested calls or that same validated final instance; unavailable or preview-only routes fail loudly and keep the portable two-step tool-loop-plus-finalization pattern.

## Opinionated where it counts, out of your way everywhere else

llmkit holds firm opinions about reliability, correctness, and observability — and deliberately none about *your* domain. It takes no position on model "roles" (strong/small/current are yours to define), on how you build prompts, on where your logs ultimately live, or on your retry budgets. The opinions exist to remove footguns, not to box you in; application policies such as retry budgets, rate limits, and log destinations are configurable. Correctness policies such as structured-output mode pinning and instructor's in-call repair budget stay fixed.

## Calling an LLM is one function

One call does it: pass a prompt — plus a Pydantic model when you want structured output — and get a result back. `structured_llm_call` returns a validated instance; `text_llm_call` returns plain text. The only bookkeeping you add is a `feature`/`label` pair, and it shows up in the logs.

## Providers switch by config; models switch per call

OpenRouter, Google AI Studio, Gemini Enterprise Agent Platform (formerly Vertex AI), Anthropic, OpenAI, DeepSeek, Cerebras, AWS Bedrock, and local Ollama sit behind one `Provider` enum and one `LLMClientConfig`, registered once at startup — so switching vendors is a config edit, not a refactor, and call sites never name a provider. Switching models is even lighter: every call takes an optional `model=` that overrides the default just for that call.

## Local-first, zero infrastructure

llmkit requires no collector, llmkit account, or hosted service of its own. Configure access to your chosen provider; logs are plain local files, and a local Ollama server keeps inference on your host. The underlying LiteLLM dependency can also fetch model-cost metadata from GitHub.

You can plug in external systems later (a [custom LogSink](README.md#write-your-own-logsink) ships records anywhere; a [custom LLMCache](README.md#write-your-own-store) stores responses wherever you keep them), but you never have to. The [response cache](README.md#response-cache) follows the same rule: off until a host configures one, the shipped store is in-process memory, and llmkit owns the request fingerprint, cache hooks, coalescing of identical buffered requests in flight, and hit records — while the host owns the store.

## Logging is on by default — and written for an AI agent

Every call is [logged with zero setup](README.md#logging-agent-readable-by-default); opt out with `configure_llm_logging(None)`. Each per-call log is laid out verdict-first: a two-line `#` header — a verdict line (`ok`/`ERROR`, feature/label, model, schema, duration, approximate cost) plus a second line carrying the timestamp, a `call=<id> attempt=<n>` correlation suffix, and `cache=hit` when the response cache answered the call — on top, the prompt and response below, plus an append-only `index.jsonl` for cross-call scans. The assumed reader is a coding agent like Claude Code debugging a run — so it can see at a glance what was sent, what came back, and what failed, and use that to diagnose problems and improve prompts.

## Async-first; sync is one call away

The call surface is async-first, because LLM calls are I/O-bound and usually fanned out. Callers that can't `await` use `structured_llm_call_sync`, which drives the same async path — and inherits the same rate limiting and logging.

## Built for reliable calls

Reliability is built into the call surface. [Automatic retries](README.md#automatic-retries-on-by-default) give transient transport failures a larger budget than schema-validation failures, so malformed answers cannot spend the entire transport budget on repeated attempts. Backoff is bounded and jittered; provider `Retry-After` delays take precedence, subject to a configurable cap (60 seconds by default) plus jitter. See [backoff and timing](README.md#backoff-and-retry-after). Streaming retries stop once an item reaches the caller: replaying already-consumed output would change the meaning of the call.

Recoverable failures are not all retryable by default. `LLM_RECOVERABLE_ERRORS` groups transport, schema-validation, tool-argument, output-limit, and backpressure errors for callers that need a common catch-set. Structured and text completions cut off by the output-token limit fail fast by default, as does an open circuit breaker. Programming errors propagate immediately. Callers can [tune retry policy](README.md#tuning-or-opting-out-per-call); instructor's [fixed in-call schema-repair budget](README.md#instructors-in-call-schema-repair) stays separate from those outer retries.

[Adaptive rate limiting](README.md#rate-limiting) is on by default and scoped per provider: overload reduces concurrency, and recovery raises it toward the configured cap. Optional per-minute ceilings support metered accounts. The [circuit breaker](README.md#circuit-breaker) is opt-in because failing fast during sustained overload is a different application policy from waiting and retrying. These controls keep reliability policy in one place while leaving budgets and failure handling with the caller.

## A thin layer, not a gateway

llmkit owns the call ergonomics, the structured-output mode pinning, the rate-limit policy, and the logging convention — and stops there. It does **not** reimplement HTTP transport or run a proxy: that is LiteLLM's job, and reimplementing it is the one thing this library deliberately won't do. Staying thin is what keeps it small, predictable, and easy to reason about.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
