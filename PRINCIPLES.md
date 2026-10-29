# Principles

The design principles behind `llmkit`: the durable promises the library makes to the people who call it. Read this to understand *why* llmkit behaves the way it does. The [README](README.md) is authoritative for the API and its details; where this page names a mechanism, it is the one that ships today.

`llmkit` is a thin, opinionated layer over LiteLLM that turns "call an LLM" into one function returning a validated result, with logging, retries, and rate limiting around it.

## The main principle

llmkit owns correctness; you own policy. Every decision in the library falls on one side of that line.

On llmkit's side is everything that is easy to get wrong and expensive to discover: which structured-output mode a provider actually honours, how a malformed answer is repaired, how retries and backoff are budgeted, what a call record must contain. These have one right answer per provider and a wrong answer that fails silently, so llmkit decides them once, measures them against the live providers, and does not hand them to you to tune. Because those decisions are made per provider and travel with it, switching providers or models is cheap for you: the correctness work has already been done on the other side of the switch.

On your side is everything that depends on your application: which models play which roles, how prompts are built, how much to retry, how hard to drive a metered account, where logs and cached answers live, and what counts as a good answer. These have no right answer outside your application, so llmkit takes no position on them. Where it must ship a default, the default is safe and the setting is yours.

In practice the line sorts every setting into one of three groups:

- **Fixed, because it is correctness:** the structured-output mode each provider runs in, instructor's in-call repair budget, the separation of transport retries from schema retries, and the content of a call record.
- **Configurable with a safe default, because it is policy:** retry budgets and backoff, concurrency and per-minute ceilings, the circuit breaker, log destination and retention, and the response-cache store.
- **Absent, because it is yours alone:** model roles, prompt construction, domain validation of a schema-valid answer, and infrastructure.

The remaining principles are this one line applied to each part of the library.

## One function call

Pass a prompt, plus a Pydantic model when you want structured output, and get a result. `structured_llm_call` returns a validated instance, `text_llm_call` plain text, and `tool_llm_call` the model's requested tool calls. The only bookkeeping you add is a `feature`/`label` pair, which identifies the call in the logs.

The surface is async-first, because LLM calls are I/O-bound and usually fanned out. Every call has a `_sync` twin that drives the same async path and inherits the same retries, rate limiting, and logging.

## Structured output always validates

You pass a Pydantic model and get back a validated instance of it, or an exception. Validation enforces your schema and validators; your application owns domain correctness. A schema-valid answer can still be empty or wrong for your purpose, so reject it with constraints or an [`on_result` hook](README.md#re-rolling-on-a-semantically-bad-result).

Making that contract hold is llmkit's side of the line. Across every provider, structured output is pinned to the mode that *actually* validates, verified against the live APIs; instructor's automatic mode silently regresses some providers to empty results, and llmkit refuses it. The same contract covers a [tool turn's final answer](README.md#tool-calls-with-a-structured-final-answer) on routes measured to support it; an unmeasured or preview-only route fails loudly rather than guess.

## Switching providers and models is easy

OpenRouter, Google AI Studio, Gemini Enterprise Agent Platform (formerly Vertex AI), Anthropic, OpenAI, DeepSeek, Cerebras, AWS Bedrock, and local Ollama sit behind one `Provider` enum and one flat [`LLMClientConfig`](README.md#configuration), registered once at startup. Switching vendors is a config edit, not a refactor, and call sites never name a provider. Switching models is lighter still: every call takes an optional `model=` that overrides the default for that call alone, and llmkit imposes no role system on top. A single call can also take a [`provider=` override](README.md#constructing-a-provider-for-a-per-call-override) built on the fly, so routing one call through another vendor or credential needs no change to global state. The same model is often reachable by more than one route: Gemini through AI Studio or Google's Agent Platform, Claude direct or through Bedrock, so a route can change for residency or billing reasons while the model stays put.

What makes the switch safe rather than merely short is the other side of the line. Each provider's structured-output mode, retry classification, rate limiter, and log record are llmkit's decisions, measured per provider, and they move with the switch. Without that, a one-line change of vendor could silently turn validated answers into empty ones.

Some things do change with a route, and each changes loudly. Model identifiers are provider-specific. Bedrock and the Agent Platform read region and project settings that no other provider accepts, and a config field the selected provider does not read is rejected at construction. A tool turn that asks for a validated final answer is enabled only on routes measured to support it, and refuses elsewhere rather than guess.

## Reliable by default

[Retries](README.md#retries) and [rate limiting](README.md#rate-limiting) are on by default, and the policy that makes them safe is fixed:

- Transient transport failures get a larger retry budget than schema-validation failures, so a deterministically malformed answer cannot spend the transport budget on doomed re-asks. Backoff is bounded and jittered, and a provider's `Retry-After` takes precedence.
- A streaming call is not retried once output has reached you, because replaying consumed output would change the meaning of the call.
- A completion cut off by its output-token limit fails fast, as does a call refused by an open circuit breaker. Programming errors propagate immediately.
- Rate limiting is adaptive and scoped per provider: overload lowers concurrency, and recovery raises it back toward your cap.

The budgets and the failure policy are yours. Tune or opt out of retries per call, set the concurrency and per-minute ceilings for your account, and arm the [circuit breaker](README.md#circuit-breaker) only if failing fast suits you; it is opt-in because it changes "eventually succeeds" into "fails fast", which is an application decision. instructor's [in-call schema repair](README.md#instructors-in-call-schema-repair) stays on the fixed side, separate from your outer retries, because it is a correctness setting rather than a budget.

## Logging by default, written for agents

Every call is [logged with zero setup](README.md#logging-agent-readable-by-default); opt out with `configure_llm_logging(None)`. The assumed reader is a coding agent debugging a run, not a dashboard. So each record leads with the verdict (ok or error, feature and label, model, duration, approximate cost), carries the full prompt and response beneath it, and an append-only index answers cross-call questions in one scan. What a record contains is llmkit's decision; where it goes is yours.

## Local-first, no infrastructure

llmkit needs no collector, llmkit account, or hosted service. Configure access to your provider and go: logs are plain local files, and a local Ollama server keeps inference on your host. (The underlying LiteLLM dependency may fetch model-cost metadata from GitHub.)

External systems can be plugged in later, never required. A [custom `LogSink`](README.md#write-your-own-logsink) ships records anywhere; a [custom `LLMCache`](README.md#write-your-own-store) stores answers wherever you keep them. The [response cache](README.md#response-cache) shows the line exactly: it is off until you configure a store, llmkit owns the request fingerprint, the hit records, and the coalescing of identical in-flight requests, and you own where the answers live.

## A thin layer, not a gateway

llmkit owns the call ergonomics, the structured-output mode pinning, the retry and rate-limit policy, and the logging convention, and stops there. It does not reimplement HTTP transport, construct provider endpoints, or run a proxy: that is LiteLLM's job, and taking it on is the one thing this library deliberately will not do. Staying thin keeps the library small and predictable, and keeps the set of correctness decisions narrow enough to measure and hold.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
