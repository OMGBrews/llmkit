# Principles

The design principles behind `llmkit`: the durable promises the library makes to the people who call it. Each section says what a principle promises and why llmkit holds it. How each promise is kept belongs to the [README](README.md), which is authoritative for the API and its details.

`llmkit` is a thin, opinionated layer over LiteLLM that turns "call an LLM" into one function returning a validated result, with logging, retries, and rate limiting around it.

## The main principle

llmkit owns correctness; you own policy. Every decision in the library falls on one side of that line.

On llmkit's side is everything that is easy to get wrong and expensive to discover: which structured-output mode a provider actually honours, how a malformed answer is repaired, how retries and backoff are budgeted, what a call record must contain. These have one right answer per provider and a wrong answer that fails silently, so llmkit decides them once, measures them against the live providers, and does not hand them to you to tune.

On your side is everything that depends on your application: which models play which roles, how prompts are built, how much to retry, how hard to drive a metered account, where logs and cached answers live, and what counts as a good answer. These have no right answer outside your application, so llmkit takes no position on them. Where it must ship a default, the default is safe and the setting is yours.

The line sorts every setting into one of three groups:

- **Fixed, because it is correctness:** the output mode each provider runs in, how a malformed answer is repaired, the separation of transport retries from answer retries, and the content of a call record.
- **Configurable with a safe default, because it is policy:** retry budgets and backoff, concurrency and per-minute ceilings, the circuit breaker, log destination and retention, and the response-cache store.
- **Absent, because it is yours alone:** model roles, prompt construction, domain validation of a schema-valid answer, and infrastructure.

The remaining principles are this one line applied to each part of the library.

## One function call

Calling an LLM is one function call: pass a prompt, and a Pydantic model when you want structured output, and get a result back. The call is async. Code that cannot await uses a sync form of the same call, which carries every promise the async form does.

llmkit holds this principle because everything that makes an LLM call hard to get right is the same for every call, so it belongs behind the function rather than repeated at every call site. The call is async because LLM calls are I/O-bound and usually fanned out. The sync form exists because a codebase that cannot await should not have to give up any of the library's promises to use it.

## Structured output always validates

Pass a Pydantic model and get back a validated instance of it, or an exception. Validation enforces your schema and its validators. Your application owns domain correctness, because a schema-valid answer can still be empty or wrong for your purpose, and only you can say so.

llmkit holds this principle because a structured answer that is silently wrong is the most expensive failure in LLM code: it passes through the call and breaks something downstream, far from the cause. Whether a given provider actually validates is a per-provider fact that can change without notice, so holding the contract across providers is llmkit's side of the line, and a route llmkit has not measured fails loudly rather than guess.

## Switching providers and models is easy

Every supported provider sits behind one configuration, registered once at startup, and call sites never name a provider. Switching vendors is a config edit, not a refactor. Switching models is lighter still: the model is an argument any call can override, and llmkit imposes no role system on top. A single call can also be routed through another provider or credential without touching global state. The same model is often reachable by more than one route, so a route can change for residency or billing reasons while the model stays put.

llmkit holds this principle because providers and models change for reasons that have nothing to do with your code: price, availability, data residency, a better model this month. The cost of acting on such a reason should match the size of the decision. What makes the switch safe rather than merely short is the main principle: each provider's correctness decisions are llmkit's, so they travel with the switch, and a one-line change of vendor cannot silently turn validated answers into empty ones. What cannot be made uniform across routes changes loudly: a setting the selected provider does not read is rejected, and a capability a route has not been measured to support is refused rather than guessed.

## Reliable by default

Retries and rate limiting are on by default, and the policy that makes them safe is fixed: what counts as retryable, what must never be retried, and how a retry is paced are llmkit's decisions. The budgets are yours. You tune or opt out of retries per call, set the ceilings for your account, and choose whether a failing provider is failed fast or waited out.

llmkit holds this principle because transient failures and rate limits are universal to LLM providers, so every caller would otherwise write the same loop and get a detail wrong. The dangerous retries are the ones that look harmless: replaying a stream whose output you have already consumed, re-asking for an answer that will be malformed every time, or retrying a programming error. Those have one right answer, so they are fixed. How many attempts a call is worth depends on your application, so it is yours. Anything that turns "eventually succeeds" into "fails fast" is opt-in, because that trade is an application decision.

## Logging is on by default

Every call llmkit makes is recorded, with nothing to set up. The record stays on your machine as a plain file. You choose where the files live, how long they are kept, and whether to send records somewhere else or turn logging off. A problem writing a record never fails the call it describes.

llmkit holds this principle because the run you most need a record of is the one you did not expect to go wrong. A debugging aid that has to be switched on in advance is missing exactly then. With every call on file, you can look back at what was actually asked and answered, and improve your work from the evidence rather than from memory.

## Logs are written for agents

The reader llmkit writes for is a coding agent, or a person, opening the log directory cold to find out what happened in a run, not a dashboard. Every record begins with the verdict: whether the call succeeded, what it was for, which model it went to, how long it took, and, when the provider reports it, roughly what it cost. The full prompt and response follow beneath. A compact index across all calls answers questions about a whole run, such as which calls failed or which were slowest or most expensive, without opening every record. What a record contains is llmkit's decision, so every reader finds the same shape in every project. Where the records go is yours.

llmkit holds this principle because a record nobody can read cold is not evidence. The reader is increasingly an agent working from files, with no dashboard and no memory of the run. A fixed, verdict-first shape lets anyone find a failing call, read exactly what was sent and what came back, and improve prompts, model choices, and spend from the record.

## Local-first, no infrastructure

llmkit needs no collector, llmkit account, or hosted service. Configure access to your provider and go. Logs are plain local files, and a local Ollama server keeps inference on your host. External systems can be plugged in later, never required: records can be shipped anywhere and cached answers stored wherever you keep them, but nothing waits on either. (The underlying LiteLLM dependency may fetch model-cost metadata from GitHub.)

llmkit holds this principle because a library that needs infrastructure before it is useful turns the first call into a project, and because the record of your prompts and answers is your data: it should leave your machine only when you send it. Infrastructure is also policy, so every optional system sits on your side of the line.

## A thin layer, not a gateway

llmkit owns the call ergonomics, the structured-output mode pinning, the retry and rate-limit policy, and the logging convention, and stops there. It does not reimplement HTTP transport, construct provider endpoints, or run a proxy: that is LiteLLM's job, and taking it on is the one thing this library deliberately will not do.

llmkit holds this principle because the set of correctness decisions it promises to hold must stay narrow enough to measure against live providers and keep measured. Every layer added beneath the call would widen that set, and a gateway would widen it past what one small library can honestly promise.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
