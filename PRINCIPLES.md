# Principles

The design principles behind `llmkit`: the durable promises the library makes to the people who call it. Each section says what a principle promises and why llmkit holds it. How each promise is kept belongs to the [README](README.md), which is authoritative for the API and its details.

`llmkit` is a thin, opinionated layer over LiteLLM that turns "call an LLM" into one function returning a validated result, with logging, retries, and rate limiting around it.

## The main principle

llmkit owns correctness; you own policy. Every decision in the library falls on one side of that line, and the remaining principles are the line applied to each part of the library.

llmkit holds this principle because the two kinds of decision fail differently. A correctness decision, such as which output mode a provider actually honours, has one right answer, and a wrong one fails silently and far from its cause. It must be made once by whoever can measure it, so llmkit decides it, measures it against live providers, and does not hand it to you to tune. A policy decision, such as how much to retry, has no right answer outside your application, and a wrong one is visible and cheap to change. So llmkit takes no position, and where it must ship a default, the default is safe and the setting is yours.

The line gives every decision one of three shapes: fixed, because it is correctness; configurable with a safe default, because it is policy the library still needs a value for; or absent, because only your application can make it.

## One function call

Calling an LLM is one function call: pass a prompt, and a Pydantic model when you want structured output, and get a result back. The call is async, and a sync form of the same call carries every promise the async form does.

llmkit holds this principle because everything that makes an LLM call hard to get right is the same for every call, so it belongs behind the function rather than repeated at every call site. The call is async because LLM calls are I/O-bound and usually fanned out. The sync form exists because a codebase that cannot await should not have to give up any of the library's promises to use it.

## Structured output always validates

Pass a Pydantic model and get back a validated instance of it, or an exception.

llmkit holds this principle because a structured answer that is silently wrong is the most expensive failure in LLM code: it passes through the call and breaks something downstream, far from the cause. Whether a provider actually validates is a per-provider fact that can change without notice, so llmkit holds the contract on every route it has measured and fails loudly on any it has not. Validation means your schema, not your meaning: a schema-valid answer can still be wrong for your purpose, and only your application can say so.

## Switching providers and models is easy

Switching the provider for a whole program is a config edit, and switching it for one call is one argument on that call, the same as switching the model. Call sites that do not care which vendor answers never name one.

llmkit holds this principle because providers and models change for reasons that have nothing to do with your code: price, availability, data residency, a better model, or a comparison across vendors. The cost of acting on such a reason should match the size of the decision, not the size of the codebase. A switch is only easy if it is also safe: each provider's correctness decisions are llmkit's, so they travel with the switch, changing a vendor cannot silently turn validated answers into empty ones, and anything a route cannot support fails loudly rather than guess.

## Reliable by default

Retries and rate limiting are on by default, with the policy that makes them safe fixed and the budgets yours.

llmkit holds this principle because transient failures and rate limits are universal to LLM providers, so without a default every caller writes the same loop and gets a detail wrong. The details that go wrong look harmless and have one right answer, so llmkit fixes them. How many attempts a call is worth, how hard to drive a metered account, and whether to keep trying or fail fast depend on your application, so those are yours. That reasoning also binds llmkit itself: nothing it adds around a call may make the call less reliable than it would be without it.

## Logging is on by default

Every call llmkit makes is recorded as a plain file on your machine, with nothing to set up and the option to turn it off.

llmkit holds this principle because the run you most need a record of is the one you did not expect to go wrong. A debugging aid that has to be switched on in advance is missing exactly then. With every call on file, you can look back at what was actually asked and answered, and improve your work from the evidence rather than from memory.

## Logs are written for agents

Every record llmkit writes is laid out so that an AI agent can read it cold and understand what happened.

llmkit holds this principle because its logs are expected to be read by AI agents more often than by people. An agent opening the logs should find a failing call, see exactly what was sent and what came back, and understand the run well enough to fix the problem, improve the prompt, or change the model. The easier that is, the more of that work anyone using llmkit can hand to their agents.

## Local-first

llmkit is complete on your machine the moment you configure access to a provider.

llmkit holds this principle because a library that needs infrastructure before it is useful turns the first call into a project, and because the record of your prompts and answers is your data, which should leave your machine only when you send it. Infrastructure is policy, so anything beyond the local default is yours to add, and only when you want it.

## A thin layer

llmkit is a thin layer over LiteLLM, owning what happens around a call and leaving the call itself to LiteLLM.

llmkit holds this principle because every correctness decision it takes on must be measured against live providers and kept measured, and that is only possible while the set of decisions stays small. Everything added beneath the call widens that set, and widens it fastest toward what one small library cannot honestly promise.

## See also

- [`README.md`](README.md) — what `llmkit` is and how to use it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventions and how to contribute.
