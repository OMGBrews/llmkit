# llmkit

A thin, opinionated, **local-first** layer over [LiteLLM](https://github.com/BerriAI/litellm) (with [instructor](https://github.com/567-labs/instructor) for structured output). It gives an application one provider-agnostic call surface across **OpenRouter, Google AI Studio, Gemini Enterprise Agent Platform (formerly Vertex AI), Anthropic, OpenAI, DeepSeek, Cerebras, AWS Bedrock, and local Ollama**, with validated structured output, per-provider rate limiting (concurrency on by default; optional requests-/tokens-per-minute), **agent-readable per-call logging**, and **transient-error retries on by default** — all out of the box — plus an opt-in, host-pluggable response cache.

LiteLLM is the implementation of the HTTP providers; llmkit owns the ergonomic call surface, the structured-output mode pinning, the rate-limit policy, and the logging convention. It is **not** a gateway and does not reimplement transport — that is solved, and reimplementing it is the thing this library deliberately does not do.

## Contents

- [Why llmkit](#why-llmkit)
- [Install](#install)
- [Quick start](#quick-start)
- [Call reference](#call-reference)
  - [When a structured answer fails validation](#when-a-structured-answer-fails-validation)
  - [When a text answer is cut off](#when-a-text-answer-is-cut-off)
  - [Tool calls with a structured final answer](#tool-calls-with-a-structured-final-answer)
  - [Choosing which tool the model calls](#choosing-which-tool-the-model-calls)
  - [Streaming a tool turn](#streaming-a-tool-turn)
  - [When a call doesn't parse](#when-a-call-doesnt-parse)
  - [Telling the model a tool failed](#telling-the-model-a-tool-failed)
  - [Tool schemas: hand them over unprocessed](#tool-schemas-hand-them-over-unprocessed)
  - [Reusing call options](#reusing-call-options)
  - [Per-request timeout](#per-request-timeout)
  - [Contracts as JSON-schema dicts](#contracts-as-json-schema-dicts)
- [Rate limiting](#rate-limiting)
  - [Adaptive concurrency](#adaptive-concurrency)
  - [Observing backpressure](#observing-backpressure)
  - [Circuit breaker](#circuit-breaker)
  - [Joining the global rate limit directly](#joining-the-global-rate-limit-directly)
- [Logging: agent-readable by default](#logging-agent-readable-by-default)
  - [Grouping calls by run](#grouping-calls-by-run)
  - [Where the logs go](#where-the-logs-go)
  - [Retention: bounded by default](#retention-bounded-by-default)
  - [Capturing call records](#capturing-call-records)
  - [Write your own `LogSink`](#write-your-own-logsink)
- [Response cache](#response-cache)
  - [What counts as the same request](#what-counts-as-the-same-request)
  - [What a hit looks like in the log](#what-a-hit-looks-like-in-the-log)
  - [Write your own store](#write-your-own-store)
- [Configuration](#configuration)
  - [`BEDROCK` and `VERTEX` do not own their endpoints](#bedrock-and-vertex-do-not-own-their-endpoints)
  - [Switching provider for a whole program](#switching-provider-for-a-whole-program)
  - [Switching provider per call](#switching-provider-per-call)
  - [OpenRouter: schema-honoring routing](#openrouter-schema-honoring-routing)
- [Retries](#retries)
  - [Automatic retries, on by default](#automatic-retries-on-by-default)
  - [Backoff and `Retry-After`](#backoff-and-retry-after)
  - [Tuning or opting out per call](#tuning-or-opting-out-per-call)
  - [Wrapping your own work: `with_retries()`](#wrapping-your-own-work-with_retries)
  - [instructor's in-call schema repair](#instructors-in-call-schema-repair)
  - [Re-rolling on a semantically-bad result](#re-rolling-on-a-semantically-bad-result)
- [Testing your app with a fake transport](#testing-your-app-with-a-fake-transport)
  - [Simulating failures](#simulating-failures)
  - [What the fake does not see](#what-the-fake-does-not-see)
- [Development](#development)
- [Status & support](#status--support)
- [License](#license)

## Why llmkit

- **Structured output that actually validates.** Each provider is pinned to an explicit, live-measured structured-output mode (never instructor's auto-`Mode.TOOLS`, which silently regresses Gemini to empty shapes), and instructor's in-call validation-retry repairs malformed JSON. You pass a Pydantic model; you get a validated instance back. (A completion *truncated by the output-token limit* is the one thing never re-asked — it fails fast as `OutputLimitError`; see [Retries](#retries).)
- **Switching providers and models is easy.** OpenRouter / Google AI Studio / Google's Agent Platform / Anthropic / OpenAI / DeepSeek / Cerebras / AWS Bedrock / Ollama behind one `Provider` enum and one `LLMClientConfig`. Switching the provider for a whole program is a config edit, and switching it for one call is one argument on that call, the same as switching the model, so comparing vendors in one program is an ordinary use. Call sites that do not care which vendor answers never name one. The same Gemini models are reachable two ways — AI Studio (a bearer key) or Google's Agent Platform (Google Cloud, with a `vertex_location` data-residency control) — just like Claude is reachable direct or via Bedrock.
- **Logging tuned for coding agents.** Every call is logged verdict-first (see [Logging](#logging-agent-readable-by-default)) — the design assumption is that the reader is usually an LLM coding agent debugging a run, not a dashboard.
- **Local-first, zero infra.** The default sink writes plain files to a directory. No collector, no account, no network. A pluggable `LogSink` lets you ship records anywhere later without touching call sites.

These four are the headline; [`PRINCIPLES.md`](PRINCIPLES.md) states the full set of design principles behind the library.

## Install

This README describes `main`, which may include unreleased changes. After
installing, check `llmkit.__version__` and read the README at the matching
`v<version>` tag in this repository. The [changelog](CHANGELOG.md) distinguishes
released changes from `[Unreleased]`.

```bash
uv add omg-llmkit          # or: pip install omg-llmkit
```

The distribution is published as **`omg-llmkit`** (the bare `llmkit` name was already
taken on PyPI), but the import name is just `llmkit`:

```python
import llmkit
```

You **install** `omg-llmkit` but **import** `llmkit` — that split trips a natural
post-install smoke test. A mistaken `import omg_llmkit` (the install name) raises
a clear one-line redirect to `import llmkit`, not a bare
`ModuleNotFoundError` that leaves you guessing.

`llmkit.__version__` reports the installed version. It is read from the
`omg-llmkit` distribution's metadata rather than hardcoded, so it cannot drift
from what you installed; a source tree with no installed metadata reports
`"0.0.0+unknown"` instead of raising at import.

Requires Python ≥ 3.12.

The supported OpenAI SDK range is `>=2.20.0,<3`;
SDK 3 uses `httpx2` and is not yet supported by llmkit's error handling.

The core install routes OpenRouter, Google AI Studio, Anthropic, OpenAI,
DeepSeek, Cerebras, and Ollama with no extra dependencies. Two providers gate their
dependencies behind opt-in extras so hosts pay only for what they call:

```bash
pip install "omg-llmkit[bedrock]"    # Claude-on-Bedrock (boto3 for SigV4 signing)
pip install "omg-llmkit[vertex]"     # Gemini via Google's Agent Platform
pip install "omg-llmkit[all]"        # both of the above
```

Bedrock's `[bedrock]` extra pulls only `boto3` (LiteLLM signs Bedrock requests
with AWS SigV4 through it), and constructing the `BedrockProvider` without it
raises a clear `install omg-llmkit[bedrock]` error at construction, not a
cryptic failure on the first call. Google's Agent Platform is the same shape: its `[vertex]`
extra pulls only `google-auth` (LiteLLM mints Google's Agent Platform OAuth token through
it), and constructing the `VertexProvider` without it raises a clear `install
omg-llmkit[vertex]` error — a host using another provider takes on no Google dependency.
Direct Anthropic routing needs no SDK at all: LiteLLM speaks the Anthropic
HTTP API itself, so there is no `[anthropic]` extra.

## Quick start

```python
from pydantic import BaseModel
from llmkit import (
    LLMClientConfig,
    Provider,
    configure_llm_client,
    structured_llm_call,
    structured_llm_call_sync,
)

# Point the library at a provider once, at startup. The key comes from
# OPENROUTER_API_KEY, and with no model set OpenRouter's default model answers.
configure_llm_client(LLMClientConfig(provider=Provider.OPENROUTER))

class Summary(BaseModel):
    title: str
    bullets: list[str]

# From synchronous code:
result: Summary = structured_llm_call_sync(
    prompt="Summarize the attached report.",
    output_schema=Summary,
    feature="reports",      # groups calls in the logs
    label="exec_summary",   # names this specific call in the logs
)

# From async code, the same call awaited:
async def summarize() -> Summary:
    return await structured_llm_call(
        prompt="Summarize the attached report.",
        output_schema=Summary,
        feature="reports",
        label="exec_summary",
    )
```

The two forms are the same call. They take the same arguments and give the
same retries, logging and result; the sync form runs the async call on
llmkit's own event loop and waits for it. Inside async code, await the async
form. The call's log record lands under `data/llm-logs/` in your project by
default — see [Where the logs go](#where-the-logs-go).

> **Two defaults worth knowing up front.**
> - **llmkit sends no `temperature` unless you set one.** A call that sets none anywhere hands LiteLLM **no `temperature` field at all**, so the provider's own default sampling applies; pass a number (e.g. `temperature=0.0`) to choose one, and it is forwarded. Some models take no temperature (GPT-5 models with reasoning active, Claude 5.5-era models, and Google's upcoming Gemini models): when LiteLLM or the provider refuses yours as unsupported, llmkit re-issues the request once without it, logs a warning the first time each model refuses, and the call record shows `temperature: null`, so no host needs LiteLLM's global `drop_params`. LiteLLM itself still adds a temperature on Gemini 3 routes — see [Provider-default sampling](#provider-default-sampling-temperaturenone).
> - **Any call switches provider with one argument, as it switches model.** `provider=Provider.ANTHROPIC` sends that call to Anthropic and leaves the registration above alone, so one program can compare several vendors on the same prompt. See [Switching provider per call](#switching-provider-per-call).

Switching this whole program to Ollama, or to a provider whose key is already
in its own environment variable, is one field: change `provider=`. The new
provider reads its own credentials and answers with its own default model, both
listed under [configuration](#configuration). Bedrock and Google's Agent
Platform also need their extra and a region or location; see
[Switching provider for a whole program](#switching-provider-for-a-whole-program).
For local calls, first start an Ollama server and pull the model as described in
[Contributing: live provider tests](CONTRIBUTING.md#live-provider-tests). Then
change the registration above to:

```python
configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA))  # llama3.2 on localhost:11434
```

Pinning `model` makes a switch two fields, because model ids are spelled per
route; see [Switching provider for a whole program](#switching-provider-for-a-whole-program).

For the next step, see [call reference](#call-reference) for tools, streaming,
and shared options; [configuration](#configuration) for provider defaults;
and [logging](#logging-agent-readable-by-default) for stored data and retention.

## Call reference

The public call surface:

| Function | Use |
|----------|-----|
| `structured_llm_call(prompt, output_schema, *, feature, label=None, ...)` | Async, returns a validated Pydantic instance |
| `structured_llm_call_sync(...)` | The same call, run synchronously |
| `text_llm_call(prompt, *, feature, label=None, ...)` | Async, returns plain text (coerces provider list-content blocks) |
| `text_llm_call_sync(...)` | The same call, run synchronously |
| `text_llm_call_stream(prompt, *, feature, label=None, ...)` | Async generator yielding text chunks, logged on completion; retries only before the first chunk |
| `text_llm_call_stream_sync(...)` | The same stream as a plain generator, one chunk per `next()` |
| `tool_llm_call(prompt, tools, *, feature, ..., output_schema=None)` | Async tool turn; with a schema, returns tool calls or a validated final answer |
| `tool_llm_call_sync(...)` | The same call, run synchronously |
| `tool_llm_call_stream(prompt, tools, *, feature, ...)` | Async generator yielding `TextDeltaEvent`s, ending with the completed `ToolCallResult`; retries only before the first event |
| `tool_llm_call_stream_sync(...)` | The same stream as a plain generator, one item per `next()` |

`feature` and everything after it are keyword-only. `feature` is the one
required name: it groups the call's records in the logs. A sync form takes
exactly the parameters of its async form, and a test holds every pair to
that.

**A stream's sync form is a plain generator.** `text_llm_call_stream_sync`
and `tool_llm_call_stream_sync` yield the same items as their async forms, one
per `next()`, and fetch each only when you ask for it. Each captures your
context when you call it, so enter `capture_llm_records()` or a run scope
before the call, not between the call and the loop. Leaving a `for` loop early,
calling `close()`, or dropping the generator records the abandonment exactly as
the async form does. Inside async code, use the async forms. A stream retries a
transient failure only before its first item reaches you; see the
[streaming caveat](#tuning-or-opting-out-per-call).

`prompt` is typed `str | Sequence[ChatMessage]` on all call functions. A plain string becomes one user message. Buffered tool repair re-asks append the rejected round, its errors, and a user instruction to retry the tool request. The sequence form is a conversation, where `ChatMessage` is `Message | AssistantToolMessage | ToolResultMessage`:

- `llmkit.Message` — a `TypedDict` whose `role` is `"system"`, `"user"`, or `"assistant"` and whose `content` is either a string or a list of content-part dicts, the multimodal shape LiteLLM accepts (`{"type": "text", ...}`, `{"type": "image_url", ...}`), forwarded verbatim;
- `AssistantToolMessage` — an assistant turn that requested tools, as `result.to_message()` returns it;
- `ToolResultMessage` — a tool's answer, as `tool_result_message(...)` builds it.

All three are exported so your own prompt builders can be annotated against them rather than against the transport's wire shape: an unknown key (`{"roel": ...}`) or a mistyped role is a type error, and multimodal content type-checks instead of being rejected. Annotate a plain conversation `list[Message]`, and a tool-loop history — which gains assistant tool turns and tool results as it runs — `list[ChatMessage]`.

### When a structured answer fails validation

`structured_llm_call` returns an instance of your model that passed its
validation, or raises. When the answer still fails validation after
instructor's in-call repair re-asks (`schema_repairs`, one by default) and the
policy's `validation_max_attempts` fresh attempts, the call raises
`StructuredOutputError`. `tool_llm_call` with
`output_schema=` raises the same type when its final answer fails validation
or the turn carries neither tool calls nor text. The error carries what
failed:

- `schema_name`: the name of your output model;
- `raw_text`: the answer text that failed, unparsed (`None` when there was no text);
- `errors`: Pydantic's errors (`ValidationError.errors()`), each naming the
  failing field in `loc` (malformed JSON reports one `json_invalid` error), or
  `None` when the failure was not a Pydantic `ValidationError`;
- `attempt`, `model` and `provider`: the attempt it ended and where it ran.

The cause is on `__cause__`: instructor's exception on the structured lane,
pydantic's `ValidationError` on the compose lane, and nothing when a compose
turn had no text. The message names each failing field and the reason, never
the answer itself. `StructuredOutputError` is in
`LLM_SCHEMA_ERRORS`, so `except LLM_RECOVERABLE_ERRORS` still catches it. A
transport or permanent provider failure is not a validation failure and keeps
its own type; see [Retries](#automatic-retries-on-by-default).

```python
from llmkit import StructuredOutputError, structured_llm_call

try:
    report = await structured_llm_call(prompt, Report, feature="reports")
except StructuredOutputError as exc:
    for error in exc.errors or []:
        print(error["loc"], error["msg"])
    print(exc.raw_text)
```

A provider response with no choices at all raises the same error, with
`raw_text` `None`, and is re-asked on the same budget.

**Validation is pydantic's strict JSON mode, on every lane**: the structured
call, a cache hit, a compose call's final answer, and tool arguments. Strict
JSON rejects coercions a model did not ask for, such as `"1"` for an `int`,
`1.0` for an `int` or `"yes"` for a `bool`, and still accepts JSON's string
forms of datetimes, dates, UUIDs, enums and decimals. The instance you get is
your class validating the provider's answer text **once**, the same step a
cache hit takes. A validator that transforms its input sees the model's text,
not a re-serialized answer, and a serializer never touches the returned value.
(instructor validates the same text first to drive its repair re-ask, so a
validator with side effects runs twice per paid answer.)

An `output_schema` llmkit cannot hold to this contract raises
`UnsupportedSchemaError` (a `TypeError`) before any request: a type that is not
a pydantic `BaseModel` subclass; one of instructor's DSL classes
(`IterableModel`, `Partial`, a simple-type adapter), which instructor turns
into a list or a bare value or validates while incomplete; or a model that
declares one of instructor's async validators (`async_field_validator`,
`async_model_validator`), on itself, a base class or a nested model. No lane
runs those validators, so a declared one would silently not validate.

Validation means your schema, not your meaning: to reject an answer that
validates but is wrong for your purpose, see
[Re-rolling on a semantically-bad result](#re-rolling-on-a-semantically-bad-result).

**The structured-output contract is measured per provider.** Each provider's
structured-output mode and request traits are fixed by llmkit and checked by the
live provider suite on that provider's default model. The model you pass is your
choice: llmkit keeps no per-model list, and it runs any model under its
provider's pinned mode. A provider you define yourself, by subclassing
`BaseProvider` or implementing `LLMProviderInterface`, is a route llmkit has not
measured: it accepts whatever instructor mode you name. Every answer is still
validated against your schema, but a mode that produces empty or
default-filled answers can pass that check, so whether the mode suits your
provider is yours to measure.

### When a text answer is cut off

When the provider reports that it stopped at the output-token limit (`finish_reason="length"`), `text_llm_call`, `text_llm_call_sync`, `text_llm_call_stream` and `text_llm_call_stream_sync` raise `OutputLimitError` instead of handing back the cut-off text as if it were the whole answer — the same error a truncated structured call raises. A missing or any other finish reason is a normal answer. The cut-off text is not lost: it is on `exc.diagnostics.partial_text`, beside the finish reason and the token counts the provider reported, and in the call's log record.

```python
from llmkit import OutputLimitError, text_llm_call

try:
    answer = await text_llm_call(prompt, feature="drafting", max_tokens=400)
except OutputLimitError as exc:
    partial = exc.diagnostics.partial_text if exc.diagnostics else None
    ...  # use the partial text, or raise max_tokens and call again
```

A stream yields every chunk it received and then raises from the iteration, so catch the error around the `async for` (or the `for` over the sync form), not the call that creates the stream. The text you already received is the partial answer:

```python
received: list[str] = []
try:
    async for chunk in text_llm_call_stream(prompt, feature="drafting", max_tokens=400):
        received.append(chunk)
except OutputLimitError:
    partial = "".join(received)  # also on exc.diagnostics.partial_text
```

The error is never retried by default and never cached; see [Retries](#automatic-retries-on-by-default). Tool calls do not raise it: they report the provider's reason as `stop_reason`. The one exception is a compose call's final answer, described below.

### Tool calls with a structured final answer

The portable pattern is still **two steps**: run a tool loop with `tool_llm_call`, then make one `structured_llm_call` for the final answer. That works across every provider, including Gemini.

For a measured compose-capable route, pass `output_schema=` to the tool call instead. Each turn is either a normal `ToolComposeResult` with `tool_calls` and `parsed is None`, or a final answer with `parsed` set to your validated Pydantic instance. Anthropic and OpenAI support the combined request; unsupported providers and unsafe legacy Claude models raise `ComposeUnsupportedError` *before any request* with a steer to the portable pattern. A registered config that cannot be built (no API key, a field the provider does not read) raises that build error instead, also before any request.

```python
from pydantic import BaseModel
from llmkit import (
    ChatMessage, Provider, ToolDefinition, make_provider,
    tool_llm_call, tool_result_message,
)

class AddArgs(BaseModel):
    a: int
    b: int

class Answer(BaseModel):
    total: int

# Compose is measured on OpenAI and Anthropic, so this loop names its provider
# on each call, whatever the program registered. The key comes from OPENAI_API_KEY.
provider = make_provider(Provider.OPENAI, model="gpt-4.1-mini")
tools = [ToolDefinition.from_model("add", AddArgs, "Add two integers")]
history: list[ChatMessage] = [
    {"role": "user", "content": "Calculate 2 + 3; use the add tool."},
]

for _ in range(5):  # bound the number of tool turns
    result = await tool_llm_call(
        history, tools, output_schema=Answer,
        feature="calculator", provider=provider,
    )
    if not result.tool_calls:
        assert result.parsed is not None
        print(result.parsed.total)
        break
    history.append(result.to_message())
    for call in result.tool_calls:
        args = call.validated
        assert isinstance(args, AddArgs)
        history.append(tool_result_message(call.id, str(args.a + args.b)))
else:
    raise RuntimeError("Tool loop did not finish within five turns")
```

Use `describe_llm(config).compose_tools_schema`, or `describe_llm(provider=...).compose_tools_schema` for a per-call provider (both from `llmkit.providers`), to choose the optimization without using exceptions. Compose calls preserve the tool lane's retry, rate-limit, usage, and log behavior — including re-asking a round in which every requested call is malformed — and their log schema is `tools+<ModelName>`. The compose lane validates the final text locally, in pydantic's strict JSON mode, and retries a malformed final answer within `validation_max_attempts`; a final-answer re-ask has no repair prompt, so the caller's complete tool history is re-sent. An all-malformed tool round instead gets the buffered tool lane's repair history. When that budget is spent it raises `StructuredOutputError`, the same type the structured lane raises. A final answer cut off by the output-token limit raises `OutputLimitError` and is not retried; a turn that requested tools keeps reporting its reason as `stop_reason`. Gemini's compose feature remains preview-only, so llmkit deliberately keeps it on the portable path; its `gemini_structured_output="json"` escape hatch is an instructor-mode setting and does not apply here.

### Choosing which tool the model calls

`tool_llm_call`, `tool_llm_call_sync`, `tool_llm_call_stream`, and
`tool_llm_call_stream_sync` take a `tool_choice=` keyword. Leave it `None` (the default) for the provider's own
behaviour, or pass `"auto"`, `"none"`, `"required"`, or `ToolName("weather")` to
force one named tool (`ToolName` is exported; `ToolChoice` is the union type). A
route whose provider does not support `tool_choice` — Ollama — accepts `"auto"`,
which is what it does anyway, and raises `ToolChoiceUnsupportedError` (an
`LlmkitError` and a `ValueError`) for `"none"`, `"required"` and `ToolName(...)`
before the first attempt, so no request is sent and no call record is written,
rather than silently ignoring the setting.

### Streaming a tool turn

A tool turn returns only once it is complete, so a chat panel rendering
assistant prose gets it in paragraph-sized blobs. `tool_llm_call_stream` yields
that prose as it arrives and then ends with the **same** `ToolCallResult` the
buffered call returns, so your tool loop is unchanged from the result onward:

```python
from llmkit import TextDeltaEvent, ToolCallResult, tool_llm_call_stream

result: ToolCallResult | None = None
async for event in tool_llm_call_stream(history, tools, feature="chat"):
    if isinstance(event, TextDeltaEvent):
        render(event.text)          # token-by-token, as it arrives
    else:
        result = event              # the completed turn; the stream is done

assert result is not None
for call in result.tool_calls:
    ...                             # your loop, exactly as before
```

The completed result is the **last item yielded**, not a return value — an
async generator cannot return one. That is why the two shapes are distinct
types: you discriminate on the type, never on position. Breaking out after the
result is the intended way to stop, and is recorded as a *completed* call;
leaving earlier records the partial transcript with `STREAM_ABANDONED_ERROR`,
never a clean `ok`. Blocking code gets the same items, with the same records,
from a plain `for` over `tool_llm_call_stream_sync`.

Three differences from `tool_llm_call`, all deliberate:

- **No `output_schema=`.** Compose is rejected at the signature, so the type
  checker enforces it rather than a runtime raise. Use the portable two-step
  pattern (tool loop, then `structured_llm_call`).
- **Retries stop at the first yielded item.** Once anything has reached you,
  a mid-stream error propagates unretried. A round in which every call is
  malformed and no prose was streamed has yielded nothing, so it is re-asked
  with the buffered lane's repair history, within `validation_max_attempts`
  and the overall attempt ceiling. If prose was streamed first, it raises
  `ToolArgumentError` unretried, whatever the policy lists. A round in which
  only some calls are malformed behaves as on the buffered lane — survivors
  on `tool_calls`, failures on `invalid_calls`.
- **Usage is real, not null.** Like the plain text stream, this lane requests
  `stream_options={"include_usage": True}`, so `result.usage`, the log record's
  `usage`, and the token-rate limiter's debit are all populated. Measured on
  Google's Agent Platform, Anthropic, OpenAI, OpenRouter, Google AI Studio and DeepSeek, and on
  Cerebras, whose LiteLLM route drops `stream_options` but which reports usage
  on the stream regardless; Bedrock and Ollama are covered by the same code path
  but that measurement is still outstanding.

Log records use the `tools-stream` schema and carry the offered `tools`, the
returned `tool_calls`, and `usage`, exactly as the buffered tool lane does.

### When a call doesn't parse

A turn can request several calls at once, and one of them can come back unusable — arguments that aren't JSON, a tool you never offered, arguments the definition's model rejects (validated from the raw JSON string in pydantic's strict JSON mode, so `{"a": "2"}` does not satisfy an `int` field). Nothing has executed at that point, so llmkit **keeps the calls that did parse** and reports each failure on `result.invalid_calls` as the `ToolArgumentError` it raised (carrying the tool name, the call id, and the raw argument string):

```python
result = await tool_llm_call(prompt, tools, feature="assistant")

for call in result.tool_calls:      # every call that parsed and validated
    ...
for error in result.invalid_calls:  # what the model got wrong, if anything
    log.warning("dropped %s: %s", error.tool_name, error)
```

`result.to_message()` is built from `tool_calls` alone, so the assistant turn you append names only the calls you can answer — pair each with a `tool_result_message` and the history has no dangling `tool_call_id` for the provider to reject. A round in which **every** requested call is malformed is re-asked within `RetryPolicy.validation_max_attempts` with the rejected assistant tool-call turn and one `ERROR:` tool result per call, naming the tool, call id and validation error, followed by a user instruction to issue a corrected tool call. Only the latest rejected round is attached; the original prompt appears once. Non-object or invalid JSON arguments are replayed as `{}`, with the raw arguments in the error result; unknown tools are named with the offered alternatives. If any call lacks an id or tool name, the re-ask keeps the original history because the round cannot be paired with results. Exhausting the budget raises `ToolArgumentError`, whose `round_errors` holds every rejection. Strict JSON validation is unchanged. The streaming lane re-asks the same way, but only when the round streamed no prose first; otherwise it raises `ToolArgumentError` unretried. Dropped calls also appear under `invalid_calls` in the call's log record, so a round that quietly lost one never reads as a clean round.

### Telling the model a tool failed

Tools fail — a timeout, a 500, a lookup that finds nothing. Say so with `is_error=True` instead of inventing an envelope:

```python
from llmkit import tool_result_message

try:
    output = run_tool(call)
except ToolFailed as exc:
    message = tool_result_message(call.id, f"{exc}", is_error=True)
else:
    message = tool_result_message(call.id, output)
```

Without a standard flag every application invents its own convention — a JSON wrapper, a bare traceback, a sentence in the result — and the same model recovers differently depending on which one it meets. This is the convention, so it doesn't.

**What the model actually receives.** The flag is llmkit's, not the wire's. There is no field to carry it: LiteLLM's Anthropic translation builds its `tool_result` block from the call id and the content alone (LiteLLM's Anthropic translation still leaves `is_error` commented out at 1.104.0, the supported floor — its own source comment records that the OpenAI-normalized shape cannot express the distinction), and an OpenAI-compatible route would forward an unknown key straight to a provider that may reject it. So the transport prefixes the content with `TOOL_ERROR_PREFIX` (`"ERROR: "`, exported so you can recognise or strip it) and drops the key — one rendering, identical on every provider. The flag itself stays on the message you hold and in the `prompt` the log records (including library-added buffered tool repair turns), so your provenance keeps the distinction structurally even though the wire cannot. Omitting `is_error` sends exactly the three-key message it always did.

### Tool schemas: hand them over unprocessed

**`ToolDefinition.from_model()` output works on every supported provider as-is. Do not pre-process it.** A Pydantic model with a nested model or an enum renders `$defs` and `$ref` in its JSON Schema, and llmkit forwards `parameters` verbatim — but the transport normalises per provider on the way to the wire. On the Gemini routes (Google's Agent Platform and Google AI Studio) that means `additionalProperties` is stripped, `$defs` is popped, `$ref` targets are inlined, and keywords outside Gemini's `Schema` subset are filtered, so a schema Gemini's API rejects outright when posted raw is accepted through llmkit. Writing your own normalisation layer on top duplicates that work and will drift from it.

The same applies to a **tool that takes no arguments**: `{"type": "object", "properties": {}}` is the portable schema, sent to Gemini as a bare `OBJECT` declaration. You do not need a dummy parameter.

```python
class Unit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class Location(BaseModel):
    city: str
    country: str

class ForecastArgs(BaseModel):        # renders $defs + $ref …
    where: Location
    unit: Unit
    days: int

tools = [
    ToolDefinition.from_model("forecast", ForecastArgs, "Look up a forecast"),
    ToolDefinition("current_time", "Get the current time", {"type": "object", "properties": {}}),
]                                     # … and both go over unchanged
```

**Where the guarantee stops.** It is delivery in the provider's accepted subset, not lossless translation: a construct the subset cannot express is *dropped, not errored*. The one to know about is that Gemini accepts `enum` only on string-typed fields, so an enum on an integer or float field is silently discarded and the model sees an unconstrained number. Keep enums as string enums when a Gemini route is in play, or validate the range yourself — `from_model()` validates the model's arguments, in strict JSON mode, locally either way, so a value outside the enum still fails validation on our side.

Both halves are pinned by tests: `tests/providers/test_gemini_tool_schema_transport.py` asserts the transform's output offline (so a transport upgrade that stopped normalising fails the suite), and `test_vertex_tool_schema_roundtrip_live` asserts Google's Agent Platform accepts it live.

### Reusing call options

The call functions (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, `text_llm_call_stream`, `text_llm_call_stream_sync`, `tool_llm_call`, `tool_llm_call_sync`, `tool_llm_call_stream`, and `tool_llm_call_stream_sync`) take a block of per-call keyword arguments. When a feature module makes many calls with the same settings, repeating that block at every site is noise. Build an `LLMCallOptions` once and pass it as `options=`:

```python
from llmkit import LLMCallOptions, Provider, structured_llm_call

# Built once per feature module.
RISK_OPTS = LLMCallOptions(
    provider=Provider.GOOGLE,
    temperature=0.0,
    model="gemini-2.5-flash",
    reasoning_effort="high",
    max_tokens=2048,
)

async def extract(prompt: str) -> RiskRegister:
    return await structured_llm_call(
        prompt, RiskRegister, feature="extraction", options=RISK_OPTS
    )
```

`LLMCallOptions` is **frozen** and carries any subset of `temperature` / `model` / `max_tokens` / `reasoning_effort` / `retry` / `provider` / `timeout` / `cache` / `schema_repairs`. `cache` is read by the buffered families and both forms of the text stream (see [Response cache](#response-cache)); the tool calls never read the response cache, so they ignore an options bundle's `cache` field. `schema_repairs` is read by `structured_llm_call` and its sync form only (see [instructor's in-call schema repair](#instructors-in-call-schema-repair)); the other families ignore a valid value but still reject an invalid one. That is what lets one bundle serve every call a feature module makes. Every field is optional and *unset* by default — an unset field defers to the call's keyword (and through it to the provider the call runs on), so a partially-filled `LLMCallOptions` only supplies the fields you set. Like the call keywords, `temperature` accepts `None` (`LLMCallOptions(temperature=None)`), which requests the provider's default sampling — the same as leaving it unset.

The bundle sets `provider` beside `model` because a model id is spelled for one route (`gemini-2.5-flash` is the Google AI Studio and Google's Agent Platform spelling), so the route and its id travel together. A member is built for each call from `GEMINI_API_KEY`, read then, so a key kept in `.env` is found; see [Switching provider per call](#switching-provider-per-call).

`feature` is intentionally **not** part of `LLMCallOptions`. It stays a required per-call keyword as a telemetry forcing function: it scopes the per-call log filename and the `index.jsonl` grouping operators grep, so it must be a conscious choice at each call site rather than something defaulted-away into a shared object.

The flat-keyword path is unchanged — pass no `options` and nothing about existing calls changes.

#### Call-vs-config precedence

`model` and `reasoning_effort` are *dual-homed*: they can be set both on `LLMClientConfig` (the app-wide default) and on the call surface. The precedence, lowest to highest, is:

**config < `options` < explicit per-call keyword**

A call that passes `provider=` takes the lowest layer from that provider instead: its own `model` and `reasoning_effort` (as given to `make_provider` or `build_provider`, or the provider's defaults for a `Provider` member), not the registered config's.

So a value passed directly as a keyword wins — *any* passed value, including `None` or one equal to the documented default. The call functions' mergeable keywords default to the `UNSET` sentinel rather than to real values, so "was it passed" is a structural fact, never inferred by comparing values: `structured_llm_call(..., temperature=0.2, options=LLMCallOptions(temperature=0.9))` runs at `0.2`, and an explicit `model=None` forces the provider/config default even when `options` carries a model. The same holds for `temperature=None`: it overrides a numeric `options` value (`structured_llm_call(..., temperature=None, options=LLMCallOptions(temperature=0.9))` sends no `temperature` field), and a numeric `LLMCallOptions(temperature=0.9)` is used when the keyword is unset. An `LLMCallOptions` field sits between the keyword and the config; when neither the keyword nor `options` supplies a value, the true default applies (`DEFAULT_TEMPERATURE`, `DEFAULT_RETRY_POLICY`, `DEFAULT_SCHEMA_REPAIRS`, or, for `model`/`reasoning_effort`, the provider the call runs on: the registered config's, or a per-call `provider=`'s own). The same precedence applies to `timeout`: `timeout=5.0` overrides `LLMCallOptions(timeout=30.0)`, while explicit `timeout=None` overrides the options value and sends no timeout. An *unset* `LLMCallOptions` field never overrides config — only a field you explicitly set on the options participates.

`Unset` (the type) and `UNSET` (the value) are exported for one idiom: your own typed wrapper can declare `temperature: float | None | Unset = UNSET` and forward it unconditionally, letting llmkit resolve "not passed" vs "provider default" (`None`) vs a number instead of re-inventing the sentinel. Compare with `is`/`is not` only — never truthiness.

#### Provider-default sampling (`temperature=None`)

llmkit chooses no temperature of its own (`DEFAULT_TEMPERATURE` is `None`), so there are two states on the wire:

| You pass | Meaning | Wire |
|---|---|---|
| *(nothing)* | provider's built-in default sampling | **no `temperature` key at all** |
| `temperature=None` | the same as passing nothing | **no `temperature` key at all** |
| `temperature=0.5` (any number, incl. `0.0`) | your value, forwarded | `temperature: 0.5`, or **no key** after the model refused it (see below) |

"Wire" is what llmkit hands LiteLLM. On every route but one that is also what
reaches the provider; the exception is Gemini 3 on Google AI Studio and Google's Agent Platform,
where LiteLLM adds a temperature of its own — see the Gemini caveat below.

`None` is accepted on every call surface — structured, plain-text, streaming, tool calls, and the sync wrappers — both as a direct keyword and through `LLMCallOptions`. When the resolved value is `None`, llmkit omits the `temperature` kwarg from the provider request entirely (an identity check, so `0.0` is never mistaken for unset). Log records reflect the omission: `LLMCallRecord.temperature` is `None` and the YAML sink writes `temperature: null`, for a default call and an explicit `None` alike.

**A refused temperature is omitted, once.** Some models accept no sampling temperature: LiteLLM refuses any value but `1` on GPT-5 models whose effective reasoning effort is not `none` (including a model whose own default effort is not `none`, such as `gpt-5-mini` with no `reasoning_effort` passed) and on Claude 5.5-era models on the Anthropic and Bedrock routes, and a provider may refuse it with an HTTP 400 of its own. On OpenRouter the refusal is a 404 instead: because every request carries `require_parameters` (see [OpenRouter: schema-honoring routing](#openrouter-schema-honoring-routing)), a model none of whose endpoints accepts a temperature answers "No endpoints found that can handle the requested parameters", with routing failed at its `Filter by Parameters` step — measured on 2026-11-03 for `temperature=0` on `openai/gpt-6-luna`. When a 400 names `temperature`, or that OpenRouter 404 arrives on a request carrying one, llmkit re-issues the same request exactly once with LiteLLM's per-call `additional_drop_params=["temperature"]` — never the process-global `litellm.drop_params` — so the call runs at the model's own sampling. The record then shows what was sent, `temperature: null`, rather than the value you asked for, and the `llmkit._litellm` logger warns the first time each model refuses, naming the model and the dropped value; later drops for that model log at DEBUG. If the re-issue fails too, the original refusal is raised. The OpenRouter 404 names no parameter, so when another one was filtered, the re-issue meets the same 404 and the original is raised after one more round trip; OpenRouter's retired-model 404 ("No endpoints found for <model>") is never re-issued. A [cache](#response-cache) hit records the temperature you asked for, because it sent nothing. One limitation: if your process sets the global `litellm.drop_params = True`, LiteLLM drops the value silently, no refusal reaches llmkit, and the record shows your value.

**Gemini caveat.** Google has announced that its upcoming Gemini models will reject requests that include `temperature`, `top_p`, or `top_k` with an error (custom sampling values have had no effect since Gemini 3.6 Flash). llmkit never sends `top_p`/`top_k` and, by default, no `temperature` either. An explicit temperature is forwarded on every provider, Gemini included, and a Gemini model that rejects it with an error naming `temperature` gets the request re-issued once without it, as above. One wrinkle remains on the Google AI Studio and Google's Agent Platform routes: through LiteLLM 1.104.0 (the supported floor), `VertexGeminiConfig.map_openai_params` re-inserts `temperature = 1.0` (Google's recommended value) for Gemini 3 models when the request carries none, so a default Gemini 3 call still sends `temperature: 1.0` on the wire. Removing that injection is an upstream LiteLLM change ([BerriAI/litellm#38663](https://github.com/BerriAI/litellm/issues/38663)); until llmkit consumes a release containing it, a default Gemini 3 call means "no warning, wire value `1.0`". OpenRouter adds no temperature when none is sent.

### Per-request timeout

Every call function, including the sync wrappers, accepts `timeout` in
seconds, directly or through
`LLMCallOptions(timeout=30.0)`. It is a deadline llmkit enforces itself: a
provider round trip still waiting when it expires is cut off and raised as a
builtin `TimeoutError` reading `no provider response within 30s`. The number
is also forwarded to LiteLLM. Unset or `None` sets no deadline and sends no
`timeout` keyword, leaving LiteLLM's own request timeout in force
(`litellm.request_timeout`, 6000 seconds on LiteLLM 1.108.0 unless its
`REQUEST_TIMEOUT` environment setting says otherwise). An explicit call value
overrides a provider subclass's `completion_kwargs()` timeout too, and a
timeout that only `completion_kwargs()` supplies is enforced the same way.
`0.0` is accepted and fires at the first wait.

llmkit enforces it because LiteLLM's timeout cannot bound an attempt on its
own. It is an HTTP read timeout, so a provider that keeps the connection open
without sending anything never trips it. On the OpenAI-compatible routes the
SDK also retries a timed-out request inside one call, by default twice.

The value bounds one provider round trip: each transport retry, instructor
schema-repair re-ask and refused-temperature re-issue gets its own timeout.
For a stream, it bounds opening the request, the wait for the first chunk,
and each later wait for the next chunk, rather than the stream's total
duration; the time your code spends between chunks never counts. A breach
before anything reached your code is retried as a fresh attempt. A breach
after a chunk reached you ends the stream with the `TimeoutError`,
unretried, like any mid-stream error. Nothing else cuts a stream between
chunks. The timeout does not include waiting for a rate-limiter slot or retry
backoff sleeps. Timeout errors are transport errors and are retried with a
fresh timeout per attempt; pass `retry=NO_RETRY` to turn those retries off.
A cut attempt counts as a timeout for the rate limiter, which treats timeouts
as neutral: they neither shrink adaptive concurrency nor count toward opening
the circuit breaker, though a half-open breaker's probe that times out
re-opens it, as any failed probe does. On `structured_llm_call` the `TimeoutError`, like every transport
error, reaches you inside instructor's `InstructorRetryException`, with the
timeout as its `__cause__`.

The attempt's [log record](#logging-agent-readable-by-default) says what happened. A cut attempt
records the `TimeoutError` as its `error`, not the abandoned-stream marker,
and a streamed attempt records `first_chunk_ms`, the time to its first chunk.

The sync wrappers behave the same way: the sync bridge adds no whole-call
ceiling of its own, and a sync stream adds no limit to each wait for its next
item beyond this one. This setting is not a whole-call deadline on either
form; the retry budget and each request's timeout are what bound a call,
with, when a response cache is configured, the cache's own `timeout` on each
store read and write (see [Response cache](#response-cache)).

### Contracts as JSON-schema dicts

If your structured-output contract is a **JSON-schema dict** — typically because the same schema is shared with a Node backend or a frontend — `model_from_json_schema(schema)` converts it to a Pydantic model at runtime, so you don't hand-write the converter (and re-discover its footguns). Build the model **once and reuse it**; `structured_llm_call` stays Pydantic-model-only and takes the result as `output_schema`.

```python
from llmkit import model_from_json_schema, structured_llm_call

INVOICE_SCHEMA = {                       # shared with Node / the frontend
    "title": "Invoice",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "total": {"type": "number"},
        "status": {"enum": ["open", "closed", "void"]},
        "note": {"type": ["string", "null"]},          # optional, nullable
        "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
    },
    "required": ["id", "total", "status", "lines"],
    "$defs": {
        "Line": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku"],
        }
    },
}

Invoice = model_from_json_schema(INVOICE_SCHEMA)   # build once, at import

result = await structured_llm_call(
    prompt="Extract the invoice.",
    output_schema=Invoice,                         # reuse on every call
    feature="billing",
)
```

The factory snapshots the original schema and builds a **schema-backed Pydantic
root model**. The original schema governs local validation before coercion and
again over the default JSON serialization. Changing your input schema later
does not change an existing model's contract. Caller-written Pydantic models
keep their existing interface.

```python
invoice = Invoice.model_validate({
    "id": "INV-1", "total": 12, "status": "open", "lines": [], "note": None,
})
invoice.root["id"]                 # "INV-1"
invoice.root["lines"]              # ordinary lists and dictionaries
invoice.model_dump()               # the original JSON object, including note: None
invoice.model_dump_json()          # JSON text with "note": null
```

**Migration from generated fields:** replace `Invoice(id=...)` with
`Invoice.model_validate({...})`, and replace `invoice.id` with
`invoice.root["id"]`. Nested objects are dictionaries and arrays are lists;
there are no generated property entries in `model_fields`. `model_dump()`
returns the JSON object directly. Explicit nulls are preserved by default and
absent keys stay absent; `exclude_none` no longer has llmkit's former optional
null preservation policy. Extra keys follow the schema: absent
`additionalProperties` permits them, `false` rejects them, and a schema value
validates their values. Propertyless objects are supported. Defaults are
annotations and never fill missing values. A title-less schema gets the name
`JsonSchemaModel`; `name=` sets it explicitly.

A schema-backed model also works with `ToolDefinition.from_model`: declare an
object schema, and read the validated arguments from the handler's `.root`.

#### Schema constraints

Local validation uses `jsonschema` over the original schema, including
`pattern`, `multipleOf`, `const`, `uniqueItems`, composition (`allOf`, `anyOf`,
`oneOf`, `not`, conditionals), and rules inside nested properties, array items,
and referenced definitions. Invalid instances raise Pydantic `ValidationError`
with instance and schema locations and use the existing bounded repair loop; an
answer still invalid when that loop is spent surfaces as
`StructuredOutputError`, with those errors on its `errors`. Validator engine
faults use `SchemaEngineError` and fail without response repair. Structured
calls may carry it inside Instructor's exception wrapper;
`llmkit.exceptions.underlying_provider_error` retrieves that cause. No legacy
path silently drops validation rules.

The supported dialects are **draft-07, 2019-09, and 2020-12**. `$schema` selects
the dialect; omission selects 2020-12, and any other dialect is rejected during
construction. Subresources must use the same dialect as the root; a nested
`$schema` selecting a different dialect is rejected. The root, after resolving
a root reference, must declare the single string `"type": "object"`. Array, scalar, nullable-object, and typeless
roots remain unsupported. Applicators alongside an object type are supported.
Malformed known keywords, unsupported required vocabularies, unavailable
references, and recursive references fail at construction with schema locations.
References resolve only against resources within the supplied document; llmkit
never fetches schemas over the network. Dialect rules govern reference siblings
(in particular, draft-07 ignores assertion siblings of `$ref`).

Unknown keywords and vendor extensions are annotations, including arbitrary JSON
payloads; they do not become validation rules. `default`, `examples`, `title`,
`description`, `$comment`, `deprecated`, `readOnly`, and `writeOnly` likewise do
not alter data. **`format` is an annotation by default.** Opt into assertions
with `model_from_json_schema(schema, assert_formats=True)`. Construction then
fails for formats absent from the installed format checker, including formats
whose optional checker dependencies are unavailable.

JSON integers remain exact Python integers, including values above `2**53`.
Floating-point values use Python double precision: `9007199254740993.0` is
rounded before validation, and floating-point `multipleOf` arithmetic can reject
`0.3` against `0.1`. Non-finite numbers (`NaN`, infinity, or overflow such as
`1e400`) are rejected. Patterns use Python `re`; ECMA-only regex syntax that
Python cannot compile fails during construction.

The schema sent to a provider is **generation guidance**, separate from the
local validation authority. Each `model_json_schema()` call returns a fresh
adapted copy: `$schema` is omitted, acyclic references are inlined (including
bare references), and the definition containers are removed. Caller composition
remains present; llmkit does not flatten arbitrary intersections. Existing route-specific LiteLLM filtering can
limit the guidance a provider sees, but never weakens local validation. A
provider rejecting guidance surfaces its error rather than causing a silent
schema downgrade. The OpenRouter strict upgrade applies only to plain object
roots with absent or false `additionalProperties` and no unsupported strict
composition, whose nested objects are all closed (`additionalProperties: false`)
and whose objects list every property in `required`; other shapes remain
non-strict and use local validation and repair.
The live suite exercises a schema with `pattern` and `const` on each shipped
provider route, including Gemini's `"json"` strategy; these cases add real calls,
cost, and potential failures to the weekly live check.

## Rate limiting

Rate limiting is **on by default**, scoped **per provider** (keyed by the effective provider name, matching how logging records it), across three independent dimensions:

- **Concurrency** — **on by default**, default cap **8 concurrent calls per provider**: enough headroom for the fan-out workloads consumers actually run, while still bounding a self-inflicted burst; lower it for a tightly-metered account, raise it for a local Ollama server. This cap is also **adaptive by default** — it backs off when a provider signals overload and recovers automatically (see [Adaptive concurrency](#adaptive-concurrency) below). The cap binds async callers and the `*_sync` wrappers alike — and because every `*_sync` call runs on llmkit's single persistent event loop, a thread-pool fan-out of sync calls shares one per-provider cap. One caveat follows from the concurrency gate being bound to a single loop: a process that drives calls on more than one loop (e.g. running the async call functions on its own event loop *and* the `*_sync` wrappers on the persistent loop) caps each loop's population independently, so it can momentarily hold up to *loops* × the cap per provider; the adaptive limit and the RPM/TPM budgets are shared across all of them.
- **Requests per minute (RPM)** — **opt-in**, off by default. A per-provider request-rate ceiling.
- **Tokens per minute (TPM)** — **opt-in**, off by default. A per-provider token-rate ceiling, debited by each call's measured token usage.

`configure_rate_limit(max_concurrent=..., enabled=..., rpm=..., tpm=..., adaptive=..., breaker=..., breaker_cooldown=...)` sets them. Each call changes only the keywords it passes and keeps the rest, so setting `rpm=` at startup and `breaker=True` later leaves both in force; pass `rpm=None` or `tpm=None` to turn that dimension off again. `max_concurrent`, `rpm` and `tpm` must be positive integers (not a bool, and not a float such as `3.5e3`), and `breaker_cooldown` must be finite and positive; any other value raises `ValueError` and leaves the configuration unchanged. Every call also clears the per-provider limiter state, so configure at startup rather than mid-run. `get_rate_limit_config()` reads back the effective `enabled` / `max_concurrent` / `rpm` / `tpm` / `adaptive` / `breaker` / `breaker_cooldown` (handy to log or assert at startup); `configure_llm_logging(sink)` swaps the log sink (below). The `breaker` switch arms the opt-in per-provider [circuit breaker](#circuit-breaker).

```python
from llmkit import configure_rate_limit

# Stay under a metered account's published per-minute limits:
configure_rate_limit(rpm=3_500, tpm=2_000_000)
```

RPM and TPM are **opt-in** because the right per-minute number is the metered limit of *your* account, with no safe default to assume. Concurrency defaults to 8; tune that cap for your account and workload too. Leaving them unset applies **no throttle** on those dimensions. The binding limit on a metered cloud account is usually RPM/TPM rather than concurrency, so a migrator coming from a requests-per-minute knob should set `rpm=` here — **the concurrency cap does not stand in for an RPM limit** (the two limit different things, and an old RPM tuning otherwise goes inert). Both use a per-provider **token bucket**, which tolerates a small burst above the configured ceiling and then smooths to the sustained rate. That burst is deliberately small — `min(max_concurrent, rpm)` requests for RPM, roughly one second of tokens for TPM — *not* a full minute's quota. Against a provider that enforces a strict fixed minute window, the burst is the worst-case overshoot, so its *relative* size scales with your limits: with the default `max_concurrent=8` it is negligible at `rpm=3_500` (~0.2%) but a meaningful fraction of a small limit (8 extra requests on `rpm=50` is 16%). A tightly-metered account should lower `max_concurrent` (which shrinks the RPM burst with it) or set `rpm=` a little below the published number to leave headroom. When the RPM ceiling does make calls wait, they are admitted in **arrival order** (FIFO): a late arrival cannot jump the queue ahead of a caller already waiting *on the same event loop*, so no caller on that loop is starved under sustained saturation. The queue is **per loop**, exactly like the per-loop concurrency caveat above — a process that drives the same provider across more than one loop (the async call functions on its own loop *and* the `*_sync` wrappers on the persistent loop) gets per-loop FIFO rather than one global arrival order across them. (Host code that [joins the limiter directly](#joining-the-global-rate-limit-directly) through the synchronous `rate_limit_acquire_sync` orders on its own independent ticket queue as well.) The aggregate rate stays exact regardless of how the queues interleave. Both streamed lanes request usage and debit TPM when the stream runs out; abandoned streams have no terminal usage to debit.

### Adaptive concurrency

The per-provider concurrency limit is **adaptive by default**: when a provider pushes back with an overload signal (HTTP **429 / 503 / 529**), llmkit lowers that provider's in-flight limit, and raises it back toward `max_concurrent` once the provider stops pushing back — a TCP-style additive-increase / multiplicative-decrease (AIMD) loop, with **zero per-account tuning**. It *discovers* a safe concurrency under a sustained-overload window that a fixed cap cannot ride out, and is the library-side generalization of a hand-tuned RPM ceiling.

It only ever lowers the limit *below* `max_concurrent`, never above, so a provider that never throttles behaves **identically to a fixed cap** — adaptive concurrency is a no-op until the provider actually pushes back. The decrease only fires when you were genuinely *saturated* (running at the limit when the throttle arrived), so an isolated throttle received while running well under the cap doesn't penalise a healthy provider; recovery is paced by wall-clock time, so it is bounded regardless of how slowly your workload trickles. It is, honestly, a **trade**, not a free win: during a real overload it deliberately runs narrower (which is correct), and the residual cost is a brief reduced-width window after the provider recovers. Turn it off with `configure_rate_limit(adaptive=False)` to pin the limit at `max_concurrent`. (Adaptive concurrency is also a strong complement to `Retry-After`, [below](#retries) — the retry waits as the provider directs, while the fleet as a whole backs off.)

```python
from llmkit import configure_rate_limit

configure_rate_limit(adaptive=False)  # fixed cap at max_concurrent, never adjusted
configure_rate_limit(adaptive=True)   # adaptive again (the default)
```

### Observing backpressure

Install a `backpressure_callback` to *see* the adaptive limiter move in real time — for metrics, a budget-visibility dashboard, or just logging. The callback receives a `BackpressureEvent(provider, old_limit, new_limit, reason)` each time a provider's limit changes (`reason` is `"throttle"` or `"recover"`) or — when the circuit breaker is armed, [below](#circuit-breaker) — the breaker changes state (`"breaker_open"`, `"breaker_half_open"`, `"breaker_closed"`). It is read from a context variable — like `retry_progress_callback` — so it propagates across the `run_sync` boundary; install it once around your fan-out.

```python
from llmkit import backpressure_callback, BackpressureEvent

def on_backpressure(event: BackpressureEvent) -> None:
    print(f"{event.provider}: {event.old_limit} -> {event.new_limit} ({event.reason})")

with backpressure_callback(on_backpressure):
    ...  # calls here report every adaptive limit change to on_backpressure
```

Ordinary callback exceptions are caught and logged; cancellation and process interrupts propagate. Callbacks run inline and must not block: there is no separate callback deadline, and a blocked callback delays the call and its event loop.

### Circuit breaker

Adaptive concurrency drives a struggling provider's limit *toward 1*; the **circuit breaker** stops calls when the provider appears unavailable. It is **opt-in and off by default** — arm it with `configure_rate_limit(breaker=True)`. Its rolling window counts successes and outages: throttles (429/503/529), transient server errors (500/502/504), and connection failures. Once at least half of the last 20 counted outcomes are outages, each new call raises `CircuitOpenError` immediately, holding no concurrency slot and deducting no RPM token. The open period lasts `breaker_cooldown` seconds, **30 by default**.

After that period, one process-wide probe tests recovery. A success closes the breaker; an outage reopens it for another cooldown. Caller deadlines, SDK timeouts without an outage HTTP status, cancellation, authentication and validation failures are neutral: they neither fill the outage window nor claim recovery. A neutral probe releases its claim so the next caller may probe immediately. A short caller-selected deadline is not evidence that a provider is down. A positive outage HTTP status takes precedence over exception inheritance: a provider HTTP 504 counts even though LiteLLM represents it as a timeout.

```python
from llmkit import configure_rate_limit, CircuitOpenError, structured_llm_call

configure_rate_limit(breaker=True, breaker_cooldown=15.0)  # opt in; default cooldown is 30s

try:
    result = await structured_llm_call(prompt, MySchema, feature="extract")
except CircuitOpenError as exc:
    ...  # the breaker is open for exc.provider — fall back fast, don't hammer it
```

It is **off by default** on purpose. Adaptive concurrency reduces load below your cap; the breaker instead refuses calls while it is open, so the host chooses whether to enable it. `CircuitOpenError` is in `LLM_RECOVERABLE_ERRORS` for fallback handling, but the default retry policy never retries it. A caller can explicitly list it in `RetryPolicy.retry_on` to opt in. Each provider has its own breaker. The window and trip fraction are internal mechanisms; `breaker_cooldown` is a caller-owned wait budget, must be finite and positive, and retains its current value when omitted from a later configuration call.

### Joining the global rate limit directly

llmkit's own call functions already pass every provider call through the
global, per-provider limit (concurrency on by default; RPM/TPM when
configured). If your app issues provider calls **outside** those functions —
for example a LangChain chat-model wrapper that calls the provider itself — you
can join the same per-provider budget by hand with the module-level acquire
functions:

```python
from llmkit import Provider
from llmkit.rate_limiting import (
    rate_limit_acquire_async,
    rate_limit_acquire_sync,
)

# Async path (e.g. an async _agenerate):
async with rate_limit_acquire_async(Provider.OPENAI) as slot:
    response = ...  # one slot held against OpenAI's budget
    slot.record_tokens(response.usage.total_tokens)  # debits TPM (no-op when off)

# Sync path (e.g. a synchronous _generate / _stream):
with rate_limit_acquire_sync(Provider.OPENAI) as slot:
    response = ...  # one slot held against OpenAI's budget
    slot.record_tokens(response.usage.total_tokens)
```

The argument is a **`Provider` member** or a **provider instance**: the member
joins the budget llmkit's own calls on that provider charge, and an instance,
including one of your own provider classes, joins the budget of its `name`. A
plain string such as `"openai"` raises `TypeError`, so a misspelt or outdated
name cannot quietly start a separate budget. Each provider has an independent
budget on every dimension. A slot held through `rate_limit_acquire_sync` counts
against the same RPM/TPM buckets, adaptive limit and circuit breaker as llmkit's
own calls, but its in-flight slots are counted separately from theirs, like a
second event loop's (see the concurrency caveat above). Each
context manager yields a `RateLimitSlot`; call its `record_tokens(...)` once you
know the call's token usage to debit the TPM budget (a no-op when TPM is off).
Both are no-ops when rate limiting is disabled, and they share the exact
throttle llmkit's own call paths use, so a hand-joined slot counts against the
same budgets.

To check whether limiting is currently active, read the effective config rather
than reaching into the limiter:

```python
from llmkit.rate_limiting import get_rate_limit_config

if get_rate_limit_config().enabled:
    ...
```

`get_rate_limit_config().enabled` is the public way to ask whether limiting is
on; `GlobalRateLimiter` in `llmkit.rate_limiting` is internal machinery, not
part of the public surface.

## Logging: agent-readable by default

The default sink stores prompts and responses without automatic payload
redaction. Treat the files as sensitive: disable persistence with
`configure_llm_logging(None)` or supply a filtering sink before sending data
you do not want stored. See [Security: scope](SECURITY.md#scope).

`LocalYamlLogSink` (the default) writes **two** things to the log directory:

1. **One YAML file per call, laid out verdict-first.** The file opens with a one-line `#` header — `ok`/`ERROR`, feature/label, resolved model, schema, duration, approximate cost — so `head -1 *.yaml` triages a whole run (the second header line carries the timestamp plus `call=<id> attempt=<n>`, so retries of one logical call — including passes of your own `with_retries` loop — are joinable from the file heads alone, and ends `cache=hit` when the [response cache](#response-cache) answered the call). Small metadata is next; the large `partial_text`, `response` and `prompt` blobs are last, so the *head* of the file is the whole story for most reads.
2. **A compact append-only `index.jsonl`** — one JSON line per call (file, timestamp, feature, label, model, provider, schema, run_id, call_id, attempt, cache_hit, duration, queue wait, time to first chunk, cost, error). Cross-call questions — "which calls errored / were slowest / most expensive / the last call for feature X" — are a single small scan instead of globbing and parsing every YAML.

```yaml
# ok | reports/exec_summary | google/gemini-3.1-flash-lite | Summary | 1840ms | $0.0007
# 2026-06-05T14:22:31.004512 | call=9f3c21ab attempt=1

timestamp: '2026-06-05T14:22:31.004512'
feature: reports
label: exec_summary
model: google/gemini-3.1-flash-lite
provider: OpenRouter
schema: Summary
run_id: nightly-eval-2026-06-05
call_id: 9f3c21ab54d64f1f8f2c14febc03a7d1
attempt: 1
cache_hit: false
source_call_id: null
temperature: null
max_tokens: null
reasoning_effort: null
tools: null
tool_calls: null
usage:
  prompt_tokens: 812
  completion_tokens: 164
  total_tokens: 976
output_limit: null
duration_ms: 1840.2
queue_wait_ms: 0.4
first_chunk_ms: null
approximate_cost: 0.0007
error: null
partial_text: null
response: ...
prompt: ...
```

`approximate_cost` is LiteLLM's per-response estimate for budget visibility — **not** a billing figure (and `None` when pricing is unavailable; `0.0` on a cache hit). `call_id` is one id per *logical* call and `attempt` the 1-based attempt within it, so the N records a retried call produces join on `call_id` — whether llmkit retried it or your own `with_retries` loop re-ran it (each pass rejoins the call it belongs to, matched by `feature`/`label`; a call from another task never joins). `duration_ms` measures the whole attempt **including** `queue_wait_ms` — the time spent queued behind llmkit's own rate limiter — so provider latency is approximately `duration_ms - queue_wait_ms` (hook time and in-call schema-repair re-asks are also inside `duration_ms`). `queue_wait_ms` is `float | None`, not always a float: it is `0.0` when the limiter is disabled and `None` when the attempt failed *before* acquiring a slot or was answered by the response cache (`cache_hit` tells the two apart), so a custom sink or `index.jsonl` parser has to handle the null rather than subtract it blindly. `first_chunk_ms` is, for a streamed attempt, the time from the attempt's start to the first frame of any kind the provider sent, a role-only or reasoning frame included. It is measured like `duration_ms`, so it includes `queue_wait_ms`, and `first_chunk_ms - queue_wait_ms` is the provider's own time to first chunk. A stream that waited 100 seconds and then streamed for 2 reads `first_chunk_ms` near 100000 against a `duration_ms` near 102000. It is `null` on every buffered call, on a cache hit, and on a streamed attempt that received nothing, which is how a stream cut off before its first chunk reads. `temperature` is the same kind of field: a call that sent no temperature (the default, `temperature=None`, or a value the model refused and llmkit omitted) records as `null`, so a typed custom sink reading `record.temperature` must handle `None` (a `float` format spec, for instance, will raise). `tools`, `tool_calls`, and `usage` are always present. The tool lanes fill `tools` and `tool_calls` with the offered tools and the returned calls, and every other call writes `null`. `usage` is the provider's own `prompt_tokens`, `completion_tokens` and `total_tokens` (each `null` when not reported) on every lane: it is filled on each successful provider attempt (a cache hit spent no tokens and writes `null`), on a structured or text attempt whose result your `on_result` hook rejected, and on a truncated structured, text or compose final-answer attempt (the same counts as `output_limit`); any other failed attempt writes `null`. Both streams, `text_llm_call_stream` and `tool_llm_call_stream` (and their sync forms), ask the provider for usage and fill it on every attempt that runs out, completed or truncated; an abandoned stream writes `null`. The text stream's usage is measured on Google's Agent Platform, Anthropic, OpenAI, OpenRouter, Google AI Studio, DeepSeek, Cerebras and Ollama; the Bedrock measurement is still outstanding. At stream end, `approximate_cost` uses a provider-reported cost or LiteLLM's calculator over the reported token counts, including cached-token details. Unmapped models and nonlocal zero-price table placeholders record `null`; local Ollama streams record `0.0`. Pricing failures never fail the stream. `output_limit` and `partial_text` are always present too, and `null` unless a **structured or text call, or a compose call's final answer, was cut off by its output limit**: then `output_limit` holds the truncated completion's `finish_reason`, `prompt_tokens`, `completion_tokens`, `total_tokens` and `reasoning_tokens` (each `null` when not reported, so a truncation that reported nothing is a mapping of nulls), and `partial_text` holds the unfinished answer. `response` stays `null` on a buffered attempt, because unfinished text is not a finished answer; the streamed lanes differ, logging their partial transcript under `response` too, because the consumer has already received it. `completion_tokens` may include reasoning tokens that never appear in `partial_text`, and a `reasoning_tokens` of `0` can also mean the provider did not break them out. Neither field reaches `index.jsonl`: find the truncations there by their `OutputLimitError` `error`, then open the file. The same values are on the raised error as `OutputLimitError.diagnostics` (an `OutputLimitDiagnostics`); the tool lanes pass `finish_reason` through as `stop_reason` instead and never raise this error, except on a compose call's final answer (`tool_llm_call` with `output_schema=`). `cache_hit` is `true` when the [response cache](#response-cache) answered the call instead of a provider, and `source_call_id` then names the paid call whose answer it returned (`null` otherwise); `index.jsonl` carries `cache_hit` but not `source_call_id`. `run_id` is the *outer* scope `call_id` does not give you — see below — and is `null` unless you set one. `error` is `"<ExceptionType>: <message>"` for a failed attempt (the bare type name when the exception has no message) and `null` for a clean one. A **cancelled** structured, text or tool attempt records `CancelledError` there, for instance when your own `asyncio.timeout` cancels the call, rather than the `null` that would make it indistinguishable from an attempt that succeeded and returned nothing. An attempt cut off by its [`timeout`](#per-request-timeout) records `TimeoutError: no provider response within <n>s` (a structured attempt records the same message inside `InstructorRetryException`), and a stream cut that way records the same rather than the abandoned-stream marker, which stays reserved for a stream your own code stopped consuming.

### Grouping calls by run

`call_id` joins the attempts of one logical call. It cannot tell you which calls belonged to one **run** — an eval sweep, a rehearsal, an incident replay — and neither can the timestamp, because a time window stops discriminating the moment two runs overlap. Set a `run_id` and every record, YAML body and `index.jsonl` line carries it, so one shared log directory stays filterable:

```bash
jq 'select(.run_id == "nightly-eval-2026-06-05")' index.jsonl
```

Precedence is **`run_scope` → `set_run_id` → `LLMKIT_RUN_ID`**. A scope overrides the process default while it is active:

```python
import llmkit

llmkit.set_run_id("nightly-eval-2026-06-05")   # process-wide, until cleared with None

with llmkit.run_scope("tuner-session-7"):      # scoped to the block
    ...

llmkit.get_run_id()                            # what's in force right now
```

…and `LLMKIT_RUN_ID` in the environment, which needs no code change at all:

```bash
LLMKIT_RUN_ID=nightly-eval-2026-06-05 python -m your_eval_sweep
```

The environment variable is the lowest layer, so an explicit `set_run_id` or `run_scope` overrides it; a blank one counts as unset. A blank *programmatic* value raises instead — pass `None` to mean "no run id". With nothing set, `run_id` is `null` and the rest of each record is unchanged.

Which of the two programmatic setters you want depends on your concurrency, because they fail in opposite directions. `run_scope` is context-scoped: it survives the sync bridge (a `*_sync` call runs on llmkit's persistent loop inside a copy of your context, which a sync stream takes when you create it), but a `threading.Thread` you start yourself gets a *fresh* context and will not see it. `set_run_id` is a process global — visible from every thread and every loop, but a single value, so it cannot express two runs overlapping in one process. Use `set_run_id` for "this process is one run" (including a thread-pool fan-out), `run_scope` for a host driving several runs at once.

`run_id` and `LLMKIT_LOG_DIR` are independent and compose freely — but tagging is what lets you keep *one* greppable history instead of walking N per-run directories.

### Where the logs go

The default directory is resolved **lazily at the first write** and then frozen for the sink's lifetime (a mid-run `chdir` can't split logs):

1. `LLMKIT_LOG_DIR`, when set;
2. `data/llm-logs/` under the nearest ancestor directory carrying a `pyproject.toml` or `.git` (nearest wins) — when the sink creates that directory it attempts to seed a `.gitignore` to reduce accidental commits (best effort; forced staging is still possible);
3. otherwise a per-user state directory (`$XDG_STATE_HOME/llmkit/llm-logs` on Linux, `~/Library/Logs/llmkit` on macOS, `%LOCALAPPDATA%\llmkit\logs` on Windows) — never a CWD-relative path.

The first successful write logs one INFO naming the absolute directory and the retention policy. `default_log_dir()` returns the currently-resolved answer; an explicit `LocalYamlLogSink(log_dir=Path(...))` — a `pathlib.Path`; a plain string raises `TypeError` — is used as given, made absolute at construction so that a *relative* path (or a relative `LLMKIT_LOG_DIR`) names one directory for the sink's lifetime instead of following the process around. Pass `configure_llm_logging(None)` to disable logging entirely.

On POSIX, the sink creates the leaf log directory with mode `0o700` and new log files with mode `0o600`. Intermediate directories and pre-existing files are outside that creation-mode guarantee. Existing directory permissions are preserved, but sharing a directory alone does not give other users read access to new log files; shared access requires separate file-permission management.

### Retention: bounded by default

By default the sink attempts to prune per-call YAML files older than **30 days** and rotates `index.jsonl` past **50 MiB** to a date-stamped sibling (which ages out under the same policy). Housekeeping is best effort and runs after successful writes, throttled to once per hour, off the event loop. Inactive applications and failed deletions can leave old files indefinitely; applications needing a fixed deletion deadline must arrange their own cleanup.

```python
LocalYamlLogSink(retention_days=None)                  # keep everything forever
LocalYamlLogSink(retention_days=7, max_index_bytes=None)  # tighter age bound, no rotation
LocalYamlLogSink(retention_days=0)                     # ValueError — see below
```

⚠️ **The sink owns its directory's `*.yaml` and `index-*.jsonl` namespace.** Pruning is a glob over `log_dir`, not a list of files the sink remembers writing: **any** `*.yaml` or rotated-index `index-*.jsonl` file in that directory whose mtime is older than `retention_days` is deleted, whoever wrote it. That is deliberate — a consumer who parks cross-reference YAML beside the per-call logs usually wants the two to rot on the same clock — but it makes `log_dir` llmkit's directory rather than shared storage. If you keep your own files there and want them to outlive the policy, give llmkit a directory of its own (`LocalYamlLogSink(log_dir=...)` or `LLMKIT_LOG_DIR`), or turn pruning off with `retention_days=None`. The active `index.jsonl` is never age-pruned — only its rotated `index-<timestamp>.jsonl` generations are.

Each bound takes a **positive** integer or `None`; `0` and negatives raise `ValueError` at construction. `0` is rejected rather than quietly interpreted because it has two opposite plausible meanings — "keep nothing" (`logrotate`'s `rotate 0`) and "no limit" (many SaaS retention settings) — and the destructive reading is what the mechanics would actually do: the prune cutoff would be *now*, so the first write would delete every log in the directory, including the file it had just written and whose path it returns. `None` is the only opt-out, for both bounds.

Managed call logging sends sink I/O to one dedicated daemon thread, including abandoned streams. Calls wait at most 100 ms for delivery by default; stream abandonment captures and submits without waiting. See the delivery policy below.

### Capturing call records

Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, `text_llm_call_stream`, `text_llm_call_stream_sync`, `tool_llm_call`, `tool_llm_call_sync`, `tool_llm_call_stream`, and `tool_llm_call_stream_sync`) builds an `LLMCallRecord` and submits it for managed delivery to the configured log sink. A higher-level orchestrator that needs to cross-reference those calls — to total approximate cost, attribute spend per feature, or weave per-call traces — has two additive capture primitives, neither of which requires authoring a sink.

**`capture_llm_records()` — records (cost / metadata).** Wrap a scope to receive the `LLMCallRecord` for every call made inside it. Each record carries `approximate_cost` (a best-effort USD estimate, `None` when the provider doesn't report it), the resolved `model`/`provider`, `duration_ms`, `error`, and the rest — so a host gets cost and metadata without writing a custom sink. Capture is sink-independent: it works even with logging disabled (`configure_llm_logging(None)`), and crosses the `run_sync` sync bridge, so `structured_llm_call_sync` is captured exactly like the async path. One record is appended per attempt (retries each produce their own), before admission and before the call returns or propagates an error. Full or unavailable delivery preserves record capture. Delivery workers never append to these capture lists after return.

```python
from llmkit import capture_llm_records, structured_llm_call

with capture_llm_records() as records:
    result = await structured_llm_call(prompt, MySchema, feature="extraction")

total_cost = sum(r.approximate_cost or 0.0 for r in records)
```

**`capture_llm_log_paths()` — file paths.** Wrap a scope to receive paths whose sink completion the caller observes within its delivery wait. The default `LocalYamlLogSink` and custom sinks exposing `write_returning_path` can supply paths. A timeout, cancellation, zero wait or abandoned-stream submission promises no captured path, and late completion never appends one. With logging disabled or a sink without path support, the list stays empty. Use `capture_llm_records()` for sink-independent cost/metadata.

```python
from llmkit import capture_llm_log_paths, structured_llm_call

with capture_llm_log_paths() as paths:
    result = await structured_llm_call(prompt, MySchema, feature="extraction")
# paths contains only paths observed before this call returned; it may be empty.
```

### Write your own `LogSink`

`LogSink` is a `Protocol` with a single, file-agnostic method:

```python
class LogSink(Protocol):
    def write(self, record: LLMCallRecord) -> None: ...
```

A custom sink (a database, a metrics pipe, an in-memory buffer) is a one-method object that returns nothing; accepted records (`LLMCallRecord`, a frozen dataclass) are handed to it by the delivery worker, and ordinary exceptions from writes are caught. To send records somewhere other than local YAML — a database, an HTTP collector, structured stdout — implement `write` and register it:

```python
import json
import sys
from llmkit import LLMCallRecord, configure_llm_logging

class StructuredStdoutSink:
    def write(self, record: LLMCallRecord) -> None:
        print(
            json.dumps({
                "feature": record.feature,
                "label": record.label,
                "model": record.model,
                "provider": record.provider,
                "schema": record.schema,
                "duration_ms": record.duration_ms,
                "approximate_cost": record.approximate_cost,
                "error": record.error,
            }),
            file=sys.stdout,
            flush=True,
        )

configure_llm_logging(StructuredStdoutSink())   # pass None to disable logging entirely
```

This example writes one JSON line directly to stdout and needs no Python
logging handlers or formatter. The host owns collecting that output.

Ordinary sink exceptions do not fail the LLM call, and llmkit does not retry the provider
because a sink raised. Built-in write failures and exceptions from custom sinks
produce diagnostic warnings, with repeated identical failures reduced to DEBUG;
host logging configuration can hide those diagnostics, so read
`get_logging_health()` (below) when you need to know whether records were lost.
The diagnostics themselves are contained too: an ordinary exception raised by a
host's `logging.Handler` while llmkit reports a problem (a sink failure, a cache
failure, a retry, a backpressure callback error) does not replace the call's
result or the provider's error, and is counted in `get_diagnostics_health()`
(below). Cancellation, `KeyboardInterrupt` and `SystemExit` still propagate. A
handler that blocks is not contained: diagnostics run inline, on the calling
thread or the delivery worker, and wait for it.
A custom sink that returns normally counts as a handoff: llmkit cannot see
whether it actually wrote anything through `write(record) -> None`. Record
capture proves a record was built, not that a sink persisted it.

**Bounded managed delivery.** One daemon worker handles sink writes, with room for
256 waiting records plus at most one active write. A full queue drops the incoming
delivery immediately and counts it in logging health; record capture still works.
A normal call waits at most 100 ms, including time queued and writing. A timeout
stops only the wait: accepted delivery can finish later. Cancellation stops waiting
immediately, including when it is already unwinding through a call finalizer.
Abandoned streams submit without waiting. These are maintainer-selected defaults,
not benchmark-derived optima.

Configure the policy before the first managed admission:

```python
from llmkit import LogDeliveryPolicy, LocalYamlLogSink, configure_llm_logging

configure_llm_logging(
    LocalYamlLogSink(),
    delivery_policy=LogDeliveryPolicy(queue_capacity=256, delivery_wait=0.1, drain_timeout=1.0),
)
```

`queue_capacity` must be a positive integer (not a bool), `delivery_wait` a finite
number from zero to 0.1 seconds, and `drain_timeout` a finite nonnegative number.
An explicit policy supplied after managed delivery starts raises a configuration
error; omitting it preserves the current policy when swapping sinks. Repeated
wait timeouts, sink swaps and sync-loop restarts never create replacement workers.
A permanently stuck sink can prevent further delivery until process restart.
A sink raising `BaseException` makes the service unavailable; pending delivery is
accounted for as lost. Reentrant calls from the worker submit without waiting,
and worker-originated flushes return immediately as incomplete.

Before queueing, llmkit privately snapshots exact built-in dictionaries, lists
and tuples, recursively, preserving shared references and cycles. Subsequent
mutations to those containers in application inputs or captured records cannot
change the queued copy. Opaque objects and container subclasses retain shared
reference semantics; no custom copy hooks run. This snapshot is taken at log
submission, not at provider invocation, and does not promise an atomic view of
concurrent mutation during copying. Snapshot preparation takes time proportional
to the payload and is outside the 100 ms delivery wait; serialization and sink
I/O remain on the worker. Cost depends on container structure: immutable text and
image strings are reused, so serialized byte size alone does not predict copying
time. Queue capacity counts preparation reservations too,
but does not bound payload bytes or caller-owned capture buffers.

Explicit low-level `write_llm_log()`, `capture.record_call()` and `sink.write()`
remain synchronous and unbounded. Managed delivery cannot forcibly terminate
arbitrary sink code or bound unrelated provider, host executor or background work.

**Flushing and exit.** `flush_llm_logs(timeout=None)` and
`await flush_llm_logs_async(timeout=None)` wait for records accepted before flush
entry, using the configured one-second drain budget by default. A finite
nonnegative override selects another budget; zero polls without waiting. The
return value says whether that accepted prefix settled, including failed writes.
Inspect health to distinguish persistence, handoff and failure. Later submissions
do not extend a flush; a timed-out flush retains pending work, and cancellation
of an async flush cancels only its wait.

`llmkit.sync.shutdown()` flushes logging even for async-only applications and
keeps the delivery worker available for later calls. An explicit smaller shutdown
timeout also limits its logging wait. Normal process exit performs one final
bounded drain after applicable sync-loop cleanup, then discards remaining queued
work and counts the loss. An active write remains unconfirmed. The daemon worker
does not keep the process alive. Abrupt termination can lose records; applications
that need delivery evidence should flush and inspect health before exiting.

`configure_llm_logging` checks the object you hand it: anything that is neither `None` nor a `LogSink` — including the sink *class* where you meant an instance — raises `TypeError` there and then, rather than being installed and silently swallowing every subsequent call's log. The check is structural, so it tests that `write` exists, not that its signature is right (a type checker catches the rest at the call site). One consequence in your own tests: a bare `MagicMock()` doesn't register as a sink, because structural checks use static attribute lookup and see through no `__getattr__` — stub with `Mock(spec=LogSink)`, a small fake class, or `configure_llm_logging(None)`.

**Installing a sink temporarily: `get_log_sink()`.** `configure_llm_logging` sets; `get_log_sink()` reads back what is installed (`LogSink | None`). Save and restore around a scope rather than guessing what to put back:

```python
from llmkit import configure_llm_logging, get_log_sink

previous = get_log_sink()
configure_llm_logging(MyCapturingSink())
try:
    ...
finally:
    configure_llm_logging(previous)
```

Restoring the returned value is what makes this correct in the two starting states a guess gets wrong: logging deliberately disabled (`None`), and the default sink, which is a *specific instance* carrying its own resolved log directory and housekeeping clock — `configure_llm_logging(LocalYamlLogSink())` puts back an equivalent sink, not the one that was there. Reach for `capture_llm_records()` above when you only want the records; this pair is for genuinely swapping the installed sink.

**Checking for lost records: `get_logging_health()`.** Logging is best-effort, so no record is guaranteed to be stored. What llmkit can detect, it counts, independently of whether any warning was displayed. `get_logging_health()` returns a frozen `LoggingHealth` snapshot:

```python
from llmkit import get_logging_health

health = get_logging_health()
if health.failed or health.dropped:
    print(f"write failures={health.failed}, dropped={health.dropped}")
```

- `sink` is `"disabled"` (logging set to `None`), `"builtin"` (`LocalYamlLogSink`), or `"custom"`.
- `written` counts records the built-in sink wrote to disk. `degraded` counts those whose index line or retention housekeeping failed afterwards: the YAML file exists, but `index.jsonl` may lack its line.
- `handed_off` counts records a custom sink accepted by returning normally. llmkit cannot see past that return, so a handoff is not evidence the record was stored.
- `failed` counts records lost to a detected failure: preparation failed, the built-in write failed, a sink raised, or a sink broke its return contract.
- `queued` includes waiting records and snapshot preparation reservations; `in_flight` counts active managed writes for the installation.
- `dropped` counts rejected or discarded delivery, including a full queue; `unavailable` identifies admission failures when the service cannot accept work. `preparation_failed` counts snapshot or worker-startup failures, also included in `failed`.
- `wait_timeouts` counts expired caller waits. A timeout is not proof of loss: a later successful write still counts in `written` or `handed_off`.
- `last_failure` is a `LoggingFailure` naming the stage (`write`, `index`, `housekeeping`, `sink_raised`, or `contract`), the `"Type: message"` error text, the record's feature and label, and the time. It never holds the prompt or response. Admission, preparation and worker-termination counters retain no exception text and do not replace this last sink failure. `last_success_at` is the time of the last write or handoff.

Counts belong to the installation bound at submission and reset on every `configure_llm_logging` call, so a temporary swap clears visible counts even when it restores the same sink. Older queued records still use their original sink and counters. Counts are lock-guarded; a write in progress has no completion outcome yet. A user `atexit` handler registered after `import llmkit` runs before llmkit’s final drain: explicitly flush and inspect health there if needed, checking the flush result rather than assuming delivery completed.

**Checking for lost diagnostics: `get_diagnostics_health()`.** Logging health answers whether records were lost; diagnostics health answers whether llmkit's own warnings were lost, so a sink failure whose warning is also lost increments both, for different reasons. When a host `logging.Handler` raises while handling a record on an `llmkit.*` logger, llmkit discards that record, keeps the call's outcome, and counts the loss in a frozen `DiagnosticsHealth` snapshot:

```python
from llmkit import get_diagnostics_health

diagnostics = get_diagnostics_health()
if diagnostics.lost:
    print(f"lost diagnostics={diagnostics.lost}, last={diagnostics.last_loss}")
```

- `lost` counts diagnostics whose logging call raised, since the process started. It is process-wide and never reset, because it is not tied to a sink installation.
- `last_loss` is a `DiagnosticLoss` naming the logger, the level of the lost record, the handler's `"Type: message"` error text and the time, or `None` when nothing was lost.

A lost diagnostic is not reported through logging again, since the broken handler would only raise again. A handler that blocks rather than raises is not detected here.

The shipped `LocalYamlLogSink` additionally exposes the path it wrote via its own `write_returning_path(record) -> Path | None` method — that file detail stays off the shared `LogSink` contract, and it is what powers `capture_llm_log_paths()` internally. A sink that defines `write_returning_path` opts into path capture and must honor that return type: anything else is treated as a failed write (one warning, no captured path), so `capture_llm_log_paths()` only ever hands you real `Path`s.

An OpenTelemetry exporter (e.g. to Langfuse/Phoenix) is a natural future `llmkit[otel]` extra; the pluggable seam makes it a non-breaking addition.

## Response cache

When an application sends the same request twice, the second one is normally paid for and waited on all over again. Configure a response cache and llmkit answers the repeat from the stored answer instead — no provider call, no rate-limiter slot, no retry budget — and still writes the call's log record, marked as a hit:

```python
from llmkit import InMemoryLLMCache, configure_llm_cache

configure_llm_cache(InMemoryLLMCache(max_entries=1024))   # None (the default) turns it off
configure_llm_cache(my_store, timeout=0.25)                # seconds per store read or write; 1.0 by default
```

The cache is read by `structured_llm_call`, `text_llm_call` and their sync wrappers, and by `text_llm_call_stream` and its sync form; the tool families do not read it. A streamed and a buffered text request for the same prompt are the same request, so either one answers the other. With no cache configured — the default — nothing changes: every request is byte-identical to one sent before this feature existed.

What a configured cache does on each call that reads it:

- **Hit.** When an earlier successful call stored an answer for the same request, the call returns it. The store holds the answer text the provider sent, and a structured hit validates that text once, with the same strict rule as the paid call, into a fresh instance of your schema, so it equals the paid answer. A stored entry that does not decode is a miss, with one warning on the `llmkit.cache` logger. Your `on_result` hook still runs on it; if the hook rejects the stored answer, this call goes to the provider as if nothing were stored, and the new answer replaces the old one. A streamed call replays the stored text as **a single chunk** and then ends — it does not re-create the chunking of the stream that stored it — so code that parses a stream incrementally receives the whole answer at once; to exercise incremental parsing, split the text in your own test double.
- **Identical requests in flight at once.** Buffered requests that start before an identical one has finished share its provider call: the first runs, the rest wait for its answer and record hits. If the first fails or is cancelled, each waiting call goes to the provider on its own retry budget. This works within one process; requests from separate processes each reach the store on their own. Streams take no part: a stream's answer is complete only when your code has read its last chunk, so an identical stream in flight is neither waited on nor waits, each pays for its own answer, and a stream never holds up a buffered call.
- **Miss.** The call runs exactly as it would without a cache. Only an answer that comes back cleanly is stored — never an attempt that raised, an answer your `on_result` rejected, a truncated answer (`OutputLimitError`, structured or text, streamed or buffered), or a stream your code stopped reading before its end. A text answer counts as truncated only when the provider says so (`finish_reason="length"`).

A cache returns the *first* sample for a request. That is the point, but it changes behaviour for a caller that re-sends a prompt to get a different draft. Opt such a call out with `cache=False`, on the call or on `LLMCallOptions`; it then neither reads, waits on, nor writes the cache. A stream has no `on_result` hook, so outside an enclosing `with_retries` loop `cache=False` is how a stream asks for a fresh sample:

```python
draft = await text_llm_call(prompt, feature="drafting", temperature=1.0, cache=False)
```

The re-roll pattern in [Wrapping your own work](#wrapping-your-own-work-with_retries) needs no opt-out: a pass that an enclosing `with_retries` loop re-runs, because your code rejected the previous answer, skips the lookup, asks the provider for a new sample, and stores it in place of the rejected one.

### What counts as the same request

llmkit computes the key, because only llmkit holds the resolved request. `llm_cache_key` is a sha256 over, in a fixed order: a key-format version (`llmkit.cache.LLM_CACHE_KEY_VERSION`, now 4, so entries an earlier release stored miss once and are replaced); the provider's name; the model as routed to LiteLLM (your `model`, or the provider default); the provider's endpoint settings — `api_base`, `extra_body`, `aws_region_name`, `vertex_project` and `vertex_location`, where present; the messages as sent; the output schema's JSON schema, or `"text"`; `temperature` (the value you asked for, even when the model refused it and llmkit omitted it on the wire); `max_tokens`; the effective reasoning effort (yours, or the provider's configured one); and, for a structured call, the provider's structured-output mode and strict-schema setting.

`timeout` is not part of the key because it does not change the answer. An identical buffered request joining one already in flight waits for the rest of the leader's provider pass — its attempts, backoff sleeps and any rate-limiter queue wait — not one round trip, and the follower's own `timeout` does not bound that wait. The leader releases its followers as soon as that pass answers, fails or is cancelled, before it writes the answer to the cache, so what bounds the pass (the leader's per-request `timeout` on each round trip and its retry budget) bounds the follower too; a rate-limiter queue wait is bounded by neither, for the leader or the follower. If the leader failed or was cancelled, the follower then runs on its own budget. A follower deadline sized like one round trip would expire while a healthy leader was retrying or queued for a rate-limiter slot and send a duplicate request, which is what joining exists to prevent, so there is none.

Cache reads and writes have their own budget, the `timeout` passed to `configure_llm_cache`: 1.0 second unless configured, `None` to turn it off. A read that overruns it is cancelled and becomes a miss, so the call goes to the provider; a write that overruns it is cancelled and dropped, and the call returns its answer unchanged. Either is reported like any other store failure, below. Omitting `timeout` on a later `configure_llm_cache` call keeps the configured value, including when you only swap the store or turn caching off. The budget works by cancelling the store's await, so it bounds an asynchronous store that honours cancellation. A store that blocks the event loop synchronously, or that catches the cancellation and carries on, is outside it; bound your own I/O as well, with your client's timeout. A caller can bound an entire async call with its own cancellation scope, but cancellation does not terminate synchronous blocked code.

For schema-backed models the key also includes the original schema snapshot, selected dialect, and format-assertion policy, so contracts with identical provider guidance still have separate cache entries. Caller-written Pydantic models keep their existing key format.

Credentials are never part of the key, so rotating an API key keeps your hits. `feature` and `label` are telemetry, not part of the request, so two features asking the identical question share one answer. Two schemas with the same fields in a different order are different requests, because the order is sent to the model and steers the order it writes fields in. A request that cannot be fingerprinted — a content part that is not JSON-serialisable, a third-party provider missing an attribute the key reads — skips the cache and runs normally.

### What a hit looks like in the log

A hit writes one record like any call, with `cache_hit: true`. It has its own `call_id`, and `source_call_id` names the paid call whose answer it returned. `approximate_cost` is `0.0`, `usage` and `queue_wait_ms` are `null` (no tokens spent, no rate-limiter slot taken), and `duration_ms` is the lookup plus any wait on an identical request in flight — for a streamed hit, also the time your code spends on its one chunk, as every stream record's duration includes the consumer's handling. A streamed hit's `schema` is `stream`, like any stream record. A replay your code abandons before its end logs as any abandoned stream does, `# ERROR` with the abandonment marker, and still carries `cache=hit` and `$0`. The YAML header's second line ends `cache=hit` (the first line keeps its shape), and the `index.jsonl` line carries `"cache_hit": true`, so one scan of the index lists every hit:

```text
# ok | reports/exec_summary | google/gemini-3.1-flash-lite | Summary | 2ms | $0
# 2026-06-05T14:25:02.118204 | call=4be0d913 attempt=1 cache=hit
```

### Write your own store

`InMemoryLLMCache` is a bounded, in-process LRU (256 entries by default): right for a script or a test run, and gone when the process exits. For persistence, sharing between workers, tenant scoping or time-based expiry, supply your own store — any object with two async methods:

```python
import dataclasses
import json

from llmkit import LLMCacheEntry, configure_llm_cache

class RedisLLMCache:
    def __init__(self, client, ttl_seconds: int) -> None:
        self._client = client          # e.g. a redis.asyncio.Redis
        self._ttl = ttl_seconds

    async def get(self, key: str) -> LLMCacheEntry | None:
        raw = await self._client.get(f"llm:{key}")
        return None if raw is None else LLMCacheEntry(**json.loads(raw))

    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        payload = json.dumps(dataclasses.asdict(entry))
        await self._client.set(f"llm:{key}", payload, ex=self._ttl)

configure_llm_cache(RedisLLMCache(redis_client, ttl_seconds=86_400))
```

`LLMCacheEntry` is a frozen dataclass of plain strings — `response` (the answer text the provider sent, for a structured call too), `schema`, `provider`, `model`, and the `call_id` of the call that stored it — so any store can persist it. It has no expiry field: retention is the store's policy, as it is a log sink's.

A store owes llmkit two things beyond the two methods: it bounds its own I/O, and its `get` and `set` honour cancellation. llmkit cancels a read or write that overruns the cache `timeout` (above) and moves on, so a store whose client has no timeout is still cut off, provided the cancellation reaches the await it is stuck in.

`configure_llm_cache` checks what you hand it, as `configure_llm_logging` does: a non-cache, or the class where you meant an instance, raises `TypeError` there and then, and a bare `MagicMock()` does not match. `get_llm_cache()` reads back what is installed, for save-and-restore. An ordinary exception from a cache read or write, or one that overruns the cache `timeout`, does not fail the provider call: a failed read becomes a miss, a failed write is dropped and the call returns its answer, and llmkit logs one warning on the `llmkit.cache` logger saying which, with repeats of the same failure at DEBUG until the cache is reconfigured. An overrun logs as a `TimeoutError` naming the operation and the budget; a `TimeoutError` your store raised itself is logged as yours.

## Configuration

`LLMClientConfig` is flat and carries only what a call needs:

```python
@dataclass(frozen=True)
class LLMClientConfig:
    provider: Provider               # a member: OPENROUTER | OLLAMA | GOOGLE | ANTHROPIC | OPENAI | DEEPSEEK | BEDROCK | VERTEX | CEREBRAS
    model: str | None = None         # None -> the provider's own default model
    api_key: str | None = None       # bearer providers only; masked in repr; else the provider env var, else raises
    base_url: str | None = None      # endpoint: config, else the provider's env var if it reads one, else its default if it has one (sent as api_base; see below); not accepted by Bedrock or Google's Agent Platform
    reasoning_effort: ReasoningEffort | None = None  # "disable" | "lowest" | "low" | "medium" | "high", or any provider value
    aws_region_name: str | None = None   # AWS Bedrock region; not accepted by any other provider
    vertex_project: str | None = None    # Google's Agent Platform GCP project; not accepted by any other provider
    vertex_location: str | None = None   # Google's Agent Platform region (data residency); not accepted by any other provider
    gemini_structured_output: Literal["schema", "json"] = "schema"  # Gemini strategy; a non-default value is not accepted by any other provider
```

When `model` is omitted or empty, these are the built-in defaults (before
LiteLLM route prefixes). Pin `model` explicitly when you need stable selection.
The last column is OpenRouter's id for the same model, from OpenRouter's model
list; *(none)* marks OpenRouter's own row and Ollama's local model:

| Provider | Default model | On OpenRouter |
|---|---|---|
| `OPENROUTER` | `google/gemini-3.1-flash-lite` | *(none)* |
| `GOOGLE` | `gemini-3.1-flash-lite` | `google/gemini-3.1-flash-lite` |
| `VERTEX` | `gemini-3.1-flash-lite` | `google/gemini-3.1-flash-lite` |
| `ANTHROPIC` | `claude-sonnet-4-6` | `anthropic/claude-sonnet-4.6` |
| `OPENAI` | `gpt-4.1-mini` | `openai/gpt-4.1-mini` |
| `DEEPSEEK` | `deepseek-chat` | `deepseek/deepseek-chat` |
| `CEREBRAS` | `gpt-oss-120b` | `openai/gpt-oss-120b` |
| `BEDROCK` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | `anthropic/claude-haiku-4.5` |
| `OLLAMA` | `llama3.2` | *(none)* |

The defaults are declared by `_default_model` in each module under
[`src/llmkit/providers/`](src/llmkit/providers/), and a test checks this table
against them. Bedrock also needs a region, and Google's Agent Platform a project
and location; each comes from the config or, when unset, from the environment.
See [their setup and endpoint caveats](#bedrock-and-vertex-do-not-own-their-endpoints).
To run the same model on another route, see
[Switching provider for a whole program](#switching-provider-for-a-whole-program).

**`provider` takes a member, and a name read from settings is parsed first.**
`Provider(value)` ignores case, so `Provider("Ollama")` is `Provider.OLLAMA`,
and an unknown name raises `ValueError` listing every member. A string passed
to `LLMClientConfig` (or `make_provider`) unparsed raises `TypeError` at
construction, naming `Provider(value)`:

```python
config = LLMClientConfig(provider=Provider(settings["llm_provider"]))
```

**Populating a knob the selected provider does not read is rejected, not
ignored.** Each provider declares which provider-shaped fields it honours, and
`build_provider` / `make_provider` raise a clear `ValueError` if the config
carries any other populated one — an `api_key` for Bedrock or Google's Agent Platform, a `base_url`
for a fixed-endpoint provider, a non-default `gemini_structured_output` for a
non-Gemini provider. Populate only the fields the active provider uses. A config
generically filled from a settings object therefore cannot *look* like it pinned
a credential or endpoint that was in fact dropped.

**`api_key` is masked in the config's `repr`.** A set key renders as
`api_key=<redacted>`, so a stray `print(config)`, log line, or traceback never
leaks the credential; its presence still shows for debugging.

**Bearer-key resolution is explicit.** The key-authenticated providers below
resolve their key from `api_key` if set, else the provider's own environment
variable, else they raise at construction naming the variable — no silent
fallback to an ambient key:

| Provider | Environment variable |
|----------|----------------------|
| `OPENROUTER` | `OPENROUTER_API_KEY` |
| `ANTHROPIC` | `ANTHROPIC_API_KEY` |
| `OPENAI` | `OPENAI_API_KEY` |
| `DEEPSEEK` | `DEEPSEEK_API_KEY` |
| `GOOGLE` (AI Studio) | `GEMINI_API_KEY` |
| `CEREBRAS` | `CEREBRAS_API_KEY` |

`OLLAMA` needs no key; `BEDROCK` and `VERTEX` authenticate through their ambient
AWS / Google credential chains ([see below](#bedrock-and-vertex-do-not-own-their-endpoints)), so none of the three accepts an
`api_key`.

**Endpoint resolution is explicit too.** Every provider that accepts a
`base_url` resolves its endpoint the same way — the configured `base_url` if
set, else the first non-empty environment variable in the order below, else
llmkit's own default — and sends the result on the wire as `api_base`. (`GOOGLE`
is the one exception, deliberately: it owns no default, so with nothing
configured it sends no `api_base` at all — see below.) So the endpoint is
llmkit's decision, readable up front from the provider's
`completion_kwargs()`, rather than something LiteLLM's internal chain resolves
later. The resolution runs **per call**, not when the provider is constructed,
so it reflects the environment the call actually goes out in: `import llmkit`
does not import LiteLLM (that is deferred to the first call) and importing
LiteLLM is what runs `load_dotenv()`, so a provider built once at startup with
`make_provider(...)` and used later would otherwise answer from an environment
the host's `.env` had not been read into yet.

| Provider | Environment variable(s), in precedence order | Default endpoint |
|----------|----------------------------------------------|------------------|
| `OPENROUTER` | *(none)* | `https://openrouter.ai/api/v1` |
| `ANTHROPIC` | `ANTHROPIC_API_BASE`, then `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` |
| `OPENAI` | `OPENAI_BASE_URL`, then `OPENAI_API_BASE` | `https://api.openai.com/v1` |
| `DEEPSEEK` | `DEEPSEEK_API_BASE` | `https://api.deepseek.com/beta` |
| `CEREBRAS` | `CEREBRAS_API_BASE` | `https://api.cerebras.ai/v1` |
| `GOOGLE` (AI Studio) | `GEMINI_API_BASE` | *(none — LiteLLM picks, see below)* |
| `OLLAMA` | *(none)* | `http://localhost:11434` |
| `BEDROCK` | `AWS_BEDROCK_RUNTIME_ENDPOINT` — read by **LiteLLM**, not llmkit | derived from `aws_region_name` by LiteLLM |
| `VERTEX` | `VERTEXAI_API_BASE`, outranked by the `litellm.api_base` global — both read by **LiteLLM**, not llmkit | derived from `vertex_location` by LiteLLM |

The last two rows (`BEDROCK`, `VERTEX`) are governed differently from the rows above them, and the
[section below](#bedrock-and-vertex-do-not-own-their-endpoints) says why and what
it costs.

**`GOOGLE` is the one provider with no llmkit default**, and the absence is
deliberate: its base is the one that cannot be a constant. The AI Studio base
carries an API *version* that LiteLLM derives **from the model** — `v1alpha` for
Gemini 3 and newer, `v1beta` otherwise — and it applies that only when no
`api_base` is given; a base that *is* given is used verbatim (that derivation
is unchanged at litellm 1.104.0, the supported floor). Measured against
litellm 1.92.0 (2026-07-21): pinning
`https://generativelanguage.googleapis.com/v1beta` sent `gemini-3-pro-preview`
to `/v1beta` where it had gone to `/v1alpha`, and pinning the bare host dropped
the version segment entirely — so any static default would silently change the
wire shape for some models, which is the one thing this resolution promises not
to do. Deriving the version inside llmkit would be endpoint *computation*, the
gateway-shaped work the library deliberately does not do. So for Google: a
configured `base_url` wins, else `GEMINI_API_BASE`, else **no `api_base` is sent
at all** and LiteLLM picks the endpoint, and with it the model's API version.
That last case has a real, scoped cost: with neither configured, Google's
endpoint can still come from a source llmkit does not read — notably the
`litellm.api_base` module global. Naming a `base_url` or setting
`GEMINI_API_BASE` closes it.

**llmkit reads the listed variables itself**, in the order shown (LiteLLM's own
measured precedence), so a host can point its endpoint with one of them. The
resolution is *closed* for `OPENAI`, `ANTHROPIC`, `DEEPSEEK`, `CEREBRAS`, and a `GOOGLE`
that has a `base_url` or `GEMINI_API_BASE`: for those, sources llmkit does not
read cannot choose the endpoint — the `litellm.api_base` module global, a
LiteLLM key-management backend serving one of these names, or an alias some
future LiteLLM release adds. That is a closed list rather than a blocklist
chasing an unversioned dependency, and it is the same bargain `api_key`
strikes: make the ambient fallback explicit, documented, and tested rather than
delete it.

Two cases sit **outside** that closure. `GOOGLE` with neither `base_url` nor
`GEMINI_API_BASE` set is one, per the paragraph above. `OLLAMA` is the other,
and structurally so: both of its LiteLLM dispatch arms order the chain
`litellm.api_base or api_base or …` — inverted relative to every other route —
so for Ollama alone the module global outranks even the `api_base`
llmkit sends (measured against litellm 1.92.0, 2026-07-21; the same ordering is
in litellm 1.104.0, the supported floor). That is accepted
rather than fixed: it is a global a host has to set deliberately inside its own
process, not something the ambient environment or a stray `.env` can reach, and
closing it would mean reaching into a dependency's globals.

`OPENROUTER` and `OLLAMA` deliberately read **no** endpoint variable: both
already named their endpoint in every configuration (a default that is a real
URL, overridable via `base_url`), so honouring extra variables would widen the
ambient surface for no gain.

### `BEDROCK` and `VERTEX` do not own their endpoints

These two reject `base_url` outright, and llmkit sends them no `api_base`:
their endpoints are *derived* rather than configured — Bedrock's from
`aws_region_name`, the endpoint for Google's Agent Platform from `vertex_location`. llmkit names an endpoint
where the endpoint is a constant it can measure once; where it is computed from
an input llmkit does not own, llmkit declines to compute it and LiteLLM's own
resolution stays in charge. (Google AI Studio with neither `base_url` nor
`GEMINI_API_BASE` set is the third case in that same family, for the API-version
reason above.) The alternative would be llmkit constructing regional URLs — the
gateway-shaped work this library [deliberately does not do](PRINCIPLES.md), and
for Google's Agent Platform it would mean running Google credential resolution just to build a
path, since the project id comes from ADC.

**The cost is scoped, and it is sharper here than elsewhere, because these are
the two providers whose region knob is a residency control.** An ambient
endpoint value overrides the region you pinned. With the current default and
the `us` multi-region location explicitly pinned in both arms:

```text
BEDROCK, aws_region_name="eu-central-1"
  nothing set   -> https://bedrock-runtime.eu-central-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse
  with the var  -> https://hijacked.invalid/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse

VERTEX, vertex_location="us"
  nothing set   -> https://aiplatform.us.rep.googleapis.com/v1/projects/<p>/locations/us/publishers/google/models/gemini-3.1-flash-lite:generateContent
  with the var  -> https://hijacked.invalid/v1:generateContent
```

Four things that matter if you pin residency:

- **What travels.** Google's Agent Platform sends the ADC-minted `Authorization: Bearer ya29.…`
  to the overridden host; Bedrock hands it a live SigV4 signature over the real
  payload. This is credential exposure as much as region drift. Bedrock's SigV4
  scope keeps following the region you pinned, so redirecting to a *different
  AWS region* fails loudly on signature; redirecting anywhere else succeeds
  silently.
- **Google's Agent Platform has two endpoint sources, Bedrock has one.** For Google's Agent Platform, LiteLLM
  resolves `api_base or litellm.api_base or VERTEXAI_API_BASE` on one line, and
  the in-process global **outranks** the variable — no environment hygiene
  reaches a global that any dependency or notebook cell can set. Bedrock's route
  never consults that global. (`VERTEX_API_BASE`, without the `AI`, is
  embeddings-only and does not affect completions.)
- **Google's Agent Platform replaces the whole URL rather than swapping the host**, composing
  `"{value}:{action}"` — so a value carrying a path or trailing slash redirects
  silently, while a *bare host* fails loudly with `Invalid port:
  'generateContent'` and sends nothing. Do not read a crash as the only failure
  mode.
- **`printenv` is not a valid check.** Importing LiteLLM runs `load_dotenv()`,
  and its search walks up from the installed package directory — so a `.env`
  sitting above your virtualenv is in force from any working directory. Both
  names are read through LiteLLM's `get_secret`, which also consults a
  configured key-management backend. An empty string is safe; a stray value is
  not.

**Google's Agent Platform has a second residency override with no environment variable
involved.** LiteLLM reroutes a pinned `vertex_location` to a model's first
`supported_regions` entry when its shipped cost map lists regions that exclude
yours, warning only on the verbose logger. llmkit's `gemini-3.1-flash-lite`
default requires the `global`, `us`, or `eu` multi-region endpoint; the
maintained configuration uses `us`. The installed cost map is dependency data,
not a guarantee — so check a region-sensitive workload against the endpoint it
actually reaches.

`aws_region_name` is the only AWS-shaped field, and it carries **only** the region. AWS Bedrock authenticates through the standard **AWS credential chain** (environment, shared config, or instance/role), so Bedrock secrets never pass through `LLMClientConfig`; leave the region `None` too and it resolves from the chain (`AWS_REGION_NAME` / `AWS_REGION`). Bedrock routing needs `boto3` for request signing — install it with the opt-in extra:

```bash
pip install "omg-llmkit[bedrock]"
```

The default model is Claude Haiku 4.5 via its **cross-region inference profile** id (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) — current Claude models on Bedrock are typically reached through inference profiles rather than plain on-demand ids. Pass a different profile- or partition-prefixed id as `model` (e.g. `eu.anthropic.claude-...`) when your account routes elsewhere.

`vertex_project` and `vertex_location` are the equivalents for Google's Agent Platform and are not accepted by any other provider. Google's Agent Platform reaches the same Gemini models as the `GOOGLE` (AI Studio) provider, but through Google Cloud — and like Bedrock it carries **no secret** here: Google credentials resolve from **Application Default Credentials** (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or a workload-identity / metadata-server token), never through `LLMClientConfig`. Leave `vertex_project` / `vertex_location` `None` to resolve them from the environment (`VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`), with the location otherwise falling back to LiteLLM's default, `us-central1`, which does not serve the default model. Google's Agent Platform routing needs `google-auth` to mint its OAuth token — install it with the opt-in extra:

```bash
pip install "omg-llmkit[vertex]"
```

**`vertex_location` is the data-processing residency control.** It selects the endpoint where the request is processed, so a regional value (e.g. `vertex_location="europe-west4"`) pins in-region processing; the `"global"` endpoint gives no residency guarantee. LiteLLM builds three endpoint shapes from it, not one: `"global"` yields `https://aiplatform.googleapis.com`, a multi-region geography (`"us"`, `"eu"`) yields `https://aiplatform.<geo>.rep.googleapis.com`, and any other value yields `https://<location>-aiplatform.googleapis.com`. llmkit does not construct that URL and does not send an `api_base` — see [`BEDROCK` and `VERTEX` do not own their endpoints](#bedrock-and-vertex-do-not-own-their-endpoints), which also covers the two ways a pinned location can be overridden. The default model is Gemini 3.1 Flash-Lite (parity with the AI Studio provider) and requires `global`, `us`, or `eu`; the maintained configuration uses `us`. As with AI Studio, reasoning can consume output tokens; choose the model's supported `reasoning_effort` and a sufficient `max_tokens` budget. Gemini 3 and later cannot turn thinking off, so `"disable"` raises there; send `"lowest"`, as described below.

> **A residency region can constrain which model you may use.** Gemini model availability is region-specific, so a region you pick for residency may not host the `gemini-3.1-flash-lite` default; it requires the `global`, `us`, or `eu` multi-region endpoint. A model that isn't deployed in your region fails with an Agent Platform `400 FAILED_PRECONDITION` ("Precondition check failed."), which is an *availability* error, not an auth one. Pin a model your region actually serves. Check the [Gemini on Google's Agent Platform locations table](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) for your region.

**`gemini_structured_output` picks Gemini's structured-output strategy** and is read only by the two Gemini providers (`VERTEX`, `GOOGLE`); a non-default value on any other provider is rejected (the default `"schema"` is always accepted). `"schema"` (the default) keeps Gemini's native JSON-schema constrained decoding (`instructor.Mode.JSON_SCHEMA`) with server-side schema enforcement — the pre-existing wire behavior, unchanged. `"json"` switches to `Mode.JSON`: the response is still server-side guaranteed to be JSON *syntax*, but the schema moves into the system prompt and is validated client-side (with instructor's in-call repair re-asks on a mismatch, `schema_repairs`, one by default). If you observe repeated output running into `max_tokens` with `"schema"`, compare `"json"` on your own model, prompts, and schemas. This is a workload-dependent workaround, not a guarantee that repetition or truncation will stop. The trade is giving up server-side *schema* enforcement for an occasional repair round-trip. An unrecognized value raises at provider construction rather than silently falling back.

```python
configure_llm_client(LLMClientConfig(
    provider=Provider.VERTEX,
    gemini_structured_output="json",   # compare this mode on your workload
))
```

Per-call `model=` overrides the default, so "strong/small/current" model roles are the host's concern — resolve them to a model string and pass it at the call site. The library has no opinion about roles.

`reasoning_effort` controls provider "thinking"/reasoning tokens. Leave it `None` (the default) for the provider's own behaviour — the outbound request is byte-identical to omitting it. The setting belongs to the provider: set it on `LLMClientConfig` (e.g. `"lowest"`) for the registered provider, or pass `make_provider(..., reasoning_effort=...)` for a per-call one, which does not inherit the registered value. A call's `reasoning_effort=` overrides either for that call. Reasoning can consume the output-token budget.

Two portable values are translated per route. **`"disable"`** means thinking off: llmkit either sends the route's off control or raises `ReasoningEffortUnsupportedError` before anything is sent, on a model that cannot stop thinking. It never substitutes a lower effort that still reasons, so switching providers cannot silently change how much a call reasons. **`"lowest"`** means the least reasoning the route supports: off where the model can turn thinking off, else its lowest effort. Use it where you want "as little as possible" on any model. Both raise errors that name the model and say why; a `"disable"` refusal on a model with a lower effort suggests `"lowest"`. Other values pass through unchanged. Per route:

- **Google AI Studio and Vertex (Gemini).** `"disable"` sends `thinkingBudget: 0` on Gemini 2.x and raises on Gemini 3 and later (by LiteLLM's own Gemini-3 test, the one its transform uses) and on `gemini-2.5-pro`, which always think. `"lowest"` sends LiteLLM's lowest thinking level with thought summaries off on Gemini 3 and later, off on Gemini 2.x, and the minimum budget on `gemini-2.5-pro`. On Vertex, partner models (Claude, Llama, Mistral and the others LiteLLM sends away from its Gemini transform) are not covered: the value is forwarded unchanged, with `"lowest"` forwarded as `"disable"`.
- **OpenRouter.** llmkit sends OpenRouter's native `reasoning.effort` object. `"disable"` sends `"none"`, and raises on Gemini 3 and later and on `google/gemini-2.5-pro`; `"lowest"` sends `"minimal"` on those and `"none"` elsewhere. OpenRouter requests always carry `require_parameters` (see [OpenRouter: schema-honoring routing](#openrouter-schema-honoring-routing)), so an effort-carrying request is routed only to endpoints that support reasoning.
- **Anthropic and Bedrock.** `"disable"` sends `"none"`, and raises on models LiteLLM flags as always requiring thinking (such as Sonnet 5.5 and Opus 5.5); `"lowest"` sends `"low"` on those and `"none"` elsewhere. Unknown or unavailable model metadata permits the translation.
- **OpenAI.** `"disable"` omits the reasoning parameter for known non-reasoning models (including the default `gpt-4.1-mini`) and sends `"none"` where model capabilities establish support (for example, `gpt-5.1`). If the model cannot disable reasoning or its capabilities are unknown, it raises. `"lowest"` follows the same capability lookup; where off is unsupported it sends `"minimal"` or `"low"` when the model's metadata flags one (for example, `"minimal"` on `gpt-5`), and it raises everywhere else `"disable"` raises, so nothing is sent blind. For unknown capabilities, choose a provider-native effort value explicitly. Other OpenAI effort values pass through without capability lookup.
- **Cerebras.** `"disable"` sends `"none"`, and raises on gpt-oss models (including the default `gpt-oss-120b`), which cannot turn reasoning off; `"lowest"` sends `"low"` on gpt-oss and `"none"` elsewhere. Reasoning tokens count against `max_tokens` on both served models.
- **DeepSeek.** Both values send `"none"`, which turns thinking off on `deepseek-chat` and `deepseek-reasoner` alike.
- **Ollama.** `"disable"` raises on gpt-oss models, which cannot turn reasoning off; on every other model llmkit sends the server's off control (`think: false`). The server is the authority there, and a model that cannot stop thinking may ignore it: `deepseek-r1:8b` returned reasoning with `think: false` (measured 2026-11-02). `"lowest"` sends `"low"` on gpt-oss and the off control elsewhere.
- **Third-party providers** that do not override `reasoning_kwargs` are not made strict: `"disable"` reaches LiteLLM unchanged, and `"lowest"` is sent as `"disable"`.

Do not assume either value eliminates reasoning tokens or prevents truncation; size `max_tokens` for the model you run.

Register the config with `configure_llm_client(LLMClientConfig(...))`. Calling it again replaces the registration, so a host that changes provider occasionally re-registers. `configure_llm_client(None)` clears the registration, after which a call without `provider=` raises `LLMClientNotConfiguredError`. Every call returns the previous registration (or `None` if there was none), so a test teardown can restore it with `configure_llm_client(previous)`. To switch one call, pass `provider=` on that call; see [Switching provider per call](#switching-provider-per-call).

### Switching provider for a whole program

With `model` unset and the key in the provider's own environment variable, a
program-wide switch is the `provider=` field alone:

```python
configure_llm_client(LLMClientConfig(provider=Provider.ANTHROPIC))  # was Provider.OPENROUTER
```

A pinned `model` is a second edit, because model ids are spelled per route.
Other pinned fields depend on the direction:

- Moving to Bedrock, Google's Agent Platform, or Ollama means removing
  `api_key`, which those three reject; Bedrock and the Agent Platform also
  reject `base_url`. Bedrock adds `aws_region_name` and the
  `bedrock` extra; Google's Agent Platform adds `vertex_project` and
  `vertex_location` and the `vertex` extra. These can also come from the
  environment, as described above, but the Agent Platform's default model needs
  `vertex_location` (or `VERTEXAI_LOCATION`) set to `global`, `us`, or `eu`.
- Moving away from them means removing the fields only they read:
  `aws_region_name`, `vertex_project`, and `vertex_location`.
- A non-default `gemini_structured_output` must go whenever the new route is
  neither `GOOGLE` nor `VERTEX`.
- Any switch between key-authenticated routes means removing or changing a
  pinned `api_key` or `base_url`, and so does a pinned `base_url` on a switch to
  or from Ollama. These routes accept the fields, so a stale key or endpoint is
  not rejected; it goes to the new vendor. Keeping each key in its provider's
  own environment variable avoids this.

**Running the same model on another route.** Use the vendor's own id on the
direct route and the platform's own id on Bedrock and Google's Agent Platform.
OpenRouter ids are `<vendor-slug>/<name>`, and the name sometimes differs from
the vendor's own (Claude versions use dots), so take it from
[OpenRouter's model list](https://openrouter.ai/models). The "On OpenRouter"
column of the [default-model table](#configuration) gives it for each default:
`model="claude-sonnet-4-6"` on `ANTHROPIC` is
`model="anthropic/claude-sonnet-4.6"` on `OPENROUTER`, and Cerebras's
`gpt-oss-120b` sits under its author, `openai/gpt-oss-120b`. Bedrock's
inference-profile ids do not carry to other routes, and Ollama's local models
have no hosted twin.

What llmkit decides changes with the provider, not with the model's name: the
new route brings its own structured-output mode, its own translation of
`reasoning_effort` (`"disable"` raises on a model that cannot stop
thinking, and `"lowest"` maps to a different native value per route, as
described above), and its own rate-limit budget, because budgets are per provider. The
limiter's `rpm`, `tpm` and `max_concurrent` settings, though, are one setting for
all providers, so they stay as you tuned them for the old account. To
confirm a switch before the first call, run `describe_llm(config)` (from
`llmkit.providers`). It builds the provider, so a missing key, a missing extra,
or a rejected field raises there. It reports the provider and model the calls
will use, and the decisions that travel with the switch: the structured-output
mode (`instructor_mode`), whether the schema is sent strictly
(`strict_json_schema`), whether a structured answer must arrive in one tool call
(`single_tool_call_per_schema`), whether the route accepts `tool_choice`
(`supports_tool_choice`), and whether it composes tools with a schema
(`compose_tools_schema`):

```python
from llmkit.providers import LLMClientConfig, Provider, describe_llm

info = describe_llm(LLMClientConfig(provider=Provider.DEEPSEEK))  # key from DEEPSEEK_API_KEY
print(info.provider_name, info.model, info.instructor_mode)  # DeepSeek deepseek-chat Mode.JSON
```

A per-call provider is described the same way, by passing what the call passes
as `provider=`: `describe_llm(provider=Provider.ANTHROPIC)` builds the provider
a call with `provider=Provider.ANTHROPIC` uses, from its environment variables
and default model, and `describe_llm(provider=make_provider(...))` describes
that instance as built. It accepts only llmkit's own provider classes; a
provider you define yourself, a subclass included, raises `TypeError`, and its
decisions are its own `instructor_mode`, `strict_json_schema`,
`single_tool_call_per_schema` and `supports_tool_choice` properties.

The snapshot does not check everything a call will do. It does not resolve the
routed model id, so a doubled prefix such as `"anthropic/claude-sonnet-4-6"` on
`ANTHROPIC` is reported unchanged. It reports decisions made per route, not per
model: Anthropic can still refuse a compose call for a model it cannot compose
with, and the translation of `reasoning_effort="disable"` and `"lowest"` depends
on the model.

### Switching provider per call

A call takes `provider=` the way it takes `model=`: one argument sends that call
to another provider and leaves the registered config alone. A `Provider` member
is built for that call from the provider's environment variables and its own
default model:

```python
from llmkit import Provider, structured_llm_call_sync

result = structured_llm_call_sync(
    prompt,
    output_schema=MyModel,
    feature="summarize",
    provider=Provider.ANTHROPIC,  # key from ANTHROPIC_API_KEY
)
```

When the call needs its own key, model or other knob, build the provider with
`make_provider` and pass the instance instead, with no `LLMClientConfig` and no
module-level config source: `provider=make_provider(Provider.ANTHROPIC,
api_key=key)`. A provider name as a plain string
(`provider="anthropic"`) raises `TypeError` naming the member to pass. A member
whose key or extra is missing raises that error before the first attempt; the
call never falls back to the registered provider.

**Comparing vendors side by side.** Because each call names its own provider,
one program can send the same prompt to several vendors and keep every answer.
No `configure_llm_client(...)` is needed when every call passes `provider=`:

```python
import asyncio

from llmkit import LLMProviderInterface, Provider, make_provider, structured_llm_call

# Each key comes from its provider's own environment variable.
contenders: dict[str, Provider | LLMProviderInterface] = {
    "anthropic": Provider.ANTHROPIC,
    "openai": Provider.OPENAI,
    "gemini": Provider.GOOGLE,
    "claude-via-openrouter": make_provider(
        Provider.OPENROUTER, model="anthropic/claude-sonnet-4.6"
    ),
}

async def compare(prompt: str) -> dict[str, Summary]:
    answers = await asyncio.gather(*(
        structured_llm_call(prompt, Summary, feature="compare", label=name, provider=p)
        for name, p in contenders.items()
    ))
    return dict(zip(contenders, answers, strict=True))
```

Each call keeps its own provider's structured-output mode, reasoning
translation, rate-limit budget, circuit breaker (when enabled), and log
identity, so one vendor's rate limit or open breaker never slows the calls to
another, and the `label` tells the records apart. Budgets are keyed by provider, not by credential: two
`make_provider(Provider.OPENAI, api_key=...)` instances with different keys
share one OpenAI budget. With a [response cache](#response-cache) installed they
also share cached answers, because the cache key leaves credentials out, so pass
`cache=False` when comparing accounts or run-to-run variation.

These providers do not inherit the registered config's `reasoning_effort`; pass
it to `make_provider` or on the call. A member passed per call is built after
llmkit imports LiteLLM, which loads a `.env` file, so it finds a key kept there
on every call. `make_provider` reads each key when it runs, and llmkit sees a
`.env` file only once its first call imports LiteLLM, so a host that builds
providers with `make_provider` before its first call and keeps its keys in
`.env` loads the file first (for example with python-dotenv's `load_dotenv()`).

`make_provider` takes the same knobs as `LLMClientConfig`, and — like
`build_provider` — **rejects** any that the selected provider does not read
rather than silently dropping them: `base_url` for Bedrock or Google's Agent Platform (whose
endpoints derive from region/location), `api_key` for Ollama or Bedrock or Google's Agent Platform
(which authenticate via their ambient AWS/Google chains), `aws_region_name` for
anything but Bedrock, and so on. Pass only the fields the provider uses. Leave
`model` unset to inherit the provider's own default; the assembled LiteLLM id is
always well-formed (e.g. `anthropic/claude-sonnet-4-6`).

**A fully per-call host needs no global config at all.** If you pass `provider=`
on *every* call, you don't have to call `configure_llm_client(...)` — there is no
global source to register, the call runs on the per-call provider alone, and the
log records that provider as the effective one. The "configure once globally" and
"provide per call" models are independent: use either, or mix them (a registered
default for the calls that do not care which vendor answers, and `provider=` on
the calls that do). A call that passes *neither* a
per-call `provider=` nor a registered global config raises
`LLMClientNotConfiguredError` before its first attempt, naming both remedies.
The class subclasses `RuntimeError`.

**A config resolved late or per context.** `configure_llm_client` also accepts
a zero-argument callable that returns an `LLMClientConfig`. llmkit calls it on
every provider construction, so it suits a host whose settings reload at
runtime or whose config depends on context, such as a tenant chosen per
request:

```python
configure_llm_client(lambda: settings.current().llm_config)
```

To build from a full config instead, use `build_provider(config)`:

```python
from llmkit import build_provider, LLMClientConfig, Provider

provider = build_provider(LLMClientConfig(provider=Provider.OPENAI, api_key=key))
```

`LLMClientConfig.model` is optional. When it is `None` (or empty), the provider
falls back to its own built-in default model rather than emitting a broken
`"<prefix>/"` id.

#### Naming: `get_*` reads, `build_*` / `make_*` construct

The accessor verbs are split by intent:

- `build_provider(config)` / `make_provider(...)` **construct** a provider —
  from a config, or from raw credentials.
- `describe_llm(config)` or `describe_llm(provider=...)` (importable from
  `llmkit.providers`) and
  `get_rate_limit_config()` **read** effective state — a snapshot for
  display/telemetry; they construct nothing you keep.

### OpenRouter: schema-honoring routing

OpenRouter is a *router* — it forwards your request to one of several **serving
providers** behind each model. There's a sharp edge for structured output:
`structured_outputs` is a **model-level** capability, but the strict
`response_format` is actually enforced by the *serving* endpoint the request
lands on. A model can advertise the capability while one of its endpoints quietly
ignores the schema and returns free-form JSON — which then surfaces only as a
confusing downstream validation failure, with nothing pointing at routing as the
cause.

`OpenRouterProvider` defends against this **by default**: it sets OpenRouter's
[`provider.require_parameters`](https://openrouter.ai/docs/features/provider-routing#requiring-providers-to-support-all-parameters)
routing preference, so a request only lands on a serving endpoint that honors
*every* parameter sent — including the structured `response_format` and native
`reasoning` control when configured. The trade-off
is that restricting routing to capable endpoints can in principle reduce
availability or shift cost. There is no switch to turn it off: whether your
schema is enforced is a correctness decision, and an endpoint that ignores it
fails silently and far from the cause. A host that knowingly wants wider routing
can subclass `OpenRouterProvider` and override `completion_kwargs`. A request
that no endpoint can serve because of a parameter fails with a 404, "No
endpoints found that can handle the requested parameters"; when the request
carried a `temperature`, llmkit re-issues it once without one, as for any
[refused temperature](#provider-default-sampling-temperaturenone).

**The strict upgrade is skipped for schemas OpenAI-served endpoints reject.**
OpenRouter's structured outputs are advisory unless the schema is sent with
`strict: true`, so llmkit upgrades a structured request to strict where the
schema allows it. It leaves the request non-strict when any nested object is
open or any object leaves a property out of `required` (a field with a
default). Measured on 2026-11-02, OpenAI-served endpoints answered HTTP 400 to
both shapes under strict and validated them sent non-strict. llmkit validates
every answer locally either way.

## Retries

Two retry layers, kept deliberately separate: llmkit's own [automatic
retries](#automatic-retries-on-by-default) across attempts, and
[instructor's in-call schema repair](#instructors-in-call-schema-repair) within
one attempt.

### Automatic retries, on by default

Every call function (`structured_llm_call`, `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, `text_llm_call_stream`, `text_llm_call_stream_sync`, `tool_llm_call`, `tool_llm_call_sync`, `tool_llm_call_stream`, `tool_llm_call_stream_sync`) retries transient provider errors on its own. **`RetryPolicy.max_attempts` is the total ceiling on outer attempts, including the first: three by default, and `1` means no outer retry.** Every attempt counts, whatever failed. Validation failures have an additional, smaller allowance inside that ceiling:

- **Transport errors** (`LLM_TRANSPORT_ERRORS`: curated rate-limit, transient server, network and timeout classes) may retry while the overall ceiling permits. A timed-out request gets a fresh per-request `timeout` on the next attempt. This is class-based classification, not a promise to retry every HTTP 5xx error.
- **Schema-validation errors** (`LLM_SCHEMA_ERRORS`: llmkit's `StructuredOutputError`, pydantic `ValidationError`, instructor `InstructorRetryException`) may cause at most `validation_max_attempts - 1` retries, **one by default**. Those retries still fit inside `max_attempts`; a transport error between validation failures does not reset the allowance. The retry layer unwraps instructor errors before classifying them: a wrapped transport error does not consume the validation allowance, and a wrapped permanent error such as 401/400/403 fails immediately.
- **Tool-argument errors** (`LLM_TOOL_ERRORS`: `ToolArgumentError`, raised for a tool round in which every requested call is malformed) use the same validation-retry allowance within the overall ceiling in `tool_llm_call` and `tool_llm_call_sync`, with or without `output_schema=`, and in `tool_llm_call_stream` and its sync form while nothing has been yielded — see the streaming caveat below. Re-asks attach the rejected calls and validation errors when every call has an id and tool name. The allowance applies whatever `validation_retry_on` lists; opt out with `validation_max_attempts=1`, `max_attempts=1` or `NO_RETRY`.
- **Output-limit truncations** (`LLM_OUTPUT_LIMIT_ERRORS`: llmkit's own `OutputLimitError`, raised when a structured or text completion, or a compose call's final answer, is cut off by the output-token limit, `finish_reason='length'`) get **zero budget**: a re-ask with an identical token budget can only truncate again (the motivating production failure was a degenerate repetition loop that burned to the provider's 65k-token ceiling on the original ask *and* on every blind re-ask, turning a seconds-long call into minutes of doomed generation). The error carries `model` / `max_tokens` / `completion_tokens`, plus `diagnostics` — the partial answer text, finish reason and token counts, also written to the call's YAML log (see [Logging](#logging-agent-readable-by-default)). A truncation does not say *why* it happened: the answer may legitimately need more room, the model may be repeating itself, reasoning may have consumed the budget (`completion_tokens` can include reasoning tokens), or the prompt may lack the evidence to finish. Read the partial text before raising the cap or retrying. Fail-fast is the default; an explicit `retry_on` listing the type opts back into retries — except that a text stream which has already yielded text is never replayed (see the streaming caveat below).
- **Backpressure** (`LLM_BACKPRESSURE_ERRORS`: llmkit's own `CircuitOpenError`, raised by the opt-in [circuit breaker](#circuit-breaker)) also gets **zero budget**: re-asking a circuit already known to be open is doomed by construction.

The two zero-budget subsets are fail-fast *defaults*, not prohibitions: a caller that genuinely wants the retry opts back in by listing `OutputLimitError` or `CircuitOpenError` explicitly in a `retry_on`. Programming errors (e.g. `TypeError`) are outside the recoverable set and propagate immediately, never retried. Each attempt is its own logged call, so `data/llm-logs/` shows one record per attempt.

**An attempt is not a provider-request or spending cap.** Instructor may issue
`schema_repairs` schema-repair requests within a structured attempt (one by
default); a refused temperature can cause one re-issue; the provider SDK can
also retry internally. For example, with the defaults persistent malformed
JSON can make two outer attempts with two instructor requests each, four
requests in total; `schema_repairs=0` makes it two. The outer ceiling does not bound money
or elapsed time: rate-limit waits, retry sleeps, logging and cache waits are
separate, and each provider round trip receives its own `timeout`.

**`LlmkitError` is the base of every exception llmkit defines**, from `TransientProviderError`, `ServiceUnavailableError` and `CircuitOpenError` to configuration errors such as `LLMClientNotConfiguredError`; `except LlmkitError` catches anything llmkit itself raises, including `StructuredOutputError` and `UnsupportedSchemaError`; an `on_result` hook that returns a value raises a plain `TypeError`. It does not catch third-party transport errors or instructor's `InstructorRetryException` around one, and it must not go in a `retry_on`, where it would re-enable retries for the zero-budget subclasses.

**`LLM_RECOVERABLE_ERRORS` is the single catch-set for transient failures** — the union of all five subsets. Keep using it in `except` clauses; the split only changes how the *retry layer* budgets them. One note on the 503 case: so that `import llmkit` doesn't pay LiteLLM's multi-second import cost, litellm's own 503 class (`litellm.exceptions.ServiceUnavailableError`) is never imported eagerly — instead, llmkit re-raises every litellm-native 503 at its transport boundary as llmkit's own **`ServiceUnavailableError`**, a plain member of `LLM_TRANSPORT_ERRORS`, so `except LLM_RECOVERABLE_ERRORS:` genuinely catches it. It carries `provider`, `model`, `status_code` (always 503), and the original `response` (so a server `Retry-After` stays honoured), with the original litellm error on `__cause__`. Around llmkit calls, catch `llmkit.ServiceUnavailableError` (or the documented tuple), not litellm's class; the raw litellm class still *classifies* as transport in `isinstance` checks — e.g. your own litellm call wrapped in `with_retries` — via a lazy stand-in resolved once litellm is loaded.

**Mapped 500, 502 and 529 gaps become `TransientProviderError`.** At the same
provider boundaries, llmkit normalizes an `openai.APIError` (including LiteLLM's
subclasses) carrying one of these integer statuses when it is not already in
the transport set. The new type belongs to both `LLM_TRANSPORT_ERRORS` and
`LLM_RECOVERABLE_ERRORS`. Already recognized transport errors retain their
types, and native 503s keep the `ServiceUnavailableError` behavior above.
Permanent 400/401/403 errors, other uncovered statuses such as 520, and
programming errors are unchanged; this does not make every HTTP 5xx retryable.

`TransientProviderError` carries the available mapped `status_code`, `provider`,
`model`, original `response` and numeric `retry_after`, with the original error
on `__cause__`. A numeric directive still takes precedence over response
headers. These are the details LiteLLM supplied, not reconstructed wire
evidence: its mapping may already have changed a status or discarded headers.
Structured calls retain instructor's outer `InstructorRetryException`; llmkit
unwraps it for classification and retry timing. Streams normalize both opening
and iteration failures, subject to the no-replay rule below.

### Backoff and `Retry-After`

All eligible retries use bounded **full-jitter** backoff: the sleep before retry *n* is a random delay in `[0, min(backoff_base_seconds * 2**(n-1), max_backoff_seconds)]`, with the per-sleep cap (`max_backoff_seconds`) defaulting to 30s. The retry number increases across transport and validation failures; switching failure classes never resets the backoff. The ceiling stays bounded however large the retry number grows.

`RetryPolicy` validates its budgets at construction and raises `ValueError` outside their domains: `max_attempts` and `validation_max_attempts` must be positive integers (not a bool, and not a float such as `3.0`); `backoff_base_seconds` must be a finite number of zero or more (`0` disables the computed backoff); `max_backoff_seconds` and `retry_after_cap` must be finite and positive. NaN, infinity and bools are refused for all five.

**`Retry-After` is honoured first.** When a retried provider error carries a `Retry-After` (a header — delta-seconds, `retry-after-ms`, or an HTTP-date — or the SDK's numeric attribute), the backoff waits *that* duration instead of the computed exponential, capped at `RetryPolicy.retry_after_cap` (default 60s) so a hostile value can't wedge a call, plus a random jitter of up to `backoff_base_seconds` so a fan-out doesn't retry in lockstep. The sleep can therefore exceed the cap by up to `backoff_base_seconds`; with a base of 0 it is exactly the capped server value. It is read from the *unwrapped* provider error, so a structured call honours it too, and it is honoured even when `backoff_base_seconds` is 0 (a server directive, not opt-in backoff); absent a header, the exponential is used unchanged. A value that parses to NaN or infinity is skipped like an unparseable one: the next source is read (the attribute, then `retry-after-ms`, then `retry-after`), and with none left the computed exponential runs; it is never clamped to the cap.

### Tuning or opting out per call

Pass `retry=` to any call function:

```python
from llmkit import structured_llm_call, RetryPolicy, NO_RETRY

# Opt this one call out of automatic retries (e.g. latency-sensitive):
result = await structured_llm_call(
    prompt="Summarize the attached report.",
    output_schema=Summary,
    feature="reports",
    label="exec_summary",
    retry=NO_RETRY,
)

# Or tune the budget / backoff for this call:
result = await structured_llm_call(
    prompt="Summarize the attached report.",
    output_schema=Summary,
    feature="reports",
    label="exec_summary",
    retry=RetryPolicy(max_attempts=5, backoff_base_seconds=1.0),
)
```

**Custom-policy compatibility:** `retry_on` matches the normalized type only;
it does not follow `TransientProviderError.__cause__` to match the original
provider type. A policy naming only `litellm.BadGatewayError`, `litellm.APIError`
or `openai.APIError` must include `TransientProviderError` to retry newly
normalized failures. For example, `RetryPolicy(retry_on=(TransientProviderError,))`
selects these transport failures; import the class from `llmkit`. An empty tuple
or an unrelated type excludes them. The default transport policy already
includes the new class.

**Streaming caveat:** `text_llm_call_stream`, `tool_llm_call_stream` and their sync forms can only retry a transient failure that happens *before the first item reaches the caller*. Once anything has been yielded, a mid-stream error propagates unretried — a partially-consumed stream can't be safely restarted. The tool stream re-asks an all-malformed round within the validation allowance and overall ceiling only when that round yielded nothing; no policy permits replay after an item was yielded. See [Streaming a tool turn](#streaming-a-tool-turn).

### Wrapping your own work: `with_retries()`

**`with_retries()`** (imported from `llmkit.retry`; see [`retry.py`](src/llmkit/retry.py)) wraps an awaitable factory representing your own unit of work. It accepts the same `RetryPolicy` and defaults as the call functions: an overall ceiling of three attempts, at most one validation-caused retry, and immediate propagation of permanent or programming errors. Pass `policy=NO_RETRY` for one pass, or supply a custom policy; the former scalar retry keywords are no longer accepted.

Normalization belongs to llmkit's provider boundaries. An independent raw
LiteLLM call inside `with_retries` does not pass through them: it retains its
raw exception types and receives only the wrapper's class-based retry policy.

```python
from llmkit.retry import with_retries
from llmkit import RetryPolicy

result = await with_retries(
    lambda: do_some_work(),
    policy=RetryPolicy(max_attempts=5, backoff_base_seconds=1.0),
)
```

Wrap a `retry_progress_callback(...)` scope around the work to observe failures before another attempt. Its `attempt` and `max_attempts` describe the overall outer count, not a separate failure-class counter.

A `RetryProgressCallback` is invoked once per non-final failed attempt with keyword arguments `label`, `attempt`, `max_attempts`, and `error`:

```python
def on_retry(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
    print(f"{label}: attempt {attempt}/{max_attempts} failed: {error}")
```

> **Don't double-wrap the call functions.** They already retry internally, so `with_retries(structured_llm_call, ...)` would otherwise multiply the budgets (the `3 × 3 = 9` trap). `with_retries` guards against this — it detects an active llmkit retry loop **owned by the current `asyncio` task** and collapses the inner layer to a single pass (warning once), so the budgets don't multiply. The task scoping bounds the guard to the pattern it warns about: a nested loop awaited inline in the same task, whose failure really does propagate out to the outer loop. A *distinct* llmkit call that merely inherited the scope across a task boundary keeps its full retry budget — an `on_result` hook calling `structured_llm_call_sync` / `text_llm_call_sync` (the sync bridge drives the coroutine as a new task on llmkit's persistent loop), or a call spawned with `asyncio.create_task` from inside an attempt. To drive retries entirely from your own wrapper instead, opt the inner call out with `retry=NO_RETRY`. Either way the call joins the loop's logical call: each pass's record carries the same `call_id` and the next `attempt`. With a [response cache](#response-cache) configured, every pass after the first skips the cache lookup, so a pass your code re-runs because it rejected the previous answer gets a new sample rather than the stored one.

### instructor's in-call schema repair

instructor re-asks the model to fix malformed JSON *within a single call*, before any `StructuredOutputError` reaches the retry layer. **`schema_repairs` sets how many re-asks one outer attempt may make: one by default (`DEFAULT_SCHEMA_REPAIRS`), so at most two sends per attempt.** Pass it to `structured_llm_call` or `structured_llm_call_sync`, or set it on `LLMCallOptions`; it counts re-asks after the first send, so `schema_repairs=0` sends once per attempt. Zero does not disable validation: a malformed answer still raises `StructuredOutputError` and draws on the policy's validation allowance. Anything but a non-negative `int` (a `bool`, a float such as `1.0`, a negative value) raises `ValueError` before any request, the rule `RetryPolicy` applies to its attempt counts. The re-ask fires for a **genuine parse failure only**: malformed JSON, a Pydantic `ValidationError`, or instructor's own `ResponseParsingError` (e.g. a blocked Gemini `Mode.JSON` response). instructor's async validators (`async_field_validator`, `async_model_validator`) are not supported: no lane runs them, so llmkit refuses a schema that declares one with `UnsupportedSchemaError` before any request (llmkit requires instructor 1.17.0 or later). A response with no choices gets no in-call re-ask; it raises `StructuredOutputError` and is re-asked on the validation budget. Every other failure is declined by instructor's in-call loop, so instructor adds no repair request: a transport failure (429/5xx/network) leaves the rate-limiter slot immediately and is retried by the automatic layer, with backoff and `Retry-After` honoured, rather than being re-sent inside the same slot with neither; a permanent failure (401/400/403) fails fast with no in-call duplicate; and a completion **truncated by the output-token limit** is never re-asked (the re-ask would run on the identical budget and can only truncate again) and surfaces immediately as `OutputLimitError`. This stays **separate** from the automatic layer: instructor repairs within one outer attempt. The policy permits at most `validation_max_attempts - 1` validation-caused outer retries (one by default), always within `max_attempts`. Refused-temperature re-issues and SDK retries can add requests independently of instructor's repair. The tool-calling lane, including `tool_llm_call` with `output_schema=`, has no in-call repair and ignores `schema_repairs`.

Four limits apply to a larger allowance. The rate limiter takes one slot and one requests-per-minute token per outer attempt, whatever the number of repair sends, so a high allowance can exceed a provider's request rate limit; token usage is debited as the attempt's total. The call record's `usage` totals every send in the attempt, but its `approximate_cost` is the last send's cost only, so a repaired attempt's cost is under-reported; this already applies at the default of one. The response cache does not key on `schema_repairs`, so a cached answer, or one shared by an identical request already in flight, is returned whatever the allowance. The live provider suite exercises one re-ask; allowances above one are verified offline only.

### Re-rolling on a semantically-bad result

A response can pass the schema and still be *wrong* — an empty register, a citation that doesn't resolve, a total that doesn't reconcile. Rather than hand-rolling an LLM-then-validate-then-re-roll loop around the call, pass an `on_result` hook: it's called with each attempt's result, and raising `ResultValidationError` from it **rejects** that result and re-rolls the call.

```python
from llmkit import structured_llm_call, ResultValidationError

def _must_have_findings(report: Report) -> None:
    if not report.findings:
        raise ResultValidationError("empty report — re-roll")

result = await structured_llm_call(
    prompt, Report, feature="reports", on_result=_must_have_findings,
)
```

The re-roll uses the **validation allowance** (`RetryPolicy.validation_max_attempts - 1`, one by default) inside the overall `max_attempts` ceiling — the same allowance a schema failure uses, so a deterministically-bad result cannot spend the full ceiling on doomed re-asks. When the budget is exhausted the last `ResultValidationError` propagates. Each attempt — including a rejected one — is its own logged call, so `data/llm-logs/` shows the rejected response alongside the error.

`on_result` is available on `structured_llm_call` and `text_llm_call`, and on both sync wrappers (`structured_llm_call_sync`, `text_llm_call_sync`); the text-path hooks receive the response *text*. It is *not* part of `LLMCallOptions` — like `feature`, it stays a conscious per-call choice.

The hook may be an `async def` function; llmkit awaits it. It must return `None`: any other return value raises `TypeError`, because a hook written as a predicate (`lambda r: bool(r.findings)`) would otherwise accept every result. The parameter's type says so, so a type checker flags such a hook. On a cache hit, an exception the call would re-roll on (`ResultValidationError`, or one in the policy's `validation_retry_on` such as a pydantic `ValidationError`) makes the hit a miss; any other exception propagates without a provider request.

The hook runs inside the call, on the event loop the call runs on, so keep it quick and non-blocking. A hook that blocks stalls that whole loop. For the sync wrappers that loop is llmkit's own, which every concurrent sync call shares, so a blocking hook in one sync call delays all the others, not only its own. A hook running on llmkit's own loop cannot iterate a sync stream form: there it raises `RuntimeError` instead of deadlocking.

## Testing your app with a fake transport

`llmkit.testing` lets your own tests answer llmkit's calls without a provider. Install a fake, and every call made while it is in place reaches the fake instead of the network. Everything around the provider round trip still runs: option merging, the response cache, the rate limiter and circuit breaker, retries, schema parsing and in-call repair, tool-argument parsing, and the call log.

```python
from llmkit import LLMClientConfig, Provider, configure_llm_client
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install

# A provider must still be configured; Ollama needs no key. The call returns
# the previous registration; configure_llm_client(previous) restores it, and
# configure_llm_client(None) clears the registration altogether.
previous = configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA, model="test"))

def fake(request: TransportRequest) -> Reply:
    if request.feature == "health":
        return Reply.text("ok")
    if request.kind == "structured":
        return Reply.json(Summary(title="t", bullets=["b"]))
    if request.kind in ("tool", "tool_stream"):
        return Reply(tool_calls=[FakeToolCall("search", {"query": "llmkit"})])
    return Reply.text("hello there")

with install(fake) as transport:        # `async with` works too
    run_the_app()
print(transport.requests)               # every request the fake answered
configure_llm_client(previous)          # put back whatever was registered before
```

The fake is a plain function or a coroutine function. It receives one frozen `TransportRequest` per provider round trip, carrying:

- **identity:** `feature`, `label`, `kind` (`"structured"`, `"text"`, `"tool"`, `"text_stream"` or `"tool_stream"`), and the `call_id` and `attempt` the call log records, so a retried call reaches the fake once per attempt;
- **inputs:** `prompt` as you passed it, `messages` as sent (structured output may add schema instructions or failed answers; tool repair adds the rejected round and its errors), `output_schema`, `tools` and `tool_choice`, and `repair`, which marks a structured call's in-call schema repair or a tool re-ask, buffered or streamed, carrying repair history;
- **effective settings:** the routed `model`, `provider`, `temperature`, `max_tokens`, `reasoning_effort` and `timeout`.

It never carries credentials or provider-native request settings.

A `Reply` answers every kind. `Reply.text(...)` gives text, and `Reply.json(model_instance)` gives the instance's JSON (`model_dump_json(by_alias=True)`). That is a re-serialization, so a model whose dump cannot be read back (computed fields with `extra="forbid"`, a serialization-only alias, an excluded required field) or whose validator transforms its input does not give back the same answer; send the JSON the model would send with `Reply.text`. `Reply(no_choice=True)` answers a buffered call with a response that has no choices, which a structured call, and a tool call with `output_schema=`, raises as `StructuredOutputError`; a text call and a plain tool call treat it as an empty answer. Raw JSON and raw tool arguments (`FakeToolCall("add", '{"left": 1')`) run the real parsing, repair and malformed-call handling. A streaming call splits the text into word-sized deltas, or uses `chunks=[...]` when you give them, then delivers the tool calls; `usage` and `cost` are reported like a provider's. `finish_reason` defaults to `"tool_calls"` or `"stop"`. Set it to `"length"` to get each kind's real truncation behavior: a structured call, a text call and a text stream raise `OutputLimitError` (the stream after delivering its chunks), as does a compose call's final answer, and any other tool call reports it as `stop_reason`.

The installed fake is **process-wide**: it reaches the sync wrappers and threads your application starts itself, such as a web server under test, and concurrent tests in one process share it. Nested installs stack, and leaving a block restores the one before. With no fake installed, the real transport runs.

### Simulating failures

The failure helpers build the exceptions providers really raise, so llmkit's retry, back-off, adaptive concurrency and circuit breaker treat them exactly like a real provider's, and so does your own `except LLM_RECOVERABLE_ERRORS`:

| Helper | Raises | Retried | Throttle signal |
|---|---|---|---|
| `rate_limit_error(retry_after=None)` | `openai.RateLimitError` (429, optional `Retry-After`) | yes, after `Retry-After` when given | yes |
| `service_unavailable_error(retry_after=None)` | `llmkit.exceptions.ServiceUnavailableError` (503) | yes | yes |
| `server_error()` | `openai.InternalServerError` (500) | yes | no |
| `timeout_error()` | `openai.APITimeoutError` | yes | no |

Raise one from the fake. `Reply(..., delay=seconds)` answers late, and the wait is spent inside the rate-limiter slot, as a slow provider's would be. The call's [`timeout`](#per-request-timeout) applies to the fake as to a provider: a delay longer than it is cut off with a `TimeoutError` and retried like a stalled request, so a host's timeout handling can be tested offline. On a stream the delay comes before the first chunk. Fault scripts compose from the request: refuse the first N attempts of one feature with `request.attempt`, or keep refusing one feature until its retry budget is spent. Tell a readiness probe from real work by its `feature` or `label`.

### What the fake does not see

- **Cache hits and shared in-flight requests.** The [response cache](#response-cache) sits above the fake. A hit never reaches it, and an identical buffered request that joins one already in flight shares that one round trip. Configure no cache, or pass `cache=False`, when a test counts requests.
- **Calls with no configured provider.** The provider is resolved before the round trip, and a key-based provider raises without its key. Configure one that needs none, such as Ollama, or give the provider a placeholder key. A `Provider` member passed as `provider=` is built from the environment, so a key-based member needs its variable set under test (`monkeypatch.setenv("OPENAI_API_KEY", "placeholder")`); `provider=Provider.OLLAMA` needs nothing.
- **Mid-stream failures.** A fake can fail a streaming call before its first chunk, not partway through.

Patching llmkit's private modules (`llmkit._litellm` and its functions) is unsupported: those are internal test seams and change without notice. `llmkit.testing` is the supported way.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run basedpyright          # recommended tier; clean with no baseline
uv run pytest
```

These four are offline checks CI runs on pushes to `main` and on pull requests; plain
`uv run pytest` needs no network and no credentials. A change to provider
behaviour (a structured-output mode pin, a default model, an adapter) also needs
the live provider suite, `uv run pytest tests/integration --run-live`, which
makes real calls and needs every provider's credentials. The full requirements
are in [`docs/work/definition-of-done.md`](docs/work/definition-of-done.md), and
the contribution workflow is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Status & support

`llmkit` is a small, opinionated, **best-effort** project, extracted from a real
application and maintained in the open. It is used in production by its author
but carries no support SLA. Bug reports and focused pull requests are welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md). For security issues, see
[SECURITY.md](SECURITY.md).

Releases follow [Semantic Versioning](https://semver.org/). While the version is
below 1.0, a minor release can carry breaking changes; every one is called out
under **Breaking** in [CHANGELOG.md](CHANGELOG.md), along with migration notes
for renamed or removed APIs. This README describes the code on `main`, so it can
run ahead of the latest release on PyPI — check the changelog's `[Unreleased]`
section for what has not shipped yet.

## License

MIT — see [LICENSE](LICENSE).
