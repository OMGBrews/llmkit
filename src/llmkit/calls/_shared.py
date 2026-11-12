"""What every call family does once, in one place.

The five families (structured, text, streamed text, tool, streamed tool) differ
in what one *attempt* is; they agree on everything around it. That agreement
lives here so the next family added does not become another copy:

* :func:`prepare_call` — the per-call resolution every family does before its
  first attempt: merge the options, build the provider exactly once, and mint
  the correlation id — or, inside a :func:`~llmkit.retry.with_retries` loop,
  join the loop's logical call;
* :func:`run_with_policy` — the :func:`~llmkit.retry.with_retries` invocation,
  which differs between families only in which errors are charged to the
  validation budget;
* :func:`run_cached` — the response-cache read-through around that pass, for
  the two buffered families that read the cache, and :func:`cache_and_key` —
  whether a call reads the cache at all, which the streamed text family shares
  with them;
* :func:`result_validation_budget` / :func:`tool_validation_budget` /
  :func:`compose_validation_budget` — the three augmented validation sets;
* :func:`build_text_record` and :func:`resolve_model_and_provider` — record
  construction shared by the text and streaming surfaces, and the effective
  model/provider resolution every family records;
* :func:`error_text` — the ``error`` string every family records for a failed
  attempt;
* :func:`parse_tool_calls` — narrowing a round's raw provider tool calls,
  shared by the buffered and streaming tool surfaces so the salvage contract
  (well-formed calls survive a partly malformed round) is written once.

The **queue-wait protocol** is a cross-module contract worth stating where the
scaffold lives: every attempt calls
:func:`~llmkit.rate_limiting.begin_queue_wait` before the transport runs and
reads :func:`~llmkit.rate_limiting.current_queue_wait_ms` when it builds the
record. Skipping the reset does not fail loudly — it silently attributes a
previous attempt's queueing to an attempt that never queued.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import NamedTuple, cast, get_args, get_type_hints, is_typeddict

from pydantic import BaseModel

from llmkit._diagnostics import report
from llmkit._types import AssistantToolMessage, ChatMessage, ReasoningEffort
from llmkit.cache import LLMCache, LLMCacheEntry, get_llm_cache
from llmkit.cache.read_through import CacheDecodeError, cache_key_or_none, read_through
from llmkit.exceptions import (
    OutputLimitDiagnostics,
    ResultValidationError,
    SchemaEngineError,
    StructuredOutputError,
    ToolArgumentError,
    UnsupportedSchemaError,
)
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, ResolvedCallArgs, Unset, resolve_call_args
from llmkit.providers import LLMProviderInterface, Provider, coerce_call_provider
from llmkit.providers.base import active_config, client_configured
from llmkit.rate_limiting import current_queue_wait_ms
from llmkit.retry import RetryPolicy, claim_logical_call, with_retries
from llmkit.run_scope import get_run_id
from llmkit.tools import TokenUsage, ToolCall, ToolDefinition, tool_result_message

# One logger for the whole call surface, named explicitly rather than via
# ``__name__`` so every family module shares a single greppable name.
logger = logging.getLogger("llmkit.calls")


def guard_output_schema(output_schema: object) -> None:
    """Refuse an ``output_schema`` no lane can hold to the contract, before any request.

    The schema must be a pydantic ``BaseModel`` subclass, not one of
    instructor's DSL classes (``IterableModel``, ``Partial``, a simple-type
    adapter), that declares no instructor async validator (``async_field_validator`` /
    ``async_model_validator``), on itself, a base, or any model reachable
    through its field annotations (including through a ``TypedDict`` or a
    dataclass): no lane runs one, so a declared validator
    would silently not validate. Raises
    :class:`~llmkit.exceptions.UnsupportedSchemaError` otherwise.
    """
    if not isinstance(output_schema, type) or not issubclass(output_schema, BaseModel):
        raise UnsupportedSchemaError(
            f"output_schema must be a pydantic BaseModel subclass, got {output_schema!r}"
        )
    from instructor.dsl.iterable import IterableBase
    from instructor.dsl.partial import PartialBase
    from instructor.dsl.simple_type import AdapterBase

    if issubclass(output_schema, (IterableBase, PartialBase, AdapterBase)):
        # instructor turns these into something other than an instance of the
        # schema (a list, a bare value) or validates an incomplete answer.
        raise UnsupportedSchemaError(
            f"output_schema {output_schema.__name__} is an instructor DSL class "
            + "(IterableModel, Partial or a simple-type adapter); pass a plain model"
        )
    offender = _async_validator_member(output_schema)
    if offender is not None:
        raise UnsupportedSchemaError(
            f"{offender}: instructor async validators are not supported; no llmkit lane "
            + "runs them. Use a pydantic validator, or validate in on_result."
        )


@functools.lru_cache(maxsize=256)
def _async_validator_member(model: type[BaseModel]) -> str | None:
    """``"<Model>.<member>"`` for the first async-validator marker reachable from *model*."""
    from instructor.validation import ASYNC_MODEL_VALIDATOR_KEY, ASYNC_VALIDATOR_KEY

    seen: set[int] = set()

    def visit(annotation: object) -> str | None:
        if id(annotation) in seen:
            return None
        seen.add(id(annotation))
        for argument in cast("tuple[object, ...]", get_args(annotation)):
            found = visit(argument)
            if found is not None:
                return found
        if not isinstance(annotation, type):
            return None
        container = cast("type[object]", annotation)
        if not issubclass(container, BaseModel):
            if not (is_typeddict(container) or dataclasses.is_dataclass(container)):
                return None
            # A model can sit inside a TypedDict or a dataclass field too.
            try:
                hints = cast("dict[str, object]", get_type_hints(container))
            except Exception:
                hints = cast("dict[str, object]", getattr(container, "__annotations__", {}))
            for hint in hints.values():
                found = visit(hint)
                if found is not None:
                    return found
            return None
        model = container
        members: dict[str, object] = {}
        for base in reversed(model.__mro__):
            members.update(vars(base))
        for name, member in members.items():
            if isinstance(member, (classmethod, staticmethod)):
                member = cast("object", member.__func__)
            if (
                getattr(member, ASYNC_VALIDATOR_KEY, None) is not None
                or getattr(member, ASYNC_MODEL_VALIDATOR_KEY, None) is not None
            ):
                return f"{model.__name__}.{name}"
        for field in model.model_fields.values():
            found = visit(field.annotation)
            if found is not None:
                return found
        return None

    return visit(model)


async def run_on_result[T](on_result: Callable[[T], object], result: T) -> None:
    """Run the caller's ``on_result`` hook on *result*, awaiting an async one.

    A hook rejects a result by raising
    :class:`~llmkit.exceptions.ResultValidationError`; its return value is
    never read. So an awaitable return is awaited (an ``async def`` hook), and
    any other value that is not ``None`` raises :class:`TypeError` rather than
    being ignored: a hook written as a predicate (``lambda r: bool(r.items)``)
    would otherwise accept every result silently.
    """
    returned = on_result(result)
    if inspect.isawaitable(returned):
        returned = cast("object", await returned)
    if returned is not None:
        raise TypeError(
            "on_result must return None (or be an async function returning None); "
            + f"it returned {type(returned).__name__}. Raise ResultValidationError to "
            + "reject a result."
        )


def result_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """The validation retry-set augmented with :class:`ResultValidationError`.

    The ``on_result`` re-roll hook charges a rejected-result against the same
    budget as a schema-validation failure (semantically the content is wrong,
    not the transport), so :class:`ResultValidationError` is folded into the
    policy's ``validation_retry_on`` for the call's :func:`with_retries` pass.
    Including it unconditionally is harmless when no ``on_result`` is supplied —
    nothing raises it — and keeps the call functions from branching on the hook.
    """
    return (*retry.validation_retry_on, ResultValidationError)


def tool_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """Tool argument errors use the same bounded repair budget as schemas."""
    return (*retry.validation_retry_on, ToolArgumentError)


def compose_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """A compose turn can fail either way, so it is charged for both.

    A turn that requested tools keeps the tool lane's contract — a round whose
    every call is malformed raises :class:`ToolArgumentError` and is re-asked —
    and a turn that answered is validated locally, raising
    :class:`StructuredOutputError` when the final answer does not validate.
    The type is listed here as well as in the default
    :data:`~llmkit.LLM_SCHEMA_ERRORS`, so a custom ``validation_retry_on``
    that omits it still re-asks a failed final answer.
    """
    return (*tool_validation_budget(retry), StructuredOutputError)


def build_call_provider(
    provider: LLMProviderInterface | None,
) -> LLMProviderInterface | None:
    """Resolve the provider for a call exactly once, best-effort.

    A ``provider`` passed for the call arrives here already an instance (see
    :func:`coerce_call_provider`) and is returned unchanged; otherwise the
    registered provider is built a single time so the transport and
    the log record share one instance — the default path no longer calls
    :func:`~llmkit.providers.build_provider` twice (once to run the call,
    again only to read ``.model``/``.name`` for the log). A build failure
    degrades to ``None`` rather than breaking the call: the compose form of
    :func:`~llmkit.calls.tool.tool_llm_call` re-resolves at once and raises the
    real configuration error before any attempt, every other call's transport
    re-resolves inside its attempt and surfaces it there, and the log path
    degrades to ``(model, None)`` via
    :func:`resolve_model_and_provider` below. Building is config +
    cached SDK checks (no I/O), so resolving it here before the retry loop is
    safe and happens once per call, not per attempt.
    """
    if provider is not None:
        return provider
    try:
        from llmkit.providers import build_provider

        return build_provider()
    except Exception:
        # The call must not break here; let the transport raise the real
        # error (logged + retried per attempt) and the log degrade to None.
        # report() contains a host handler that raises on this debug.
        _ = report(
            logger, logging.DEBUG, "Could not pre-resolve provider for LLM call", exc_info=True
        )
        return None


class PreparedCall(NamedTuple):
    """What a call resolves once, before its first attempt.

    Attributes:
        args: the merged per-call arguments (config < options < keyword).
        provider: the provider instance this call runs on, built exactly once
            so the transport and every attempt's log record share it: a
            per-call instance as given, one built from a per-call
            :class:`~llmkit.Provider` member, the registered provider, or
            ``None`` when the registered build failed (the transport then
            surfaces that error).
        call_id: one ``uuid4`` hex per *logical* call. Every retry attempt
            shares it and numbers itself, so the N records a retried call
            produces join on ``call_id`` rather than on feature + timestamp
            proximity, which breaks under concurrent same-feature fan-out. A
            call re-run by an outer :func:`~llmkit.retry.with_retries` reuses
            the id its earlier passes recorded.
        attempt_offset: the attempts this logical call already made on earlier
            passes of an outer :func:`~llmkit.retry.with_retries` (``0``
            outside one). Each family records ``attempt_offset`` plus its own
            1-based attempt number.
    """

    args: ResolvedCallArgs
    provider: LLMProviderInterface | None
    call_id: str
    attempt_offset: int


def prepare_call(
    options: LLMCallOptions | None,
    *,
    temperature: float | None | Unset,
    model: str | None | Unset,
    max_tokens: int | None | Unset,
    reasoning_effort: ReasoningEffort | None | Unset,
    retry: RetryPolicy | Unset,
    provider: LLMProviderInterface | Provider | None | Unset,
    feature: str,
    label: str | None,
    cache: bool | Unset = UNSET,
    timeout: float | None | Unset = UNSET,
    schema_repairs: int | Unset = UNSET,
) -> PreparedCall:
    """Resolve everything a call decides once, before any attempt runs.

    The preamble every family shares. Each keyword arrives exactly as the
    call function received it — :data:`~llmkit.UNSET` when the caller did not
    pass it — and is merged by :func:`~llmkit.options.resolve_call_args`.

    Building the provider here rather than per attempt is what lets the
    transport and the log record name the same instance; it is config plus
    cached SDK checks with no I/O, so once per call is both correct and cheap.

    *feature* and *label* identify the call to an enclosing
    :func:`~llmkit.retry.with_retries` loop owned by this task (see
    :func:`~llmkit.retry.claim_logical_call`), which is how a call re-run on
    every pass keeps one ``call_id``. For a stream this runs on the generator's
    first ``__anext__``, in the consumer's task — a read of the loop's scope,
    never a write.

    A per-call ``provider`` is coerced here by :func:`coerce_call_provider`,
    before the not-configured check and before the call is claimed from an
    enclosing retry loop, so a refused string or a member that cannot be built
    raises before any attempt is made, logged or counted.

    *cache* is passed by the families that read the response cache — the two
    buffered ones and the streamed text family; the tool families leave it
    unset. *schema_repairs* is passed by the structured family only; the
    others leave it unset, so only an ``options`` value reaches the merge,
    where an invalid one is still rejected.
    """
    args = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
        cache=cache,
        schema_repairs=schema_repairs,
    )
    # Before the claim: inside an outer with_retries a claim counts an attempt,
    # so an error raised after it would count a phantom one.
    call_provider = coerce_call_provider(args.provider)
    if call_provider is None and not client_configured():
        # The one piece of setup a call needs, missing: fail before any attempt
        # is made or logged, naming it. Other provider build failures keep the
        # best-effort path below.
        _ = active_config()
    call_id, attempt_offset = claim_logical_call(feature, label)
    return PreparedCall(args, build_call_provider(call_provider), call_id, attempt_offset)


async def run_with_policy[T](
    attempt: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    tag: str,
    validation_retry_on: tuple[type[BaseException], ...],
) -> T:
    """Run *attempt* under the call's retry budget.

    The families differ only in which errors are charged to the *validation*
    budget, so that is the one parameter; everything else comes off the policy.
    Note that this is one pass of the retry loop, not one attempt: each attempt
    is its own logged call, because the loop wraps the logging call functions
    rather than living inside them.
    """
    return await with_retries(
        attempt,
        policy=dataclasses.replace(policy, validation_retry_on=validation_retry_on),
        label=tag,
    )


def cache_and_key(
    prepared: PreparedCall,
    prompt: str | Sequence[ChatMessage],
    output_schema: type[BaseModel] | None,
) -> tuple[LLMCache, str] | None:
    """The configured cache and this request's key, or ``None`` to bypass it.

    The cache participates only when the call did not opt out
    (``args.cache``), a cache is configured, and the provider resolved — a
    ``None`` provider means the build failed and the transport must be left to
    raise the real error — and only when the request can be fingerprinted.
    Every lane that reads the cache decides here, so they cannot drift on when
    a request is byte-identical to a library without a cache.
    """
    args, provider, _, _ = prepared
    cache = get_llm_cache() if args.cache else None
    if cache is None or provider is None:
        return None
    key = cache_key_or_none(
        provider=provider,
        prompt=prompt,
        output_schema=output_schema,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    return None if key is None else (cache, key)


async def run_cached[T](
    run: Callable[[], Awaitable[T]],
    *,
    prepared: PreparedCall,
    prompt: str | Sequence[ChatMessage],
    output_schema: type[BaseModel] | None,
    encode: Callable[[T], str | None],
    decode: Callable[[str], T],
    on_result: Callable[[T], object] | None,
    record_hit: Callable[[T, str | None, datetime, float], Awaitable[None]],
) -> T:
    """Run *run* (the call's whole retry pass) through the response cache.

    When :func:`cache_and_key` bypasses the cache this is exactly
    ``await run()``: the request is byte-identical to a library without a
    cache.

    A hit sits above :func:`run_with_policy` and therefore above the rate
    limiter, so it spends no retry budget and takes no slot. It still runs the
    caller's *on_result* on the decoded answer — a rejection there makes it a
    miss for this caller, as does an exception in the policy's
    ``validation_retry_on`` (the paid path would re-roll on it), while any other
    exception from the hook propagates without a provider request; only a
    decode failure is a cache failure — and
    writes its own record through *record_hit*,
    called with the decoded answer, the ``call_id`` of the call that paid for
    it, and the hit's start time (wall clock, then monotonic).

    *encode* and *decode* turn a result into the stored string and back;
    *encode* returning ``None`` stores nothing.
    """
    args, provider, call_id, attempt_offset = prepared
    participation = cache_and_key(prepared, prompt, output_schema)
    if participation is None:
        return await run()
    cache, key = participation
    started_at = datetime.now(UTC)
    start_t = time.monotonic()
    model, provider_name = resolve_model_and_provider(args.model, provider)
    schema = output_schema.__name__ if output_schema is not None else "text"

    def _encode(result: T) -> LLMCacheEntry | None:
        response = encode(result)
        if response is None:
            # Nothing a hit could validate: answered, just not stored.
            return None
        return LLMCacheEntry(
            response=response,
            schema=schema,
            provider=provider_name,
            model=model,
            call_id=call_id,
        )

    async def _accept(entry: LLMCacheEntry) -> T:
        try:
            result = decode(entry.response)
        except Exception as exc:
            # Only a decode failure is a cache failure; anything the caller's
            # hook raises below propagates as it would on a paid answer.
            raise CacheDecodeError from exc
        if on_result is not None:
            try:
                await run_on_result(on_result, result)
            except args.retry.validation_retry_on as exc:
                # On a paid answer the call would re-roll on this exception
                # (it is in the validation budget), so on a hit it is a miss,
                # like a ResultValidationError; anything else propagates.
                raise ResultValidationError(str(exc)) from exc
        await record_hit(result, entry.call_id, started_at, start_t)
        return result

    # A pass re-run by an enclosing ``with_retries`` loop exists because the
    # host rejected the earlier answer; serving it the stored answer again
    # would make the documented re-roll pattern a fixed point.
    return await read_through(
        cache, key, lookup=attempt_offset == 0, run=run, encode=_encode, accept=_accept
    )


def _tool_call_from_raw(raw: object, definitions: dict[str, ToolDefinition]) -> ToolCall:
    """Narrow LiteLLM's intentionally open tool-call objects at one boundary."""
    call_id = getattr(raw, "id", None)
    function = getattr(raw, "function", None)
    name = getattr(function, "name", None)
    arguments_raw = getattr(function, "arguments", None)
    if (
        not isinstance(call_id, str)
        or not isinstance(name, str)
        or not isinstance(arguments_raw, str)
    ):
        raise ToolArgumentError(
            name if isinstance(name, str) else None,
            call_id if isinstance(call_id, str) else None,
            arguments_raw if isinstance(arguments_raw, str) else None,
            "provider returned a malformed tool call",
        )
    definition = definitions.get(name)
    if definition is None:
        raise ToolArgumentError(
            name, call_id, arguments_raw, f"model requested unknown tool {name!r}"
        )
    try:
        parsed = cast("object", json.loads(arguments_raw))
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(
            name, call_id, arguments_raw, "tool arguments are not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolArgumentError(
            name, call_id, arguments_raw, "tool arguments must be a JSON object"
        )
    try:
        # JSON mode on the raw string, strict, like every other lane: the same
        # rule for a tool's model as for an ``output_schema``.
        validated = (
            definition.model.model_validate_json(arguments_raw, strict=True)
            if definition.model is not None
            else None
        )
    except SchemaEngineError:
        raise
    except Exception as exc:
        raise ToolArgumentError(
            name, call_id, arguments_raw, f"tool arguments failed validation: {exc}"
        ) from exc
    return ToolCall(call_id, name, arguments_raw, cast("dict[str, object]", parsed), validated)


def parse_tool_calls(
    raw_calls: list[object], definitions: dict[str, ToolDefinition]
) -> tuple[list[ToolCall], list[ToolArgumentError]]:
    """Narrow a round's raw calls per call, not as an all-or-nothing batch.

    Parsing used to be a list comprehension, so the first unparseable call
    raised out of it and took its well-formed siblings with it. Nothing has
    executed at this point, which makes that lossy rather than unsafe — but
    with parallel calls one bad argument string cost the entire round and a
    re-ask. Each call is narrowed on its own instead, and the failures are
    returned beside the survivors for :class:`~llmkit.ToolCallResult` to carry.

    The one case that still raises is a round in which **every** call failed:
    the turn produced nothing actionable, so it stays a whole-round
    :class:`~llmkit.ToolArgumentError` and is re-asked on the validation
    budget with the rejected calls and their errors attached when replayable.
    A round with no calls at all is not a failure — a text-only or compose
    answer takes that path.

    Public-named inside a private module, the repo's convention for a seam two
    families share: :func:`~llmkit.tool_llm_call` and
    :func:`~llmkit.tool_llm_call_stream` both terminate in a
    :class:`~llmkit.ToolCallResult`, so parsing once here is what keeps the
    buffered and streamed lanes' salvage semantics from drifting apart. The
    streaming lane re-asks on the same validation budget only while nothing
    has been yielded (a partially consumed stream cannot be restarted), so an
    all-malformed round that followed streamed prose reaches its caller
    unretried — the contract is the same, its enforcement differs.
    """
    parsed: list[ToolCall] = []
    invalid: list[ToolArgumentError] = []
    for raw in raw_calls:
        try:
            parsed.append(_tool_call_from_raw(raw, definitions))
        except ToolArgumentError as exc:
            invalid.append(exc)
    if invalid and not parsed:
        first = invalid[0]
        raise ToolArgumentError(
            first.tool_name,
            first.call_id,
            first.arguments_raw,
            str(first),
            round_errors=tuple(invalid),
        ) from first
    if invalid:
        _ = report(
            logger,
            logging.WARNING,
            "Dropped %d malformed tool call(s) from a round of %d: %s",
            len(invalid),
            len(raw_calls),
            "; ".join(f"{error.tool_name!r}: {error}" for error in invalid),
        )
    return parsed, invalid


def resolve_model_and_provider(
    model: str | None, provider: LLMProviderInterface | None = None
) -> tuple[str | None, str | None]:
    """Resolve the *effective* model + provider name for the log record.

    When the caller passes ``model=None`` the provider's configured
    default is what actually ran — record that instead of ``null`` so
    cost attribution is a ``grep | sort | uniq -c`` over the logs, not a
    code trace. An explicit ``provider`` (passed for the call) is used
    as-is so the log names the provider that *actually* ran, not the
    registered one. Best-effort: any failure resolving the
    provider degrades to ``(model, None)`` rather than breaking the log
    write, and its debug goes through :func:`~llmkit._diagnostics.report`, so
    an ordinary exception from a host handler is contained and counted too (a
    handler that blocks is not).
    """
    try:
        if provider is None:
            from llmkit.providers import build_provider

            provider = build_provider()
        return (model or provider.model, provider.name)
    except Exception:
        # Degrade to (model, None); report() contains a raising handler.
        _ = report(logger, logging.DEBUG, "Could not resolve provider for LLM log", exc_info=True)
        return (model, None)


def error_text(exc: BaseException) -> str:
    """The record's ``error`` for *exc*: ``"<Type>: <message>"``, or the bare type.

    One formatter for every family, so a failure reads the same on each lane.
    The message is dropped when empty, so a bare ``asyncio.CancelledError``
    records ``"CancelledError"`` rather than ``"CancelledError: "``.
    """
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def usage_log_dict(usage: TokenUsage | None) -> dict[str, int | None] | None:
    """The log record's ``usage`` mapping, ``None`` when no usage was received.

    One builder for every lane that logs token counts, so the structured, text
    and tool records cannot drift on key names.
    """
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def build_text_record(
    *,
    started_at: datetime,
    feature: str,
    label: str | None,
    prompt: str | Sequence[ChatMessage],
    text: str | None,
    start_t: float,
    temperature: float | None,
    model: str | None,
    provider: LLMProviderInterface | None,
    error: str | None,
    approximate_cost: float | None,
    schema: str = "text",
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
    usage: TokenUsage | None = None,
    cache_hit: bool = False,
    source_call_id: str | None = None,
    output_limit: OutputLimitDiagnostics | None = None,
    partial_text: str | None = None,
    native_finish_reason: str | None = None,
    first_chunk_at: float | None = None,
) -> LLMCallRecord:
    """Build the ``LLMCallRecord`` for a plain-text/stream call.

    Shared by :func:`text_llm_call` and :func:`text_llm_call_stream`: the
    ``schema`` distinguishes the two surfaces in the log — ``"text"`` for a
    buffered plain-text call, ``"stream"`` for a streamed one — and
    ``response`` is the accumulated text rather than a Pydantic dump
    (``None`` when a buffered attempt failed before any content; the
    streaming surface instead passes the partial transcript). ``model``
    is resolved to
    the effective model (provider default substituted when the caller
    passed ``None``); ``provider`` is the one passed for the call (``None``
    uses the registered one) and names the provider the log records.
    ``max_tokens``/``reasoning_effort`` are recorded as on the structured
    path, so the cap and thinking setting appear in the log for these calls
    too (both default ``None`` — absent from the request and unset on the
    record). ``usage`` is the call's token counts as the provider reported
    them, buffered or streamed (``None`` where an attempt never received
    them, such as an abandoned stream). ``cache_hit``
    marks a response-cache hit — buffered or streamed, whichever lane
    answered — whose ``queue_wait_ms`` is ``None`` because it never queued
    (the queue-wait stamp would still hold the previous attempt's value), and ``source_call_id`` names the call that paid
    for its answer. ``output_limit`` is a truncated attempt's
    :class:`~llmkit.OutputLimitDiagnostics`: it fills the record's
    ``output_limit`` and ``partial_text`` fields, and its token counts become
    ``usage`` when the caller passes none, as on the structured path.
    ``partial_text`` also carries filtered or failed text when no output limit
    occurred; ``native_finish_reason`` preserves the provider reason where delivered.
    ``first_chunk_at`` is the :func:`time.monotonic` instant a streamed
    attempt's first wire frame arrived, recorded as ``first_chunk_ms`` from
    ``start_t``; ``None`` (every buffered call, every hit) records ``None``.

    Building is separate from recording on purpose: the buffered path hands
    the record to :func:`~llmkit.capture.record_call_async` for bounded waiting,
    while abandoned streams capture and submit without waiting. Both deliver
    sink I/O on the same worker; each caller picks its waiting behavior.
    """
    duration_ms = (time.monotonic() - start_t) * 1000
    resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
    if output_limit is not None and usage is None:
        usage = TokenUsage(
            output_limit.prompt_tokens,
            output_limit.completion_tokens,
            output_limit.total_tokens,
        )
    return LLMCallRecord(
        started_at=started_at,
        feature=feature,
        label=label,
        model=resolved_model,
        provider=resolved_provider,
        temperature=temperature,
        duration_ms=duration_ms,
        schema=schema,
        prompt=prompt,
        response=text,
        error=error,
        approximate_cost=approximate_cost,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        call_id=call_id,
        attempt=attempt,
        queue_wait_ms=None if cache_hit else current_queue_wait_ms(),
        run_id=get_run_id(),
        usage=usage_log_dict(usage),
        cache_hit=cache_hit,
        source_call_id=source_call_id,
        output_limit=output_limit.to_log_dict() if output_limit is not None else None,
        partial_text=output_limit.partial_text if output_limit is not None else partial_text,
        native_finish_reason=native_finish_reason,
        first_chunk_ms=(first_chunk_at - start_t) * 1000 if first_chunk_at is not None else None,
    )


def tool_repair_messages(
    errors: Sequence[ToolArgumentError], offered: Sequence[str]
) -> list[ChatMessage]:
    """Render one rejected round without replaying non-object JSON arguments.

    Missing ids or names make the entire round unpairable, so it stays blind.
    A final user turn requests another tool call rather than a prose answer.
    Unknown names are preserved; whether Anthropic or Gemini accept history
    naming an undeclared tool has not been measured.
    """
    assistant: AssistantToolMessage = {"role": "assistant", "content": None, "tool_calls": []}
    results: list[ChatMessage] = []
    for error in errors:
        if error.call_id is None or error.tool_name is None:
            return []
        arguments = error.arguments_raw
        try:
            replayable = arguments is not None and isinstance(json.loads(arguments), dict)
        except json.JSONDecodeError:
            replayable = False
        assistant["tool_calls"].append(
            {
                "id": error.call_id,
                "type": "function",
                "function": {
                    "name": error.tool_name,
                    "arguments": arguments if replayable and arguments is not None else "{}",
                },
            }
        )
        text = (
            f"Tool {error.tool_name!r}, call {error.call_id!r}: {error}. "
            "The tool did not execute. Request the tool again with corrected arguments."
        )
        if not replayable:
            text += (
                f". Raw arguments: {arguments!r}"
                if arguments is not None
                else ". No arguments arrived"
            )
        if error.tool_name not in offered:
            text += f". Tool was not offered. Offered tools: {', '.join(offered)}"
        results.append(tool_result_message(error.call_id, text, is_error=True))
    if not results:
        return []
    # A tool result normally invites a final answer. Small models can describe
    # corrected arguments in prose unless a new user turn requests another call.
    reminder: ChatMessage = {
        "role": "user",
        "content": (
            "Retry the rejected tool request now using the tool interface. "
            "Correct the arguments according to the errors above. "
            "Respect the declared JSON types: an integer must be a JSON number "
            "without quotes. Send a tool call rather than prose or a code block."
        ),
    }
    return [assistant, *results, reminder]
