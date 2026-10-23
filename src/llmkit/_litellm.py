"""Internal LiteLLM call layer.

The single place that talks to LiteLLM (and, for structured output,
``instructor`` over LiteLLM). The public call functions in
:mod:`llmkit.calls` build/log :class:`LLMCallRecord`s
around these helpers; this module owns provider routing, the rate-limit
semaphore, structured-output mode pinning, and best-effort cost extraction.

**The transport boundary.** Every provider round trip crosses :func:`_send`
(buffered) or :func:`_stream` (streamed), inside the rate-limiter slot, as
llmkit's own types from :mod:`llmkit._transport`. Behind the boundary,
:func:`_acompletion` is the LiteLLM adapter, and the ``_wire_*`` readers are
the only code that reads LiteLLM's response objects; everything after them
reads wire types. A fake installed through :mod:`llmkit.testing` answers at
the boundary instead, so the limiter, breaker, retry and parsing still run.
That is the supported way for downstream code to fake a provider; patching
this module is not.

**Internal test seams.** The library's own tests patch three levels here,
and they stay internal: the five transport coroutines
(``acompletion_structured`` / ``acompletion_text`` / ``acompletion_tools`` /
``astream_text`` / ``astream_tools``), :func:`_acompletion`, and
``litellm.acompletion`` itself. A test belongs on these seams when it tests
the LiteLLM adapter itself, needs input the public fake cannot produce (a
LiteLLM-shaped response, a mid-stream failure), or deliberately isolates the
call layer from the limiter and breaker. A test of behavior observable
through the public call functions, whose inputs the public fake can express,
belongs on :mod:`llmkit.testing`.

**Every reach into this module is a *function-local* ``import
llmkit._litellm as _litellm``** (the call modules under :mod:`llmkit.calls`
and :mod:`llmkit.sync`), never a module-level import. That is not style: this
module imports ``litellm`` eagerly at line scope below, and ``import llmkit``
must leave litellm out of ``sys.modules`` entirely — a multi-second import a
local-first library consumed by CLIs should not pay until the first provider
call. ``tests/packaging/test_litellm_lazy_import.py`` pins that contract in a
fresh interpreter. Binding the *module object* (rather than
``from llmkit._litellm import acompletion_structured``) is what additionally
keeps the patch seam live: the attribute lookup resolves at call time, so a
patch installed after import is seen. The absolute ``import llmkit._litellm``
form is deliberate too — ``from llmkit import _litellm`` would route through
the package ``__init__`` and reintroduce the static import cycle that used to
force a file-level ``reportImportCycles`` waiver there.

LiteLLM's ``acompletion`` and instructor's ``create_with_completion`` carry
very strict, heavily-overloaded type stubs that reject this module's generic
``**credential-kwargs`` and ``list[Message]`` message shapes. Those
call expressions therefore carry a single ``reportArgumentType`` suppression
each, tagged ``raw-llm`` — the boundary where our thin wrapper meets the
provider SDK's exhaustive parameter surface.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import cast

import instructor
import litellm
from instructor.core.exceptions import IncompleteOutputException
from litellm import CustomStreamWrapper
from litellm.exceptions import BadRequestError
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from llmkit._messages import resolve_reasoning_effort, wire_messages
from llmkit._transport import (
    CallContext,
    CallKind,
    WireChunk,
    WireFunction,
    WireRequest,
    WireResponse,
    WireStreamEnd,
    WireToolCall,
    WireToolFragment,
    WireUsage,
    active_double,
)
from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.exceptions import (
    REPAIRABLE_PARSE_ERRORS,
    OutputLimitDiagnostics,
    OutputLimitError,
    normalize_service_unavailable,
)
from llmkit.providers import LLMProviderInterface, build_provider
from llmkit.rate_limiting import GlobalRateLimiter
from llmkit.tools import ToolChoice, ToolDefinition, ToolName

logger = logging.getLogger(__name__)


class _NotSent:
    """Sentinel type: no transport call in this context has returned yet."""


_NOT_SENT = _NotSent()

# The temperature the funnel actually sent on its last *successful* call in
# this context: the caller's value, or ``None`` when the model refused it and
# the request was re-issued without it. The same three-party protocol as the
# queue-wait stamp (:mod:`llmkit.rate_limiting._observability`): the call
# layer calls :func:`begin_sent_temperature` before each transport await,
# :func:`_acompletion` stamps on return, and the call layer copies
# :func:`sent_temperature` into a plain local *immediately* after the await —
# never at record-build time, because a streaming record may be written from
# another task's context (see ``llmkit.calls.stream``).
_sent_temperature: ContextVar[float | None | _NotSent] = ContextVar(
    "_llmkit_sent_temperature", default=_NOT_SENT
)

_TEMPERATURE = "temperature"
_DROP_PARAMS = "additional_drop_params"
# LiteLLM's per-call opt-out of its proxy MCP gateway check; see _acompletion.
_SKIP_MCP_HANDLER = "_skip_mcp_handler"


def begin_sent_temperature() -> None:
    """Clear the sent-temperature stamp before an attempt reaches the transport.

    Without the reset an attempt that raised, or a test double that never
    reached :func:`_acompletion`, would inherit an earlier call's stamp.
    """
    _ = _sent_temperature.set(_NOT_SENT)


def sent_temperature(requested: float | None) -> float | None:
    """The temperature the last transport call in this context sent.

    Falls back to *requested* when nothing stamped since
    :func:`begin_sent_temperature`: the attempt raised, or a patched transport
    bypassed the funnel.
    """
    sent = _sent_temperature.get()
    return requested if isinstance(sent, _NotSent) else sent


def _refuses_temperature(exc: Exception, kwargs: dict[str, object]) -> bool:
    """Whether *exc* is a 400 refusing the ``temperature`` this request carried.

    Covers LiteLLM's client-side ``UnsupportedParamsError`` (a
    ``BadRequestError`` subclass) and a provider's own 400 alike, so the
    behaviour does not depend on which LiteLLM model map loaded. A request that
    already asks LiteLLM to drop ``temperature`` is never re-issued.
    """
    if _TEMPERATURE not in kwargs or not isinstance(exc, BadRequestError):
        return False
    if getattr(exc, "status_code", None) != 400:
        return False
    already_dropped = kwargs.get(_DROP_PARAMS)
    if isinstance(already_dropped, list) and _TEMPERATURE in cast("list[object]", already_dropped):
        return False
    return _TEMPERATURE in str(exc).lower()


def _without_temperature(kwargs: dict[str, object]) -> dict[str, object]:
    """*kwargs* plus LiteLLM's per-call ``additional_drop_params=["temperature"]``.

    The ``temperature`` key itself stays, so the re-issued request differs from
    the refused one only in the parameter it named; a host-supplied drop list
    is extended, not replaced.
    """
    existing = kwargs.get(_DROP_PARAMS)
    drop: list[object] = list(cast("list[object]", existing)) if isinstance(existing, list) else []
    return {**kwargs, _DROP_PARAMS: [*drop, _TEMPERATURE]}


async def _acompletion(**kwargs: object) -> ModelResponse | CustomStreamWrapper:
    """Typed boundary over ``litellm.acompletion``.

    LiteLLM's shipped stub types ``acompletion`` with ``Unknown`` in several
    parameter slots (``messages``/``stop``/``model_list`` …) and exposes the
    attribute itself with a partially-unknown type, which would otherwise leak
    ``reportUnknownMemberType`` into every call site. Funnelling the call
    through here narrows that member access (and the result) in one place to the
    single fact callers rely on: it resolves to a ``ModelResponse`` (or, with
    ``stream=True``, a ``CustomStreamWrapper``).

    ``litellm.acompletion`` is resolved *inside* the call so unit tests can keep
    patching ``llmkit._litellm.litellm.acompletion`` to stub the provider; the
    per-call ``reportArgumentType`` suppressions at each caller cover the
    over-strict argument shapes (credential splat, message-list typing).

    This is also the boundary where a litellm-native 503 is re-owned: litellm's
    ``ServiceUnavailableError`` matches the transport tuple only via a metaclass
    ``isinstance`` hook that ``except`` clauses bypass, so left raw it would slip
    through a host's documented ``except LLM_RECOVERABLE_ERRORS`` net. It is
    re-raised as :class:`llmkit.exceptions.ServiceUnavailableError` (original on
    ``__cause__``); every other exception propagates untouched. This cannot be an
    ``except <type>`` clause — that is the very matching the litellm class fails —
    hence the catch-all plus ``isinstance`` inside the normalizer.

    It is also where a refused ``temperature`` is omitted. When LiteLLM
    (client-side) or the provider (an HTTP 400) refuses the request's
    ``temperature`` as unsupported for the model, the same request is re-issued
    exactly once with LiteLLM's per-call ``additional_drop_params`` naming it —
    never the process-global ``litellm.drop_params``. If the re-issue fails too,
    the *original* refusal is raised, the second error chained as its context.
    The temperature actually sent is stamped for the call record (see
    :func:`sent_temperature`). A host that sets the global
    ``litellm.drop_params`` makes LiteLLM drop silently instead, so no refusal
    reaches here and the record shows the caller's value.

    It is also where a request carrying ``tools`` opts out of LiteLLM's proxy
    MCP gateway, with LiteLLM's per-call ``_skip_mcp_handler=True``. Unless that
    switch is set, ``completion`` enters a check that asks whether any tool names
    the gateway, and in LiteLLM 1.106.0 through 1.106.2 the check's module
    imports the ``mcp`` package at module level. ``mcp`` ships only in LiteLLM's
    ``mcp``/``proxy`` extras, so on a plain install every tool call failed
    before anything was sent, as an ``APIConnectionError`` reading
    ``No module named 'mcp'`` (BerriAI/litellm#46045, fixed on LiteLLM's
    ``main`` by #45961). Skipping is safe because llmkit builds only
    ``{"type": "function", ...}`` tools, which that check classifies as non-MCP
    anyway. ``completion`` pops the key before parameter mapping, so it never
    reaches a provider, and it is set on these *kwargs* so the temperature
    re-issue carries it too. Why llmkit does this rather than cap LiteLLM or
    require the ``mcp`` extra is recorded on the ``litellm`` requirement in
    ``pyproject.toml``.
    """
    acompletion = cast(
        "Callable[..., Coroutine[object, object, ModelResponse | CustomStreamWrapper]]",
        litellm.acompletion,
    )
    if kwargs.get("tools"):
        _ = kwargs.setdefault(_SKIP_MCP_HANDLER, True)
    try:
        try:
            result = await acompletion(**kwargs)
        except Exception as refused:
            if not _refuses_temperature(refused, kwargs):
                raise
            try:
                result = await acompletion(**_without_temperature(kwargs))
            except Exception:
                # Raised inside this handler, so the second error rides on
                # ``__context__`` while the caller sees the refusal it caused.
                raise refused  # noqa: B904 — the original is the error to surface
            _ = _sent_temperature.set(None)
            return result
        sent = kwargs.get(_TEMPERATURE)
        _ = _sent_temperature.set(sent if isinstance(sent, (int, float)) else None)
        return result
    except Exception as e:
        normalized = normalize_service_unavailable(e)
        if normalized is not e:
            raise normalized from e
        raise


async def _send(request: WireRequest) -> WireResponse:
    """One buffered provider round trip, as llmkit's own response type.

    The boundary the buffered transports await, always inside the
    rate-limiter slot. A double installed through :mod:`llmkit.testing`
    answers here in place of the provider; otherwise the LiteLLM adapter does:
    :func:`_acompletion`, whose duties (503 re-ownership, the temperature
    re-issue and stamp, the MCP-handler opt-out) this only wraps with a
    conversion.
    """
    double = active_double()
    if double is not None and request.call is not None:
        return await double.respond(request.call)
    return _wire_response(await _acompletion(**request.kwargs))


async def _stream(request: WireRequest) -> AsyncGenerator[WireChunk | WireStreamEnd]:
    """One streamed provider round trip: wire chunks, then one :class:`WireStreamEnd`.

    A 503 can surface mid-iteration, because the wrapper fetches chunks lazily
    after :func:`_acompletion` has returned. It is re-owned exactly as the
    call-time boundary does, so the stream path keeps the
    ``except LLM_RECOVERABLE_ERRORS`` contract too.
    """
    double = active_double()
    if double is not None and request.call is not None:
        async for item in double.stream(request.call):
            yield item
        return
    # stream=True makes acompletion return a CustomStreamWrapper, whose async
    # iteration yields typed ModelResponseStream chunks.
    stream = cast("CustomStreamWrapper", await _acompletion(**request.kwargs))
    try:
        async for chunk in stream:
            yield _wire_chunk(chunk)
    except Exception as e:
        normalized = normalize_service_unavailable(e)
        if normalized is not e:
            raise normalized from e
        raise
    yield WireStreamEnd(usage=_wire_usage(stream), cost=_response_cost(stream))


def _wire_usage(raw: object) -> WireUsage:
    """Read *raw*'s ``usage`` into :class:`WireUsage`, pure ``getattr``.

    Any missing or non-``int`` count is ``None``; nothing here raises.
    """
    usage = getattr(raw, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return WireUsage(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
        total_tokens=total if isinstance(total, int) else None,
        reasoning_tokens=reasoning if isinstance(reasoning, int) else None,
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _wire_tool_call(raw: object) -> WireToolCall:
    """One LiteLLM tool call; a field that is not a ``str`` becomes ``None``.

    The parser rejects a non-``str`` field as a malformed call either way, so
    narrowing here changes no outcome.
    """
    function = getattr(raw, "function", None)
    return WireToolCall(
        id=_str_or_none(getattr(raw, "id", None)),
        function=WireFunction(
            name=_str_or_none(getattr(function, "name", None)),
            arguments=_str_or_none(getattr(function, "arguments", None)),
        ),
    )


def _wire_response(raw: object) -> WireResponse:
    """Convert a LiteLLM completion to :class:`WireResponse`.

    Total ``getattr`` rather than typed, so a test double shaped like a
    completion (a ``SimpleNamespace``, a ``MagicMock``) converts as a real
    ``ModelResponse`` does. Message content becomes ``None`` only when absent;
    anything else is coerced by :func:`_coerce_text_content`.
    """
    usage = _wire_usage(raw)
    cost = _response_cost(raw)
    choices = cast("object", getattr(raw, "choices", None))
    if not choices:
        # A degenerate empty-``choices`` response (e.g. a malformed
        # OpenAI-compatible proxy returning ``{"choices": []}``).
        return WireResponse(has_choice=False, usage=usage, cost=cost)
    choice = cast("Sequence[object]", choices)[0]
    message = getattr(choice, "message", None)
    content = cast("object", getattr(message, "content", None))
    raw_calls = cast("object", getattr(message, "tool_calls", None))
    return WireResponse(
        text=_coerce_text_content(content) if content is not None else None,
        tool_calls=(
            tuple(_wire_tool_call(call) for call in cast("list[object]", raw_calls))
            if isinstance(raw_calls, list)
            else ()
        ),
        finish_reason=_str_or_none(getattr(choice, "finish_reason", None)),
        usage=usage,
        cost=cost,
    )


def _wire_chunk(chunk: ModelResponseStream) -> WireChunk:
    """Convert one LiteLLM stream frame to :class:`WireChunk`."""
    delta = _chunk_delta_text(chunk)
    return WireChunk(
        text=delta if isinstance(delta, str) else None,
        tool_fragments=_chunk_tool_fragments(chunk),
        finish_reason=_chunk_finish_reason(chunk),
        # ``include_usage`` puts usage on a *final* frame that usually carries
        # no choices; a frame without any usage reports ``None``.
        usage=_wire_usage(chunk) if getattr(chunk, "usage", None) is not None else None,
        cost=_response_cost(chunk),
    )


def _render_model_response(response: WireResponse) -> ModelResponse:
    """Present a :class:`WireResponse` in the shape instructor parses.

    instructor's handlers for the modes the providers pin (``JSON_SCHEMA``,
    ``JSON``) read only the first choice's content and finish reason; a tool
    mode reads its tool calls. Its usage tally reads ``usage``, and the
    structured transport reads cost from ``_hidden_params["response_cost"]``,
    which a hand-built ``ModelResponse`` starts without. ``usage`` is set only
    when some count was reported: a ``ModelResponse`` built without one has no
    ``usage`` attribute at all, which is what the original reported too.
    """
    message: dict[str, object] = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in response.tool_calls
        ]
    choices: list[dict[str, object]] = (
        [{"index": 0, "finish_reason": response.finish_reason, "message": message}]
        if response.has_choice
        else []
    )
    usage = response.usage
    reported = any(count is not None for count in (*usage.counts(), usage.reasoning_tokens))
    rendered = (
        ModelResponse(
            choices=choices,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            ),
        )
        if reported
        else ModelResponse(choices=choices)
    )
    rendered._hidden_params = (  # pyright: ignore[reportPrivateUsage]  # raw-llm — litellm's cost slot
        {"response_cost": response.cost} if response.cost is not None else {}
    )
    return rendered


@dataclass
class _Invocations:
    """How many times instructor has called the completion callable this attempt."""

    count: int = 0


async def _instructor_completion(
    *, strict: bool, call: CallContext, invocations: _Invocations, **kwargs: object
) -> ModelResponse:
    """The completion callable handed to ``instructor.from_litellm``.

    Sends instructor's request through :func:`_send` and presents the answer
    with :func:`_render_model_response`, so the structured path crosses the
    same boundary as every other transport. :func:`acompletion_structured`
    binds the keyword-only arguments per call: ``strict`` (see
    :func:`_stricten_json_schema_response_format`), the call's identity, and
    the invocation counter that marks instructor's in-attempt schema-repair
    re-ask. The identity carries instructor's messages, which differ from the
    caller's prompt in ``Mode.JSON``, where a schema system message is
    prepended.
    """
    if strict:
        _stricten_json_schema_response_format(kwargs)
    invocations.count += 1
    messages = kwargs.get("messages")
    call = replace(
        call,
        messages=(
            cast("list[ChatMessage]", messages) if isinstance(messages, list) else call.messages
        ),
        repair=invocations.count > 1,
    )
    return _render_model_response(await _send(WireRequest(kwargs=kwargs, call=call)))


def _stricten_json_schema_response_format(kwargs: dict[str, object]) -> None:
    """Upgrade a ``json_schema`` ``response_format`` in *kwargs* to strict.

    instructor's ``JSON_SCHEMA`` handler emits ``response_format`` without
    ``"strict": true`` or ``additionalProperties: false``, and a provider that
    treats a non-strict json_schema as *advisory* (OpenRouter) then lets weak
    models drift — measured 2026-07-14: ``mistralai/mistral-nemo``
    stochastically echoing the schema itself. The pre-1.15.3
    ``OPENROUTER_STRUCTURED_OUTPUTS`` handler sent exactly this strict shape,
    so this restores the measured wire contract at the one seam the library
    owns (the completion callable handed to ``from_litellm``). Used only for
    providers that opt in via
    :pyattr:`~llmkit.providers.base.BaseProvider.strict_json_schema`; the
    mutation targets the request dict instructor builds fresh per call.
    """
    response_format = kwargs.get("response_format")
    if (
        isinstance(response_format, dict)
        and cast("dict[str, object]", response_format).get("type") == "json_schema"
    ):
        json_schema = cast("dict[str, object]", response_format).get("json_schema")
        if isinstance(json_schema, dict):
            json_schema = cast("dict[str, object]", json_schema)
            json_schema["strict"] = True
            schema = json_schema.get("schema")
            if isinstance(schema, dict):
                cast("dict[str, object]", schema)["additionalProperties"] = False


async def drain_async_logging(*, timeout: float | None) -> None:
    """Flush LiteLLM's pending async logging on the current event loop.

    LiteLLM doesn't ``await`` its success logging inline. After a successful
    async call it *eagerly constructs* the ``Logging.async_success_handler``
    coroutine and hands it to a module-global ``LoggingWorker`` queue
    (``GLOBAL_LOGGING_WORKER``) to be run as a background task. On the sync
    bridge that coroutine can still be sitting in the queue — created but never
    awaited — when :func:`llmkit.sync.run_sync` closes the loop, and Python then
    emits ``RuntimeWarning: coroutine 'Logging.async_success_handler' was never
    awaited`` to stderr. That is visible noise on an otherwise clean call.

    Calling the worker's own ``flush`` *awaits* every queued coroutine to
    completion, which both performs the logging and clears the queue so no
    coroutine is destroyed unawaited. We bound it with ``timeout`` so a wedged
    callback can't hang the bridge (``flush`` is an unbounded ``queue.join`` on
    its own); past the deadline we give up draining rather than block.

    Best-effort by contract: this reaches into a LiteLLM internal, so any
    failure (the internal moving, no queue yet, the flush erroring) is swallowed
    — draining logging must never turn a successful call into a failed one.
    """
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except Exception:  # pragma: no cover - litellm internal moved
        return
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=timeout)
    except TimeoutError:
        logger.debug("LiteLLM async-logging drain timed out after %ss", timeout)
    except Exception:  # pragma: no cover - best-effort drain
        logger.debug("LiteLLM async-logging drain failed", exc_info=True)


def _reasoning_request_kwargs(
    provider: LLMProviderInterface, effort: ReasoningEffort | None, model: str | None
) -> dict[str, object]:
    """Build provider-owned reasoning kwargs, retaining old providers' fallback.

    ``reasoning_kwargs`` was added after the provider interface shipped, so
    third-party providers may not implement it yet. They retain the original
    flat LiteLLM request shape until they opt into a native translation.
    """
    if effort is None:
        return {}
    try:
        translate = provider.reasoning_kwargs
    except AttributeError:
        return {"reasoning_effort": effort}
    return translate(effort, model or provider.model)


def _completion_request_kwargs(
    provider: LLMProviderInterface, effort: ReasoningEffort | None, model: str | None
) -> dict[str, object]:
    """Merge provider credentials and a reasoning translation for one request."""
    creds = provider.completion_kwargs()
    reasoning = _reasoning_request_kwargs(provider, effort, model)
    native_body = reasoning.pop("extra_body", None)
    if native_body is None:
        return {**creds, **reasoning}
    if not isinstance(native_body, dict):
        raise TypeError("Provider reasoning extra_body must be a dict")
    native_body = cast("dict[str, object]", native_body)
    existing_body = creds.get("extra_body")
    if existing_body is None:
        merged_body = native_body
    elif isinstance(existing_body, dict):
        existing_body = cast("dict[str, object]", existing_body)
        merged_body = {**existing_body, **native_body}
    else:
        raise TypeError("Provider completion extra_body must be a dict")
    return {**creds, **reasoning, "extra_body": merged_body}


def _response_cost(
    raw: object,
) -> float | None:
    """Best-effort USD cost for a completion from its ``_hidden_params``.

    LiteLLM stamps ``response_cost`` onto the completion's
    ``_hidden_params`` (token usage x model pricing). Best-effort: any
    missing/odd shape degrades to ``None`` rather than breaking the call.
    """
    hidden = getattr(raw, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost: object = cast("dict[str, object]", hidden).get("response_cost")
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def _completion_tokens(raw: object) -> int | None:
    """Best-effort completion-token count from a completion's ``usage``.

    For :class:`OutputLimitError`'s
    diagnostics: how many tokens the provider actually generated before the
    output-token limit cut the completion off. Returns ``None`` for any
    missing/odd shape — the diagnostics degrade, never break the raise.
    """
    tokens = getattr(getattr(raw, "usage", None), "completion_tokens", None)
    return tokens if isinstance(tokens, int) else None


def _output_limit_diagnostics(last_completion: object) -> OutputLimitDiagnostics:
    """Best-effort :class:`OutputLimitDiagnostics` from a truncated completion.

    ``last_completion`` is whatever instructor attached to its
    ``IncompleteOutputException``: normally the whole litellm
    ``ModelResponse`` (the OpenAI JSON / JSON-schema handlers every shipped
    provider reaches), but one handler passes a bare ``choices[0]`` and it can
    be ``None``. Pure ``getattr`` like :func:`_wire_usage`, and it never
    raises — the diagnostics degrade, never break or mask the
    :class:`OutputLimitError`. Shape rules: string content is kept as-is
    (``""`` stays ``""``); list-of-blocks content joins its text blocks as
    :func:`_coerce_text_content` does (``""`` when none carry text); absent,
    ``None`` or any other content yields ``partial_text=None`` while the token
    counts are still read; a shape that raises mid-extraction blanks every
    field. A third-party provider in instructor's tool mode puts the partial
    answer in tool-call arguments, which are not read: its ``partial_text`` is
    ``None``.
    """
    try:
        choices = getattr(last_completion, "choices", None)
        choice: object = (
            cast("list[object]", choices)[0]
            if isinstance(choices, list) and choices
            else last_completion
        )
        finish_reason = getattr(choice, "finish_reason", None)
        content = getattr(getattr(choice, "message", None), "content", None)
        partial_text = (
            _coerce_text_content(cast("object", content))
            if isinstance(content, str | list)
            else None
        )
        usage = _wire_usage(last_completion)
        return OutputLimitDiagnostics(
            partial_text=partial_text,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )
    except Exception:
        logger.debug("Could not read output-limit diagnostics", exc_info=True)
        return OutputLimitDiagnostics()


def _schema_repair_retrying() -> AsyncRetrying:
    """instructor's in-call repair budget: one schema-repair re-ask, for a
    genuine parse failure only.

    ``stop_after_attempt(2)`` keeps the historical budget (instructor feeds
    this to tenacity as *total attempts*, so 2 = one repair re-ask). The
    predicate retries **only** the parse failures in
    :data:`REPAIRABLE_PARSE_ERRORS` — the same set instructor's own int-based
    path restricts to (mirrored, not imported from its private tuple). This is
    the whole point of the custom retrying: instructor's stock ``int``
    ``max_retries`` would re-ask parse errors too, but llmkit added the custom
    object for the truncation carve-out and must not, in doing so, silently
    widen the re-ask to *every* exception. A transport (429/5xx/network) or
    permanent (401/400/403) error therefore fails the predicate and is *not*
    re-asked — otherwise it would be blindly re-sent inside the single
    rate-limiter slot with zero backoff, ignoring ``Retry-After``, doubling the
    real requests one structured call makes.

    The explicit ``IncompleteOutputException`` exclusion is **defense in
    depth**: that class (instructor's truncated-by-the-output-token-limit
    signal, raised *before parsing* for both the OpenAI ``finish_reason ==
    "length"`` and google-genai ``FinishReason.MAX_TOKENS`` shapes) already
    subclasses none of :data:`REPAIRABLE_PARSE_ERRORS`, so the parse-only
    predicate declines it today. The exclusion stays load-bearing in case a
    future instructor reparents the class under a parse error — the
    no-re-ask-on-truncation guarantee (a re-ask on an identical token budget
    can only truncate again) must not silently vanish. When declined, tenacity
    propagates ``IncompleteOutputException`` **bare** (no wrap);
    :func:`acompletion_structured` re-raises
    :class:`~llmkit.exceptions.OutputLimitError` at the boundary.

    ``reraise=True`` matches instructor's own int path exactly. It changes only
    the cause chain, not whether instructor wraps: any non-``IncompleteOutput``
    exception escaping this loop — a declined transport/permanent error (**1**
    call) or an exhausted parse re-ask (**2** calls) — is caught by
    instructor's blanket ``except Exception as e: raise
    InstructorRetryException(...) from e``, so the escaping type is always
    ``InstructorRetryException``. What ``reraise`` controls is ``__cause__``:
    the raw last error (``reraise=True``) rather than a tenacity ``RetryError``
    hop. The wrap shape is therefore uniform — ``__cause__`` is the raw provider
    error on the declined path and the raw last parse error on the exhausted
    path — and :func:`~llmkit.exceptions.underlying_provider_error` unwraps both.

    Built fresh per call — tenacity retrying objects mutate internal iteration
    state, so a shared module-level instance would race across concurrent
    structured calls on the same loop.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(2),
        retry=retry_if_exception(
            lambda e: (
                isinstance(e, REPAIRABLE_PARSE_ERRORS)
                and not isinstance(e, IncompleteOutputException)
            )
        ),
        reraise=True,
    )


def _coerce_text_content(content: object) -> str:
    """Coerce a completion's ``message.content`` to a single string.

    Most providers return a plain ``str`` for a text completion, but some
    return a *list* of content blocks — dicts shaped like
    ``{"type": "text", "text": "..."}`` (or SDK objects exposing a ``text``
    attribute). This concatenates the text of every such block so
    :func:`acompletion_text` always honours its ``str`` return annotation
    (and the README's "coerces provider list-content blocks" promise), and
    degrades to ``""`` for ``None``/empty. Best-effort: blocks without
    string text (images, thinking, unknown shapes) are skipped rather than
    breaking the call.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in cast("list[object]", content):
            text = (
                cast("dict[str, object]", block).get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _chunk_delta_text(chunk: ModelResponseStream) -> str | None:
    """Extract the first choice's textual delta from a stream chunk.

    LiteLLM assigns ``StreamingChoices.delta`` inside ``__init__`` rather than
    as an annotated class attribute, so the static type of
    ``chunk.choices[0].delta`` is ``Unknown``. We narrow that hop to its real
    runtime type here so the streaming loop reads a precise ``str | None`` —
    the one place that knows the litellm stream object's shape (raw-llm).

    ``Delta.content`` needed the same treatment until litellm 1.95.0 annotated
    it as ``str | None``; the cast is now redundant and ``reportUnnecessaryCast``
    rejects it.
    """
    if not chunk.choices:
        # Metadata-only / keepalive frames carry no choice to index — e.g. the
        # Gemini provider emits ``ModelResponseStream(choices=[])`` for frames
        # with no candidate. Return ``None`` so the streaming loop's ``if delta:``
        # skips the frame, rather than raising ``IndexError`` and killing the
        # stream mid-flight (the error is not retried once content has yielded).
        return None
    choice: StreamingChoices = chunk.choices[0]
    delta: Delta = choice.delta
    return delta.content


def _call_context(
    kind: CallKind,
    prompt: str | Sequence[ChatMessage],
    messages: Sequence[ChatMessage],
    provider: LLMProviderInterface,
    *,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
    feature: str | None,
    label: str | None,
    call_id: str | None,
    attempt: int | None,
    output_schema: type[BaseModel] | None = None,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice | None = None,
) -> CallContext:
    """The :class:`CallContext` a transport attaches to its wire request."""
    return CallContext(
        kind=kind,
        prompt=prompt,
        messages=messages,
        model=model,
        provider=provider.name,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
        output_schema=output_schema,
        tools=tools,
        tool_choice=tool_choice,
    )


async def acompletion_structured[T: BaseModel](
    prompt: str | Sequence[ChatMessage],
    output_schema: type[T],
    *,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
    feature: str | None = None,
    label: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> tuple[T, float | None, tuple[int | None, int | None, int | None]]:
    """Structured completion via instructor pinned to the provider's mode.

    Uses ``create_with_completion`` so the parsed model *and* the raw
    completion (for cost) are both in hand. instructor's in-call schema-repair
    budget is pinned to two total attempts (one schema-repair re-ask,
    deliberately low) and restricted to genuine *parse* failures — a transport
    or permanent provider error is **not** re-asked in-call (it would only be a
    blind, backoff-free re-send inside the rate-limiter slot; the cross-call
    layer handles it). A completion truncated by the output-token limit
    (``finish_reason='length'``) is likewise **never** re-asked, because an
    identical token budget can only truncate again; it surfaces immediately as
    :class:`~llmkit.exceptions.OutputLimitError` carrying the truncated
    attempt's diagnostics (see :func:`_schema_repair_retrying`).
    The budget is kept separate from the transient-error retry layer
    (``with_retries`` in :mod:`llmkit.retry`), which owns the cross-call
    429/503/5xx and schema-validation budgets — and which likewise never
    retries an ``OutputLimitError``.

    ``max_tokens`` caps the completion length when set; it is only included
    in the underlying call when not ``None``, so the default produces a
    request byte-identical to the prior behaviour (no ``max_tokens`` key).
    LiteLLM accepts ``max_tokens`` as a standard cross-provider kwarg and
    instructor forwards extra kwargs to the wrapped completion, so no
    per-provider branching is needed.

    ``temperature`` is forwarded the same way: only when not ``None``. A
    resolved ``None`` — llmkit's :data:`~llmkit.DEFAULT_TEMPERATURE` — sends
    no ``temperature`` key at all, so the provider's default sampling applies
    (Google is retiring sampling parameters on Gemini). An explicit number is
    forwarded; a refusal of it as unsupported for the model is absorbed by
    :func:`_acompletion`, which re-issues the request once without it.

    ``reasoning_effort`` controls provider thinking/reasoning tokens (e.g.
    ``"disable"`` turns Gemini thinking off). When ``None``, the provider's
    configured :pyattr:`~llmkit.providers.LLMProviderInterface.reasoning_effort`
    (from :class:`~llmkit.LLMClientConfig`) is used; an explicit value here
    overrides it for this call. The kwarg is only included when the resolved
    value is not ``None``, so the default request is byte-identical to before.

    ``provider`` overrides the configured provider for this call only
    (``None`` uses the globally-configured one) — the seam that lets a
    single call route through a different provider family without changing
    the app-wide registration.

    Returns ``(parsed, approximate_cost, usage)``, where ``usage`` is the
    completion's ``(prompt_tokens, completion_tokens, total_tokens)`` from
    :class:`~llmkit._transport.WireUsage` (each ``None`` when not reported) for the call log.
    """
    provider = provider if provider is not None else build_provider()
    effort = resolve_reasoning_effort(reasoning_effort, provider)
    litellm_model = provider.litellm_model(model)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    # The completion callable is the library's seam under instructor: it sends
    # through the transport boundary, and providers that opt in via
    # ``strict_json_schema`` (OpenRouter) get their ``response_format``
    # upgraded to strict enforcement on the way. ``inspect.iscoroutinefunction``
    # sees through the partial, so instructor still builds its async client.
    messages = wire_messages(prompt)
    call = _call_context(
        "structured",
        prompt,
        messages,
        provider,
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
        output_schema=output_schema,
    )
    completion = functools.partial(
        _instructor_completion,
        strict=provider.strict_json_schema,
        call=call,
        invocations=_Invocations(),
    )
    # instructor's ``from_litellm`` is overloaded on whether the completion is
    # sync or async; basedpyright has been seen to resolve the *sync* overload
    # for our async completion callable on some platforms (CI x86_64 vs local
    # arm64), which then types ``create_with_completion`` as a non-awaitable
    # ``tuple`` and breaks the ``await`` below. Pin the async client across
    # platforms by casting through ``object`` — the escape pyright itself
    # recommends when the two inferred types don't overlap, and which is neither
    # a redundant cast (on the async platform) nor an invalid one (on the sync
    # platform). raw-llm — the instructor boundary.
    client = cast(
        "instructor.AsyncInstructor",
        cast("object", instructor.from_litellm(completion, mode=provider.instructor_mode)),
    )
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        try:
            result = await client.chat.completions.create_with_completion(
                model=litellm_model,
                messages=messages,  # pyright: ignore[reportArgumentType]  # raw-llm — instructor over-strict ChatCompletionMessageParam
                response_model=output_schema,
                # instructor's in-call schema-repair budget: two total attempts
                # = exactly one schema-repair re-ask, and only for a genuine
                # parse failure — never for a length truncation, a transport, or
                # a permanent error (see _schema_repair_retrying). Built per
                # call — tenacity retrying objects mutate iteration state. The
                # cross-call transient/validation budgets live in llmkit.retry.
                max_retries=_schema_repair_retrying(),
                **request_kwargs,  # pyright: ignore[reportArgumentType]  # raw-llm — provider-owned credential/native kwargs
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                # Gate temperature like max_tokens / reasoning_effort: a
                # ``None`` resolved value sends no ``temperature`` key at all
                # (not an explicit ``None`` kwarg), so the provider's default
                # sampling applies — see options.resolve_call_args. The
                # identity check keeps ``0.0`` a real, forwarded value.
                **({"temperature": temperature} if temperature is not None else {}),
                # One schema, one tool call: on the Claude routes a schema can
                # travel as a synthetic tool the model may call twice, splitting
                # the answer (see BaseProvider._single_tool_call_per_schema).
                # Gated like the two above, so every other provider's request is
                # unchanged; never sent by the tool-calling functions.
                **({"parallel_tool_calls": False} if provider.single_tool_call_per_schema else {}),
            )
        except IncompleteOutputException as e:
            # The declined re-ask propagates instructor's exception bare (no
            # InstructorRetryException wrap) — re-own it at the boundary so
            # callers see one llmkit type carrying what the truncated
            # completion reported (partial text, finish reason, token counts)
            # for the caller and the call log to judge — a truncation alone
            # does not say whether the budget or the output was at fault —
            # with the instructor original on the cause chain.
            raise OutputLimitError(
                model=litellm_model,
                max_tokens=max_tokens,
                completion_tokens=_completion_tokens(e.last_completion),
                diagnostics=_output_limit_diagnostics(e.last_completion),
            ) from e
        # instructor types the raw completion half of the tuple as Any. It is
        # the last rendered response, whose usage instructor's tally has
        # rewritten to the whole in-call total, so it is read back rather than
        # the last wire response.
        parsed, raw_completion = cast("tuple[T, object]", result)
        response = _wire_response(raw_completion)
        slot.record_tokens(response.usage.debit())
    return parsed, response.cost, response.usage.counts()


async def acompletion_text(
    prompt: str | Sequence[ChatMessage],
    *,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
    feature: str | None = None,
    label: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> tuple[str, float | None, tuple[int | None, int | None, int | None]]:
    """Plain-text completion via LiteLLM.

    ``reasoning_effort`` controls provider thinking tokens, resolved against
    the provider's configured value when ``None`` (an explicit value
    overrides it); the kwarg is only forwarded when the resolved value is
    not ``None``. ``temperature`` is forwarded only when not ``None`` — a
    resolved ``None`` sends no ``temperature`` key at all (the provider's
    default sampling applies). ``provider`` overrides the configured
    provider for this call only (``None`` uses the globally-configured one).

    Returns ``(text, approximate_cost, usage)``, ``usage`` being the
    :class:`~llmkit._transport.WireUsage` triple the call log records. The text is the first
    choice's message content coerced to a string via
    :func:`_coerce_text_content` (list content blocks joined; an empty string
    when the provider returns none).
    """
    provider = provider if provider is not None else build_provider()
    effort = resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    litellm_model = provider.litellm_model(model)
    messages = wire_messages(prompt)
    call = _call_context(
        "text",
        prompt,
        messages,
        provider,
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
    )
    request = WireRequest(
        kwargs={
            "model": litellm_model,
            "messages": messages,
            # Gate temperature like max_tokens / reasoning_effort: a
            # ``None`` resolved value sends no ``temperature`` key at all,
            # so the provider's default sampling applies. Identity check
            # keeps ``0.0`` a real, forwarded value.
            **({"temperature": temperature} if temperature is not None else {}),
            **request_kwargs,
            # Gate max_tokens like the structured/stream paths: an unset cap
            # sends no ``max_tokens`` key at all (not an explicit ``None``), so
            # the request stays byte-identical to prior behaviour. Creds are
            # splatted first to match the structured/stream helpers exactly.
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        },
        call=call,
    )
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        response = await _send(request)
        slot.record_tokens(response.usage.debit())
    # A response with no choice, or no content, is an empty answer.
    text = response.text if response.text is not None else ""
    return text, response.cost, response.usage.counts()


async def acompletion_tools(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    tool_choice: ToolChoice | None,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
    response_format: dict[str, object] | None = None,
    output_schema: type[BaseModel] | None = None,
    feature: str | None = None,
    label: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> tuple[
    str | None, list[object], str | None, tuple[int | None, int | None, int | None], float | None
]:
    """One raw-LiteLLM tool completion, with the usual limiter and TPM debit."""
    provider = provider if provider is not None else build_provider()
    if tool_choice is not None and not getattr(provider, "supports_tool_choice", True):
        raise ValueError(f"{provider.name} does not support tool_choice on this route")
    effort = resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    if response_format is not None and provider.strict_json_schema:
        _stricten_json_schema_response_format({"response_format": response_format})
    choice: object = tool_choice
    if isinstance(tool_choice, ToolName):
        choice = {"type": "function", "function": {"name": tool_choice.value}}
    litellm_model = provider.litellm_model(model)
    messages = wire_messages(prompt)
    call = _call_context(
        "tool",
        prompt,
        messages,
        provider,
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
        output_schema=output_schema,
        tools=tools,
        tool_choice=tool_choice,
    )
    request = WireRequest(
        kwargs={
            "model": litellm_model,
            "messages": messages,
            "tools": [definition.to_litellm() for definition in tools],
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tool_choice": choice} if choice is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        },
        call=call,
    )
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        response = await _send(request)
        slot.record_tokens(response.usage.debit())
    return (
        response.text,
        list[object](response.tool_calls),
        response.finish_reason,
        response.usage.counts(),
        response.cost,
    )


@dataclass
class _ToolCallFragments:
    """One streamed tool call, accumulated across chunks by its ``index``.

    LiteLLM's Anthropic and Gemini adapters both keep a *cumulative* index over
    a turn's tool calls, so the index is the join key: the id and the function
    name arrive once (usually on the fragment that opens the call) and the
    argument string arrives in one or more pieces that concatenate in order.
    """

    id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _StreamedFunction:
    name: str | None
    arguments: str


@dataclass(frozen=True)
class _StreamedToolCall:
    """An assembled tool call in the attribute shape the parser narrows.

    :func:`~llmkit.calls._shared.parse_tool_calls` reads ``.id``,
    ``.function.name`` and ``.function.arguments`` off whatever the transport
    hands it, and rejects anything that is not a ``str`` as a malformed call.
    Assembling into this shape rather than a dict is what lets the streamed and
    buffered lanes share one parser — and keeps a call whose id or name never
    arrived an honest ``ToolArgumentError`` rather than a silent default.
    """

    id: str | None
    function: _StreamedFunction


@dataclass(frozen=True)
class StreamedToolTurn:
    """Everything but the prose from a streamed tool turn.

    The final item :func:`astream_tools` yields, after the last text delta:
    an async generator cannot ``return`` a value, so the assembled turn is a
    terminal *yield* of a distinct type. Mirrors what
    :func:`acompletion_tools` returns as a tuple.
    """

    raw_calls: list[object]
    stop_reason: str | None
    usage: tuple[int | None, int | None, int | None]
    approximate_cost: float | None


def _chunk_tool_fragments(chunk: ModelResponseStream) -> tuple[WireToolFragment, ...]:
    """The first choice's tool-call deltas as :class:`WireToolFragment`\\ s.

    Deliberately total-getattr rather than typed: LiteLLM annotates
    ``Delta.tool_calls`` loosely and providers differ in which fields a given
    fragment carries (Gemini sends a complete argument JSON in one fragment;
    Anthropic and the OpenAI-compatible routes split it). Anything missing or
    empty is ``None`` rather than defaulted, so a call that never received an
    id or a name stays malformed for the parser to reject.
    """
    if not chunk.choices:
        return ()
    delta: Delta = chunk.choices[0].delta
    raw_calls = cast("object", getattr(delta, "tool_calls", None))
    if not isinstance(raw_calls, list):
        return ()
    fragments: list[WireToolFragment] = []
    for position, raw in enumerate(cast("list[object]", raw_calls)):
        index = getattr(raw, "index", None)
        function = getattr(raw, "function", None)
        fragments.append(
            WireToolFragment(
                # A provider that omits ``index`` on a single-call turn falls
                # back to the fragment's position within this chunk, which is
                # the same key for the common one-call case and never merges
                # two distinct calls.
                index=index if isinstance(index, int) else position,
                id=_str_or_none(getattr(raw, "id", None)) or None,
                name=_str_or_none(getattr(function, "name", None)) or None,
                arguments=_str_or_none(getattr(function, "arguments", None)) or None,
            )
        )
    return tuple(fragments)


def _absorb_tool_fragments(
    fragments: tuple[WireToolFragment, ...], accumulated: dict[int, _ToolCallFragments]
) -> None:
    """Merge one chunk's tool-call fragments into the per-index accumulator.

    The id and the name are kept from the first fragment that carries them;
    argument pieces concatenate in arrival order.
    """
    for piece in fragments:
        fragment = accumulated.setdefault(piece.index, _ToolCallFragments())
        if piece.id is not None and fragment.id is None:
            fragment.id = piece.id
        if piece.name is not None and fragment.name is None:
            fragment.name = piece.name
        if piece.arguments is not None:
            fragment.arguments.append(piece.arguments)


def _assemble_tool_calls(fragments: dict[int, _ToolCallFragments]) -> list[object]:
    """Close the accumulator into parser-shaped calls, in provider order."""
    return [
        _StreamedToolCall(
            id=fragment.id,
            function=_StreamedFunction(
                name=fragment.name,
                # A tool that takes no arguments streams no argument fragments
                # on some routes, where the buffered lane receives ``"{}"``.
                # Normalising here keeps the two lanes' parse results identical;
                # a *truncated* argument stream is not affected, since partial
                # JSON is non-empty and still fails to parse.
                arguments="".join(fragment.arguments) or "{}",
            ),
        )
        for _index, fragment in sorted(fragments.items())
    ]


def _chunk_finish_reason(chunk: ModelResponseStream) -> str | None:
    """The first choice's ``finish_reason``, or ``None`` on a choice-less frame."""
    if not chunk.choices:
        return None
    reason = cast("object", getattr(chunk.choices[0], "finish_reason", None))
    return reason if isinstance(reason, str) else None


async def astream_tools(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    tool_choice: ToolChoice | None,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
    feature: str | None = None,
    label: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> AsyncGenerator[str | StreamedToolTurn]:
    """Stream a tool-enabled turn: text deltas, then one :class:`StreamedToolTurn`.

    The streaming sibling of :func:`acompletion_tools`, sharing its
    ``tool_choice`` guard: a route whose provider declares no ``tool_choice``
    support raises before any request, so the streaming surface cannot
    silently accept what the buffered one rejects.

    Unlike :func:`astream_text` this **does** request
    ``stream_options={"include_usage": True}``, so the turn's token usage
    reaches the terminal :class:`~llmkit.ToolCallResult`, the log record's
    ``usage`` field, and the TPM limiter's debit — the tool lane records usage
    and a streamed tool turn should not be the one that does not.

    The rate-limit slot is released **before** the terminal yield: a consumer
    parked on the assembled result (the idiomatic stopping point) is running
    its own tools, and holding a per-provider concurrency slot across that is
    exactly the deadlock the limiter exists to avoid.
    """
    provider = provider if provider is not None else build_provider()
    if tool_choice is not None and not getattr(provider, "supports_tool_choice", True):
        raise ValueError(f"{provider.name} does not support tool_choice on this route")
    effort = resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    choice: object = tool_choice
    if isinstance(tool_choice, ToolName):
        choice = {"type": "function", "function": {"name": tool_choice.value}}
    litellm_model = provider.litellm_model(model)
    messages = wire_messages(prompt)
    call = _call_context(
        "tool_stream",
        prompt,
        messages,
        provider,
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
        tools=tools,
        tool_choice=tool_choice,
    )
    request = WireRequest(
        kwargs={
            "model": litellm_model,
            "messages": messages,
            "tools": [definition.to_litellm() for definition in tools],
            **({"tool_choice": choice} if choice is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            "stream": True,
            "stream_options": {"include_usage": True},
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        },
        call=call,
    )
    fragments: dict[int, _ToolCallFragments] = {}
    stop_reason: str | None = None
    usage: WireUsage | None = None
    cost: float | None = None
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        async with aclosing(_stream(request)) as stream:
            async for item in stream:
                if isinstance(item, WireStreamEnd):
                    # Usage on a frame wins; the stream's own accumulation is
                    # the fallback.
                    if usage is None:
                        usage = item.usage
                    if cost is None:
                        cost = item.cost
                    continue
                _absorb_tool_fragments(item.tool_fragments, fragments)
                if item.finish_reason is not None:
                    stop_reason = item.finish_reason
                # ``include_usage`` puts usage on a *final* frame that usually
                # carries no choices; keep the last frame that has any.
                if item.usage is not None:
                    usage = item.usage
                if item.cost is not None:
                    cost = item.cost
                if item.text:
                    yield item.text
        usage = usage if usage is not None else WireUsage()
        slot.record_tokens(usage.debit())
    yield StreamedToolTurn(
        raw_calls=_assemble_tool_calls(fragments),
        stop_reason=stop_reason,
        usage=usage.counts(),
        approximate_cost=cost,
    )


async def astream_text(
    prompt: str | Sequence[ChatMessage],
    *,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
    feature: str | None = None,
    label: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> AsyncIterator[str]:
    """Stream plain-text deltas via LiteLLM.

    Yields each chunk's textual delta as it arrives. The rate-limit slot
    is held for the lifetime of the stream. ``provider`` overrides the
    configured provider for this call only (``None`` uses the global one).

    ``max_tokens`` caps the streamed completion length and ``reasoning_effort``
    controls provider thinking tokens — parity with the non-streaming text and
    structured paths. Each is only forwarded when set (``reasoning_effort``
    resolved against the provider's configured value when ``None``), so the
    default request is byte-identical to the prior stream call. ``temperature``
    is forwarded only when not ``None`` — a resolved ``None`` sends no
    ``temperature`` key at all (the provider's default sampling applies).
    """
    provider = provider if provider is not None else build_provider()
    effort = resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    litellm_model = provider.litellm_model(model)
    messages = wire_messages(prompt)
    call = _call_context(
        "text_stream",
        prompt,
        messages,
        provider,
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        feature=feature,
        label=label,
        call_id=call_id,
        attempt=attempt,
    )
    request = WireRequest(
        kwargs={
            "model": litellm_model,
            "messages": messages,
            # Gate temperature like max_tokens / reasoning_effort: a
            # ``None`` resolved value sends no ``temperature`` key at all,
            # so the provider's default sampling applies. Identity check
            # keeps ``0.0`` a real, forwarded value.
            **({"temperature": temperature} if temperature is not None else {}),
            "stream": True,
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        },
        call=call,
    )
    async with (
        GlobalRateLimiter.acquire_async(provider.name) as slot,
        aclosing(_stream(request)) as stream,
    ):
        async for item in stream:
            if isinstance(item, WireStreamEnd):
                # Best-effort TPM accounting: a stream usually reports no
                # usage (we don't request stream_options=include_usage), so
                # this is typically a no-op — consistent with cost being None
                # for streamed calls. When the stream does expose a final
                # usage, debit it before releasing the slot.
                slot.record_tokens(item.usage.debit())
            elif item.text:
                yield item.text
