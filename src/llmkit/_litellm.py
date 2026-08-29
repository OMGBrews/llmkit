"""Internal LiteLLM call layer.

The single place that talks to LiteLLM (and, for structured output,
``instructor`` over LiteLLM). The public call functions in
:mod:`llmkit.calls` build/log :class:`LLMCallRecord`s
around these helpers; this module owns provider routing, the rate-limit
semaphore, structured-output mode pinning, and best-effort cost extraction.

It is also the **test seam**: unit tests patch these entry points
(``acompletion_structured`` / ``acompletion_text`` / ``acompletion_tools`` /
``astream_text`` / ``astream_tools``) so the real call-function bodies —
logging, retry, content coercion — still run over a faked provider response
(see ``tests/_support`` ``patch_llm``).

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
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import cast

import instructor
import litellm
from instructor.core.exceptions import IncompleteOutputException
from litellm import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from llmkit._types import TOOL_ERROR_PREFIX, ChatMessage, ReasoningEffort
from llmkit.exceptions import (
    REPAIRABLE_PARSE_ERRORS,
    OutputLimitError,
    normalize_service_unavailable,
)
from llmkit.providers import LLMProviderInterface, build_provider
from llmkit.rate_limiting import GlobalRateLimiter
from llmkit.tools import ToolChoice, ToolDefinition, ToolName

logger = logging.getLogger(__name__)


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
    """
    acompletion = cast(
        "Callable[..., Coroutine[object, object, ModelResponse | CustomStreamWrapper]]",
        litellm.acompletion,
    )
    try:
        return await acompletion(**kwargs)
    except Exception as e:
        normalized = normalize_service_unavailable(e)
        if normalized is not e:
            raise normalized from e
        raise


async def _acompletion_strict_json_schema(
    **kwargs: object,
) -> ModelResponse | CustomStreamWrapper:
    """:func:`_acompletion`, upgrading a ``json_schema`` ``response_format`` to strict.

    instructor's ``JSON_SCHEMA`` handler emits ``response_format`` without
    ``"strict": true`` or ``additionalProperties: false``, and a provider that
    treats a non-strict json_schema as *advisory* (OpenRouter) then lets weak
    models drift — measured 2026-07-14: ``mistralai/mistral-nemo``
    stochastically echoing the schema itself. The pre-1.15.3
    ``OPENROUTER_STRUCTURED_OUTPUTS`` handler sent exactly this strict shape,
    so this wrapper restores the measured wire contract at the one seam the
    library owns (the completion callable handed to ``from_litellm``). Used
    only for providers that opt in via
    :pyattr:`~llmkit.providers.base.BaseProvider.strict_json_schema`; the
    mutation targets the request dict instructor builds fresh per call.
    """
    _stricten_json_schema_response_format(kwargs)
    return await _acompletion(**kwargs)


def _stricten_json_schema_response_format(kwargs: dict[str, object]) -> None:
    """Mutate a JSON-schema response format into the strict wire shape."""
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


def _wire_message(message: ChatMessage) -> ChatMessage:
    """Render llmkit's own message extensions into what LiteLLM can send.

    Today that is exactly one: :class:`~llmkit.ToolResultMessage`'s optional
    ``is_error``. It has no wire equivalent — LiteLLM's Anthropic translation
    builds its ``tool_result`` block from the id and content alone (measured
    against litellm 1.95.0, the declared floor), and an OpenAI-compatible route
    would forward the unknown key to a provider that may reject it — so the
    flag becomes a :data:`~llmkit._types.TOOL_ERROR_PREFIX` prefix on the
    content, which every provider receives identically, and the key is dropped.

    Returns a **new** dict for a message it rewrites: the caller's history is
    theirs, is what the log records as ``prompt``, and is fed back into the
    next turn — flattening it in place would strip the flag from the record and
    double the prefix on the following call.
    """
    if message["role"] != "tool" or not message.get("is_error", False):
        return message
    return {
        "role": "tool",
        "tool_call_id": message["tool_call_id"],
        "content": f"{TOOL_ERROR_PREFIX}{message['content']}",
    }


def _messages(prompt: str | Sequence[ChatMessage]) -> list[ChatMessage]:
    """Normalise a prompt into LiteLLM's message-list shape."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [_wire_message(message) for message in prompt]


def _resolve_reasoning_effort(
    override: ReasoningEffort | None, provider: LLMProviderInterface
) -> ReasoningEffort | None:
    """Resolve the effective reasoning effort for a call.

    A per-call ``override`` wins when set; otherwise the provider's
    configured value (from :class:`~llmkit.LLMClientConfig`) applies. Both
    ``None`` means no reasoning kwarg is forwarded — byte-identical to the
    pre-feature request. ``getattr`` keeps third-party providers that predate
    the ``reasoning_effort`` property working (they degrade to ``None``).
    """
    if override is not None:
        return override
    return getattr(provider, "reasoning_effort", None)


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


def _total_tokens(raw: object) -> int | None:
    """Best-effort total token count for a completion from its ``usage``.

    LiteLLM stamps a ``usage`` object (``prompt_tokens`` / ``completion_tokens``
    / ``total_tokens``) onto a non-streamed completion. This reads
    ``usage.total_tokens`` for the tokens-per-minute limiter to debit, falling
    back to summing ``prompt_tokens`` + ``completion_tokens`` (treating a
    missing one as 0) when ``total_tokens`` is absent or non-int. Returns
    ``None`` for any missing/odd shape (e.g. a streamed response that reports no
    usage) — TPM accounting is best-effort and never breaks the call.
    """
    usage = getattr(raw, "usage", None)
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int):
        return total
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if isinstance(prompt, int) or isinstance(completion, int):
        return (prompt if isinstance(prompt, int) else 0) + (
            completion if isinstance(completion, int) else 0
        )
    return None


def _completion_tokens(raw: object) -> int | None:
    """Best-effort completion-token count from a completion's ``usage``.

    Sibling of :func:`_total_tokens`, for :class:`OutputLimitError`'s
    diagnostics: how many tokens the provider actually generated before the
    output-token limit cut the completion off. Returns ``None`` for any
    missing/odd shape — the diagnostics degrade, never break the raise.
    """
    tokens = getattr(getattr(raw, "usage", None), "completion_tokens", None)
    return tokens if isinstance(tokens, int) else None


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
    rejects it. The declared floor is older (``litellm>=1.87.1``), but only the
    newest-resolution ``check`` job type-checks, so the floor is unaffected.
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


async def acompletion_structured[T: BaseModel](
    prompt: str | Sequence[ChatMessage],
    output_schema: type[T],
    *,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
) -> tuple[T, float | None]:
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
    resolved ``None`` sends no ``temperature`` key at all (the provider's
    default sampling applies) — the escape hatch from llmkit's
    :data:`~llmkit.DEFAULT_TEMPERATURE` for providers whose guidance says to
    omit the field (Gemini 3.x deprecates it).

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

    Returns ``(parsed, approximate_cost)``.
    """
    provider = provider if provider is not None else build_provider()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    litellm_model = provider.litellm_model(model)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    # The completion callable is the library's seam under instructor: providers
    # that opt in via ``strict_json_schema`` (OpenRouter) get their
    # ``response_format`` upgraded to strict enforcement on the way to LiteLLM.
    completion = _acompletion_strict_json_schema if provider.strict_json_schema else _acompletion
    # instructor's ``from_litellm`` is overloaded on whether the completion is
    # sync or async; basedpyright has been seen to resolve the *sync* overload
    # for our ``async def _acompletion`` on some platforms (CI x86_64 vs local
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
                messages=_messages(prompt),  # pyright: ignore[reportArgumentType]  # raw-llm — instructor over-strict ChatCompletionMessageParam
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
            )
        except IncompleteOutputException as e:
            # The declined re-ask propagates instructor's exception bare (no
            # InstructorRetryException wrap) — re-own it at the boundary so
            # callers see one llmkit type carrying the diagnostics that make
            # the fix legible ("cap too snug" vs "prompt causes runaway
            # output"), with the instructor original on the cause chain.
            raise OutputLimitError(
                model=litellm_model,
                max_tokens=max_tokens,
                completion_tokens=_completion_tokens(e.last_completion),
            ) from e
        # instructor types the raw completion half of the tuple as Any; it is a
        # litellm ModelResponse. Narrow once so cost/usage read a real type.
        parsed, completion = cast("tuple[T, ModelResponse]", result)
        slot.record_tokens(_total_tokens(completion))
    return parsed, _response_cost(completion)


async def acompletion_text(
    prompt: str | Sequence[ChatMessage],
    *,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    provider: LLMProviderInterface | None = None,
) -> tuple[str, float | None]:
    """Plain-text completion via LiteLLM.

    ``reasoning_effort`` controls provider thinking tokens, resolved against
    the provider's configured value when ``None`` (an explicit value
    overrides it); the kwarg is only forwarded when the resolved value is
    not ``None``. ``temperature`` is forwarded only when not ``None`` — a
    resolved ``None`` sends no ``temperature`` key at all (the provider's
    default sampling applies). ``provider`` overrides the configured
    provider for this call only (``None`` uses the globally-configured one).

    Returns ``(text, approximate_cost)``. The text is the first choice's
    message content coerced to a string via :func:`_coerce_text_content`
    (list content blocks joined; an empty string when the provider returns
    none).
    """
    provider = provider if provider is not None else build_provider()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
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
        )
        # Non-streaming acompletion returns a ModelResponse (the stub's union
        # also admits CustomStreamWrapper, only reachable with stream=True);
        # narrow it so the typed .choices[0].message.content chain (str | None)
        # and .usage are precise.
        response = cast("ModelResponse", resp)
        slot.record_tokens(_total_tokens(response))
    # Guard the index: a degenerate empty-``choices`` response (e.g. a malformed
    # OpenAI-compatible proxy returning ``{"choices": []}``) would otherwise raise
    # ``IndexError`` here, bypassing the ``None`` -> "" coercion below.
    content = response.choices[0].message.content if response.choices else None
    return _coerce_text_content(content), _response_cost(response)


def _usage_counts(response: object) -> tuple[int | None, int | None, int | None]:
    """Extract portable usage fields without trusting a provider-specific model.

    Typed ``object`` rather than ``ModelResponse`` because it is pure
    ``getattr``: the streaming lane hands it a ``ModelResponseStream`` frame or
    the stream wrapper itself, neither of which is a ``ModelResponse``.
    """
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
        total if isinstance(total, int) else None,
    )


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
) -> tuple[
    str | None, list[object], str | None, tuple[int | None, int | None, int | None], float | None
]:
    """One raw-LiteLLM tool completion, with the usual limiter and TPM debit."""
    provider = provider if provider is not None else build_provider()
    if tool_choice is not None and not getattr(provider, "supports_tool_choice", True):
        raise ValueError(f"{provider.name} does not support tool_choice on this route")
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    if response_format is not None and provider.strict_json_schema:
        _stricten_json_schema_response_format({"response_format": response_format})
    choice: object = tool_choice
    if isinstance(tool_choice, ToolName):
        choice = {"type": "function", "function": {"name": tool_choice.value}}
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            tools=[definition.to_litellm() for definition in tools],
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tool_choice": choice} if choice is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        )
        response = cast("ModelResponse", resp)
        slot.record_tokens(_total_tokens(response))
    if not response.choices:
        return None, [], None, _usage_counts(response), _response_cost(response)
    message = response.choices[0].message
    content = cast("object", getattr(message, "content", None))
    raw_calls = cast("object", getattr(message, "tool_calls", None))
    return (
        _coerce_text_content(content) if content is not None else None,
        cast("list[object]", raw_calls) if isinstance(raw_calls, list) else [],
        getattr(response.choices[0], "finish_reason", None),
        _usage_counts(response),
        _response_cost(response),
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


def _absorb_tool_fragments(
    chunk: ModelResponseStream, fragments: dict[int, _ToolCallFragments]
) -> None:
    """Merge one chunk's tool-call deltas into the per-index accumulator.

    Deliberately total-getattr rather than typed: LiteLLM annotates
    ``Delta.tool_calls`` loosely and providers differ in which fields a given
    fragment carries (Gemini sends a complete argument JSON in one fragment;
    Anthropic and the OpenAI-compatible routes split it). Anything missing is
    skipped rather than defaulted, so a call that never received an id or a
    name stays malformed for the parser to reject.
    """
    if not chunk.choices:
        return
    delta: Delta = chunk.choices[0].delta
    raw_calls = cast("object", getattr(delta, "tool_calls", None))
    if not isinstance(raw_calls, list):
        return
    for position, raw in enumerate(cast("list[object]", raw_calls)):
        index = getattr(raw, "index", None)
        # A provider that omits ``index`` on a single-call turn falls back to
        # the fragment's position within this chunk, which is the same key for
        # the common one-call case and never merges two distinct calls.
        key = index if isinstance(index, int) else position
        fragment = fragments.setdefault(key, _ToolCallFragments())
        call_id = getattr(raw, "id", None)
        if isinstance(call_id, str) and call_id and fragment.id is None:
            fragment.id = call_id
        function = getattr(raw, "function", None)
        name = getattr(function, "name", None)
        if isinstance(name, str) and name and fragment.name is None:
            fragment.name = name
        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str) and arguments:
            fragment.arguments.append(arguments)


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
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    choice: object = tool_choice
    if isinstance(tool_choice, ToolName):
        choice = {"type": "function", "function": {"name": tool_choice.value}}
    fragments: dict[int, _ToolCallFragments] = {}
    stop_reason: str | None = None
    usage_source: object = None
    cost: float | None = None
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            tools=[definition.to_litellm() for definition in tools],
            **({"tool_choice": choice} if choice is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        )
        stream = cast("CustomStreamWrapper", resp)
        try:
            async for chunk in stream:
                _absorb_tool_fragments(chunk, fragments)
                reason = _chunk_finish_reason(chunk)
                if reason is not None:
                    stop_reason = reason
                # ``include_usage`` puts usage on a *final* frame that usually
                # carries no choices; keep the last frame that has any, and
                # fall back to the wrapper's own accumulation below.
                if getattr(chunk, "usage", None) is not None:
                    usage_source = chunk
                chunk_cost = _response_cost(chunk)
                if chunk_cost is not None:
                    cost = chunk_cost
                delta = _chunk_delta_text(chunk)
                if delta:
                    yield delta
        except Exception as e:
            # Same re-ownership as ``astream_text``: a 503 can surface
            # mid-iteration, after ``_acompletion`` has already returned.
            normalized = normalize_service_unavailable(e)
            if normalized is not e:
                raise normalized from e
            raise
        if usage_source is None:
            usage_source = stream
        slot.record_tokens(_total_tokens(usage_source))
        if cost is None:
            cost = _response_cost(stream)
    yield StreamedToolTurn(
        raw_calls=_assemble_tool_calls(fragments),
        stop_reason=stop_reason,
        usage=_usage_counts(usage_source),
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
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    request_kwargs = _completion_request_kwargs(provider, effort, model)
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            # Gate temperature like max_tokens / reasoning_effort: a
            # ``None`` resolved value sends no ``temperature`` key at all,
            # so the provider's default sampling applies. Identity check
            # keeps ``0.0`` a real, forwarded value.
            **({"temperature": temperature} if temperature is not None else {}),
            stream=True,
            **request_kwargs,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        )
        # stream=True makes acompletion return a CustomStreamWrapper, whose
        # async iteration yields typed ModelResponseStream chunks.
        stream = cast("CustomStreamWrapper", resp)
        try:
            async for chunk in stream:
                delta = _chunk_delta_text(chunk)
                if delta:
                    yield delta
        except Exception as e:
            # A 503 can also surface mid-iteration (the wrapper fetches chunks
            # lazily, after ``_acompletion`` has returned) — re-own it exactly
            # as the call-time boundary does, so the stream path keeps the
            # ``except LLM_RECOVERABLE_ERRORS`` contract too.
            normalized = normalize_service_unavailable(e)
            if normalized is not e:
                raise normalized from e
            raise
        # Best-effort TPM accounting: a stream usually reports no usage (we
        # don't request stream_options=include_usage), so this is typically a
        # no-op — consistent with cost being None for streamed calls. When the
        # wrapper does expose a final usage, debit it before releasing the slot.
        slot.record_tokens(_total_tokens(stream))
