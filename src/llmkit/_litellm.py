"""Internal LiteLLM call layer.

The single place that talks to LiteLLM (and, for structured output,
``instructor`` over LiteLLM). The public call functions in
:mod:`llmkit.structured_output` build/log :class:`LLMCallRecord`s
around these helpers; this module owns provider routing, the rate-limit
semaphore, structured-output mode pinning, and best-effort cost extraction.

It is also the **test seam**: unit tests patch these three coroutines
(``acompletion_structured`` / ``acompletion_text`` / ``astream_text``) so
the real call-function bodies — logging, retry, content coercion — still
run over a faked provider response (see ``tests/_support`` ``patch_llm``).

LiteLLM's ``acompletion`` and instructor's ``create_with_completion`` carry
very strict, heavily-overloaded type stubs that reject this module's generic
``**credential-kwargs`` and ``list[dict[str, str]]`` message shapes. Those
call expressions therefore carry a single ``reportArgumentType`` suppression
each, tagged ``raw-llm`` — the boundary where our thin wrapper meets the
provider SDK's exhaustive parameter surface.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import cast

import instructor
import litellm
from litellm import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices
from pydantic import BaseModel

from llmkit.providers import LLMProviderInterface, build_provider
from llmkit.rate_limiting import GlobalRateLimiter

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
    """
    acompletion = cast(
        "Callable[..., Coroutine[object, object, ModelResponse | CustomStreamWrapper]]",
        litellm.acompletion,
    )
    return await acompletion(**kwargs)


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


def _messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalise a prompt into LiteLLM's message-list shape."""
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt


def _resolve_reasoning_effort(override: str | None, provider: LLMProviderInterface) -> str | None:
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

    LiteLLM assigns ``StreamingChoices.delta`` and ``Delta.content`` inside
    ``__init__`` rather than as annotated class attributes, so the static type
    of ``chunk.choices[0].delta.content`` is ``Unknown``. We narrow the two
    dynamic hops to their real runtime types (``Delta``, then ``str | None``)
    here so the streaming loop reads a precise ``str | None`` — the one place
    that knows the litellm stream object's shape (raw-llm).
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
    return cast("str | None", delta.content)


async def acompletion_structured[T: BaseModel](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
) -> tuple[T, float | None]:
    """Structured completion via instructor pinned to the provider's mode.

    Uses ``create_with_completion`` so the parsed model *and* the raw
    completion (for cost) are both in hand. instructor's in-call schema-repair
    budget is pinned to ``max_retries=2`` — instructor counts *total attempts*
    (it feeds the int to tenacity's ``stop_after_attempt``), so 2 means two
    attempts total = one schema-repair re-ask, deliberately low — and kept
    separate from the transient-error retry layer (``with_retries`` in
    :mod:`llmkit.retry`), which owns the cross-call 429/503/5xx and
    schema-validation budgets.

    ``max_tokens`` caps the completion length when set; it is only included
    in the underlying call when not ``None``, so the default produces a
    request byte-identical to the prior behaviour (no ``max_tokens`` key).
    LiteLLM accepts ``max_tokens`` as a standard cross-provider kwarg and
    instructor forwards extra kwargs to the wrapped completion, so no
    per-provider branching is needed.

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
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
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
        cast("object", instructor.from_litellm(_acompletion, mode=provider.instructor_mode)),
    )
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        result = await client.chat.completions.create_with_completion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),  # pyright: ignore[reportArgumentType]  # raw-llm — instructor over-strict ChatCompletionMessageParam
            response_model=output_schema,
            temperature=temperature,
            # instructor's in-call schema-repair budget. instructor counts
            # *total attempts* (the int becomes tenacity stop_after_attempt),
            # so 2 = two attempts total = exactly one schema-repair re-ask;
            # 1 would mean no repair ever happens. The cross-call
            # transient/validation budgets live in llmkit.retry.
            max_retries=2,
            **creds,  # pyright: ignore[reportArgumentType]  # raw-llm — provider-owned credential kwargs (api_key / api_base / aws_region_name)
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **({"reasoning_effort": effort} if effort is not None else {}),
        )
        # instructor types the raw completion half of the tuple as Any; it is a
        # litellm ModelResponse. Narrow once so cost/usage read a real type.
        parsed, completion = cast("tuple[T, ModelResponse]", result)
        slot.record_tokens(_total_tokens(completion))
    return parsed, _response_cost(completion)


async def acompletion_text(
    prompt: str | list[dict[str, str]],
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
) -> tuple[str, float | None]:
    """Plain-text completion via LiteLLM.

    ``reasoning_effort`` controls provider thinking tokens, resolved against
    the provider's configured value when ``None`` (an explicit value
    overrides it); the kwarg is only forwarded when the resolved value is
    not ``None``. ``provider`` overrides the configured provider for this
    call only (``None`` uses the globally-configured one).

    Returns ``(text, approximate_cost)``. The text is the first choice's
    message content coerced to a string via :func:`_coerce_text_content`
    (list content blocks joined; an empty string when the provider returns
    none).
    """
    provider = provider if provider is not None else build_provider()
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            temperature=temperature,
            **creds,
            # Gate max_tokens like the structured/stream paths: an unset cap
            # sends no ``max_tokens`` key at all (not an explicit ``None``), so
            # the request stays byte-identical to prior behaviour. Creds are
            # splatted first to match the structured/stream helpers exactly.
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **({"reasoning_effort": effort} if effort is not None else {}),
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


async def astream_text(
    prompt: str | list[dict[str, str]],
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
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
    default request is byte-identical to the prior stream call.
    """
    provider = provider if provider is not None else build_provider()
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    async with GlobalRateLimiter.acquire_async(provider.name) as slot:
        resp = await _acompletion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            temperature=temperature,
            stream=True,
            **creds,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **({"reasoning_effort": effort} if effort is not None else {}),
        )
        # stream=True makes acompletion return a CustomStreamWrapper, whose
        # async iteration yields typed ModelResponseStream chunks.
        stream = cast("CustomStreamWrapper", resp)
        async for chunk in stream:
            delta = _chunk_delta_text(chunk)
            if delta:
                yield delta
        # Best-effort TPM accounting: a stream usually reports no usage (we
        # don't request stream_options=include_usage), so this is typically a
        # no-op — consistent with cost being None for streamed calls. When the
        # wrapper does expose a final usage, debit it before releasing the slot.
        slot.record_tokens(_total_tokens(stream))
