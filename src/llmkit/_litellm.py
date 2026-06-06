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

import logging
from collections.abc import AsyncIterator

import instructor
import litellm
from pydantic import BaseModel

from llmkit.providers import LLMProviderInterface, get_provider
from llmkit.rate_limiting import GlobalRateLimiter

logger = logging.getLogger(__name__)


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
        cost = hidden.get("response_cost")  # pyright: ignore[reportUnknownMemberType]  # raw-llm — litellm hidden-params dict
        if isinstance(cost, (int, float)):
            return float(cost)
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
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


async def acompletion_structured[T: BaseModel](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    validation_retries: int = 1,
) -> tuple[T, float | None]:
    """Structured completion via instructor pinned to the provider's mode.

    Uses ``create_with_completion`` so the parsed model *and* the raw
    completion (for cost) are both in hand. ``validation_retries`` is
    instructor's in-call schema-repair budget — deliberately low and kept
    separate from the transient-error retry layer (``with_retries`` in
    :mod:`llmkit.retry`), which handles 429/503/5xx.

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
    provider = provider if provider is not None else get_provider()
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    client = instructor.from_litellm(litellm.acompletion, mode=provider.instructor_mode)
    async with GlobalRateLimiter.acquire_async():
        parsed, completion = await client.chat.completions.create_with_completion(
            model=provider.litellm_model(model),
            messages=_messages(prompt),  # pyright: ignore[reportArgumentType]  # raw-llm — instructor over-strict ChatCompletionMessageParam
            response_model=output_schema,
            temperature=temperature,
            max_retries=validation_retries,
            api_key=creds.get("api_key"),
            api_base=creds.get("api_base"),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),  # pyright: ignore[reportArgumentType]  # raw-llm — instructor **kwargs passthrough for optional max_tokens
            **({"reasoning_effort": effort} if effort is not None else {}),  # pyright: ignore[reportArgumentType]  # raw-llm — instructor **kwargs passthrough for optional reasoning_effort
        )
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
    provider = provider if provider is not None else get_provider()
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    async with GlobalRateLimiter.acquire_async():
        resp = await litellm.acompletion(  # pyright: ignore[reportArgumentType]  # raw-llm — litellm over-strict signature
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=creds.get("api_key"),
            api_base=creds.get("api_base"),
            **({"reasoning_effort": effort} if effort is not None else {}),  # pyright: ignore[reportArgumentType]  # raw-llm — litellm **kwargs passthrough for optional reasoning_effort
        )
    content = resp.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]  # raw-llm — litellm ModelResponse
    return _coerce_text_content(content), _response_cost(resp)


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
    provider = provider if provider is not None else get_provider()
    creds = provider.completion_kwargs()
    effort = _resolve_reasoning_effort(reasoning_effort, provider)
    async with GlobalRateLimiter.acquire_async():
        stream = await litellm.acompletion(  # pyright: ignore[reportArgumentType]  # raw-llm — litellm over-strict signature
            model=provider.litellm_model(model),
            messages=_messages(prompt),
            temperature=temperature,
            stream=True,
            api_key=creds.get("api_key"),
            api_base=creds.get("api_base"),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),  # pyright: ignore[reportArgumentType]  # raw-llm — litellm **kwargs passthrough for optional max_tokens
            **({"reasoning_effort": effort} if effort is not None else {}),  # pyright: ignore[reportArgumentType]  # raw-llm — litellm **kwargs passthrough for optional reasoning_effort
        )
        async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]  # raw-llm — litellm stream wrapper is async-iterable
            delta = chunk.choices[0].delta.content  # pyright: ignore[reportAttributeAccessIssue]  # raw-llm — litellm stream chunk
            if delta:
                yield delta
