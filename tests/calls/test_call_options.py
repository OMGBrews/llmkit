"""Tests for the opt-in :class:`LLMCallOptions` call-options bundle.

A feature module builds one :class:`LLMCallOptions` and passes it as
``options=`` to every call instead of repeating the nine-keyword block.
These tests pin the three-layer precedence the audit made explicit —
**config < options < explicit per-call keyword** — by patching the
transport seam and asserting what each call forwards: an ``LLMCallOptions``
reused across calls, an explicit keyword overriding a field set in options
(including one whose value equals the old signature default — detection is
the ``UNSET`` sentinel, never value equality), and an unset options field
falling through to the config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import patch

from llmkit import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TEMPERATURE,
    NO_RETRY,
    UNSET,
    LLMCallOptions,
)
from llmkit import (
    calls as llm_calls,
)
from llmkit.options import resolve_call_args
from tests._support import OkSchema, provider_mock

type _CallKwargs = dict[str, object]
type _StructuredFake = Callable[..., Awaitable[tuple[OkSchema, float | None]]]


def _structured_recorder() -> tuple[_StructuredFake, list[_CallKwargs]]:
    """An async ``acompletion_structured`` stub that records its kwargs.

    Returns the stub (drop it into ``side_effect``) and the list its calls
    accumulate into — a plain ``async def`` closure, matching the existing
    capture tests' shape so the patched transport stays awaitable.
    """
    calls: list[_CallKwargs] = []

    async def _fake(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        calls.append(dict(kwargs))
        return OkSchema(ok=True), 0.0

    return _fake, calls


def test_options_reused_across_calls_supplies_each_call() -> None:
    """One ``LLMCallOptions`` built once feeds every call's transport."""
    fake, calls = _structured_recorder()
    options = LLMCallOptions(
        temperature=0.9,
        model="custom-model",
        max_tokens=512,
        reasoning_effort="high",
        retry=NO_RETRY,
    )
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            for _ in range(3):
                _ = await llm_calls.structured_llm_call(
                    "hi", OkSchema, feature="extraction", options=options
                )

        asyncio.run(_run())

    assert len(calls) == 3
    for call in calls:
        assert call["temperature"] == 0.9
        assert call["model"] == "custom-model"
        assert call["max_tokens"] == 512
        assert call["reasoning_effort"] == "high"


def test_explicit_keyword_overrides_options_field() -> None:
    """An explicit per-call keyword wins over the same field set in options."""
    fake, calls = _structured_recorder()
    options = LLMCallOptions(model="options-model", temperature=0.9)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi",
                OkSchema,
                feature="extraction",
                model="explicit-model",
                options=options,
            )

        asyncio.run(_run())

    assert calls[0]["model"] == "explicit-model"
    # The untouched field still comes from options.
    assert calls[0]["temperature"] == 0.9


def test_explicit_keyword_equal_to_old_default_overrides_options() -> None:
    """Regression for the audit's precedence bug: an explicitly-passed keyword
    wins over ``options`` even when its value equals the old signature default.

    Detection is structural (the ``UNSET`` sentinel), not value equality, so
    ``max_tokens=None`` and ``temperature=0.2`` — both former signature
    defaults — override the matching options fields, exactly as the README's
    **config < options < explicit keyword** promises."""
    fake, calls = _structured_recorder()
    options = LLMCallOptions(max_tokens=512, temperature=0.9, model="options-model")
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi",
                OkSchema,
                feature="extraction",
                max_tokens=None,  # explicit -> wins despite equalling the old default
                temperature=DEFAULT_TEMPERATURE,  # explicit 0.2 -> wins over 0.9
                model=None,  # explicit None -> provider default, not options-model
                options=options,
            )

        asyncio.run(_run())

    assert calls[0]["max_tokens"] is None
    assert calls[0]["temperature"] == DEFAULT_TEMPERATURE
    assert calls[0]["model"] is None


def test_unset_options_field_falls_through_to_config() -> None:
    """An unset options field leaves the keyword at its default, so the
    transport receives ``model=None`` and resolves it against the config."""
    fake, calls = _structured_recorder()
    # ``model`` is deliberately unset; only ``temperature`` is supplied.
    options = LLMCallOptions(temperature=0.7)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi", OkSchema, feature="extraction", options=options
            )

        asyncio.run(_run())

    # Unset ``model`` does not clobber config: ``None`` is forwarded so the
    # transport substitutes the provider/config default.
    assert calls[0]["model"] is None
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["temperature"] == 0.7


def test_no_options_leaves_flat_kwargs_unchanged() -> None:
    """With ``options=None`` the flat-keyword path forwards the same values
    as ever: passed keywords as given, unset ones at their true defaults."""
    fake, calls = _structured_recorder()
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi", OkSchema, feature="extraction", model="flat-model", temperature=0.4
            )

        asyncio.run(_run())

    assert calls[0]["model"] == "flat-model"
    assert calls[0]["temperature"] == 0.4
    # Defaults untouched.
    assert calls[0]["max_tokens"] is None


def test_options_retry_applied_when_keyword_unset() -> None:
    """A ``retry`` set on options is honored when the keyword is not passed;
    proves the budget field merges like the rest."""
    options = LLMCallOptions(retry=NO_RETRY)
    resolved = resolve_call_args(
        options,
        temperature=UNSET,
        model=UNSET,
        max_tokens=UNSET,
        reasoning_effort=UNSET,
        retry=UNSET,
        provider=UNSET,
    )

    assert resolved.retry is NO_RETRY


def test_all_unset_resolves_to_true_defaults() -> None:
    """Nothing passed anywhere resolves to the documented defaults — applied
    in ``resolve_call_args``, the single definition site, not in signatures."""
    resolved = resolve_call_args(
        None,
        temperature=UNSET,
        model=UNSET,
        max_tokens=UNSET,
        reasoning_effort=UNSET,
        retry=UNSET,
        provider=UNSET,
    )

    assert resolved.temperature == DEFAULT_TEMPERATURE
    assert resolved.model is None
    assert resolved.max_tokens is None
    assert resolved.reasoning_effort is None
    assert resolved.retry is DEFAULT_RETRY_POLICY
    assert resolved.provider is None


def test_repr_shows_only_set_fields_and_sentinel_reads_clean() -> None:
    """``LLMCallOptions`` repr prints only the fields actually set, and the
    sentinel itself reprs as ``UNSET`` — no internal enum-member noise in
    debug output or ``help()`` signatures."""
    assert repr(UNSET) == "UNSET"
    assert repr(LLMCallOptions()) == "LLMCallOptions()"
    assert repr(LLMCallOptions(temperature=0.9)) == "LLMCallOptions(temperature=0.9)"
    # An explicit None is a *set* field (distinct from unset) and shows up.
    assert repr(LLMCallOptions(model=None)) == "LLMCallOptions(model=None)"


def test_options_threads_through_text_and_sync() -> None:
    """``options`` is honored by text and structured sync wrappers."""
    text_recorder: list[dict[str, object]] = []

    async def _fake_text(*_args: object, **kwargs: object) -> tuple[str, float | None]:
        text_recorder.append(dict(kwargs))
        return "hello", None

    struct_fake, struct_calls = _structured_recorder()
    options = LLMCallOptions(model="shared-model", max_tokens=128)
    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_fake_text),
        patch("llmkit._litellm.acompletion_structured", side_effect=struct_fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run_text() -> None:
            _ = await llm_calls.text_llm_call("hi", feature="summary", options=options)

        asyncio.run(_run_text())
        text_sync_result = llm_calls.text_llm_call_sync("hi", feature="summary", options=options)
        result = llm_calls.structured_llm_call_sync(
            "hi", OkSchema, feature="classification", options=options
        )

    assert result.ok is True
    assert text_sync_result == "hello"
    assert text_recorder[0]["model"] == "shared-model"
    assert text_recorder[0]["max_tokens"] == 128
    assert text_recorder[1]["model"] == "shared-model"
    assert text_recorder[1]["max_tokens"] == 128
    assert struct_calls[0]["model"] == "shared-model"
    assert struct_calls[0]["max_tokens"] == 128


def test_explicit_temperature_none_overrides_options_value() -> None:
    """An explicit ``temperature=None`` keyword wins over a numeric options
    value — the ``model=None`` idiom applied to the sampling temperature.
    Resolution yields ``None`` (not options' ``0.9``); the transport's
    gating then sends no ``temperature`` key at all (pinned at the wire
    seam in ``test_temperature.py``)."""
    fake, calls = _structured_recorder()
    options = LLMCallOptions(temperature=0.9)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi",
                OkSchema,
                feature="extraction",
                temperature=None,  # explicit -> wins over options' 0.9
                options=options,
            )

        asyncio.run(_run())

    assert calls[0]["temperature"] is None


def test_options_temperature_none_overrides_default() -> None:
    """``LLMCallOptions(temperature=None)`` wins over
    ``DEFAULT_TEMPERATURE`` when the keyword is unset — the options field
    resolves to ``None`` (the transport then omits the key)."""
    fake, calls = _structured_recorder()
    options = LLMCallOptions(temperature=None)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> None:
            _ = await llm_calls.structured_llm_call(
                "hi",
                OkSchema,
                feature="extraction",
                temperature=UNSET,
                options=options,
            )

        asyncio.run(_run())

    assert calls[0]["temperature"] is None
