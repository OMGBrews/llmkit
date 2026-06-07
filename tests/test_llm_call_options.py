"""Tests for the opt-in :class:`LLMCallOptions` call-options bundle.

A feature module builds one :class:`LLMCallOptions` and passes it as
``options=`` to every call instead of repeating the nine-keyword block.
These tests pin the three-layer precedence the audit made explicit —
**config < options < explicit per-call keyword** — by patching the
transport seam and asserting what each call forwards: an ``LLMCallOptions``
reused across calls, an explicit keyword overriding a field set in options,
and an unset options field falling through (``model=None``) to the config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    LocalYamlLogSink,
    configure_llm_logging,
    structured_output,
)

# Imported from the submodule rather than the package root: the integration
# phase owns adding ``LLMCallOptions`` to ``llmkit.__all__``.
from llmkit.structured_output import LLMCallOptions


class _Schema(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    """Point logging at a no-op sink so these tests don't touch the disk."""
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.model = "gemini-2.5-flash-lite"
    provider.name = "Google AI Studio"
    return provider


type _CallKwargs = dict[str, object]
type _StructuredFake = Callable[..., Awaitable[tuple[_Schema, float | None]]]


def _structured_recorder() -> tuple[_StructuredFake, list[_CallKwargs]]:
    """An async ``acompletion_structured`` stub that records its kwargs.

    Returns the stub (drop it into ``side_effect``) and the list its calls
    accumulate into — a plain ``async def`` closure, matching the existing
    capture tests' shape so the patched transport stays awaitable.
    """
    calls: list[_CallKwargs] = []

    async def _fake(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        calls.append(dict(kwargs))
        return _Schema(ok=True), 0.0

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
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> None:
            for _ in range(3):
                _ = await structured_output.structured_llm_call(
                    "hi", _Schema, feature="extraction", options=options
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
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> None:
            _ = await structured_output.structured_llm_call(
                "hi",
                _Schema,
                feature="extraction",
                model="explicit-model",
                options=options,
            )

        asyncio.run(_run())

    assert calls[0]["model"] == "explicit-model"
    # The untouched field still comes from options.
    assert calls[0]["temperature"] == 0.9


def test_unset_options_field_falls_through_to_config() -> None:
    """An unset options field leaves the keyword at its default, so the
    transport receives ``model=None`` and resolves it against the config."""
    fake, calls = _structured_recorder()
    # ``model`` is deliberately unset; only ``temperature`` is supplied.
    options = LLMCallOptions(temperature=0.7)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> None:
            _ = await structured_output.structured_llm_call(
                "hi", _Schema, feature="extraction", options=options
            )

        asyncio.run(_run())

    # Unset ``model`` does not clobber config: ``None`` is forwarded so the
    # transport substitutes the provider/config default.
    assert calls[0]["model"] is None
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["temperature"] == 0.7


def test_no_options_leaves_flat_kwargs_unchanged() -> None:
    """With ``options=None`` the flat-keyword path is byte-for-byte unchanged."""
    fake, calls = _structured_recorder()
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> None:
            _ = await structured_output.structured_llm_call(
                "hi", _Schema, feature="extraction", model="flat-model", temperature=0.4
            )

        asyncio.run(_run())

    assert calls[0]["model"] == "flat-model"
    assert calls[0]["temperature"] == 0.4
    # Defaults untouched.
    assert calls[0]["max_tokens"] is None


def test_options_retry_applied_when_keyword_default() -> None:
    """A ``retry`` set on options is honored when the keyword is left default;
    proves the budget field merges like the rest."""
    options = LLMCallOptions(retry=NO_RETRY)
    resolved = structured_output._resolve_call_args(
        options,
        temperature=0.2,
        model=None,
        max_tokens=None,
        reasoning_effort=None,
        retry=structured_output.DEFAULT_RETRY_POLICY,
        provider=None,
    )

    assert resolved.retry is NO_RETRY


def test_options_threads_through_text_and_sync() -> None:
    """``options`` is honored by ``text_llm_call`` and ``structured_llm_call_sync``."""
    text_recorder: list[dict[str, object]] = []

    async def _fake_text(*_args: object, **kwargs: object) -> tuple[str, float | None]:
        text_recorder.append(dict(kwargs))
        return "hello", None

    struct_fake, struct_calls = _structured_recorder()
    options = LLMCallOptions(model="shared-model", max_tokens=128)
    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_fake_text),
        patch("llmkit._litellm.acompletion_structured", side_effect=struct_fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run_text() -> None:
            _ = await structured_output.text_llm_call("hi", feature="summary", options=options)

        asyncio.run(_run_text())
        result = structured_output.structured_llm_call_sync(
            "hi", _Schema, feature="classification", options=options
        )

    assert cast("_Schema", result).ok is True
    assert text_recorder[0]["model"] == "shared-model"
    assert text_recorder[0]["max_tokens"] == 128
    assert struct_calls[0]["model"] == "shared-model"
    assert struct_calls[0]["max_tokens"] == 128
