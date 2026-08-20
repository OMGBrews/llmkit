"""Tests for the ``temperature=None`` "use the provider default" escape hatch.

llmkit historically always forwarded a resolved ``temperature`` (defaulting to
:data:`~llmkit.DEFAULT_TEMPERATURE`, ``0.2``), so a caller could not ask for a
provider's built-in sampling. An explicit ``None`` — passed as a call keyword
or through ``LLMCallOptions`` — now resolves to **no** ``temperature`` key on
the outgoing provider request, while the unset path keeps forwarding ``0.2``.

These tests pin three seams:

* every public call surface (structured, plain-text, streaming, the sync
  wrappers, and the deprecated ``stream_text_with_log`` alias) accepts
  ``temperature=None``;
* the three transports (``acompletion_structured``, ``acompletion_text``,
  ``astream_text``) omit the ``temperature`` key when the resolved value is
  ``None`` and forward explicit numerics — including ``0.0`` — unchanged; and
* the default (neither keyword nor ``options`` value) still resolves to and
  forwards :data:`~llmkit.DEFAULT_TEMPERATURE`.

The Gemini-3 consequence (no ``DeprecationWarning`` from LiteLLM when the key
is absent) is documented, not asserted mechanically here: the warning fires
inside LiteLLM's request transformation, which no llmkit-boundary test can
observe — that assertion lives in the follow-up task
``contribute-and-consume-litellm-gemini-3-temperature-fix``'s
transformation-level tests.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from llmkit import DEFAULT_TEMPERATURE, LLMCallOptions, structured_output
from tests._support import (
    OkSchema,
    capture_stream_provider_kwargs,
    capture_structured_provider_kwargs,
    capture_text_provider_kwargs,
    capturing_sink,
)


def test_signatures_expose_temperature() -> None:
    """Every public surface carries a ``temperature`` parameter."""
    for surface in (
        structured_output.structured_llm_call,
        structured_output.structured_llm_call_sync,
        structured_output.text_llm_call,
        structured_output.text_llm_call_sync,
        structured_output.text_llm_call_stream,
        structured_output.stream_text_with_log,
    ):
        assert "temperature" in inspect.signature(surface).parameters


# --- Transport seams: the absent-key invariant -----------------------------


def test_transport_omits_temperature_when_none_structured() -> None:
    """At the ``acompletion_structured`` seam, a ``None`` temperature sends
    **no** ``temperature`` kwarg (absent, not an explicit ``None``)."""
    seen = capture_structured_provider_kwargs(temperature=None)
    assert "temperature" not in seen


def test_transport_omits_temperature_when_none_text() -> None:
    """Parity: at the ``acompletion_text`` seam a ``None`` temperature sends
    **no** ``temperature`` kwarg."""
    seen = capture_text_provider_kwargs(temperature=None)
    assert "temperature" not in seen


def test_transport_omits_temperature_when_none_stream() -> None:
    """Parity: at the ``astream_text`` seam a ``None`` temperature sends
    **no** ``temperature`` kwarg."""
    seen = capture_stream_provider_kwargs(temperature=None)
    assert "temperature" not in seen


def test_transport_forwards_explicit_zero_unchanged_structured() -> None:
    """``0.0`` is a real, forwarded value — the gating checks identity, not
    truthiness, so an explicit zero is never mistaken for "unset"."""
    seen = capture_structured_provider_kwargs(temperature=0.0)
    assert seen["temperature"] == 0.0


def test_transport_forwards_explicit_zero_unchanged_text() -> None:
    """The ``acompletion_text`` seam forwards an explicit ``0.0`` unchanged."""
    seen = capture_text_provider_kwargs(temperature=0.0)
    assert seen["temperature"] == 0.0


def test_transport_forwards_explicit_zero_unchanged_stream() -> None:
    """The ``astream_text`` seam forwards an explicit ``0.0`` unchanged."""
    seen = capture_stream_provider_kwargs(temperature=0.0)
    assert seen["temperature"] == 0.0


def test_transport_forwards_other_numeric_unchanged() -> None:
    """Any non-zero numeric reaches the provider request unchanged."""
    assert capture_structured_provider_kwargs(temperature=0.8)["temperature"] == 0.8
    assert capture_text_provider_kwargs(temperature=1.5)["temperature"] == 1.5
    assert capture_stream_provider_kwargs(temperature=0.3)["temperature"] == 0.3


# --- The unset path still resolves to DEFAULT_TEMPERATURE ------------------


def test_unset_resolves_and_forwards_default_structured() -> None:
    """Neither keyword nor ``options`` value → the transport receives
    ``DEFAULT_TEMPERATURE`` (``0.2``), preserving existing behavior."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(structured_output.structured_llm_call("hi", OkSchema, feature="test"))

    assert seen["temperature"] == DEFAULT_TEMPERATURE


def test_unset_resolves_and_forwards_default_text() -> None:
    """The plain-text path forwards the legacy default too."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[str, float | None]:
        seen.update(kwargs)
        return "ok", None

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = asyncio.run(structured_output.text_llm_call("hi", feature="test"))

    assert seen["temperature"] == DEFAULT_TEMPERATURE


# --- Keyword and options paths through the full call surface ----------------


def test_keyword_none_reaches_transport_structured() -> None:
    """``structured_llm_call(..., temperature=None)`` resolves to ``None`` —
    the value the transport's gating then drops (absence is asserted at the
    real wire seam by ``capture_structured_provider_kwargs``)."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(
            structured_output.structured_llm_call("hi", OkSchema, feature="test", temperature=None)
        )

    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_text() -> None:
    """``text_llm_call(..., temperature=None)`` resolves to ``None`` — the
    value the transport's gating then drops (absence asserted at the real
    seam by ``capture_text_provider_kwargs``)."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[str, float | None]:
        seen.update(kwargs)
        return "ok", None

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = asyncio.run(structured_output.text_llm_call("hi", feature="test", temperature=None))

    assert seen["temperature"] is None


def test_keyword_none_omits_key_stream() -> None:
    """``text_llm_call_stream(..., temperature=None)`` sends no key at all."""

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[str]:
            async def _gen() -> AsyncIterator[str]:
                yield "he"
                yield "llo"

            return _gen()

    seen: dict[str, object] = {}

    def _fake_transport(*_args: object, **kwargs: object) -> _FakeStream:
        seen.update(kwargs)
        return _FakeStream()

    async def _drive() -> list[str]:
        chunks: list[str] = []
        async for chunk in structured_output.text_llm_call_stream(
            "hi", feature="test", temperature=None
        ):
            chunks.append(chunk)
        return chunks

    with patch("llmkit._litellm.astream_text", side_effect=_fake_transport):
        chunks = asyncio.run(_drive())

    assert chunks == ["he", "llo"]
    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_sync_structured() -> None:
    """The sync wrapper propagates ``temperature=None`` to the transport."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = structured_output.structured_llm_call_sync(
            "hi", OkSchema, feature="test", temperature=None
        )

    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_sync_text() -> None:
    """``text_llm_call_sync(..., temperature=None)`` resolves to ``None`` at
    the transport."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[str, float | None]:
        seen.update(kwargs)
        return "ok", None

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = structured_output.text_llm_call_sync("hi", feature="test", temperature=None)

    assert seen["temperature"] is None


def test_deprecated_alias_omits_key() -> None:
    """The deprecated ``stream_text_with_log`` alias forwards
    ``temperature=None`` to the streaming path."""

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[str]:
            async def _gen() -> AsyncIterator[str]:
                yield "ok"

            return _gen()

    def _fake_stream(*_args: object, **kwargs: object) -> _FakeStream:
        seen.update(kwargs)
        return _FakeStream()

    seen: dict[str, object] = {}

    async def _drive() -> list[str]:
        chunks: list[str] = []
        async for chunk in structured_output.stream_text_with_log(
            "hi", feature="test", temperature=None
        ):
            chunks.append(chunk)
        return chunks

    with patch("llmkit._litellm.astream_text", side_effect=_fake_stream):
        with pytest.warns(DeprecationWarning, match="text_llm_call_stream"):
            chunks = asyncio.run(_drive())

    assert chunks == ["ok"]
    assert seen["temperature"] is None


def test_options_numeric_still_forwards_when_keyword_unset() -> None:
    """A numeric ``LLMCallOptions`` value still reaches the transport when the
    keyword is unset."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(
            structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", options=LLMCallOptions(temperature=0.6)
            )
        )

    assert seen["temperature"] == 0.6


# --- Log records: the omitted state is observable and distinct -----------------


def test_log_record_carries_none_for_omitted_temperature() -> None:
    """An intentionally-omitted temperature is recorded as ``None`` on the
    ``LLMCallRecord`` — distinct from the ``0.2`` default."""

    async def _fake_transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport),
    ):
        _ = asyncio.run(
            structured_output.structured_llm_call("hi", OkSchema, feature="test", temperature=None)
        )

    assert len(captured) == 1
    assert captured[0].temperature is None


def test_log_record_carries_default_for_unset_temperature() -> None:
    """An unset temperature is recorded as ``DEFAULT_TEMPERATURE`` — the
    legacy value, unchanged."""

    async def _fake_transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport),
    ):
        _ = asyncio.run(structured_output.structured_llm_call("hi", OkSchema, feature="test"))

    assert len(captured) == 1
    assert captured[0].temperature == DEFAULT_TEMPERATURE
