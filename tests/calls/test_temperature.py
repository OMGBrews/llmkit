"""Tests for llmkit's "no default temperature" behavior and ``temperature=None``.

llmkit chooses no sampling temperature: :data:`~llmkit.DEFAULT_TEMPERATURE` is
``None``, so a call that sets no temperature anywhere sends **no**
``temperature`` key and the provider's default sampling applies. (It was
``0.2`` until Google announced that upcoming Gemini models reject sampling
parameters outright.) An explicit ``None`` — passed as a call keyword or
through ``LLMCallOptions`` — resolves the same way, and an explicit number is
forwarded unchanged.

These tests pin three seams:

* every public call surface (structured, plain-text, streaming, the sync
  wrappers, and the deprecated ``stream_text_with_log`` alias) accepts
  ``temperature=None``;
* the three transports (``acompletion_structured``, ``acompletion_text``,
  ``astream_text``) omit the ``temperature`` key when the resolved value is
  ``None`` and forward explicit numerics — including ``0.0`` — unchanged; and
* the default (neither keyword nor ``options`` value) resolves to ``None`` and
  reaches the request with no ``temperature`` key on every shipped provider.

On Gemini 3+ the wire result additionally depends on LiteLLM, which re-injects
``temperature = 1.0`` when the key is absent; that is asserted (and fixed) by
the follow-up task ``contribute-and-consume-litellm-gemini-3-temperature-fix``,
not here.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from unittest.mock import MagicMock, patch

import pytest

from llmkit import DEFAULT_TEMPERATURE, LLMCallOptions, Provider, make_provider
from llmkit import calls as llm_calls
from llmkit.providers import LLMProviderInterface
from tests._support import (
    NO_USAGE,
    OkSchema,
    UsageCounts,
    capture_stream_provider_kwargs,
    capture_structured_provider_kwargs,
    capture_text_provider_kwargs,
    capturing_sink,
)


def test_signatures_expose_temperature() -> None:
    """Every public surface carries a ``temperature`` parameter."""
    for surface in (
        llm_calls.structured_llm_call,
        llm_calls.structured_llm_call_sync,
        llm_calls.text_llm_call,
        llm_calls.text_llm_call_sync,
        llm_calls.text_llm_call_stream,
        llm_calls.stream_text_with_log,
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


# --- The unset path sends no temperature on any provider --------------------


def test_default_temperature_is_none() -> None:
    """llmkit chooses no temperature: the exported default is ``None``."""
    assert DEFAULT_TEMPERATURE is None


def test_unset_resolves_to_none_structured() -> None:
    """Neither keyword nor ``options`` value → the transport receives
    ``None``, which its gating turns into an absent key."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        seen.update(kwargs)
        return OkSchema(ok=True), None, NO_USAGE

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(llm_calls.structured_llm_call("hi", OkSchema, feature="test"))

    assert "temperature" in seen
    assert seen["temperature"] is None


def test_unset_resolves_to_none_text() -> None:
    """The plain-text path resolves an unset temperature to ``None`` too."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[str, float | None, UsageCounts]:
        seen.update(kwargs)
        return "ok", None, NO_USAGE

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = asyncio.run(llm_calls.text_llm_call("hi", feature="test"))

    assert "temperature" in seen
    assert seen["temperature"] is None


#: A builder for every shipped provider — only the arguments each one's knob
#: contract accepts, so ``make_provider`` builds the real class.
_PROVIDER_BUILDERS: dict[Provider, Callable[[], LLMProviderInterface]] = {
    Provider.ANTHROPIC: lambda: make_provider(Provider.ANTHROPIC, api_key="k"),
    Provider.BEDROCK: lambda: make_provider(Provider.BEDROCK, aws_region_name="us-east-1"),
    Provider.DEEPSEEK: lambda: make_provider(Provider.DEEPSEEK, api_key="k"),
    Provider.GOOGLE: lambda: make_provider(Provider.GOOGLE, api_key="k"),
    Provider.OLLAMA: lambda: make_provider(Provider.OLLAMA),
    Provider.OPENAI: lambda: make_provider(Provider.OPENAI, api_key="k"),
    Provider.OPENROUTER: lambda: make_provider(Provider.OPENROUTER, api_key="k"),
    Provider.VERTEX: lambda: make_provider(
        Provider.VERTEX, vertex_project="p", vertex_location="us-central1"
    ),
}


def test_provider_build_table_covers_every_provider() -> None:
    """A new ``Provider`` member must join the no-default-temperature sweep."""
    assert set(_PROVIDER_BUILDERS) == set(Provider)


@pytest.mark.parametrize("provider_enum", list(Provider))
def test_unset_call_sends_no_temperature_on_any_provider_text(provider_enum: Provider) -> None:
    """A default ``text_llm_call`` through each real provider reaches
    ``litellm.acompletion`` with no ``temperature`` key — neither the
    resolution nor any provider hook supplies one."""
    provider = _PROVIDER_BUILDERS[provider_enum]()
    seen: dict[str, object] = {}
    fake_resp = MagicMock(_hidden_params={})
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def _fake_acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        return fake_resp

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text = asyncio.run(llm_calls.text_llm_call("hi", feature="test", provider=provider))

    assert text == "ok"
    assert "messages" in seen  # the request reached the seam
    assert "temperature" not in seen


@pytest.mark.parametrize("provider_enum", list(Provider))
def test_unset_call_sends_no_temperature_on_any_provider_structured(
    provider_enum: Provider,
) -> None:
    """The structured path, through each real provider, reaches the
    instructor completion with no ``temperature`` key."""
    provider = _PROVIDER_BUILDERS[provider_enum]()
    seen: dict[str, object] = {}

    async def _fake_create_with_completion(**kwargs: object) -> tuple[OkSchema, MagicMock]:
        seen.update(kwargs)
        return OkSchema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat = MagicMock(
        completions=MagicMock(create_with_completion=_fake_create_with_completion)
    )

    with patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client):
        parsed = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", provider=provider)
        )

    assert parsed.ok is True
    assert "messages" in seen  # the request reached the seam
    assert "temperature" not in seen


def test_unset_call_sends_no_temperature_stream() -> None:
    """The streaming transport, driven at its real seam with the value an
    unset call resolves to, sends no ``temperature`` key."""
    seen = capture_stream_provider_kwargs(temperature=DEFAULT_TEMPERATURE)
    assert "temperature" not in seen


# --- Keyword and options paths through the full call surface ----------------


def test_keyword_none_reaches_transport_structured() -> None:
    """``structured_llm_call(..., temperature=None)`` resolves to ``None`` —
    the value the transport's gating then drops (absence is asserted at the
    real wire seam by ``capture_structured_provider_kwargs``)."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        seen.update(kwargs)
        return OkSchema(ok=True), None, NO_USAGE

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", temperature=None)
        )

    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_text() -> None:
    """``text_llm_call(..., temperature=None)`` resolves to ``None`` — the
    value the transport's gating then drops (absence asserted at the real
    seam by ``capture_text_provider_kwargs``)."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[str, float | None, UsageCounts]:
        seen.update(kwargs)
        return "ok", None, NO_USAGE

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = asyncio.run(llm_calls.text_llm_call("hi", feature="test", temperature=None))

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
        async for chunk in llm_calls.text_llm_call_stream("hi", feature="test", temperature=None):
            chunks.append(chunk)
        return chunks

    with patch("llmkit._litellm.astream_text", side_effect=_fake_transport):
        chunks = asyncio.run(_drive())

    assert chunks == ["he", "llo"]
    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_sync_structured() -> None:
    """The sync wrapper propagates ``temperature=None`` to the transport."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        seen.update(kwargs)
        return OkSchema(ok=True), None, NO_USAGE

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = llm_calls.structured_llm_call_sync("hi", OkSchema, feature="test", temperature=None)

    assert seen["temperature"] is None


def test_keyword_none_reaches_transport_sync_text() -> None:
    """``text_llm_call_sync(..., temperature=None)`` resolves to ``None`` at
    the transport."""
    seen: dict[str, object] = {}

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[str, float | None, UsageCounts]:
        seen.update(kwargs)
        return "ok", None, NO_USAGE

    with patch("llmkit._litellm.acompletion_text", side_effect=_fake_transport):
        _ = llm_calls.text_llm_call_sync("hi", feature="test", temperature=None)

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
        async for chunk in llm_calls.stream_text_with_log("hi", feature="test", temperature=None):
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

    async def _fake_transport(
        *_args: object, **kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        seen.update(kwargs)
        return OkSchema(ok=True), None, NO_USAGE

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        _ = asyncio.run(
            llm_calls.structured_llm_call(
                "hi", OkSchema, feature="test", options=LLMCallOptions(temperature=0.6)
            )
        )

    assert seen["temperature"] == 0.6


# --- Log records: the omitted state is observable and distinct -----------------


def test_log_record_carries_none_for_omitted_temperature() -> None:
    """An explicitly-omitted temperature is recorded as ``None`` on the
    ``LLMCallRecord``."""

    async def _fake_transport(
        *_args: object, **_kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        return OkSchema(ok=True), None, NO_USAGE

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport),
    ):
        _ = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", temperature=None)
        )

    assert len(captured) == 1
    assert captured[0].temperature is None


def test_log_record_carries_none_for_unset_temperature() -> None:
    """An unset temperature sends none, so it is recorded as ``None`` — the
    same state as an explicit ``temperature=None``."""

    async def _fake_transport(
        *_args: object, **_kwargs: object
    ) -> tuple[OkSchema, float | None, UsageCounts]:
        return OkSchema(ok=True), None, NO_USAGE

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport),
    ):
        _ = asyncio.run(llm_calls.structured_llm_call("hi", OkSchema, feature="test"))

    assert len(captured) == 1
    assert captured[0].temperature is None
