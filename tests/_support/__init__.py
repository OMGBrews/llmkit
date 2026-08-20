"""Shared, non-test scaffolding for the offline suite (import as ``tests._support``).

Holds the fake provider, the fake-transport capture helpers, the capturing
log sink, and the log-record builder that the offline tests would otherwise
each re-derive. Deliberately small: only genuinely repeated, mechanical
scaffolding lives here. A stub that is specific to one test — a bespoke schema,
a one-off ``side_effect`` — stays inline where it reads clearer next to the
assertion it serves.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
)


class OkSchema(BaseModel):
    """The minimal structured-output target: a single boolean field."""

    ok: bool


def model_attr(instance: BaseModel, name: str) -> object:
    """Read a dynamically-built model field as a statically-known ``object``.

    ``model_from_json_schema`` returns ``type[BaseModel]``, so an instance's
    runtime fields (``.score``, ``.tags``, …) are not statically known and a
    bare ``inst.score`` is an unknown-member access. Routing each assertion's
    read through here gives it a real ``object`` type — which still supports the
    ``==`` / ``is`` checks at call sites — without weakening the assertion.
    """
    return cast("object", getattr(instance, name))


def provider_mock(
    *, model: str = "gemini-2.5-flash-lite", name: str = "Google AI Studio", **overrides: object
) -> MagicMock:
    """A fake provider as ``build_provider`` would return.

    Defaults to the Google shape the call tests resolve model/provider against;
    pass ``model=``/``name=`` or any extra attribute via ``overrides``.
    """
    provider = MagicMock()
    provider.model = model
    provider.name = name
    for attr, value in overrides.items():
        setattr(provider, attr, value)
    return provider


def make_record(**overrides: object) -> LLMCallRecord:
    """Build an :class:`LLMCallRecord` from sensible defaults, overriding any field."""
    base: dict[str, object] = {
        "started_at": datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC),
        "feature": "extraction",
        "label": "summary",
        "model": "gemini-2.5-flash-lite",
        "provider": "Google AI Studio",
        "temperature": 0.0,
        "duration_ms": 12.3,
        "schema": "Schema",
        "prompt": "hi",
        "response": None,
        "error": None,
    }
    base.update(overrides)
    return LLMCallRecord(**base)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat


@contextmanager
def quiet_logging() -> Generator[None]:
    """Disable logging (no-op sink) for the duration, restoring the default after.

    Keeps offline call tests from writing YAML to disk; the autouse fixtures in
    ``calls/conftest.py`` and ``logging/conftest.py`` wrap this.
    """
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


@contextmanager
def capturing_sink() -> Generator[list[LLMCallRecord]]:
    """Install a sink that appends every written record to the yielded list.

    Restores the default :class:`LocalYamlLogSink` on exit, so a test can assert
    on the ``LLMCallRecord`` a call produced without writing YAML to disk.
    """
    captured: list[LLMCallRecord] = []

    class _CapturingSink:
        def write(self, record: LLMCallRecord) -> None:
            captured.append(record)

    configure_llm_logging(_CapturingSink())
    try:
        yield captured
    finally:
        configure_llm_logging(LocalYamlLogSink())


def _transport_provider(
    *, reasoning_effort: str | None, instructor_mode: str = "json"
) -> MagicMock:
    """The fake provider the transport seam reads (kwargs / model / effort)."""
    provider = MagicMock()
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", "api_base": None})
    provider.instructor_mode = instructor_mode
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = reasoning_effort
    return provider


def capture_structured_provider_kwargs(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider_effort: str | None = None,
) -> dict[str, object]:
    """Drive the real ``acompletion_structured`` over a faked instructor client
    and return the kwargs the provider call (``create_with_completion``) saw.

    ``temperature`` / ``max_tokens`` / ``reasoning_effort`` are the per-call
    values forwarded to the transport; ``provider_effort`` sets the provider's
    own configured reasoning effort (the fallback when no per-call value is
    given). The default ``temperature=None`` forwards **no** ``temperature``
    key at all (the transport's gating), matching the public ``None``
    semantics.
    """
    from llmkit import _litellm

    seen: dict[str, object] = {}

    async def _fake_create_with_completion(**kwargs: object) -> tuple[OkSchema, MagicMock]:
        seen.update(kwargs)
        return OkSchema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat = MagicMock(
        completions=MagicMock(create_with_completion=_fake_create_with_completion)
    )
    provider = _transport_provider(reasoning_effort=provider_effort)

    with patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client):
        parsed, _cost = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                OkSchema,
                temperature=temperature,
                model=None,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        )
    assert parsed.ok is True
    return seen


def capture_text_provider_kwargs(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider_effort: str | None = None,
) -> dict[str, object]:
    """Drive the real ``acompletion_text`` over a faked ``litellm.acompletion``
    and return the kwargs the provider request saw.

    ``temperature`` / ``max_tokens`` / ``reasoning_effort`` are the per-call
    values forwarded to the transport; ``provider_effort`` sets the provider's
    own configured reasoning effort. The default ``temperature=None``
    forwards **no** ``temperature`` key at all.
    """
    from llmkit import _litellm

    seen: dict[str, object] = {}
    fake_resp = MagicMock(_hidden_params={})
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def _fake_acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        return fake_resp

    provider = _transport_provider(reasoning_effort=provider_effort)

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text(
                "hi",
                temperature=temperature,
                model=None,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        )
    assert text == "ok"
    return seen


def capture_stream_provider_kwargs(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider_effort: str | None = None,
) -> dict[str, object]:
    """Drive the real ``astream_text`` over a faked LiteLLM stream and return
    the kwargs the provider call (``litellm.acompletion``) saw.

    ``temperature`` / ``max_tokens`` / ``reasoning_effort`` are the per-call
    values forwarded to the transport; ``provider_effort`` sets the provider's
    own configured reasoning effort. The default ``temperature=None``
    forwards **no** ``temperature`` key at all.
    """
    from llmkit import _litellm

    seen: dict[str, object] = {}

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[MagicMock]:
            async def _gen() -> AsyncIterator[MagicMock]:
                for delta in ("he", "llo"):
                    yield MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])

            return _gen()

    async def _fake_acompletion(**kwargs: object) -> _FakeStream:
        seen.update(kwargs)
        return _FakeStream()

    provider = _transport_provider(reasoning_effort=provider_effort)

    async def _drive() -> list[str]:
        return [
            chunk
            async for chunk in _litellm.astream_text(
                "hi",
                temperature=temperature,
                model=None,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        ]

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        chunks = asyncio.run(_drive())

    assert chunks == ["he", "llo"]
    return seen
