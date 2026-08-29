"""Tests for honoring a provider ``Retry-After`` in the backoff layer.

When a retried provider error carries ``Retry-After`` (a numeric attribute, a
``retry-after`` / ``retry-after-ms`` header, or an HTTP-date), the shared backoff
routine (:func:`llmkit.retry.handle_retry_failure`, used by both
:func:`~llmkit.retry.with_retries` and the streaming loop) sleeps *that* duration
instead of the blind exponential — capped at ``retry_after_cap`` and read from the
**unwrapped** provider error so structured calls honour it too. These pin:

* the parser (:func:`llmkit.retry._retry_after_seconds`) across the attribute /
  seconds-header / ms-header / HTTP-date / absent / unparseable / wrapped cases;
* the sleep behaviour: a header overrides the exponential, is capped, is honoured
  even when ``backoff_base_seconds == 0`` (a server directive, not opt-in backoff),
  keeps its jitter, and — absent a header — falls back byte-identically; and
* the end-to-end thread-through on the structured and streaming call surfaces.

All deterministic: ``asyncio.sleep`` is captured, ``random.uniform`` is pinned, and
the HTTP-date clock (``retry._utcnow``) is frozen — nothing sleeps for real.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import calls as llm_calls
from llmkit import retry
from llmkit.retry import _retry_after_seconds, with_retries
from tests._support import OkSchema, quiet_logging


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.test/v1/chat/completions")


def _rate_limit(headers: dict[str, str] | None = None) -> openai.RateLimitError:
    """A real ``openai.RateLimitError`` (429) carrying *headers* on its response."""
    response = httpx.Response(429, headers=headers or {}, request=_request())
    return openai.RateLimitError("slow down", response=response, body=None)


class _ErrWithRetryAfter(Exception):
    """A minimal exception exposing a numeric ``retry_after`` attribute."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("retry me")
        self.retry_after: float = retry_after


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in stream]


def _no_jitter(lo: float, _hi: float) -> float:
    return lo


def _max_jitter(_lo: float, hi: float) -> float:
    return hi


# --- _retry_after_seconds: parsing ---------------------------------------


def test_numeric_attribute_is_read() -> None:
    """A numeric ``retry_after`` attribute is read directly."""
    assert _retry_after_seconds(_ErrWithRetryAfter(7.0)) == 7.0


def test_retry_after_seconds_header() -> None:
    """A ``retry-after`` header of integer seconds is parsed."""
    assert _retry_after_seconds(_rate_limit({"retry-after": "5"})) == 5.0


def test_retry_after_ms_header_divided_to_seconds() -> None:
    """A ``retry-after-ms`` header is divided to seconds and wins over none."""
    assert _retry_after_seconds(_rate_limit({"retry-after-ms": "2500"})) == 2.5


def test_retry_after_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTP-date ``retry-after`` resolves to the delta from a frozen now."""
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(retry, "_utcnow", lambda: base)
    when = format_datetime(base + timedelta(seconds=30), usegmt=True)
    assert _retry_after_seconds(_rate_limit({"retry-after": when})) == 30.0


def test_retry_after_http_date_in_past_clamps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A past HTTP-date clamps to zero (retry now), never negative."""
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(retry, "_utcnow", lambda: base)
    when = format_datetime(base - timedelta(seconds=30), usegmt=True)
    assert _retry_after_seconds(_rate_limit({"retry-after": when})) == 0.0


def test_absent_header_returns_none() -> None:
    """No ``Retry-After`` anywhere → ``None`` (caller falls back to exponential)."""
    assert _retry_after_seconds(_rate_limit()) is None


def test_no_response_returns_none() -> None:
    """An error without a ``.response`` (a timeout) carries no header → ``None``."""
    assert _retry_after_seconds(TimeoutError("boom")) is None


def test_unparseable_header_returns_none() -> None:
    """A non-numeric, non-date ``retry-after`` is unparseable → ``None``."""
    assert _retry_after_seconds(_rate_limit({"retry-after": "soon"})) is None


def test_negative_seconds_clamped_to_zero() -> None:
    """A negative delta-seconds value clamps to zero."""
    assert _retry_after_seconds(_rate_limit({"retry-after": "-5"})) == 0.0


def test_retry_after_unwraps_instructor_wrapper() -> None:
    """A structured-call throttle wrapped in ``InstructorRetryException`` still
    surfaces its ``Retry-After`` — the parser unwraps first (D0)."""
    wrapped = InstructorRetryException(
        _rate_limit({"retry-after": "5"}), n_attempts=1, total_usage=0
    )
    assert _retry_after_seconds(wrapped) == 5.0


# --- the sleep: Retry-After overrides the exponential --------------------


@pytest.mark.asyncio
async def test_header_overrides_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a ``Retry-After: 5`` the sleep is 5s, not the ~0.5s exponential ceiling."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)

    async def _fn() -> str:
        raise _rate_limit({"retry-after": "5"})

    with pytest.raises(openai.RateLimitError):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(openai.RateLimitError,), backoff_base_seconds=0.5
        )

    assert slept == [5.0]  # the server directive, not 0.5


@pytest.mark.asyncio
async def test_capped_at_retry_after_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile ``Retry-After: 9999`` is capped at ``retry_after_cap`` (default 60)."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)

    async def _fn() -> str:
        raise _rate_limit({"retry-after": "9999"})

    with pytest.raises(openai.RateLimitError):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(openai.RateLimitError,), backoff_base_seconds=0.5
        )

    assert slept == [60.0]


@pytest.mark.asyncio
async def test_custom_retry_after_cap_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``retry_after_cap`` caps the server value in place of 60s."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)

    async def _fn() -> str:
        raise _rate_limit({"retry-after": "9999"})

    with pytest.raises(openai.RateLimitError):
        _ = await with_retries(
            _fn,
            max_attempts=2,
            retry_on=(openai.RateLimitError,),
            backoff_base_seconds=0.5,
            retry_after_cap=10.0,
        )

    assert slept == [10.0]


@pytest.mark.asyncio
async def test_honored_even_when_backoff_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server ``Retry-After`` is honoured even with ``backoff_base_seconds == 0``
    (a directive, not opt-in backoff) — the jitter band just collapses to 0."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    async def _fn() -> str:
        raise _rate_limit({"retry-after": "5"})

    with pytest.raises(openai.RateLimitError):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(openai.RateLimitError,), backoff_base_seconds=0.0
        )

    assert slept == [5.0]


@pytest.mark.asyncio
async def test_jitter_added_to_server_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A little jitter (from the ``backoff_base_seconds`` band) rides on the server
    value so a fan-out doesn't synchronize its retries."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)  # jitter -> backoff_base

    async def _fn() -> str:
        raise _rate_limit({"retry-after": "5"})

    with pytest.raises(openai.RateLimitError):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(openai.RateLimitError,), backoff_base_seconds=0.5
        )

    assert slept == [5.5]  # 5 (capped server value) + 0.5 (max jitter)


@pytest.mark.asyncio
async def test_absent_header_falls_back_to_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``Retry-After``, the computed exponential is used unchanged."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    async def _fn() -> str:
        raise TimeoutError("transient")  # no response, no header

    with pytest.raises(TimeoutError):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(TimeoutError,), backoff_base_seconds=2.0
        )

    assert slept == [2.0]  # exponential ceiling at attempt 1, not a server value


@pytest.mark.asyncio
async def test_wrapped_header_honored_through_with_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``InstructorRetryException`` wrapping a 429-with-header sleeps the
    server value — the unwrap is load-bearing on the structured path."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)

    async def _fn() -> str:
        raise InstructorRetryException(
            _rate_limit({"retry-after": "5"}), n_attempts=1, total_usage=0
        )

    with pytest.raises(InstructorRetryException):
        _ = await with_retries(
            _fn, max_attempts=2, retry_on=(InstructorRetryException,), backoff_base_seconds=0.5
        )

    assert slept == [5.0]


# --- end-to-end thread-through on the call surfaces ----------------------


@pytest.mark.asyncio
async def test_retry_after_honored_in_structured_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``structured_llm_call`` honours a ``Retry-After`` from its transport: the
    default-policy backoff sleeps the server value before the successful retry."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise _rate_limit({"retry-after": "4"})
        return OkSchema(ok=True), None

    with quiet_logging(), patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")

    assert result.ok is True
    assert calls[0] == 2
    assert slept == [4.0]


@pytest.mark.asyncio
async def test_retry_after_honored_in_streaming_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``text_llm_call_stream`` honours a ``Retry-After`` on a pre-first-chunk
    failure: the retry waits the server value before delivering the stream.

    Covers ``tool_llm_call_stream`` too, and deliberately without a second
    case: ``Retry-After`` parsing and the sleep that honours it live entirely
    inside :func:`~llmkit.retry.with_retries_stream`, which both streaming
    families call with the same policy object. The only parameters that differ
    between them — ``label``, ``surface``, ``warn_stacklevel`` — do not reach
    the delay path, and the tool lane's own pre-first-event retry behaviour is
    pinned in ``test_stream_retry_guard.py``. A duplicate here would exercise
    the same lines through a longer wrapper.
    """
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "uniform", _no_jitter)
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        if calls[0] == 1:
            raise _rate_limit({"retry-after": "4"})
            # Unreachable: the bare yield makes this an async generator.
            yield  # pyright: ignore[reportUnreachable]  # pragma: no cover
        for delta in ("he", "llo"):
            yield delta

    with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
        chunks = await _drain(llm_calls.text_llm_call_stream("hi", feature="test"))

    assert chunks == ["he", "llo"]
    assert calls[0] == 2
    assert slept == [4.0]
