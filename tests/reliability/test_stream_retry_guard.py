"""Tests for the streaming nested-retry guard.

``text_llm_call_stream`` participates in the same ``_retry_scope`` guard as
:func:`with_retries`, so a host wrapping stream consumption in an outer llmkit
retry loop (the documented composable path, since mid-stream errors propagate
unretried) does not multiply the two budgets. These pin:

* under an outer ``with_retries``, the stream loop collapses to a single
  pre-first-chunk attempt per outer pass (no N x N multiplication), and the
  accidental double-wrap emits a ``RuntimeWarning``;
* the warning fires only when the stream policy *would* have retried — an
  explicit ``retry=NO_RETRY`` inner and a mid-stream failure both stay silent;
* without an outer loop, the stream's own retry behaviour is unchanged; and
* the guard flag never leaks into the consumer's context — not between
  chunks, not after a full drain, not after an early break.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from llmkit import NO_RETRY, RetryPolicy
from llmkit import calls as llm_calls
from llmkit.retry import _retry_scope, _RetryScope, with_retries
from tests._support import quiet_logging

_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    """Consume an async text stream into a list of chunks."""
    return [chunk async for chunk in stream]


# --- under an outer with_retries: collapse to a single pass ---------------


@pytest.mark.asyncio
async def test_outer_with_retries_collapses_stream_to_single_pass() -> None:
    """Wrapping stream consumption (which already retries 3x pre-first-chunk)
    in ``with_retries(max_attempts=3)`` must NOT yield 3x3=9 provider calls.
    The stream loop collapses to a single pass per outer attempt, and the
    accidental double-wrap warns."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        raise TimeoutError("always")
        # Unreachable by design: the bare ``yield`` makes this an async
        # generator; the ``raise`` above precedes it on every attempt.
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    async def _wrapped() -> list[str]:
        return await _drain(llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF))

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_text", _transport),
        pytest.warns(RuntimeWarning, match="nested inside an already-retrying"),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await with_retries(_wrapped, max_attempts=3, retry_on=(TimeoutError,))

    # Outer budget of 3, inner collapsed to a single pass each -> 3 total,
    # NOT 9 (3 x 3). The guard prevented the multiplication.
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_double_wrap_warning_blames_the_caller_not_llmkit() -> None:
    """The warning must name the *consumer's* ``async for``, once per call site.

    ``pytest.warns`` cannot see either property: it installs the ``always``
    filter and resets the registry, so it reports every warning regardless of
    origin, and it never inspects ``filename``/``lineno``. Both are what the
    warning is *for* — a host is being told which of its double-wrapped calls to
    fix, and Python's default filter de-duplicates per warn-site, so an origin
    inside llmkit would collapse every caller's warning into one.

    ``text_llm_call_stream`` re-yields from :func:`with_retries_stream`, so the
    warning's frame count depends on that wrapper; this pins the
    ``warn_stacklevel`` that compensates for it.
    """

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        raise TimeoutError("always")
        # Unreachable by design (see above).
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    # Two callers with their OWN ``async for`` lines: the default filter keys on
    # the warn site, so sharing ``_drain`` would legitimately collapse them to
    # one and the count below would prove nothing.
    async def _site_one() -> list[str]:
        return [
            c async for c in llm_calls.text_llm_call_stream("a", feature="t", retry=_NO_BACKOFF)
        ]

    async def _site_two() -> list[str]:
        return [
            c async for c in llm_calls.text_llm_call_stream("b", feature="t", retry=_NO_BACKOFF)
        ]

    caught: list[warnings.WarningMessage] = []
    with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
        with warnings.catch_warnings(record=True) as recorded:
            # The real default filter, not pytest.warns' ``always`` — the
            # per-warn-site de-duplication is half of what is being pinned.
            warnings.simplefilter("default")
            for site in (_site_one, _site_two):
                with contextlib.suppress(TimeoutError):
                    _ = await with_retries(site, max_attempts=2, retry_on=(TimeoutError,))
            caught = [w for w in recorded if issubclass(w.category, RuntimeWarning)]

    # Two distinct caller sites -> two warnings. One would mean the origin
    # collapsed onto a single line of llmkit source.
    assert len(caught) == 2, [f"{w.filename}:{w.lineno}" for w in caught]
    # Every one attributed to this file, not to anything under src/llmkit.
    assert all(w.filename == __file__ for w in caught), [w.filename for w in caught]
    assert len({w.lineno for w in caught}) == 2


@pytest.mark.asyncio
async def test_no_retry_inner_under_outer_wrapper_stays_silent() -> None:
    """The documented escape hatch: drive retries from an outer ``with_retries``
    by opting the stream out with ``retry=NO_RETRY``. The inner is already a
    single pass, so there is nothing to multiply and no ``RuntimeWarning``."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        raise TimeoutError("always")
        # Unreachable by design (see above).
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    async def _wrapped() -> list[str]:
        return await _drain(llm_calls.text_llm_call_stream("hi", feature="test", retry=NO_RETRY))

    with warnings.catch_warnings():
        # Any nested-guard RuntimeWarning on this intended path fails the test.
        warnings.simplefilter("error", RuntimeWarning)
        with (
            quiet_logging(),
            patch("llmkit._litellm.astream_text", _transport),
            pytest.raises(TimeoutError, match="always"),
        ):
            _ = await with_retries(_wrapped, max_attempts=3, retry_on=(TimeoutError,))

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_mid_stream_failure_under_outer_wrapper_does_not_warn() -> None:
    """A mid-stream failure (chunks already delivered) is never retried by the
    stream loop even standalone, so the guard has nothing to warn about: the
    error propagates silently to the outer wrapper, which owns the retries."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        yield "he"
        raise TimeoutError("mid-stream")

    async def _wrapped() -> list[str]:
        return await _drain(llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with (
            quiet_logging(),
            patch("llmkit._litellm.astream_text", _transport),
            pytest.raises(TimeoutError, match="mid-stream"),
        ):
            _ = await with_retries(_wrapped, max_attempts=2, retry_on=(TimeoutError,))

    # One provider call per outer attempt — the inner loop never re-dialed.
    assert calls[0] == 2


# --- without an outer loop: behaviour unchanged ----------------------------


@pytest.mark.asyncio
async def test_stream_retries_normally_without_outer_loop() -> None:
    """Standalone, a pre-first-chunk transient failure is still retried and
    the second attempt delivers the full output (the guard changed nothing)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
            # Unreachable by design (see above).
            yield  # pyright: ignore[reportUnreachable]  # pragma: no cover
        for delta in ("he", "llo"):
            yield delta

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
            chunks = await _drain(
                llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF)
            )

    assert chunks == ["he", "llo"]
    assert calls[0] == 2
    # The loop's own guard flag was reset on completion.
    assert _retry_scope.get() is None


@pytest.mark.asyncio
async def test_stream_exhausts_full_budget_without_outer_loop() -> None:
    """Standalone, a persistent pre-first-chunk failure still spends the full
    transport budget (3 attempts) before propagating."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        raise TimeoutError("always")
        # Unreachable by design (see above).
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_text", _transport),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await _drain(llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF))

    assert calls[0] == 3
    assert _retry_scope.get() is None


# --- the guard flag never leaks into the consumer's context ----------------


@pytest.mark.asyncio
async def test_guard_flag_not_visible_to_consumer_between_chunks() -> None:
    """An async generator body runs in its consumer's context, so the loop
    releases the flag around each ``yield``: code the host runs between chunks
    (e.g. its own llmkit calls) must not see an active retry loop."""

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for delta in ("he", "llo"):
            yield delta

    seen_between_chunks: list[_RetryScope | None] = []
    with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
        async for _chunk in llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF):
            seen_between_chunks.append(_retry_scope.get())

    assert seen_between_chunks == [None, None]
    assert _retry_scope.get() is None


@pytest.mark.asyncio
async def test_guard_flag_clear_after_early_break() -> None:
    """A consumer that stops reading mid-stream must not be left with the
    guard flag set in its context — its later llmkit calls retry normally."""

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for delta in ("he", "llo"):
            yield delta

    with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
        # The public annotation is AsyncGenerator, so aclose() — the
        # early-abandonment path under test — is on the surface directly.
        stream = llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF)
        async for _chunk in stream:
            break
        assert _retry_scope.get() is None
        await stream.aclose()

    assert _retry_scope.get() is None
