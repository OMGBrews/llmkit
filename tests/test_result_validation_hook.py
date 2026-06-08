"""Tests for the ``on_result`` semantic-validation re-roll hook.

The call functions take an optional ``on_result`` callback that inspects each
attempt's parsed result and raises :class:`~llmkit.ResultValidationError` to
**reject** a result that parsed cleanly but is semantically wrong, re-rolling
the call. These tests pin the contract over the patched transport seam:

* a rejected-then-accepted result re-rolls and returns the accepted one;
* a persistently-rejected result is charged against the *validation* budget
  (the lower one), NOT the transport budget;
* a result the hook accepts is returned with no re-roll;
* ``retry=NO_RETRY`` makes a rejection propagate after a single attempt;
* the hook works on ``text_llm_call`` (text);
* a rejected attempt is still its own logged record.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    LocalYamlLogSink,
    ResultValidationError,
    RetryPolicy,
    capture_llm_records,
    configure_llm_logging,
)
from llmkit.structured_output import (
    structured_llm_call,
    text_llm_call,
)


class _Schema(BaseModel):
    """Minimal structured-output schema carrying a single int."""

    n: int


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse pytest fixture
    """Point logging at a no-op sink so these tests don't touch the disk."""
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


# Keep the default attempt budgets (transport 3 / validation 2) but drop the
# backoff so the tests run without real sleeps.
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


def _reject_below_two(result: _Schema) -> None:
    """Reject any result whose ``n`` is below 2 (forces a re-roll)."""
    if result.n < 2:
        raise ResultValidationError(f"n={result.n} too low")


@pytest.mark.asyncio
async def test_structured_on_result_rejects_then_accepts() -> None:
    """A bad-then-good result re-rolls and returns the accepted one."""
    calls = [0]

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        calls[0] += 1
        return _Schema(n=calls[0]), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_reject_below_two
        )

    assert result.n == 2
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_persistent_rejection_uses_validation_budget_not_transport() -> None:
    """A persistently-rejected result spends the *validation* budget (2),
    not the transport budget (3) — a deterministically-bad result can't burn
    the full transport budget on doomed re-rolls."""
    calls = [0]

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        calls[0] += 1
        return _Schema(n=0), None

    def _always_reject(_r: _Schema) -> None:
        raise ResultValidationError("never good enough")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(ResultValidationError, match="never good enough"),
    ):
        _ = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_always_reject
        )

    # validation_max_attempts default is 2 -> two attempts, NOT three (transport).
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_on_result_that_accepts_does_not_re_roll() -> None:
    """A result the hook accepts is returned after a single attempt."""
    calls = [0]
    seen: list[int] = []

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        calls[0] += 1
        return _Schema(n=5), None

    def _accept(result: _Schema) -> None:
        seen.append(result.n)

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_accept
        )

    assert result.n == 5
    assert calls[0] == 1
    assert seen == [5]


@pytest.mark.asyncio
async def test_no_retry_makes_rejection_propagate_after_one_attempt() -> None:
    """``retry=NO_RETRY`` disables the re-roll: one attempt, then propagate."""
    calls = [0]

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        calls[0] += 1
        return _Schema(n=0), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(ResultValidationError, match="too low"),
    ):
        _ = await structured_llm_call(
            "hi", _Schema, feature="t", retry=NO_RETRY, on_result=_reject_below_two
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_text_on_result_rejects_then_accepts() -> None:
    """The hook re-rolls a plain-text call too (e.g. reject empty output)."""
    calls = [0]

    async def _transport(*_a: object, **_k: object) -> tuple[str, float | None]:
        calls[0] += 1
        return ("" if calls[0] == 1 else "ok"), None

    def _reject_empty(text: str) -> None:
        if not text:
            raise ResultValidationError("empty response")

    with patch("llmkit._litellm.acompletion_text", side_effect=_transport):
        result = await text_llm_call("hi", feature="t", retry=_NO_BACKOFF, on_result=_reject_empty)

    assert result == "ok"
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_rejected_attempt_is_its_own_logged_record() -> None:
    """Each attempt is logged, the rejected one included: a bad-then-good run
    yields two captured records — the first carrying the rejection error and
    the response that was rejected, the second clean."""
    calls = [0]

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        calls[0] += 1
        return _Schema(n=calls[0]), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        capture_llm_records() as records,
    ):
        result = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_reject_below_two
        )

    assert result.n == 2
    assert len(records) == 2
    assert records[0].error is not None
    assert "ResultValidationError" in records[0].error
    # The rejected attempt recorded the response it rejected.
    assert cast("object", records[0].response) == {"n": 1}
    assert records[1].error is None
