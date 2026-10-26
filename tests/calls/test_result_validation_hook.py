"""Tests for the ``on_result`` semantic-validation re-roll hook.

The call functions take an optional ``on_result`` callback that inspects each
attempt's parsed result and raises :class:`~llmkit.ResultValidationError` to
**reject** a result that parsed cleanly but is semantically wrong, re-rolling
the call. These tests pin the contract through an :mod:`llmkit.testing` fake:

* a rejected-then-accepted result re-rolls and returns the accepted one;
* a persistently-rejected result is charged against the *validation* budget
  (the lower one), NOT the transport budget;
* a result the hook accepts is returned with no re-roll;
* ``retry=NO_RETRY`` makes a rejection propagate after a single attempt;
* the hook works on ``text_llm_call`` (text);
* a rejected attempt is still its own logged record.
"""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ResultValidationError,
    RetryPolicy,
    capture_llm_records,
)
from llmkit.calls import (
    structured_llm_call,
    text_llm_call,
)
from llmkit.testing import Reply, TransportRequest, install


class _Schema(BaseModel):
    """Minimal structured-output schema carrying a single int."""

    n: int


# Keep the default attempt budgets (transport 3 / validation 2) but drop the
# backoff so the tests run without real sleeps.
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


def _n_is_attempt(request: TransportRequest) -> Reply:
    """Answer ``n`` equal to the attempt number: 1, then 2, ..."""
    return Reply.json(_Schema(n=request.attempt or 0))


def _attempts(requests: list[TransportRequest]) -> list[int | None]:
    return [request.attempt for request in requests]


def _reject_below_two(result: _Schema) -> None:
    """Reject any result whose ``n`` is below 2 (forces a re-roll)."""
    if result.n < 2:
        raise ResultValidationError(f"n={result.n} too low")


async def test_structured_on_result_rejects_then_accepts() -> None:
    """A bad-then-good result re-rolls and returns the accepted one."""
    with install(_n_is_attempt) as installed:
        result = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_reject_below_two
        )

    assert result.n == 2
    assert _attempts(installed.requests) == [1, 2]


async def test_persistent_rejection_uses_validation_budget_not_transport() -> None:
    """A persistently-rejected result spends the *validation* budget (2),
    not the transport budget (3) — a deterministically-bad result can't burn
    the full transport budget on doomed re-rolls."""

    def _always_reject(_r: _Schema) -> None:
        raise ResultValidationError("never good enough")

    with (
        install(lambda _request: Reply.json(_Schema(n=0))) as installed,
        pytest.raises(ResultValidationError, match="never good enough"),
    ):
        _ = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_always_reject
        )

    # validation_max_attempts default is 2 -> two attempts, NOT three (transport).
    assert _attempts(installed.requests) == [1, 2]


async def test_on_result_that_accepts_does_not_re_roll() -> None:
    """A result the hook accepts is returned after a single attempt."""
    seen: list[int] = []

    def _accept(result: _Schema) -> None:
        seen.append(result.n)

    with install(lambda _request: Reply.json(_Schema(n=5))) as installed:
        result = await structured_llm_call(
            "hi", _Schema, feature="t", retry=_NO_BACKOFF, on_result=_accept
        )

    assert result.n == 5
    assert _attempts(installed.requests) == [1]
    assert seen == [5]


async def test_no_retry_makes_rejection_propagate_after_one_attempt() -> None:
    """``retry=NO_RETRY`` disables the re-roll: one attempt, then propagate."""
    with (
        install(lambda _request: Reply.json(_Schema(n=0))) as installed,
        pytest.raises(ResultValidationError, match="too low"),
    ):
        _ = await structured_llm_call(
            "hi", _Schema, feature="t", retry=NO_RETRY, on_result=_reject_below_two
        )

    assert _attempts(installed.requests) == [1]


async def test_text_on_result_rejects_then_accepts() -> None:
    """The hook re-rolls a plain-text call too (e.g. reject empty output)."""

    def fake(request: TransportRequest) -> Reply:
        return Reply.text("" if request.attempt == 1 else "ok")

    def _reject_empty(text: str) -> None:
        if not text:
            raise ResultValidationError("empty response")

    with install(fake) as installed:
        result = await text_llm_call("hi", feature="t", retry=_NO_BACKOFF, on_result=_reject_empty)

    assert result == "ok"
    assert _attempts(installed.requests) == [1, 2]


async def test_rejected_attempt_is_its_own_logged_record() -> None:
    """Each attempt is logged, the rejected one included: a bad-then-good run
    yields two captured records — the first carrying the rejection error and
    the response that was rejected, the second clean."""
    with install(_n_is_attempt), capture_llm_records() as records:
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
