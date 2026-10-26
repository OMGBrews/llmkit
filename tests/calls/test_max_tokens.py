"""Tests for the optional ``max_tokens`` cap on the structured-output path.

``structured_llm_call`` / ``structured_llm_call_sync`` gained a keyword-only
``max_tokens`` (parity with ``text_llm_call``). These tests pin two seams:

* the public call functions thread ``max_tokens`` down to the transport
  (observed through an :mod:`llmkit.testing` fake's request), and
* the transport (``acompletion_structured``) only puts ``max_tokens`` on the
  provider request when it is not ``None`` — so the default produces a
  request byte-identical to the prior behaviour (no ``max_tokens`` key).

The plain-text transport (``acompletion_text``) gates ``max_tokens`` the same
way, so all three call paths agree on the "unset cap sends no key" invariant.
"""

from __future__ import annotations

import asyncio
import inspect

from llmkit import calls as llm_calls
from llmkit.testing import Reply, install
from tests._support import (
    OkSchema,
    capture_structured_provider_kwargs,
    capture_text_provider_kwargs,
    capturing_sink,
)

_OK = Reply.json(OkSchema(ok=True))


def test_signatures_expose_max_tokens() -> None:
    """Acceptance: both public functions carry a ``max_tokens`` parameter."""
    assert "max_tokens" in inspect.signature(llm_calls.structured_llm_call).parameters
    assert "max_tokens" in inspect.signature(llm_calls.structured_llm_call_sync).parameters


def test_sync_call_threads_max_tokens_to_transport() -> None:
    """``structured_llm_call_sync(..., max_tokens=256)`` forwards the cap."""
    with install(lambda _request: _OK) as installed:
        result = llm_calls.structured_llm_call_sync("hi", OkSchema, feature="test", max_tokens=256)

    assert [r.max_tokens for r in installed.requests] == [256]
    # max_tokens does not disturb parsing — a validated instance comes back.
    assert isinstance(result, OkSchema)
    assert result.ok is True


def test_async_call_threads_max_tokens_to_transport() -> None:
    """The async ``structured_llm_call(..., max_tokens=N)`` does the same."""
    with install(lambda _request: _OK) as installed:
        result = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", max_tokens=42)
        )

    assert [r.max_tokens for r in installed.requests] == [42]
    assert result.ok is True


def test_transport_omits_max_tokens_when_none() -> None:
    """At the provider seam, a ``None`` cap sends **no** ``max_tokens`` kwarg
    (absent, not ``None``) — byte-identical to the pre-feature request."""
    seen = capture_structured_provider_kwargs(max_tokens=None)
    assert "max_tokens" not in seen


def test_transport_includes_max_tokens_when_set() -> None:
    """A concrete cap reaches the provider call kwargs unchanged."""
    seen = capture_structured_provider_kwargs(max_tokens=8)
    assert seen["max_tokens"] == 8


def test_log_record_carries_max_tokens() -> None:
    """The ``LLMCallRecord`` built for a structured call records the cap."""
    with capturing_sink() as captured, install(lambda _request: _OK):
        _ = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", max_tokens=256)
        )

    assert len(captured) == 1
    assert captured[0].max_tokens == 256


def test_text_transport_omits_max_tokens_when_none() -> None:
    """Parity with the structured path: at the ``acompletion_text`` seam a
    ``None`` cap sends **no** ``max_tokens`` kwarg (absent, not an explicit
    ``None``) — byte-identical to the pre-feature request."""
    seen = capture_text_provider_kwargs(max_tokens=None)
    assert "max_tokens" not in seen


def test_text_transport_includes_max_tokens_when_set() -> None:
    """A concrete cap reaches the plain-text provider call kwargs unchanged."""
    seen = capture_text_provider_kwargs(max_tokens=16)
    assert seen["max_tokens"] == 16
