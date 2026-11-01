"""The sync forms are the async calls, run synchronously.

PRINCIPLES.md promises that the sync form of a call carries every promise the
async form does. Two properties keep that true, and this module pins both:

* **Same signature.** Every ``*_sync`` function in :mod:`llmkit.calls` takes
  exactly the parameters of its async twin (names, kinds, defaults,
  annotations, type parameters, overloads) and returns what the async form's
  coroutine returns. A keyword added to one side and not the other fails here.
* **Same path.** Each sync form drives the async coroutine on llmkit's
  persistent loop, so retries, per-attempt logging, record capture and the
  retry progress callback behave identically. The loop identity check is what
  makes the test fail under a broken bridge: a wrapper that ran the coroutine
  under :func:`asyncio.run` would still retry and still capture (the fake is
  process-wide and the context carries the capture list), so only the loop it
  ran on would differ. :func:`test_a_bridge_that_bypasses_the_persistent_loop_fails`
  demonstrates that.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from collections.abc import Callable, Coroutine, Iterator
from typing import TypeVar, cast

import pytest

from llmkit import calls, capture_llm_records, retry_progress_callback
from llmkit.retry import RetryPolicy
from llmkit.sync import run_sync
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install, rate_limit_error
from llmkit.tools import ToolDefinition
from tests._support import OkSchema, ollama_call_defaults

# The async call functions that have no sync form yet. Adding their sync forms
# is filed as its own task; when one lands, its name leaves this set and the
# parity check below covers it automatically.
_ASYNC_ONLY = frozenset({"text_llm_call_stream", "tool_llm_call_stream"})


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """A keyless Ollama config, no cache, muted logging, default limits."""
    with ollama_call_defaults():
        yield


_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)
_TOOLS = [ToolDefinition("lookup", "Look a thing up.", {"type": "object"})]


_Fn = Callable[..., object]


def _fn(name: str) -> _Fn:
    return cast("_Fn", getattr(calls, name))


def _call_functions() -> dict[str, _Fn]:
    return {
        name: _fn(name)
        for name in calls.__all__
        if name.endswith("_llm_call") or "_llm_call_" in name
    }


def _sync_pairs() -> list[tuple[str, str]]:
    names = _call_functions()
    return sorted((name.removesuffix("_sync"), name) for name in names if name.endswith("_sync"))


def test_every_async_call_has_a_sync_form_or_is_listed() -> None:
    names = _call_functions()
    async_names = {
        name for name, fn in names.items() if inspect.iscoroutinefunction(fn) or name in _ASYNC_ONLY
    }
    paired = {base for base, _ in _sync_pairs()}
    assert paired <= async_names, "a *_sync function has no async twin"
    assert async_names - paired == _ASYNC_ONLY


def _shape(fn: _Fn) -> tuple[object, ...]:
    """A comparable rendering of *fn*'s signature, return type included.

    Annotations are compared as source strings (``from __future__ import
    annotations`` keeps them unevaluated in the call modules), and type
    parameters by name and bound, since each function owns its own ``T``.
    """
    sig = inspect.signature(fn)
    params = tuple(
        (p.name, p.kind, repr(cast("object", p.default)), str(cast("object", p.annotation)))
        for p in sig.parameters.values()
    )
    declared = cast("tuple[TypeVar, ...]", getattr(fn, "__type_params__", ()))
    type_params = tuple((tp.__name__, str(tp.__bound__)) for tp in declared)
    return params, str(cast("object", sig.return_annotation)), type_params


@pytest.mark.parametrize(("async_name", "sync_name"), _sync_pairs())
def test_sync_signature_matches_async(async_name: str, sync_name: str) -> None:
    async_fn = _fn(async_name)
    sync_fn = _fn(sync_name)
    assert inspect.iscoroutinefunction(async_fn)
    assert not inspect.iscoroutinefunction(sync_fn)
    assert _shape(sync_fn) == _shape(async_fn)
    async_overloads = typing.get_overloads(async_fn)
    sync_overloads = typing.get_overloads(sync_fn)
    assert len(sync_overloads) == len(async_overloads)
    for sync_overload, async_overload in zip(sync_overloads, async_overloads, strict=True):
        assert _shape(sync_overload) == _shape(async_overload)


def test_the_parity_check_sees_a_divergent_keyword() -> None:
    """The comparison is not vacuous: one changed default is a mismatch."""

    async def base(prompt: str, *, label: str | None = None) -> str:
        return f"{prompt}{label}"

    def drifted(prompt: str, *, label: str | None = "x") -> str:
        return f"{prompt}{label}"

    assert _shape(drifted) != _shape(base)


# --- Same path -------------------------------------------------------------------

_Lane = Callable[[], object]


def _structured() -> object:
    return calls.structured_llm_call_sync("hi", OkSchema, feature="parity", retry=_NO_BACKOFF)


def _text() -> object:
    return calls.text_llm_call_sync("hi", feature="parity", retry=_NO_BACKOFF)


def _tool() -> object:
    return calls.tool_llm_call_sync("hi", _TOOLS, feature="parity", retry=_NO_BACKOFF)


_LANES: dict[str, _Lane] = {"structured": _structured, "text": _text, "tool": _tool}


def _answer(request: TransportRequest) -> Reply:
    if request.kind == "structured":
        return Reply.json(OkSchema(ok=True))
    if request.kind == "tool":
        return Reply(tool_calls=[FakeToolCall("lookup", {})])
    return Reply.text("hello")


async def _loop_id() -> int:
    return id(asyncio.get_running_loop())


def _check_sync_lane_drives_async_path(lane: _Lane) -> None:
    loops: list[int] = []
    progress: list[int] = []

    def fake(request: TransportRequest) -> Reply:
        loops.append(id(asyncio.get_running_loop()))
        if request.attempt == 1:
            raise rate_limit_error()
        return _answer(request)

    def on_progress(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
        del label, max_attempts, error
        progress.append(attempt)

    with install(fake) as transport:
        with capture_llm_records() as records, retry_progress_callback(on_progress):
            result = lane()

    assert result is not None
    assert [r.attempt for r in transport.requests] == [1, 2], "the retry did not run"
    assert [r.attempt for r in records] == [1, 2], "each attempt is captured"
    assert records[0].error is not None and records[1].error is None
    assert progress == [1], "the retry progress callback crossed the bridge"
    persistent = run_sync(_loop_id())
    assert loops == [persistent, persistent], "the call did not run on the persistent loop"


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_sync_form_drives_the_async_path(lane: str) -> None:
    _check_sync_lane_drives_async_path(_LANES[lane])


def test_sync_form_from_a_foreign_loop_uses_the_persistent_loop() -> None:
    """A sync form called inside another running loop still takes the same path."""

    async def host() -> None:
        _check_sync_lane_drives_async_path(_text)

    asyncio.run(host())


def test_a_bridge_that_bypasses_the_persistent_loop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The targeted break: a wrapper that runs its coroutine under ``asyncio.run``."""

    def fresh_loop(coro: Coroutine[object, object, object], **_: object) -> object:
        return asyncio.run(coro)

    import llmkit.calls.text as text_module

    monkeypatch.setattr(text_module, "run_sync", fresh_loop)
    with pytest.raises(AssertionError, match="persistent loop"):
        _check_sync_lane_drives_async_path(_text)
