"""Call family x outcome: exactly one complete record per attempt reaches disk.

The matrix is an explicit list of cells, not a Cartesian product: some
outcomes cannot happen on some lanes, and :data:`CELLS` says so cell by cell
(``applicable=False`` with the reason) so the list stays exhaustive. Every
applicable cell follows one sequence (``tests/logging/_record_disk.py``):
install a fresh :class:`~llmkit.LocalYamlLogSink` on ``tmp_path`` (restoring
the previous sink afterwards), drive the outcome through an
:mod:`llmkit.testing` fake (or, for a mid-stream failure, the private stream
seam the fake cannot reach), flush delivery and require it to return true,
require :func:`~llmkit.get_logging_health` to show nothing dropped or failed,
and only then read the directory. A cell then checks, for each record it
expects:

* header line 1 against the pinned verdict shape, and header line 2;
* the YAML body's complete key set, in order, against :data:`YAML_KEYS`;
* ``requests`` present and a list (its content belongs to
  ``tests/calls/test_request_history.py`` and ``test_input_snapshots.py``);
* the ``index.jsonl`` line's key set, and its values equal to the YAML's on
  every shared key but ``file``;
* the cell's own key states.

The sync stream forms run through the same lanes; their abandonment is pinned
in ``tests/runtime/test_sync_stream.py`` and they get no cells here.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, get_args, override
from unittest.mock import patch

import pytest

from llmkit import (
    STREAM_ABANDONED_ERROR,
    InMemoryLLMCache,
    LocalYamlLogSink,
    Provider,
    ResultValidationError,
    RetryPolicy,
    TokenUsage,
    configure_llm_cache,
    configure_rate_limit,
    make_provider,
)
from llmkit import calls as llm_calls
from llmkit.rate_limiting._tuning import BREAKER_MIN_SAMPLES
from llmkit.testing import (
    FakeToolCall,
    Reply,
    TransportRequest,
    install,
    rate_limit_error,
    server_error,
    service_unavailable_error,
    timeout_error,
)
from tests._support import OkSchema
from tests.logging._record_disk import (
    FAMILIES,
    FEATURE,
    SCHEMA,
    TOOLS,
    DiskRecord,
    Family,
    assert_delivery_available,
    call,
    check_contract,
    fake_answer,
    read_disk,
    settle,
    sink_installed,
)

# --- Key states --------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """An expected state of one YAML key, checked against the cell's records."""

    name: str
    holds: Callable[[object, list[DiskRecord]], bool]

    @override
    def __repr__(self) -> str:
        return self.name


NULL = State("null", lambda v, _r: v is None)
FLOAT = State("float", lambda v, _r: isinstance(v, float))
MAPPING = State("mapping", lambda v, _r: isinstance(v, dict))
LIST = State("list", lambda v, _r: isinstance(v, list))
NONEMPTY_LIST = State(
    "non-empty list", lambda v, _r: isinstance(v, list) and len(cast("list[object]", v)) > 0
)


def eq(expected: object) -> State:
    return State(f"== {expected!r}", lambda v, _r: v == expected)


def matches(pattern: str) -> State:
    return State(
        f"~ {pattern!r}", lambda v, _r: isinstance(v, str) and re.fullmatch(pattern, v) is not None
    )


def same_as(key: str, record: int) -> State:
    """Equal to *key* of the cell's *record*-th target record."""
    return State(f"== records[{record}].{key}", lambda v, r: v == r[record].body[key])


def tool_calls_named(*ids: str) -> State:
    def holds(v: object, _r: list[DiskRecord]) -> bool:
        return isinstance(v, list) and [
            cast("dict[str, object]", c)["id"] for c in cast("list[object]", v)
        ] == list(ids)

    return State(f"tool calls {ids}", holds)


#: Every lane records the transport error itself: the structured lane
#: unwraps instructor's ``InstructorRetryException`` as the retry layer does
#: (``llmkit/exceptions.py``, the ``InstructorRetryException`` caveat).
def transport_error(type_name: str) -> State:
    return matches(rf"{type_name}: .+")


BUFFERED: tuple[Family, ...] = ("structured", "text", "tool")
STREAMS: tuple[Family, ...] = ("text_stream", "tool_stream")
TOOL_LANES: tuple[Family, ...] = ("tool", "tool_stream")


def no_answer(family: Family) -> object:
    """What ``response`` holds when an attempt produced no answer."""
    if family == "text_stream":
        return ""
    if family == "tool_stream":
        return {"text": None, "tool_calls": []}
    return None


def base_states(family: Family) -> dict[str, State]:
    """States every record of *family* has unless a cell says otherwise."""
    tool = family in TOOL_LANES
    return {
        "feature": eq(FEATURE),
        "model": eq("fake-model"),
        "provider": eq("Ollama"),
        "schema": eq(SCHEMA[family]),
        "call_id": matches(r"[0-9a-f-]{8,}"),
        "attempt": eq(1),
        "cache_hit": eq(False),
        "source_call_id": NULL,
        "tools": NONEMPTY_LIST if tool else NULL,
        "duration_ms": FLOAT,
        "requests": LIST,
        "prompt": eq("hi"),
    }


# --- Cells -------------------------------------------------------------------

type Outcome = Literal[
    "success",
    "rate_limit",
    "service_unavailable",
    "server_error",
    "timeout",
    "invalid_json",
    "on_result_rejects",
    "all_tool_calls_malformed",
    "some_tool_calls_malformed",
    "retry_attempt",
    "cache_hit",
    "rejected_cache_hit",
    "output_limit",
    "mid_stream_failure",
    "abandoned_stream",
    "cancellation",
    "circuit_open",
    "compose_round",
]
OUTCOMES = cast("tuple[Outcome, ...]", get_args(cast("object", Outcome.__value__)))


@dataclass(frozen=True)
class Expect:
    """One record a cell expects, in the order the attempts were made."""

    status: Literal["ok", "ERROR"]
    states: Mapping[str, State]


@dataclass(frozen=True)
class Cell:
    family: Family
    outcome: Outcome
    expects: tuple[Expect, ...] = ()
    applicable: bool = True
    reason: str = ""
    header_cost: str | None = None
    overrides: Mapping[str, State] = field(default_factory=dict[str, State])

    @property
    def id(self) -> str:
        return f"{self.family}-{self.outcome}"


def na(family: Family, outcome: Outcome, reason: str) -> Cell:
    return Cell(family, outcome, applicable=False, reason=reason)


def one(status: Literal["ok", "ERROR"], **states: State) -> tuple[Expect, ...]:
    return (Expect(status, states),)


_USAGE = TokenUsage(3, 2, 5)
_USAGE_DUMP = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
_EMPTY_USAGE = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
_TOOL_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"q": "a"}'},
}
_SUCCESS_RESPONSE: dict[Family, object] = {
    "structured": {"ok": True},
    "text": "ok",
    "text_stream": "ok",
    "tool": {"text": "ok", "tool_calls": [_TOOL_CALL]},
    "tool_stream": {"text": "ok", "tool_calls": [_TOOL_CALL]},
}
_TRUNCATED = '{"ok": tr'
_TRUNCATED_RESPONSE: dict[Family, object] = {
    "structured": None,
    "text": None,
    "text_stream": _TRUNCATED,
    "tool": None,
    "tool_stream": {"text": _TRUNCATED, "tool_calls": []},
}
_TRANSPORT: dict[Outcome, str] = {
    "rate_limit": "RateLimitError",
    "service_unavailable": "ServiceUnavailableError",
    "server_error": "InternalServerError",
    "timeout": "APITimeoutError",
}
_NO_STREAM = "a buffered lane has no stream to fail, abandon or cut mid-way"
_NO_HOOK = "this lane has no on_result hook"
_NO_TOOLS = "only the tool lanes parse tool-call arguments"
_NO_CACHE = "the tool lanes never read or write the response cache"
_NO_STREAM_HOOK = "a stream has no on_result hook, so nothing rejects its replayed answer"
_NO_SCHEMA = "only structured_llm_call parses a schema answer from text"
_NO_COMPOSE = "only tool_llm_call takes output_schema= (compose)"


def _first_chunk(family: Family) -> State:
    return FLOAT if family in STREAMS else NULL


def _cells() -> list[Cell]:
    cells: list[Cell] = []
    for family in FAMILIES:
        tool = family in TOOL_LANES
        # Success; on the tool lanes this is also the tool round: ``tools``
        # and ``tool_calls`` are populated.
        cells.append(
            Cell(
                family,
                "success",
                one(
                    "ok",
                    error=NULL,
                    response=eq(_SUCCESS_RESPONSE[family]),
                    usage=eq(_USAGE_DUMP),
                    approximate_cost=eq(0.001),
                    queue_wait_ms=FLOAT,
                    first_chunk_ms=_first_chunk(family),
                    tool_calls=eq([_TOOL_CALL]) if tool else NULL,
                    output_limit=NULL,
                    partial_text=NULL,
                ),
                header_cost="0.001",
            )
        )
        # Transport failures, each with retry=NO_RETRY.
        for outcome, type_name in _TRANSPORT.items():
            cells.append(
                Cell(
                    family,
                    outcome,
                    one(
                        "ERROR",
                        error=transport_error(type_name),
                        usage=NULL,
                        approximate_cost=NULL,
                        queue_wait_ms=FLOAT,
                        response=eq(no_answer(family)),
                        tool_calls=NULL,
                    ),
                )
            )
        # Validation failures. A failed parse was paid for: its usage (the
        # in-call total over the default one repair re-ask, two sends) and
        # cost are recorded, and the rejected answer is in ``partial_text``.
        cells.append(
            Cell(
                family,
                "invalid_json",
                one(
                    "ERROR",
                    error=matches(r"StructuredOutputError: .+"),
                    response=NULL,
                    queue_wait_ms=FLOAT,
                    usage=eq({"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}),
                    approximate_cost=eq(0.001),
                    partial_text=eq("not json"),
                ),
            )
            if family == "structured"
            else na(family, "invalid_json", _NO_SCHEMA)
        )
        cells.append(
            Cell(
                family,
                "on_result_rejects",
                one(
                    "ERROR",
                    error=eq("ValueError: rejected"),
                    response=eq(_SUCCESS_RESPONSE[family]),
                    usage=eq(_EMPTY_USAGE),
                ),
            )
            if family in ("structured", "text")
            else na(family, "on_result_rejects", _NO_HOOK)
        )
        cells.append(
            Cell(
                family,
                "all_tool_calls_malformed",
                one(
                    "ERROR",
                    error=eq("ToolArgumentError: tool arguments are not valid JSON"),
                    tool_calls=NULL,
                    response=eq(no_answer(family)),
                    first_chunk_ms=_first_chunk(family),
                    # The rejected round's raw arguments.
                    partial_text=eq("{"),
                ),
            )
            if tool
            else na(family, "all_tool_calls_malformed", _NO_TOOLS)
        )
        cells.append(
            Cell(
                family,
                "some_tool_calls_malformed",
                one(
                    "ok",
                    error=NULL,
                    tool_calls=tool_calls_named("call_1"),
                    response=MAPPING,
                ),
                reason=(
                    "ok by design: parse_tool_calls (calls/_shared.py) drops the "
                    "malformed call, keeps the survivor and lists the dropped one "
                    "under response.invalid_calls"
                ),
            )
            if tool
            else na(family, "some_tool_calls_malformed", _NO_TOOLS)
        )
        # Retry attempt: two records sharing call_id, attempts 1 and 2.
        cells.append(
            Cell(
                family,
                "retry_attempt",
                (
                    Expect(
                        "ERROR",
                        {
                            "error": transport_error("InternalServerError"),
                            "attempt": eq(1),
                        },
                    ),
                    Expect(
                        "ok",
                        {
                            "error": NULL,
                            "attempt": eq(2),
                            "call_id": same_as("call_id", 0),
                            "response": MAPPING if tool or family == "structured" else eq("ok"),
                        },
                    ),
                ),
            )
        )
        # Cache hit: the paid attempt, then the hit.
        cells.append(
            Cell(
                family,
                "cache_hit",
                (
                    Expect("ok", {"cache_hit": eq(False), "queue_wait_ms": FLOAT}),
                    Expect(
                        "ok",
                        {
                            "cache_hit": eq(True),
                            "approximate_cost": eq(0.0),
                            "usage": NULL,
                            "queue_wait_ms": NULL,
                            "source_call_id": same_as("call_id", 0),
                            "error": NULL,
                            "response": eq(_SUCCESS_RESPONSE[family]),
                        },
                    ),
                ),
            )
            if not tool
            else na(family, "cache_hit", _NO_CACHE)
        )
        # Rejected cache hit: a paid call under another label stores the
        # answer; this call's on_result rejects the stored answer, which is
        # its own attempt, and then accepts the live one numbered after it.
        cells.append(
            Cell(
                family,
                "rejected_cache_hit",
                (
                    Expect(
                        "ERROR",
                        {
                            "cache_hit": eq(True),
                            "attempt": eq(1),
                            "error": eq("ResultValidationError: stored answer rejected"),
                            "response": eq(_SUCCESS_RESPONSE[family]),
                            "approximate_cost": eq(0.0),
                            "usage": NULL,
                            "queue_wait_ms": NULL,
                            "source_call_id": matches(r"[0-9a-f]{32}"),
                        },
                    ),
                    Expect(
                        "ok",
                        {
                            "cache_hit": eq(False),
                            "attempt": eq(2),
                            "call_id": same_as("call_id", 0),
                            "error": NULL,
                            "queue_wait_ms": FLOAT,
                            "response": eq(_SUCCESS_RESPONSE[family]),
                        },
                    ),
                ),
            )
            if family in ("structured", "text")
            else na(
                family,
                "rejected_cache_hit",
                _NO_STREAM_HOOK if family == "text_stream" else _NO_CACHE,
            )
        )
        cells.append(
            Cell(
                family,
                "output_limit",
                one(
                    "ERROR",
                    error=matches(r"OutputLimitError: .+"),
                    output_limit=MAPPING,
                    partial_text=eq(_TRUNCATED),
                    response=eq(_TRUNCATED_RESPONSE[family]),
                ),
            )
        )
        cells.append(
            Cell(
                family,
                "mid_stream_failure",
                one(
                    "ERROR",
                    error=eq("RuntimeError: stream died"),
                    response=eq(
                        "partial"
                        if family == "text_stream"
                        else {"text": "partial", "tool_calls": []}
                    ),
                ),
                reason="driven through the private stream seam: the fake fails only before chunk 1",
            )
            if family in STREAMS
            else na(family, "mid_stream_failure", _NO_STREAM)
        )
        cells.append(
            Cell(
                family,
                "abandoned_stream",
                one(
                    "ERROR",
                    error=eq(STREAM_ABANDONED_ERROR),
                    response=eq(
                        "a " if family == "text_stream" else {"text": "a ", "tool_calls": []}
                    ),
                    first_chunk_ms=FLOAT,
                ),
            )
            if family in STREAMS
            else na(family, "abandoned_stream", _NO_STREAM)
        )
        cells.append(
            Cell(
                family,
                "cancellation",
                one(
                    "ERROR",
                    error=eq(STREAM_ABANDONED_ERROR if family in STREAMS else "CancelledError"),
                    response=eq(no_answer(family)),
                    usage=NULL,
                ),
            )
        )
        cells.append(
            Cell(
                family,
                "circuit_open",
                one(
                    "ERROR",
                    error=matches(r"CircuitOpenError: .+"),
                    queue_wait_ms=NULL,
                    usage=NULL,
                    approximate_cost=NULL,
                    label=eq("target"),
                ),
            )
        )
        cells.append(
            Cell(
                family,
                "compose_round",
                one(
                    "ok",
                    error=NULL,
                    response=MAPPING,
                    tool_calls=LIST,
                ),
                overrides={
                    "schema": eq("tools+OkSchema"),
                    "model": eq("gpt-4.1-mini"),
                    "provider": eq("OpenAI"),
                },
            )
            if family == "tool"
            else na(family, "compose_round", _NO_COMPOSE)
        )
    return cells


CELLS: list[Cell] = _cells()


# --- Drivers -------------------------------------------------------------------

_LABEL = "cell"
_BACKOFF_FREE_RETRY = RetryPolicy(max_attempts=2, backoff_base_seconds=0)


async def _expect_raise(call_: Awaitable[object]) -> None:
    try:
        _ = await call_
    except Exception:
        return
    raise AssertionError("the outcome was expected to raise")


def _reply(request: TransportRequest, **kwargs: object) -> Reply:
    content = '{"ok": true}' if request.kind == "structured" else "ok"
    tool_calls = [FakeToolCall("lookup", {"q": "a"})] if request.kind.startswith("tool") else []
    return Reply(content=content, tool_calls=tool_calls, **kwargs)  # pyright: ignore[reportArgumentType]  # test-helper kwargs


def _raising(make: Callable[[], Exception]) -> Callable[[TransportRequest], Reply]:
    def fake(_request: TransportRequest) -> Reply:
        raise make()

    return fake


async def _drive_success(family: Family) -> None:
    with install(lambda r: _reply(r, usage=_USAGE, cost=0.001)):
        _ = await call(family, label=_LABEL)


def _transport_driver(make: Callable[[], Exception]) -> Callable[[Family], Awaitable[None]]:
    async def drive(family: Family) -> None:
        with install(_raising(make)):
            await _expect_raise(call(family, label=_LABEL))

    return drive


async def _drive_invalid_json(family: Family) -> None:
    with install(lambda _r: Reply.text("not json", usage=_USAGE, cost=0.001)):
        await _expect_raise(call(family, label=_LABEL))


async def _drive_on_result(family: Family) -> None:
    def reject(_value: object) -> None:
        raise ValueError("rejected")

    with install(fake_answer):
        await _expect_raise(call(family, label=_LABEL, on_result=reject))


async def _drive_all_malformed(family: Family) -> None:
    with install(lambda _r: Reply(tool_calls=[FakeToolCall("lookup", "{")])):
        await _expect_raise(call(family, label=_LABEL))


async def _drive_some_malformed(family: Family) -> None:
    reply = Reply(tool_calls=[FakeToolCall("lookup", "{"), FakeToolCall("lookup", {"q": "a"})])
    with install(lambda _r: reply):
        _ = await call(family, label=_LABEL)


async def _drive_retry(family: Family) -> None:
    def fake(request: TransportRequest) -> Reply:
        if request.attempt == 1:
            raise server_error()
        return _reply(request) if family in TOOL_LANES else fake_answer(request)

    with install(fake):
        _ = await call(family, label=_LABEL, retry=_BACKOFF_FREE_RETRY)


async def _drive_cache_hit(family: Family) -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(fake_answer) as installed:
        _ = await call(family, label=_LABEL)
        _ = await call(family, label=_LABEL)
    assert len(installed.requests) == 1, "the second call was not a cache hit"


async def _drive_rejected_cache_hit(family: Family) -> None:
    configure_llm_cache(InMemoryLLMCache())
    seen: list[object] = []

    def reject_the_stored_answer(value: object) -> None:
        seen.append(value)
        if len(seen) == 1:
            raise ResultValidationError("stored answer rejected")

    with install(fake_answer) as installed:
        _ = await call(family, label="seed")
        _ = await call(family, label=_LABEL, on_result=reject_the_stored_answer)
    assert len(installed.requests) == 2, "the rejected hit did not go to the provider"
    assert len(seen) == 2, "the hook did not see the stored answer and then the live one"


async def _drive_output_limit(family: Family) -> None:
    with install(lambda _r: Reply.text(_TRUNCATED, finish_reason="length")):
        await _expect_raise(call(family, label=_LABEL))


async def _drive_mid_stream(family: Family) -> None:
    def seam(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield "par"
            yield "tial"
            raise RuntimeError("stream died")

        return gen()

    target = "astream_text" if family == "text_stream" else "astream_tools"
    with patch(f"llmkit._litellm.{target}", side_effect=seam):
        await _expect_raise(call(family, label=_LABEL))


async def _drive_abandoned(family: Family) -> None:
    async def nothing() -> None:
        return None

    with install(lambda _r: Reply(chunks=["a ", "b ", "c"])):
        _ = await call(family, label=_LABEL, after_first=nothing)


async def _drive_cancellation(family: Family) -> None:
    started = asyncio.Event()

    async def stall(request: TransportRequest) -> Reply:
        started.set()
        await asyncio.sleep(30)
        return fake_answer(request)

    with install(stall):
        task = asyncio.create_task(call(family, label=_LABEL))
        _ = await asyncio.wait_for(started.wait(), timeout=5)
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def _drive_circuit_open(family: Family) -> None:
    configure_rate_limit(breaker=True)
    with install(_raising(rate_limit_error)) as installed:
        for _ in range(BREAKER_MIN_SAMPLES):
            await _expect_raise(call("text", label="trip"))
        tripped = len(installed.requests)
        await _expect_raise(call(family, label="target"))
    assert len(installed.requests) == tripped, "the target call reached the provider"


async def _drive_compose(family: Family) -> None:
    assert family == "tool"
    provider = make_provider(Provider.OPENAI, api_key="placeholder", model="gpt-4.1-mini")
    with install(lambda _r: Reply.json(OkSchema(ok=True))):
        _ = await llm_calls.tool_llm_call(
            "hi",
            TOOLS,
            feature=FEATURE,
            label=_LABEL,
            output_schema=OkSchema,
            provider=provider,
            retry=_BACKOFF_FREE_RETRY,
        )


DRIVERS: dict[Outcome, Callable[[Family], Awaitable[None]]] = {
    "success": _drive_success,
    "rate_limit": _transport_driver(rate_limit_error),
    "service_unavailable": _transport_driver(service_unavailable_error),
    "server_error": _transport_driver(server_error),
    "timeout": _transport_driver(timeout_error),
    "invalid_json": _drive_invalid_json,
    "on_result_rejects": _drive_on_result,
    "all_tool_calls_malformed": _drive_all_malformed,
    "some_tool_calls_malformed": _drive_some_malformed,
    "retry_attempt": _drive_retry,
    "cache_hit": _drive_cache_hit,
    "rejected_cache_hit": _drive_rejected_cache_hit,
    "output_limit": _drive_output_limit,
    "mid_stream_failure": _drive_mid_stream,
    "abandoned_stream": _drive_abandoned,
    "cancellation": _drive_cancellation,
    "circuit_open": _drive_circuit_open,
    "compose_round": _drive_compose,
}


# --- Checks ---------------------------------------------------------------------


def check_cell(cell: Cell, records: list[DiskRecord]) -> None:
    assert len(records) == len(cell.expects), [r.path.name for r in records]
    if len(records) > 1 and cell.outcome in ("retry_attempt", "rejected_cache_hit"):
        assert len({r.body["call_id"] for r in records}) == 1
    for record, expect in zip(records, cell.expects, strict=True):
        assert record.status == expect.status, record.header[0]
        states = {**base_states(cell.family), **cell.overrides, **expect.states}
        failures = {
            key: (state, record.body[key])
            for key, state in states.items()
            if not state.holds(record.body[key], records)
        }
        assert not failures, (record.path.name, failures)
    if cell.header_cost is not None:
        assert records[0].header[0].endswith(f"| ${cell.header_cost}")


# --- Tests ----------------------------------------------------------------------


def test_delivery_is_available() -> None:
    """First in the module: a dead worker would drop every record below."""
    assert_delivery_available()


def test_the_cell_list_covers_every_family_and_outcome_once() -> None:
    seen = [(cell.family, cell.outcome) for cell in CELLS]
    assert sorted(seen) == sorted((f, o) for f in FAMILIES for o in OUTCOMES)
    assert len(set(seen)) == len(seen)
    assert set(DRIVERS) == set(OUTCOMES)
    for cell in CELLS:
        assert cell.applicable == bool(cell.expects), cell.id
        assert cell.applicable or cell.reason, f"{cell.id}: a not-applicable cell needs a reason"


@pytest.fixture
def log_dir(tmp_path: Path) -> Iterator[Path]:
    assert_delivery_available()
    directory = tmp_path / "logs"
    with sink_installed(LocalYamlLogSink(directory)):
        yield directory


_APPLICABLE = [c for c in CELLS if c.applicable]


@pytest.mark.parametrize("cell", _APPLICABLE, ids=[c.id for c in _APPLICABLE])
async def test_cell_writes_one_complete_record_per_attempt(cell: Cell, log_dir: Path) -> None:
    await DRIVERS[cell.outcome](cell.family)
    health = await settle()

    every = read_disk(log_dir)
    assert health.written == len(every), (health, len(every))
    for record in every:
        check_contract(record)
    label = "target" if cell.outcome == "circuit_open" else _LABEL
    check_cell(cell, [r for r in every if r.body["label"] == label])
