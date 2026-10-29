"""The plain text stream asks for usage and carries it to the call record.

Transport-level tests drive the real ``astream_text`` over a faked LiteLLM
stream, because the :mod:`llmkit.testing` double ignores request kwargs and so
cannot see the ``stream_options`` flag. The inputs that select the behaviour
vary: usage on the final text-less frame, usage only on the stream wrapper, no
usage at all, and a cost on a frame or only the wrapper's ``0.0`` placeholder.

Mutation check: with ``"stream_options"`` removed from ``astream_text``'s
request, ``test_transport_requests_usage_on_the_stream`` fails; with the
terminal ``StreamedTextEnd`` yield removed, every test reading ``items[-1]``
fails and the record tests see ``usage=None``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import cast, final
from unittest.mock import MagicMock, patch

import pytest

from llmkit import NO_RETRY, OutputLimitError, capture_llm_records, text_llm_call_stream
from llmkit._litellm import StreamedTextEnd


@final
class _FakeUsage:
    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


@final
class _FakeChunk:
    """A stream frame in the shape LiteLLM produces (loose by design)."""

    def __init__(
        self,
        *,
        content: str | None = None,
        finish_reason: str | None = None,
        usage: object = None,
        cost: float | None = None,
    ) -> None:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        self.choices = [choice] if (content or finish_reason) else []
        self.usage = usage
        self._hidden_params: dict[str, object] = {"response_cost": cost} if cost is not None else {}


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.name = "openai"
    provider.model = "fake-model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.litellm_model = MagicMock(return_value="openai/fake")
    provider.reasoning_effort = None
    return provider


@final
class _FakeStream:
    """The wrapper LiteLLM returns; ``usage`` and cost may sit on it instead."""

    def __init__(
        self, chunks: list[_FakeChunk], usage: object = None, cost: float | None = None
    ) -> None:
        self._chunks = chunks
        self.usage = usage
        self._hidden_params: dict[str, object] = {"response_cost": cost} if cost is not None else {}

    def __aiter__(self) -> AsyncIterator[_FakeChunk]:
        async def _gen() -> AsyncIterator[_FakeChunk]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


@pytest.fixture
def debits() -> Iterator[list[int | None]]:
    """Every token count ``astream_text`` debits on its rate-limit slot."""
    recorded: list[int | None] = []
    slot = MagicMock()
    slot.record_tokens = recorded.append

    @asynccontextmanager
    async def _acquire(_name: str) -> AsyncGenerator[MagicMock]:
        yield slot

    with patch("llmkit._litellm.GlobalRateLimiter.acquire_async", _acquire):
        yield recorded


async def _drive(
    stream: _FakeStream, *, max_tokens: int | None = None
) -> tuple[list[str | StreamedTextEnd], dict[str, object]]:
    from llmkit import _litellm

    seen: dict[str, object] = {}

    async def _fake_acompletion(**request: object) -> _FakeStream:
        seen.update(request)
        return stream

    items: list[str | StreamedTextEnd] = []
    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        async for item in _litellm.astream_text(
            "hi", temperature=None, model=None, max_tokens=max_tokens, provider=_provider()
        ):
            items.append(item)
    return items, seen


async def test_transport_requests_usage_on_the_stream(debits: list[int | None]) -> None:
    """The flag is sent, and the final text-less usage frame yields no delta."""
    stream = _FakeStream(
        [
            _FakeChunk(content="he"),
            _FakeChunk(content="llo", finish_reason="stop"),
            _FakeChunk(usage=_FakeUsage(11, 22, 33), cost=0.004),
        ]
    )
    items, request = await _drive(stream)

    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert items == ["he", "llo", StreamedTextEnd(usage=(11, 22, 33), approximate_cost=0.004)]
    assert debits == [33]


async def test_transport_falls_back_to_the_wrappers_usage(debits: list[int | None]) -> None:
    """A route that reports usage only on the stream object still records it."""
    stream = _FakeStream(
        [_FakeChunk(content="hi", finish_reason="stop")], usage=_FakeUsage(5, 6, 11)
    )
    items, _request = await _drive(stream)

    assert items == ["hi", StreamedTextEnd(usage=(5, 6, 11), approximate_cost=None)]
    assert debits == [11]


async def test_the_wrappers_placeholder_cost_is_not_recorded(debits: list[int | None]) -> None:
    """LiteLLM seeds the stream wrapper's ``response_cost`` with ``0.0`` and never
    prices the stream; recording it would log a paid call as free."""
    stream = _FakeStream(
        [_FakeChunk(content="hi", finish_reason="stop"), _FakeChunk(usage=_FakeUsage(1, 2, 3))],
        usage=_FakeUsage(7, 8, 15),
        cost=0.0,
    )
    items, _request = await _drive(stream)

    # Usage on a frame wins over the wrapper's; the placeholder cost is dropped.
    assert items[-1] == StreamedTextEnd(usage=(1, 2, 3), approximate_cost=None)
    assert debits == [3]


async def test_transport_reports_nulls_when_the_provider_reports_nothing(
    debits: list[int | None],
) -> None:
    stream = _FakeStream([_FakeChunk(content="hi", finish_reason="stop")])
    items, _request = await _drive(stream)

    assert items == ["hi", StreamedTextEnd(usage=(None, None, None), approximate_cost=None)]
    assert debits == [None]


async def test_a_truncated_stream_yields_its_counts_before_raising(
    debits: list[int | None],
) -> None:
    from llmkit import _litellm

    stream = _FakeStream(
        [
            _FakeChunk(content="partial", finish_reason="length"),
            _FakeChunk(usage=_FakeUsage(4, 8, 12)),
        ]
    )

    async def _fake_acompletion(**_request: object) -> _FakeStream:
        return stream

    items: list[str | StreamedTextEnd] = []
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
        pytest.raises(OutputLimitError) as caught,
    ):
        async for item in _litellm.astream_text(
            "hi", temperature=None, model=None, max_tokens=8, provider=_provider()
        ):
            items.append(item)

    assert items == ["partial", StreamedTextEnd(usage=(4, 8, 12), approximate_cost=None)]
    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.completion_tokens == 8
    assert debits == [12]


async def test_the_record_carries_the_streams_usage_and_cost(debits: list[int | None]) -> None:
    """End to end through ``text_llm_call_stream``: the terminal item never
    reaches the consumer, and its counts reach the record."""
    stream = _FakeStream(
        [
            _FakeChunk(content="he"),
            _FakeChunk(content="llo", finish_reason="stop"),
            _FakeChunk(usage=_FakeUsage(11, 22, 33), cost=0.004),
        ]
    )

    async def _fake_acompletion(**_request: object) -> _FakeStream:
        return stream

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
        capture_llm_records() as records,
    ):
        chunks = [
            chunk
            async for chunk in text_llm_call_stream(
                "hi", feature="usage", provider=_provider(), retry=NO_RETRY, cache=False
            )
        ]

    assert chunks == ["he", "llo"]
    assert len(records) == 1
    assert records[0].usage == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    assert records[0].approximate_cost == 0.004
    assert cast("object", records[0].response) == "hello"
    assert debits == [33]
