"""A stream interrupted by a ``KeyboardInterrupt`` or ``SystemExit`` says so.

Both are ``BaseException`` subclasses that are neither a cancellation nor a
consumer's ``GeneratorExit``, so they bypass a stream attempt's ``except
Exception`` clause; without a clause of their own the ``finally`` would write
the attempt as a clean ``ok`` with ``error: null``. Each lane records
``"<Type>: <message>"`` instead, as the buffered lanes do, and re-raises the
interrupt untouched. The cache-hit branch of ``text_llm_call_stream`` is
pinned in ``tests/cache/test_stream.py``.

The interrupt is raised from the private stream seam after one chunk, because
a :mod:`llmkit.testing` fake can fail a stream only before its first chunk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import patch

import pytest

from llmkit import (
    NO_RETRY,
    ToolDefinition,
    capture_llm_records,
    text_llm_call_stream,
    tool_llm_call_stream,
)

_TOOLS = [
    ToolDefinition(
        "lookup", "Look a term up.", {"type": "object", "properties": {"q": {"type": "string"}}}
    )
]


def _seam(interrupt: type[BaseException]) -> object:
    def seam(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield "par"
            raise interrupt("stop")

        return gen()

    return seam


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
async def test_text_stream_records_the_interrupt(interrupt: type[BaseException]) -> None:
    with (
        patch("llmkit._litellm.astream_text", side_effect=_seam(interrupt)),
        capture_llm_records() as records,
        pytest.raises(interrupt),
    ):
        async for _chunk in text_llm_call_stream("hi", feature="f", retry=NO_RETRY):
            pass

    [record] = records
    assert record.error == f"{interrupt.__name__}: stop"
    assert cast("object", record.response) == "par"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
async def test_tool_stream_records_the_interrupt(interrupt: type[BaseException]) -> None:
    with (
        patch("llmkit._litellm.astream_tools", side_effect=_seam(interrupt)),
        capture_llm_records() as records,
        pytest.raises(interrupt),
    ):
        async for _event in tool_llm_call_stream("hi", _TOOLS, feature="f", retry=NO_RETRY):
            pass

    [record] = records
    assert record.error == f"{interrupt.__name__}: stop"
    assert cast("object", record.response) == {"text": "par", "tool_calls": []}
