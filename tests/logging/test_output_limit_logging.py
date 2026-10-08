"""Output-limit diagnostics delivered through the logger, end to end.

A structured completion cut off by its output limit runs through instructor's
real machinery (the provider seam ``llmkit._litellm.litellm.acompletion`` is
patched, as in ``tests/reliability/test_output_limit_failfast.py``), raises
:class:`~llmkit.OutputLimitError`, and is read back from the YAML a real
:class:`~llmkit.LocalYamlLogSink` wrote: the partial text and token counts are
in the file, ``response`` stays null, and the compact index carries neither.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import pytest
import yaml
from litellm.types.utils import ModelResponse, Usage

from llmkit import (
    NO_RETRY,
    LocalYamlLogSink,
    OutputLimitError,
    configure_llm_logging,
)
from llmkit import calls as llm_calls
from llmkit.logging import INDEX_FILENAME
from llmkit.retry import with_retries
from tests._support import OkSchema


def _response(content: str, *, finish_reason: str, completion_tokens: int) -> ModelResponse:
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage=Usage(
            prompt_tokens=7,
            completion_tokens=completion_tokens,
            total_tokens=7 + completion_tokens,
        ),
    )
    resp._hidden_params = {}  # mirror litellm's cost-lookup attribute
    return resp


def _fake_provider() -> MagicMock:
    """A real instructor mode over the patched seam, with a string ``model``
    so the sink's summary header can be built."""
    provider = MagicMock()
    provider.name = "fake-provider"
    provider.model = "fake/model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None
    return provider


@pytest.fixture
def log_dir(tmp_path: Path) -> Iterator[Path]:
    """A real YAML sink for this test, overriding the package's quiet one."""
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        yield tmp_path
    finally:
        configure_llm_logging(None)


def _read_logs(log_dir: Path) -> list[dict[str, object]]:
    docs = [
        cast("dict[str, object]", yaml.safe_load(path.read_text()))
        for path in sorted(log_dir.glob("*.yaml"))
    ]
    return sorted(docs, key=lambda doc: cast("int", doc["attempt"]))


def _index_lines(log_dir: Path) -> list[str]:
    return (log_dir / INDEX_FILENAME).read_text().splitlines()


async def test_truncated_call_yaml_keeps_partial_text_and_usage(log_dir: Path) -> None:
    """Completion -> exception -> record -> YAML with known partial text."""
    truncated = _response('{"ok": tru', finish_reason="length", completion_tokens=64)
    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=truncated)),
        pytest.raises(OutputLimitError),
    ):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="trunc", max_tokens=64, provider=_fake_provider()
        )

    [doc] = _read_logs(log_dir)
    assert doc["partial_text"] == '{"ok": tru'
    assert doc["output_limit"] == {
        "finish_reason": "length",
        "prompt_tokens": 7,
        "completion_tokens": 64,
        "total_tokens": 71,
        # instructor's usage accumulator zero-fills a missing breakdown.
        "reasoning_tokens": 0,
    }
    assert str(doc["error"]).startswith("OutputLimitError: ")
    assert doc["response"] is None
    assert doc["call_id"] is not None
    assert doc["attempt"] == 1
    assert doc["usage"] is None  # tool-lane only; unchanged

    [line] = _index_lines(log_dir)
    entry = cast("dict[str, object]", json.loads(line))
    assert not {"partial_text", "output_limit"} & entry.keys()
    assert all('{"ok": tru' not in str(value) for value in entry.values())


async def test_successful_call_writes_null_diagnostics(log_dir: Path) -> None:
    """A clean structured call is unchanged apart from the two null keys."""
    ok = _response('{"ok": true}', finish_reason="stop", completion_tokens=5)
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)):
        result = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="ok", provider=_fake_provider()
        )

    assert result.ok is True
    [doc] = _read_logs(log_dir)
    assert doc["output_limit"] is None
    assert doc["partial_text"] is None
    assert doc["response"] == {"ok": True}
    assert doc["error"] is None


async def test_default_policy_logs_one_attempt(log_dir: Path) -> None:
    """Fail-fast stays the default: one attempt, one file, no backoff."""
    truncated = _response('{"ok": tru', finish_reason="length", completion_tokens=64)
    sleep_mock = AsyncMock()
    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=truncated)),
        patch("llmkit.retry.asyncio.sleep", sleep_mock),
        pytest.raises(OutputLimitError),
    ):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="trunc", provider=_fake_provider()
        )

    assert len(_read_logs(log_dir)) == 1
    assert sleep_mock.await_count == 0


async def test_opted_in_retry_logs_each_attempts_own_diagnostics(log_dir: Path) -> None:
    """An explicit outer ``retry_on`` opt-in still retries, and each attempt's
    file carries that attempt's own partial text and counts under one
    ``call_id`` — the two generations differ, so a record reusing the first
    attempt's payload would fail."""
    generations = [
        _response('{"ok": tr', finish_reason="length", completion_tokens=40),
        _response('{"ok": tru', finish_reason="length", completion_tokens=64),
    ]
    sleep_mock = AsyncMock()

    async def call() -> OkSchema:
        return await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="trunc", retry=NO_RETRY, provider=_fake_provider()
        )

    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(side_effect=generations)),
        patch("llmkit.retry.asyncio.sleep", sleep_mock),
        pytest.raises(OutputLimitError),
    ):
        _ = await with_retries(call, max_attempts=2, retry_on=(OutputLimitError,))

    first, second = _read_logs(log_dir)
    assert first["call_id"] == second["call_id"]
    assert (first["attempt"], second["attempt"]) == (1, 2)
    assert first["partial_text"] == '{"ok": tr'
    assert second["partial_text"] == '{"ok": tru'
    assert cast("dict[str, object]", first["output_limit"])["completion_tokens"] == 40
    assert cast("dict[str, object]", second["output_limit"])["completion_tokens"] == 64
