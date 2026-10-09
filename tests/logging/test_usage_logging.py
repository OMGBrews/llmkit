"""Token usage delivered through the logger on the structured and text lanes.

The provider seam ``llmkit._litellm.litellm.acompletion`` is patched, so each
call runs through the real helper (and, on the structured lane, instructor's
real machinery) and is read back from the YAML a real
:class:`~llmkit.LocalYamlLogSink` wrote. ``usage`` carries the provider's own
counts in the shape the tool lanes write, a mapping of nulls when the provider
reported none, ``null`` when the attempt failed before any response arrived,
and never reaches ``index.jsonl``.
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

from llmkit import NO_RETRY, LocalYamlLogSink, ResultValidationError, configure_llm_logging
from llmkit import calls as llm_calls
from llmkit.logging import INDEX_FILENAME
from tests._support import OkSchema

_NO_COUNTS = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _response(content: str, *, usage: Usage | None) -> ModelResponse:
    """A completion; ``usage=None`` leaves it without a usage object at all."""
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        **({"usage": usage} if usage is not None else {}),
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


def _read_log(log_dir: Path) -> dict[str, object]:
    [path] = log_dir.glob("*.yaml")
    return cast("dict[str, object]", yaml.safe_load(path.read_text()))


def _index_entry(log_dir: Path) -> dict[str, object]:
    [line] = (log_dir / INDEX_FILENAME).read_text().splitlines()
    return cast("dict[str, object]", json.loads(line))


async def test_structured_call_logs_provider_usage(log_dir: Path) -> None:
    ok = _response(
        '{"ok": true}', usage=Usage(prompt_tokens=11, completion_tokens=5, total_tokens=16)
    )
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="usage", provider=_fake_provider()
        )

    doc = _read_log(log_dir)
    assert doc["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    assert doc["output_limit"] is None
    assert "usage" not in _index_entry(log_dir)


async def test_text_call_logs_provider_usage(log_dir: Path) -> None:
    ok = _response("hello", usage=Usage(prompt_tokens=9, completion_tokens=2, total_tokens=11))
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)):
        _ = await llm_calls.text_llm_call("hi", feature="usage", provider=_fake_provider())

    doc = _read_log(log_dir)
    assert doc["usage"] == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
    assert "usage" not in _index_entry(log_dir)


async def test_structured_call_without_reported_usage_logs_nulls(log_dir: Path) -> None:
    ok = _response('{"ok": true}', usage=None)
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="usage", provider=_fake_provider()
        )

    assert _read_log(log_dir)["usage"] == _NO_COUNTS


async def test_text_call_without_reported_usage_logs_nulls(log_dir: Path) -> None:
    ok = _response("hello", usage=None)
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)):
        _ = await llm_calls.text_llm_call("hi", feature="usage", provider=_fake_provider())

    assert _read_log(log_dir)["usage"] == _NO_COUNTS


@pytest.mark.parametrize("lane", ["structured", "text"])
async def test_attempt_failing_before_a_response_logs_null_usage(log_dir: Path, lane: str) -> None:
    boom = AsyncMock(side_effect=ValueError("permanent"))
    with patch("llmkit._litellm.litellm.acompletion", boom), pytest.raises(Exception):
        if lane == "structured":
            _ = await llm_calls.structured_llm_call(
                "hi", OkSchema, feature="usage", retry=NO_RETRY, provider=_fake_provider()
            )
        else:
            _ = await llm_calls.text_llm_call(
                "hi", feature="usage", retry=NO_RETRY, provider=_fake_provider()
            )

    doc = _read_log(log_dir)
    assert doc["error"] is not None
    assert doc["usage"] is None


def _reject(_result: object) -> None:
    raise ResultValidationError("not what we wanted")


@pytest.mark.parametrize("lane", ["structured", "text"])
async def test_result_rejected_by_on_result_keeps_its_usage(log_dir: Path, lane: str) -> None:
    """The completion arrived and was paid for; only the hook turned it down."""
    content = '{"ok": true}' if lane == "structured" else "hello"
    ok = _response(content, usage=Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7))
    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=ok)),
        pytest.raises(ResultValidationError),
    ):
        if lane == "structured":
            _ = await llm_calls.structured_llm_call(
                "hi",
                OkSchema,
                feature="usage",
                retry=NO_RETRY,
                on_result=_reject,
                provider=_fake_provider(),
            )
        else:
            _ = await llm_calls.text_llm_call(
                "hi", feature="usage", retry=NO_RETRY, on_result=_reject, provider=_fake_provider()
            )

    doc = _read_log(log_dir)
    assert str(doc["error"]).startswith("ResultValidationError")
    assert doc["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
