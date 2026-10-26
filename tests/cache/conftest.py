"""Shared fixtures and doubles for the response-cache tests.

Nothing else resets the process-global cache, so every test here starts and
ends with none configured. The autouse fixture also gives each test a keyless
Ollama provider config, muted logging and the shipped rate-limit defaults, so
the call tests can fake the provider through :mod:`llmkit.testing`; a test
turns the cache on in its own body, after that setup has turned it off. A test
that inspects records opts back in with ``capture_llm_records`` or its own sink.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import instructor
import pytest

from tests._support import ollama_call_defaults, provider_mock

#: The structured-output mode for a double that runs the real structured
#: transport, which needs an ``instructor.Mode``. ``cache_provider``'s plain
#: string default stays for the key tests, whose pinned digest hashes it.
CALL_MODE = instructor.Mode.JSON_SCHEMA


@pytest.fixture(autouse=True)
def _ollama_no_cache_and_quiet_logging() -> Iterator[None]:
    """Ollama config, no cache and no YAML written, before and after each test."""
    with ollama_call_defaults():
        yield


def cache_provider(
    *,
    name: str = "Google AI Studio",
    model: str = "gemini-3.1-flash-lite",
    request_kwargs: dict[str, object] | None = None,
    instructor_mode: instructor.Mode | str = "json_schema",
    strict_json_schema: bool = False,
    reasoning_effort: str | None = None,
) -> MagicMock:
    """A provider double whose every attribute the cache key and the request path read is set.

    A bare ``MagicMock`` attribute is not JSON-serialisable, so the key would
    fail, the call would quietly bypass the cache, and a hit test would pass
    straight through to the fake transport. Tests assert the cache warning is
    absent to keep that from happening unnoticed. The request-path hooks
    return real values too (``reasoning_kwargs`` a dict, the parallel-tool-call
    trait a bool), so the double survives the real transport up to an
    installed :mod:`llmkit.testing` fake.
    """
    provider = provider_mock(
        model=model,
        name=name,
        instructor_mode=instructor_mode,
        strict_json_schema=strict_json_schema,
        reasoning_effort=reasoning_effort,
        single_tool_call_per_schema=False,
    )

    def _litellm_model(override: str | None = None) -> str:
        return f"gemini/{override or model}"

    def _reasoning_kwargs(effort: str, _model: str) -> dict[str, object]:
        return {"reasoning_effort": effort}

    provider.litellm_model = MagicMock(side_effect=_litellm_model)
    provider.reasoning_kwargs = MagicMock(side_effect=_reasoning_kwargs)
    provider.completion_kwargs = MagicMock(
        return_value=request_kwargs
        if request_kwargs is not None
        else {"api_key": "secret", "api_base": "https://example.test/v1"}
    )
    return provider
