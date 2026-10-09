"""Shared fixtures and doubles for the response-cache tests.

Nothing else resets the process-global cache, so every test here starts and
ends with none configured. Logging is muted as in the call tests; a test that
inspects records opts back in with ``capture_llm_records`` or its own sink.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from llmkit import configure_llm_cache
from tests._support import provider_mock, quiet_logging


@pytest.fixture(autouse=True)
def _no_cache_and_quiet_logging() -> Iterator[None]:
    """No cache configured and no YAML written, before and after each test."""
    configure_llm_cache(None)
    try:
        with quiet_logging():
            yield
    finally:
        configure_llm_cache(None)


def cache_provider(
    *,
    name: str = "Google AI Studio",
    model: str = "gemini-3.1-flash-lite",
    request_kwargs: dict[str, object] | None = None,
    instructor_mode: str = "json_schema",
    strict_json_schema: bool = False,
    reasoning_effort: str | None = None,
) -> MagicMock:
    """A provider double whose every attribute the cache key reads is set.

    A bare ``MagicMock`` attribute is not JSON-serialisable, so the key would
    fail, the call would quietly bypass the cache, and a hit test would pass
    straight through to the fake transport. Tests assert the cache warning is
    absent to keep that from happening unnoticed.
    """
    provider = provider_mock(
        model=model,
        name=name,
        instructor_mode=instructor_mode,
        strict_json_schema=strict_json_schema,
        reasoning_effort=reasoning_effort,
    )

    def _litellm_model(override: str | None = None) -> str:
        return f"gemini/{override or model}"

    provider.litellm_model = MagicMock(side_effect=_litellm_model)
    provider.completion_kwargs = MagicMock(
        return_value=request_kwargs
        if request_kwargs is not None
        else {"api_key": "secret", "api_base": "https://example.test/v1"}
    )
    return provider
