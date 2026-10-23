"""Shared fixtures for the public testing-transport tests.

These tests drive the public call functions the way a downstream project
would: a real provider config (Ollama, which needs no key), the fake installed
through :mod:`llmkit.testing`, and nothing patched. The autouse fixture gives
each test that config, no cache, no YAML log, and the shipped rate-limit
defaults, and restores all four afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llmkit import LLMClientConfig, Provider, configure_llm_cache, configure_llm_client
from llmkit.providers import base as providers_base
from llmkit.rate_limiting import GlobalRateLimiter
from tests._support import quiet_logging


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """An Ollama provider config, no cache, muted logging, default limits."""
    previous = providers_base._config_source
    configure_llm_client(lambda: LLMClientConfig(provider=Provider.OLLAMA, model="fake-model"))
    configure_llm_cache(None)
    GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
    try:
        with quiet_logging():
            yield
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
        configure_llm_cache(None)
        providers_base._config_source = previous
