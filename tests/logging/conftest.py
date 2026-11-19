"""Shared fixtures for the logging tests.

The disk-facing tests use an explicit ``LocalYamlLogSink(tmp_path)`` and so are
unaffected by the global sink; the call-driven ones fake the provider through
:mod:`llmkit.testing` and should not write a stray log. The autouse fixture
mutes the global sink by default and gives each test a keyless Ollama provider
config, no cache and the shipped rate-limiter defaults, so those calls run the
real limiter, retry and parsing around the fake. A second autouse fixture
clears the logging environment switches, so a developer's ``LLMKIT_LOG`` or
``LLMKIT_LOG_RETENTION_DAYS`` cannot change what these tests see.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llmkit.logging import LOG_ENV_VAR, LOG_RETENTION_ENV_VAR
from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _no_logging_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the logging environment switches for the test and its children."""
    monkeypatch.delenv(LOG_ENV_VAR, raising=False)
    monkeypatch.delenv(LOG_RETENTION_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """Mute the global sink and configure Ollama, no cache and default limits."""
    with ollama_call_defaults():
        yield
