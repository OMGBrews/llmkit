"""Shared fixtures for the reliability tests.

Every reliability test runs against the *same* process-global
:class:`~llmkit.rate_limiting.GlobalRateLimiter`, so one autouse fixture resets
it to the shipped default around each test. The same fixture gives each test a
keyless Ollama provider config, no cache and muted logging, so the call-function
tests can fake the provider through :mod:`llmkit.testing` and run the real
limiter, breaker, retry and parsing around it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    """Restore the shipped default around each test: cap 8, enabled, adaptive on,
    breaker off; and configure Ollama, no cache and muted logging.

    The limiter is a process-global; ``configure`` also clears the gate / state /
    bucket registries, so no gate or adaptive state constructed in one test can
    leak into the next (and any slot-acquiring test gains isolation).
    """
    with ollama_call_defaults():
        yield
