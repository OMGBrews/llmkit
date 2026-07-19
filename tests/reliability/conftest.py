"""Shared fixtures for the reliability tests.

Every reliability test runs against the *same* process-global
:class:`~llmkit.rate_limiting.GlobalRateLimiter`, so one autouse fixture resets
it to the shipped default around each test — the reset the four reliability
suites (rate limiting, adaptive concurrency, backpressure observability, circuit
breaker) previously copy-pasted verbatim.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llmkit.rate_limiting import GlobalRateLimiter


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    """Restore the shipped default around each test: cap 8, enabled, adaptive on,
    breaker off.

    The limiter is a process-global; ``configure`` also clears the gate / state /
    bucket registries, so no gate or adaptive state constructed in one test can
    leak into the next (and any slot-acquiring test gains isolation).
    """
    GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
    try:
        yield
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
