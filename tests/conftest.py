"""Pytest configuration: the explicit live / offline test split.

llmkit's tests fall into two disjoint, explicitly-marked groups:

* **Offline tests** (the default) never touch the network and never read a
  provider credential. They run everywhere — contributor laptops, CI, no keys —
  and behave identically regardless of environment.
* **Live tests** (``@pytest.mark.live``) make real API calls and need real
  credentials. They run *only* when ``--run-live`` is passed, and then every one
  must pass: a missing key or unreachable server is a hard failure, never a skip.

Whether a live test runs is decided solely by ``--run-live`` — never by whether
a key happens to be present. That matters because importing ``litellm`` runs
``load_dotenv()``, which can inject keys from a stray ``.env`` into the process;
gating on key-presence would make the same command behave differently on
different machines. ``--run-live`` removes that ambiguity: same command, same
behavior, everywhere.
"""

from __future__ import annotations

import pytest

_RUN_LIVE = "--run-live"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-live`` opt-in for the live API tests."""
    parser.addoption(
        _RUN_LIVE,
        action="store_true",
        default=False,
        help=(
            "Run @pytest.mark.live tests, which make real API calls and require "
            "real provider credentials (and a reachable Ollama server). Off by "
            "default; a missing dependency is then a failure, not a skip."
        ),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every ``live`` test unless ``--run-live`` was passed.

    Deterministic and environment-independent: with the flag, live tests run;
    without it, they are skipped at collection no matter what credentials are in
    the environment.
    """
    if config.getoption(_RUN_LIVE):
        return
    skip_live = pytest.mark.skip(reason=f"live test — pass {_RUN_LIVE} to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
