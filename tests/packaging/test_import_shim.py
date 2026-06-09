"""Test the ``omg_llmkit`` install-name import shim.

The distribution installs as ``omg-llmkit`` but is imported as ``llmkit``. A
mistaken ``import omg_llmkit`` (e.g. a post-install smoke test that uses the
install name) should surface an actionable redirect, not a bare
``ModuleNotFoundError`` that gives no hint about the split.
"""

from __future__ import annotations

import importlib

import pytest


def test_import_omg_llmkit_raises_actionable_redirect() -> None:
    """Importing the install name raises an ``ImportError`` pointing at ``llmkit``."""
    with pytest.raises(ImportError, match=r"imported as 'llmkit', not 'omg_llmkit'"):
        _ = importlib.import_module("omg_llmkit")
