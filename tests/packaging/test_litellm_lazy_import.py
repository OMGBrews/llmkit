"""Import-time laziness of litellm.

``import llmkit`` must not import litellm: the multi-second litellm import is
deferred to the first provider call (``llmkit._litellm``). The transport
classification in :mod:`llmkit.exceptions` participates via a lazy stand-in
for ``litellm.exceptions.ServiceUnavailableError`` that resolves the real
class only once litellm is genuinely loaded. These tests run fresh
interpreters so the parent test session's own litellm import can't mask a
regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_python(code: str) -> None:
    """Run *code* in a fresh interpreter, failing the test on a non-zero exit."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_import_llmkit_does_not_import_litellm() -> None:
    """``import llmkit`` leaves litellm out of ``sys.modules``.

    Regression test: a module-level ``import litellm`` in
    ``llmkit.exceptions`` (added for the 503 transport entry) made every
    ``import llmkit`` pay the full multi-second litellm import, roughly
    doubling import time for a local-first library consumed by CLIs.
    """
    _run_python("import llmkit, sys; assert 'litellm' not in sys.modules")


def test_transport_classification_works_without_litellm_loaded() -> None:
    """The lazy 503 entry is inert before litellm loads.

    Classifying against ``LLM_TRANSPORT_ERRORS`` neither imports litellm nor
    crashes: a litellm exception instance can't exist until litellm is in
    ``sys.modules``, so the stand-in simply matches nothing yet.
    """
    _run_python(
        textwrap.dedent(
            """
            import sys
            import llmkit
            assert isinstance(TimeoutError(), llmkit.LLM_TRANSPORT_ERRORS)
            assert not isinstance(ValueError(), llmkit.LLM_TRANSPORT_ERRORS)
            assert 'litellm' not in sys.modules
            """
        )
    )


def test_litellm_503_classifies_as_transport_once_litellm_imported() -> None:
    """Once litellm IS loaded, its real 503 still classifies as transport.

    The stand-in resolves ``litellm.exceptions.ServiceUnavailableError`` at
    ``isinstance`` time, so a genuine litellm 503 raised after the (lazy)
    litellm import matches ``LLM_TRANSPORT_ERRORS`` exactly as the previous
    eager listing did.
    """
    _run_python(
        textwrap.dedent(
            """
            import sys
            import llmkit
            assert 'litellm' not in sys.modules
            import litellm
            err = litellm.exceptions.ServiceUnavailableError(
                message='service unavailable', llm_provider='openai', model='gpt-4o-mini'
            )
            assert isinstance(err, llmkit.LLM_TRANSPORT_ERRORS)
            assert isinstance(err, llmkit.LLM_RECOVERABLE_ERRORS)
            """
        )
    )
