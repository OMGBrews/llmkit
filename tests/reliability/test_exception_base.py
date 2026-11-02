"""Every exception class llmkit defines shares one base, ``LlmkitError``.

``except LlmkitError`` is the one-name handler for failures llmkit itself
raises. The lazy litellm stand-in is excluded: its metaclass exists only for
``isinstance`` classification of litellm's own class, which llmkit does not
define. The base must not change any catch-set, so the tuples are pinned too.
"""

from __future__ import annotations

import inspect
from typing import cast

import llmkit
import llmkit.exceptions as exceptions_module
from llmkit import LlmkitError


def _defined_exception_classes() -> list[type[BaseException]]:
    return [
        cls
        for _, cls in inspect.getmembers(exceptions_module, inspect.isclass)
        if cls.__module__ == exceptions_module.__name__ and issubclass(cls, BaseException)
    ]


def test_every_llmkit_exception_inherits_the_base() -> None:
    classes = _defined_exception_classes()
    assert len(classes) > 5, "the scan found the exception classes"
    outside = {
        cls.__name__
        for cls in classes
        if not issubclass(cls, LlmkitError) and cls is not LlmkitError
    }
    assert outside == {"_LiteLLMServiceUnavailableError"}


def test_every_exported_exception_inherits_the_base() -> None:
    exported = [
        obj
        for name in llmkit.__all__
        if inspect.isclass(obj := cast("object", getattr(llmkit, name)))
        and issubclass(obj, BaseException)
    ]
    assert exported, "the scan found exported exception classes"
    assert all(issubclass(cls, LlmkitError) for cls in exported)


def test_the_base_is_in_no_catch_set_or_retry_default() -> None:
    for name in dir(exceptions_module):
        if name.startswith("LLM_") and name.endswith("_ERRORS"):
            members = cast("tuple[type[BaseException], ...]", getattr(exceptions_module, name))
            assert LlmkitError not in members, name
    assert LlmkitError not in llmkit.DEFAULT_RETRY_POLICY.retry_on
