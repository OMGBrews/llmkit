"""litellm-native 503s are re-owned so ``except LLM_RECOVERABLE_ERRORS`` works.

litellm's ``ServiceUnavailableError`` participates in the transport set only
through a metaclass ``isinstance`` hook (the lazy stand-in that keeps
``import llmkit`` litellm-free) — and CPython's ``except`` matching bypasses
metaclass hooks. So before normalization, the *documented* host pattern
``except LLM_RECOVERABLE_ERRORS:`` silently missed a genuine litellm 503:
the retry layer classified and retried it fine (``isinstance``), then on
budget exhaustion the raw litellm class blew straight through the host's
degrade path.

The transport boundary (:func:`llmkit._litellm._acompletion` and the stream
iteration in :func:`llmkit._litellm.astream_text`) now re-raises every
litellm-native 503 as llmkit's own :class:`~llmkit.ServiceUnavailableError`
(original on ``__cause__``). These tests pin that contract end-to-end with
**literal ``try``/``except`` clauses** — not ``pytest.raises`` or
``isinstance``, both of which honour the metaclass hook and would pass even
against the raw litellm class, exactly the decorative-check trap this bug
lived in — plus the two behaviour-preservation properties: a server
``Retry-After`` stays readable through the copied ``response``, and
``status_code == 503`` keeps the AIMD/breaker throttle classification intact.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import instructor
import litellm
import pytest
from instructor.core.exceptions import InstructorRetryException
from litellm.types.utils import ModelResponseStream

from llmkit import LLM_RECOVERABLE_ERRORS, ServiceUnavailableError, _litellm
from llmkit.exceptions import normalize_service_unavailable, underlying_provider_error
from llmkit.rate_limiting import _is_throttle_signal
from llmkit.retry import _retry_after_seconds
from tests._support import OkSchema


def _fake_provider() -> MagicMock:
    """Same shape as ``test_in_call_retry_scope._fake_provider``: a real
    instructor mode so the structured path builds a genuine client over the
    patched ``litellm.acompletion``."""
    provider = MagicMock()
    provider.name = "fake-provider"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None
    provider.strict_json_schema = False
    return provider


def _litellm_503(*, headers: dict[str, str] | None = None) -> Exception:
    """A genuine ``litellm.exceptions.ServiceUnavailableError`` carrying a real
    ``httpx`` 503 response — the exact shape LiteLLM surfaces in production."""
    request = httpx.Request("POST", "https://fake/v1/chat/completions")
    response = httpx.Response(503, headers=headers or {}, request=request)
    return litellm.exceptions.ServiceUnavailableError(
        message="the provider is down",
        llm_provider="fake-provider",
        model="fake/model",
        response=response,
    )


class _ExplodingStream:
    """A stream that yields one keepalive frame, then raises mid-iteration."""

    def __init__(self, error: Exception) -> None:
        self._error: Exception = error
        self._yielded: bool = False

    def __aiter__(self) -> _ExplodingStream:
        return self

    async def __anext__(self) -> ModelResponseStream:
        if not self._yielded:
            self._yielded = True
            return ModelResponseStream(choices=[])
        raise self._error


async def test_text_path_503_caught_by_documented_except_clause() -> None:
    """The regression this file exists for: a litellm 503 from the text path
    lands in a literal ``except LLM_RECOVERABLE_ERRORS`` clause as llmkit's
    owned class, original on ``__cause__``, diagnostics preserved. Before the
    boundary normalization this clause was never entered."""
    original = _litellm_503()

    async def fake_acompletion(**_kwargs: object) -> object:
        raise original

    caught: Exception | None = None
    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion):
        try:
            _ = await _litellm.acompletion_text(
                "hi", temperature=0.0, model=None, provider=_fake_provider()
            )
        except LLM_RECOVERABLE_ERRORS as e:
            caught = e

    assert isinstance(caught, ServiceUnavailableError)
    assert caught.__cause__ is original
    assert caught.provider == "fake-provider"
    assert caught.model == "fake/model"
    assert caught.status_code == 503


async def test_mid_stream_503_caught_by_documented_except_clause() -> None:
    """A 503 raised *during* stream iteration — after ``_acompletion`` has
    already returned the wrapper — is normalized by the streaming loop's own
    boundary, so the documented ``except`` net covers the stream path too."""
    original = _litellm_503()

    async def fake_acompletion(**_kwargs: object) -> object:
        return _ExplodingStream(original)

    caught: Exception | None = None
    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion):
        try:
            async for _ in _litellm.astream_text(
                "hi", temperature=0.0, model=None, provider=_fake_provider()
            ):
                pass
        except LLM_RECOVERABLE_ERRORS as e:
            caught = e

    assert isinstance(caught, ServiceUnavailableError)
    assert caught.__cause__ is original


async def test_structured_path_wraps_the_owned_class() -> None:
    """On the structured path instructor wraps whatever escapes its loop, so
    the host-visible type is ``InstructorRetryException`` (already catchable) —
    but the unwrapped cause must now be llmkit's owned 503, keeping transport
    budget routing and ``Retry-After`` recovery on the normalized shape."""
    original = _litellm_503()

    async def fake_acompletion(**_kwargs: object) -> object:
        raise original

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await _litellm.acompletion_structured(
            "hi", OkSchema, temperature=0.0, model=None, provider=_fake_provider()
        )

    root = underlying_provider_error(excinfo.value)
    assert isinstance(root, ServiceUnavailableError)
    assert root.__cause__ is original


def test_retry_after_survives_normalization() -> None:
    """The copied ``response`` keeps a server ``Retry-After`` readable by the
    retry layer's backoff — normalization must never cost the server-directed
    wait that distinguishes a 503 from a blind exponential."""
    normalized = normalize_service_unavailable(_litellm_503(headers={"retry-after": "7"}))
    assert isinstance(normalized, ServiceUnavailableError)
    assert _retry_after_seconds(normalized) == 7.0


def test_throttle_classification_survives_normalization() -> None:
    """``status_code == 503`` keeps the AIMD/breaker feedback classifying the
    normalized error as an overload signal, exactly as it did the raw one."""
    original = _litellm_503()
    normalized = normalize_service_unavailable(original)
    assert normalized is not original
    assert _is_throttle_signal(original)
    assert _is_throttle_signal(normalized)


def test_non_503_errors_pass_through_untouched() -> None:
    """The normalizer re-owns only the litellm-native 503; any other exception
    is returned as the same object, so the boundary adds no wrapping noise."""
    plain = ValueError("not a 503")
    assert normalize_service_unavailable(plain) is plain
