"""Synthetic native errors traverse LiteLLM's real mapper before normalization.

These inputs are deliberately neutral: a fixed message, no structured error
body and empty headers. They measure dependency mapping, not errors observed
from live providers. The fixtures were measured on LiteLLM 1.106.6 / OpenAI
2.54.0 and the floors LiteLLM 1.104.0 / OpenAI 2.20.0, both with instructor
1.17.0. These two resolutions map every fixture identically; future differences
must be measured rather than silently weakening the expected types or statuses.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Literal, Never, Protocol, cast

import httpx
import openai
import pytest

from llmkit import Provider, ServiceUnavailableError, TransientProviderError
from llmkit.exceptions import LLM_TRANSPORT_ERRORS, normalize_transient_provider_error

_STATUSES = (400, 401, 403, 500, 502, 503, 504, 520, 529)
_ROUTES = {
    Provider.OPENAI: "openai",
    Provider.OPENROUTER: "openrouter",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
    Provider.VERTEX: "vertex_ai",
    Provider.BEDROCK: "bedrock",
    Provider.DEEPSEEK: "deepseek",
    Provider.CEREBRAS: "cerebras",
    Provider.OLLAMA: "ollama_chat",
}


@dataclass(frozen=True)
class _Mapping:
    name: str
    status: int
    outcome: Literal["excluded", "unchanged", "owned-503", "new-transient"]


# Explicit observations, before llmkit normalizes anything. In particular,
# incoming and mapped statuses need not agree: Anthropic 529 becomes 500,
# Bedrock 500 becomes 503, and Google's 502 becomes a connection error/500.
_COMMON = {
    400: _Mapping("BadRequestError", 400, "excluded"),
    401: _Mapping("AuthenticationError", 401, "excluded"),
    403: _Mapping("APIError", 403, "excluded"),
    500: _Mapping("InternalServerError", 500, "unchanged"),
    502: _Mapping("BadGatewayError", 502, "new-transient"),
    503: _Mapping("ServiceUnavailableError", 503, "owned-503"),
    504: _Mapping("Timeout", 504, "unchanged"),
    520: _Mapping("APIError", 520, "excluded"),
    529: _Mapping("APIError", 529, "new-transient"),
}
_OVERRIDES: dict[Provider, dict[int, _Mapping]] = {
    Provider.OPENROUTER: {
        500: _Mapping("APIError", 500, "new-transient"),
        502: _Mapping("APIError", 502, "new-transient"),
    },
    Provider.ANTHROPIC: {
        403: _Mapping("PermissionDeniedError", 403, "excluded"),
        529: _Mapping("InternalServerError", 500, "unchanged"),
    },
    Provider.GOOGLE: {
        403: _Mapping("PermissionDeniedError", 403, "excluded"),
        502: _Mapping("APIConnectionError", 500, "unchanged"),
    },
    Provider.VERTEX: {
        403: _Mapping("PermissionDeniedError", 403, "excluded"),
        502: _Mapping("APIConnectionError", 500, "unchanged"),
    },
    Provider.BEDROCK: {
        403: _Mapping("PermissionDeniedError", 403, "excluded"),
        500: _Mapping("ServiceUnavailableError", 503, "owned-503"),
    },
    Provider.OLLAMA: {403: _Mapping("PermissionDeniedError", 403, "excluded")},
}
_CASES = [(provider, status) for provider in _ROUTES for status in _STATUSES]


class _ExceptionMapper(Protocol):
    def __call__(
        self,
        *,
        model: str,
        original_exception: Exception,
        custom_llm_provider: str,
        completion_kwargs: dict[str, object],
        extra_kwargs: dict[str, object],
    ) -> object: ...


def _native_error(provider: Provider, status: int) -> Exception:
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.llms.bedrock.common_utils import BedrockError
    from litellm.llms.ollama.common_utils import OllamaError
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    message = "audit synthetic status"
    headers = httpx.Headers()
    match provider:
        case Provider.ANTHROPIC:
            return AnthropicError(status_code=status, message=message, headers=headers)
        case Provider.GOOGLE | Provider.VERTEX:
            return VertexAIError(status_code=status, message=message, headers=headers)
        case Provider.BEDROCK:
            return BedrockError(status_code=status, message=message, headers=headers)
        case Provider.OLLAMA:
            return OllamaError(status_code=status, message=message, headers=headers)
        case _:
            request = httpx.Request("POST", "https://audit.invalid/completions")
            response = httpx.Response(status, request=request)
            return openai.APIStatusError(message, response=response, body=None)


def _deny_network(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("synthetic exception mapping attempted network I/O")


def test_mapping_inventory_covers_every_shipped_provider() -> None:
    assert set(_ROUTES) == set(Provider)
    assert set(_COMMON) == set(_STATUSES)
    assert len(_CASES) == len(set(_CASES)) == len(Provider) * len(_STATUSES) > 0
    assert set(_OVERRIDES) <= set(_ROUTES)
    assert all(set(overrides) <= set(_STATUSES) for overrides in _OVERRIDES.values())


@pytest.mark.parametrize(
    ("provider", "status"),
    _CASES,
    ids=[f"{_ROUTES[provider]}-{status}" for provider, status in _CASES],
)
def test_native_mapping_and_normalization(
    provider: Provider, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    import litellm
    from litellm.litellm_core_utils.exception_mapping_utils import (
        exception_type,  # pyright: ignore[reportUnknownVariableType]  # raw-llm: mapper has untyped parameters, narrowed below
    )

    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    # raw-llm: the real mapper has untyped parameters; narrow its boundary once.
    map_exception = cast(_ExceptionMapper, exception_type)
    original = _native_error(provider, status)
    try:
        mapped = map_exception(
            model="audit-model",
            original_exception=original,
            custom_llm_provider=_ROUTES[provider],
            completion_kwargs={},
            extra_kwargs={},
        )
    except Exception as error:
        mapped = error
    assert isinstance(mapped, Exception)
    expected = _OVERRIDES.get(provider, {}).get(status, _COMMON[status])
    assert (type(mapped).__module__, type(mapped).__name__) == (
        "litellm.exceptions",
        expected.name,
    )
    assert getattr(mapped, "status_code", None) == expected.status
    assert isinstance(mapped, LLM_TRANSPORT_ERRORS) is (
        expected.outcome in {"unchanged", "owned-503"}
    )

    normalized = normalize_transient_provider_error(mapped)
    if expected.outcome == "new-transient":
        assert type(normalized) is TransientProviderError
        assert normalized.status_code == expected.status
    elif expected.outcome == "owned-503":
        assert type(normalized) is ServiceUnavailableError
        assert normalized.status_code == 503
    else:
        assert normalized is mapped
    assert isinstance(normalized, LLM_TRANSPORT_ERRORS) is (expected.outcome != "excluded")
