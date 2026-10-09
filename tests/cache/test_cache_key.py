"""The request fingerprint: every input that shapes the request changes it.

``llm_cache_key`` is public and its canonical form is a stored-data contract —
a host's persistent store holds keys computed by an earlier release — so a
pinned digest makes any change to the form a deliberate version bump.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel, Field

from llmkit import ChatMessage, llm_cache_key
from llmkit.cache import LLM_CACHE_KEY_VERSION
from tests.cache.conftest import cache_provider


class _Verdict(BaseModel):
    ok: bool
    reason: str


class _VerdictReordered(BaseModel):
    reason: str
    ok: bool


_PROMPT: list[ChatMessage] = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Is water wet?"},
]


def _key(**overrides: object) -> str:
    inputs: dict[str, object] = {
        "provider": cache_provider(),
        "prompt": _PROMPT,
        "output_schema": _Verdict,
        "model": None,
        "temperature": 1.0,
        "max_tokens": 512,
        "reasoning_effort": None,
    }
    inputs.update(overrides)
    return llm_cache_key(**inputs)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat


def test_pinned_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical form is pinned: a change here needs a new
    ``LLM_CACHE_KEY_VERSION`` and a changelog entry, never a silent re-key.

    The schema JSON is fixed for the pin, so the digest pins llmkit's form
    rather than the schema pydantic generates in whichever version is
    installed (the floor tests run an older one).
    """
    fixed_schema: dict[str, object] = {"properties": {"ok": {"type": "boolean"}}, "type": "object"}

    def _fixed(_cls: type[BaseModel]) -> dict[str, object]:
        return fixed_schema

    monkeypatch.setattr(_Verdict, "model_json_schema", classmethod(_fixed))
    assert LLM_CACHE_KEY_VERSION == 1
    assert _key() == "beed3cc8e19235df8d16b64c63cec05ae172e56d0a2338538f0716e1de01937c"
    assert (
        _key(output_schema=None)
        == "06bc783a9655238d77e9a8c818f1b30a8b810ec31dc930c1fb9dd0ce363569da"
    )


def test_identical_inputs_give_identical_keys() -> None:
    assert _key() == _key()
    assert len(_key()) == 64


_CHANGES: dict[str, Callable[[], str]] = {
    "provider name": lambda: _key(provider=cache_provider(name="OpenRouter")),
    "default model": lambda: _key(provider=cache_provider(model="gemini-3.1-pro")),
    "model override": lambda: _key(model="gemini-3.1-pro"),
    "api_base": lambda: _key(
        provider=cache_provider(request_kwargs={"api_key": "secret", "api_base": "https://b.test"})
    ),
    "extra_body": lambda: _key(
        provider=cache_provider(
            request_kwargs={
                "api_key": "secret",
                "api_base": "https://example.test/v1",
                "extra_body": {"provider": {"require_parameters": True}},
            }
        )
    ),
    "aws_region_name": lambda: _key(
        provider=cache_provider(request_kwargs={"aws_region_name": "eu-west-1"})
    ),
    "vertex_project": lambda: _key(provider=cache_provider(request_kwargs={"vertex_project": "p"})),
    "vertex_location": lambda: _key(
        provider=cache_provider(request_kwargs={"vertex_location": "us-central1"})
    ),
    "messages": lambda: _key(prompt=[*_PROMPT, {"role": "user", "content": "Really?"}]),
    "plain-string prompt": lambda: _key(prompt="Is water wet?"),
    "output schema": lambda: _key(output_schema=None),
    "schema field order": lambda: _key(output_schema=_VerdictReordered),
    "temperature": lambda: _key(temperature=0.0),
    "temperature unset": lambda: _key(temperature=None),
    "max_tokens": lambda: _key(max_tokens=None),
    "reasoning effort": lambda: _key(reasoning_effort="low"),
    "provider reasoning effort": lambda: _key(provider=cache_provider(reasoning_effort="high")),
    "structured-output mode": lambda: _key(provider=cache_provider(instructor_mode="tools")),
    "strict json schema": lambda: _key(provider=cache_provider(strict_json_schema=True)),
}


@pytest.mark.parametrize("change", list(_CHANGES))
def test_every_request_input_changes_the_key(change: str) -> None:
    assert _CHANGES[change]() != _key()


def test_schema_field_order_alone_changes_the_key() -> None:
    """The property order is sent to the provider and steers generation, so
    the key is serialised without sorting: same fields, different order,
    different request."""
    assert set(_Verdict.model_fields) == set(_VerdictReordered.model_fields)
    renamed = type("_Verdict", (BaseModel,), {"__annotations__": {"reason": str, "ok": bool}})
    assert _key(output_schema=renamed) != _key()


def test_schema_constraints_change_the_key() -> None:
    class _Bounded(BaseModel):
        ok: bool
        reason: str = Field(max_length=10)

    class _Unbounded(BaseModel):
        ok: bool
        reason: str

    _Bounded.__name__ = _Unbounded.__name__ = "Same"
    assert _key(output_schema=_Bounded) != _key(output_schema=_Unbounded)


def test_credentials_never_enter_the_key() -> None:
    """Only the endpoint allowlist is hashed, so rotating a key keeps hits."""
    rotated = cache_provider(
        request_kwargs={"api_key": "rotated", "api_base": "https://example.test/v1"}
    )
    assert _key(provider=rotated) == _key()


def test_effective_reasoning_effort_is_what_counts() -> None:
    """An unset per-call effort resolves to the provider's configured one, as
    the transport resolves it, so the two spellings are one request."""
    configured = cache_provider(reasoning_effort="low")
    assert _key(provider=configured) == _key(reasoning_effort="low")


def test_text_key_ignores_the_structured_output_mode() -> None:
    """A text request carries no structured-output mode, so the mode cannot
    split its key."""
    text_json = _key(output_schema=None)
    text_tools = _key(output_schema=None, provider=cache_provider(instructor_mode="tools"))
    assert text_json == text_tools


def test_tool_error_flag_is_keyed_as_sent() -> None:
    """Messages are keyed in their wire form, so the ``is_error`` flag (sent as
    a content prefix) distinguishes two otherwise identical tool results."""
    ok_turn: ChatMessage = {"role": "tool", "tool_call_id": "c1", "content": "done"}
    error_turn: ChatMessage = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "done",
        "is_error": True,
    }
    assert _key(prompt=[*_PROMPT, ok_turn]) != _key(prompt=[*_PROMPT, error_turn])


def test_feature_and_label_are_not_inputs() -> None:
    """Telemetry is not part of the request, so it is not a parameter."""
    with pytest.raises(TypeError):
        _ = _key(feature="extraction")
