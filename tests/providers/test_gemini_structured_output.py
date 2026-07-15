"""Host-selectable Gemini structured-output strategy (``schema`` vs ``json``).

The two Gemini providers (Vertex, Google AI Studio) resolve their instructor
``Mode`` from a host-configurable strategy instead of a fixed ClassVar, so a host
can escape Gemini's constrained-decoding repetition-loop trap without subclassing
a provider or overriding the private ``_mode`` ClassVar. These tests pin:

- the default preserves 0.7.0 behavior (``Mode.JSON_SCHEMA``) byte-for-byte;
- ``"json"`` selects ``Mode.JSON`` and ``"schema"`` selects ``Mode.JSON_SCHEMA``;
- an invalid strategy fails loudly at construction (never a silent fallback);
- the strategy threads through ``build_provider`` / ``LLMClientConfig`` and the
  ``make_provider`` per-call seam;
- the ``Mode.JSON`` pin is registry-valid over ``instructor.from_litellm`` (it is
  a core mode, so it survives instructor's construction-time registry check);
- non-Gemini providers ignore the field entirely.

Mocked-shape assertions only; a real round-trip lives in
``tests/integration/test_live_providers.py`` (which exercises the ``"schema"``
default).
"""

from __future__ import annotations

import instructor
import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import (
    DeepSeekProvider,
    GoogleProvider,
    VertexProvider,
    make_provider,
)
from llmkit.providers.base import BaseProvider


def _google(structured_output: str = "schema") -> GoogleProvider:
    return GoogleProvider(api_key="k", structured_output=structured_output)


def _vertex(structured_output: str = "schema") -> VertexProvider:
    return VertexProvider(structured_output=structured_output)


# --- default preserves 0.7.0 behavior -------------------------------------


def test_google_default_mode_is_json_schema() -> None:
    """Default construction keeps Gemini's native JSON-schema mode."""
    assert _google().instructor_mode is instructor.Mode.JSON_SCHEMA


def test_vertex_default_mode_is_json_schema() -> None:
    """Default construction keeps Gemini's native JSON-schema mode."""
    assert _vertex().instructor_mode is instructor.Mode.JSON_SCHEMA


def test_schema_strategy_is_the_explicit_default() -> None:
    """``structured_output="schema"`` is byte-identical to the default."""
    assert _google("schema").instructor_mode is instructor.Mode.JSON_SCHEMA
    assert _vertex("schema").instructor_mode is instructor.Mode.JSON_SCHEMA


def test_classvar_mode_is_unchanged_on_both_providers() -> None:
    """The ``_mode`` ClassVar stays the declared contract value (JSON_SCHEMA).

    The selection mechanism is the overridden ``instructor_mode`` property; the
    ClassVar is never dropped or instance-shadowed (``__init_subclass__`` still
    requires it, and the other six providers keep the ClassVar → base-property
    path).
    """
    assert GoogleProvider._mode is instructor.Mode.JSON_SCHEMA
    assert VertexProvider._mode is instructor.Mode.JSON_SCHEMA


# --- "json" selects Mode.JSON ---------------------------------------------


def test_google_json_strategy_selects_mode_json() -> None:
    assert _google("json").instructor_mode is instructor.Mode.JSON


def test_vertex_json_strategy_selects_mode_json() -> None:
    assert _vertex("json").instructor_mode is instructor.Mode.JSON


# --- invalid value fails loudly -------------------------------------------


@pytest.mark.parametrize("bad", ["JSON", "schemaa", "tools", ""])
def test_google_invalid_strategy_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="structured-output strategy"):
        _ = _google(bad)


@pytest.mark.parametrize("bad", ["JSON", "schemaa", "tools", ""])
def test_vertex_invalid_strategy_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="structured-output strategy"):
        _ = _vertex(bad)


def test_invalid_strategy_message_names_the_valid_values() -> None:
    """The error lists the accepted strategies so the fix is obvious."""
    with pytest.raises(ValueError) as excinfo:
        _ = _vertex("nope")
    message = str(excinfo.value)
    assert "'schema'" in message
    assert "'json'" in message


# --- threads through build_provider / LLMClientConfig ----------------------


def test_build_provider_google_threads_json_strategy() -> None:
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.GOOGLE,
            api_key="k",
            gemini_structured_output="json",
        )
    )
    assert isinstance(provider, GoogleProvider)
    assert provider.instructor_mode is instructor.Mode.JSON


def test_build_provider_vertex_threads_json_strategy() -> None:
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.VERTEX,
            gemini_structured_output="json",
        )
    )
    assert isinstance(provider, VertexProvider)
    assert provider.instructor_mode is instructor.Mode.JSON


def test_config_field_defaults_to_schema() -> None:
    """A config that never names the field yields today's JSON-schema behavior."""
    config = LLMClientConfig(provider=Provider.GOOGLE, api_key="k")
    assert config.gemini_structured_output == "schema"
    provider = build_provider(config)
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA


def test_make_provider_threads_json_strategy() -> None:
    """The per-call ``make_provider`` seam carries the strategy too."""
    provider = make_provider(
        Provider.VERTEX,
        model="gemini-2.5-flash",
        gemini_structured_output="json",
    )
    assert isinstance(provider, VertexProvider)
    assert provider.instructor_mode is instructor.Mode.JSON


# --- registry validity of the Mode.JSON pin --------------------------------


@pytest.mark.parametrize("provider", [_google("json"), _vertex("json")])
def test_gemini_json_mode_constructs_an_instructor_client(provider: BaseProvider) -> None:
    """Gemini under ``"json"`` yields a mode ``from_litellm`` accepts at construction.

    instructor >= 1.15.3 validates the mode against its LiteLLM registry when the
    client is built, so a non-core mode would raise ``RegistryError`` here.
    ``Mode.JSON`` is core (``DeepSeekProvider`` already ships it), so this pins
    that the Gemini ``"json"`` strategy stays registry-valid across instructor
    bumps — mirroring ``test_provider_contract`` for the shipped pins.
    """

    async def _never_called(**_kwargs: object) -> object:
        raise AssertionError("construction is the contract under test; no call expected")

    client = instructor.from_litellm(_never_called, mode=provider.instructor_mode)
    assert isinstance(client, instructor.AsyncInstructor)


# --- non-Gemini providers ignore the field ---------------------------------


def test_non_gemini_provider_ignores_the_field() -> None:
    """A non-Gemini provider keeps its own pinned mode regardless of the field.

    DeepSeek pins ``Mode.JSON`` for its own (documented) reason; the point is
    that ``gemini_structured_output`` never touches it — the field is read only
    by the two Gemini providers' ``build``.
    """
    schema = build_provider(
        LLMClientConfig(provider=Provider.DEEPSEEK, api_key="k", gemini_structured_output="schema")
    )
    json = build_provider(
        LLMClientConfig(provider=Provider.DEEPSEEK, api_key="k", gemini_structured_output="json")
    )
    assert isinstance(schema, DeepSeekProvider)
    assert isinstance(json, DeepSeekProvider)
    # DeepSeek's mode is its own pin, unaffected by the Gemini field either way.
    assert schema.instructor_mode is instructor.Mode.JSON
    assert json.instructor_mode is instructor.Mode.JSON
