"""Unit tests for :func:`describe_llm` and its :class:`LLMInfo` snapshot.

``describe_llm`` is the read accessor (renamed from ``get_llm_config`` in
0.2.0) that reads back the provider a call would use, without making a call.
It takes the registered config (no argument), an explicit ``config``, or, as a
keyword-only ``provider=``, the same values a call's ``provider=`` takes: a
:class:`Provider` member, built the way a call builds it, or a built instance
of a shipped provider class, whose member is recovered from its exact class.
A host's own provider class, a subclass of a shipped class included, a plain
string, and ``config`` and ``provider`` together each raise ``TypeError``.

These tests pin, for every member and through every input form, the provider,
its display name, the resolved model, ``is_local`` (true only for the local
Ollama provider), and the route's correctness decisions the snapshot reports:
``compose_tools_schema``, ``instructor_mode``, ``strict_json_schema``,
``single_tool_call_per_schema`` and ``supports_tool_choice``, read from the
instance so Google and Gemini Enterprise Agent Platform with
``gemini_structured_output="json"`` report ``Mode.JSON``. They also pin that an
``LLMInfo`` built with only its first five fields still constructs and reports
``None`` (not reported) for each decision added after them.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, TypedDict, cast

import instructor
import pytest

from llmkit.providers import (
    AnthropicProvider,
    LLMClientConfig,
    LLMInfo,
    LLMProviderInterface,
    Provider,
    _bare_provider_string_message,
    _provider_class,
    coerce_call_provider,
    configure_llm_client,
    describe_llm,
    make_provider,
)


class _Decisions(NamedTuple):
    mode: instructor.Mode
    strict: bool
    single_tool_call: bool
    tool_choice: bool
    compose: bool
    local: bool


# The route decisions at the default settings, measured from built instances.
_DECISIONS: dict[Provider, _Decisions] = {
    Provider.OPENROUTER: _Decisions(instructor.Mode.JSON_SCHEMA, True, False, True, False, False),
    Provider.OLLAMA: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, False, False, True),
    Provider.GOOGLE: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, True, False, False),
    Provider.ANTHROPIC: _Decisions(instructor.Mode.JSON_SCHEMA, False, True, True, True, False),
    Provider.OPENAI: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, True, True, False),
    Provider.DEEPSEEK: _Decisions(instructor.Mode.JSON, False, False, True, False, False),
    Provider.BEDROCK: _Decisions(instructor.Mode.JSON_SCHEMA, False, True, True, False, False),
    Provider.VERTEX: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, True, False, False),
    Provider.CEREBRAS: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, True, False, False),
    Provider.MOONSHOT: _Decisions(instructor.Mode.JSON, False, False, True, False, False),
    Provider.ALIBABA: _Decisions(instructor.Mode.JSON_SCHEMA, False, False, True, False, False),
}


class _BuildKwargs(TypedDict, total=False):
    api_key: str
    vertex_location: str
    aws_region_name: str
    gemini_structured_output: Literal["schema", "json"]


def _build_kwargs(member: Provider) -> _BuildKwargs:
    """The fewest explicit fields that build ``member`` offline."""
    if member is Provider.VERTEX:
        return {"vertex_location": "us-central1"}
    if member is Provider.BEDROCK:
        return {"aws_region_name": "us-east-1"}
    if member is Provider.OLLAMA:
        return {}
    return {"api_key": "k"}


def _assert_reports(info: LLMInfo, member: Provider) -> None:
    """``info`` names ``member`` and reports its route's every decision."""
    expected = _DECISIONS[member]
    assert info.provider is member
    assert info.provider_name == _provider_class(member).display_name()
    assert info.is_local is expected.local
    assert info.compose_tools_schema is expected.compose
    assert info.instructor_mode is expected.mode
    assert info.strict_json_schema is expected.strict
    assert info.single_tool_call_per_schema is expected.single_tool_call
    assert info.supports_tool_choice is expected.tool_choice


def test_the_decision_table_covers_every_member() -> None:
    assert set(_DECISIONS) == set(Provider)


@pytest.mark.parametrize("member", list(Provider))
def test_describe_llm_reports_a_shipped_instance(member: Provider) -> None:
    """An instance names its member, display name and configured model.

    The member is recovered from the instance's exact class, so a wrong entry
    in the reverse map fails that member's case.
    """
    instance = make_provider(member, model="configured-model", **_build_kwargs(member))
    info = describe_llm(provider=instance)
    _assert_reports(info, member)
    assert info.model == "configured-model"


@pytest.mark.parametrize("member", list(Provider))
def test_describe_llm_reports_a_config(member: Provider) -> None:
    """A config reports the same decisions as an instance of its provider."""
    info = describe_llm(LLMClientConfig(provider=member, **_build_kwargs(member)))
    _assert_reports(info, member)
    assert info.model == _provider_class(member)._default_model


@pytest.mark.parametrize("member", list(Provider))
def test_describe_llm_builds_a_member_the_way_a_call_does(
    member: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member is built from the environment with the provider's default model."""
    env_var = cast(str | None, getattr(_provider_class(member), "_api_key_env_var", None))
    if env_var is not None:
        monkeypatch.setenv(env_var, "k")
    info = describe_llm(provider=member)
    _assert_reports(info, member)
    assert info.model == _provider_class(member)._default_model
    # The same instance a call's ``provider=member`` runs on.
    call_instance = coerce_call_provider(member)
    assert call_instance is not None
    assert info.model == call_instance.model
    assert info.provider_name == call_instance.name


@pytest.mark.parametrize("member", [Provider.GOOGLE, Provider.VERTEX])
def test_gemini_json_structured_output_reports_json_mode(member: Provider) -> None:
    """The mode is read from the instance: ``"json"`` gives ``Mode.JSON``."""
    kwargs = _build_kwargs(member)
    kwargs["gemini_structured_output"] = "json"
    from_config = describe_llm(LLMClientConfig(provider=member, **kwargs))
    from_instance = describe_llm(provider=make_provider(member, **kwargs))
    assert from_config.instructor_mode is instructor.Mode.JSON
    assert from_instance.instructor_mode is instructor.Mode.JSON
    # The class-level pin is unchanged; the snapshot does not read it.
    assert _provider_class(member)._mode is instructor.Mode.JSON_SCHEMA


def test_a_host_subclass_of_a_shipped_class_is_refused() -> None:
    """A subclass is a route llmkit has not measured, so it has no member."""

    class HostAnthropic(AnthropicProvider):
        pass

    instance = HostAnthropic.build(LLMClientConfig(provider=Provider.ANTHROPIC, api_key="k"))
    assert isinstance(instance, AnthropicProvider)
    with pytest.raises(TypeError) as caught:
        _ = describe_llm(provider=instance)
    message = str(caught.value)
    assert "HostAnthropic" in message
    assert "LLMClientConfig" in message
    assert "Provider member" in message
    for member in Provider:
        assert _provider_class(member).__name__ in message
    for decision in (
        "instructor_mode",
        "strict_json_schema",
        "single_tool_call_per_schema",
        "supports_tool_choice",
    ):
        assert decision in message


def test_a_provider_that_is_not_a_base_provider_is_refused() -> None:
    class HostProvider:
        pass

    with pytest.raises(TypeError, match="HostProvider"):
        _ = describe_llm(provider=cast(LLMProviderInterface, cast(object, HostProvider())))


def test_a_provider_string_raises_the_calls_type_error() -> None:
    with pytest.raises(TypeError) as caught:
        _ = describe_llm(provider=cast(Provider, cast(object, "anthropic")))
    assert str(caught.value) == _bare_provider_string_message("anthropic")
    assert "Provider.ANTHROPIC" in str(caught.value)


def test_config_and_provider_together_are_refused() -> None:
    config = LLMClientConfig(provider=Provider.OLLAMA)
    with pytest.raises(TypeError, match="not both"):
        _ = describe_llm(config, provider=make_provider(Provider.OLLAMA))
    with pytest.raises(TypeError, match="not both"):
        _ = describe_llm(config=config, provider=Provider.OLLAMA)


def test_describe_llm_reads_the_registered_config_with_no_argument() -> None:
    previous = configure_llm_client(
        LLMClientConfig(provider=Provider.ANTHROPIC, api_key="k", model="registered-model")
    )
    try:
        info = describe_llm()
    finally:
        _ = configure_llm_client(previous)
    _assert_reports(info, Provider.ANTHROPIC)
    assert info.model == "registered-model"


def test_describe_llm_takes_a_positional_config() -> None:
    info = describe_llm(LLMClientConfig(provider=Provider.OPENAI, api_key="k", model="m"))
    _assert_reports(info, Provider.OPENAI)
    assert info.model == "m"


def test_describe_llm_takes_config_by_keyword() -> None:
    info = describe_llm(config=LLMClientConfig(provider=Provider.DEEPSEEK, api_key="k"))
    _assert_reports(info, Provider.DEEPSEEK)


def test_describe_llm_resolves_provider_default_model() -> None:
    """An unset ``model`` resolves to the provider's own default in the snapshot."""
    info = describe_llm(LLMClientConfig(provider=Provider.GOOGLE, api_key="k"))
    assert isinstance(info, LLMInfo)
    assert info.provider == Provider.GOOGLE
    assert info.provider_name == "Google AI Studio"
    assert info.model == "gemini-3.1-flash-lite"
    assert info.is_local is False
    assert info.compose_tools_schema is False


def test_describe_llm_honors_explicit_model() -> None:
    """An explicit ``model`` is carried through to the snapshot verbatim."""
    info = describe_llm(
        LLMClientConfig(provider=Provider.GOOGLE, api_key="k", model="gemini-2.5-pro")
    )
    assert info.model == "gemini-2.5-pro"


def test_describe_llm_snapshots_vertex() -> None:
    """Gemini Enterprise Agent Platform (formerly Vertex AI) resolves its name + default model in
    the snapshot; residency fields don't affect the read accessor (they're routing, not
    display)."""
    info = describe_llm(LLMClientConfig(provider=Provider.VERTEX, vertex_location="europe-west4"))
    assert info.provider == Provider.VERTEX
    assert info.provider_name == "Google Agent Platform"
    assert info.model == "gemini-3.1-flash-lite"
    assert info.is_local is False


def test_a_five_field_llm_info_still_constructs() -> None:
    """The decisions added after the first five fields default to ``None``."""
    positional = LLMInfo(Provider.OLLAMA, "Ollama", "m", True, False)
    keyword = LLMInfo(
        provider=Provider.OLLAMA,
        provider_name="Ollama",
        model="m",
        is_local=True,
        compose_tools_schema=False,
    )
    for info in (positional, keyword):
        assert info.instructor_mode is None
        assert info.strict_json_schema is None
        assert info.single_tool_call_per_schema is None
        assert info.supports_tool_choice is None
