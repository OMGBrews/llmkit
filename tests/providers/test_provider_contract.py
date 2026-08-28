"""The provider routing-contract guard (``BaseProvider.__init_subclass__``).

Every concrete provider must name its ``_provider_name``, ``_mode``, and
``_default_model`` — a silent class-level default would be a footgun (a
forgotten ``_mode`` silently regresses Gemini/DeepSeek structured output; a
forgotten ``_default_model`` assembles a dangling ``"<prefix>/"`` id; a
forgotten ``_provider_name`` degrades logging and rate-limit keying). The guard
turns that convention into a contract enforced at class-definition (import)
time. These tests pin the loud failure for an incomplete provider, the
exemption for an intermediate ABC, and the happy path for the shipped
providers.

The deliberately-incomplete subclasses are built with ``type(...)`` rather than
``class`` statements: the contract guard fires identically either way (it runs
in ``__init_subclass__``), and the dynamic form keeps an intentionally-broken
provider from tripping class-statement lints in the test module.
"""

from __future__ import annotations

import typing
from typing import cast

import instructor
import pytest

from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
    DeepSeekProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VertexProvider,
)
from llmkit.providers.base import BaseProvider

# Every concrete provider the library ships. Listed by class (not via the
# ``Provider`` enum) so a real subclass that silently lost a hook is caught
# here independent of the dispatch wiring.
_SHIPPED_PROVIDERS = [
    AnthropicProvider,
    BedrockProvider,
    DeepSeekProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VertexProvider,
]


def _define_provider(name: str, bases: tuple[type, ...], **hooks: object) -> type[BaseProvider]:
    """Define a *concrete* ``BaseProvider`` subclass with ``hooks`` as class attrs.

    Supplies the two abstract methods (``completion_kwargs`` and ``build``) so the
    class is concrete (the guard exempts abstract classes), plus an
    ``_accepted_config_fields`` default so the knob-contract check is satisfied —
    then sets whatever hooks the caller passes (which override these defaults), so
    a caller can omit exactly the one hook whose loud failure it is exercising.
    """

    def _completion_kwargs(_self: BaseProvider) -> dict[str, object]:
        return {}

    def _build(cls: type[BaseProvider], _config: object) -> BaseProvider:
        return cls()

    namespace: dict[str, object] = {
        "completion_kwargs": _completion_kwargs,
        "build": classmethod(_build),
        "_accepted_config_fields": frozenset(),
        **hooks,
    }
    return cast(type[BaseProvider], type(name, bases, namespace))


def test_missing_provider_name_fails_loudly() -> None:
    """A concrete provider that omits ``_provider_name`` fails at definition."""
    with pytest.raises(TypeError, match=r"_Incomplete.*_provider_name"):
        _ = _define_provider(
            "_Incomplete", (BaseProvider,), _mode=instructor.Mode.JSON_SCHEMA, _default_model="m"
        )


def test_missing_mode_fails_loudly() -> None:
    """A concrete provider that omits ``_mode`` fails at definition."""
    with pytest.raises(TypeError, match=r"_Incomplete.*_mode"):
        _ = _define_provider("_Incomplete", (BaseProvider,), _provider_name="X", _default_model="m")


def test_missing_default_model_fails_loudly() -> None:
    """A concrete provider that omits ``_default_model`` fails at definition."""
    with pytest.raises(TypeError, match=r"_Incomplete.*_default_model"):
        _ = _define_provider(
            "_Incomplete", (BaseProvider,), _provider_name="X", _mode=instructor.Mode.JSON_SCHEMA
        )


def test_empty_string_hook_is_rejected_like_omission() -> None:
    """An explicit falsy ``_default_model`` is a footgun too, so it is rejected.

    Makes the ctor's "never a dangling ``<prefix>/``" promise true by
    construction: an empty ``_default_model`` plus a falsy config model is
    exactly the combination that would assemble ``"<prefix>/"``.
    """
    with pytest.raises(TypeError, match=r"_default_model"):
        _ = _define_provider(
            "_Incomplete",
            (BaseProvider,),
            _provider_name="X",
            _mode=instructor.Mode.JSON_SCHEMA,
            _default_model="",
        )


def test_all_missing_hooks_are_named() -> None:
    """The message lists *every* missing hook, not just the first."""
    with pytest.raises(TypeError) as excinfo:
        _ = _define_provider("_Incomplete", (BaseProvider,))

    message = str(excinfo.value)
    assert "_provider_name" in message
    assert "_mode" in message
    assert "_default_model" in message


def test_missing_accepted_config_fields_fails_loudly() -> None:
    """A concrete provider that omits ``_accepted_config_fields`` fails at definition.

    The knob-acceptance contract is enforced the same way as the routing hooks:
    a provider must declare which ``LLMClientConfig`` knobs it reads so
    ``build_provider`` can reject a field it would silently ignore. Built here via
    a raw ``type(...)`` (not ``_define_provider``, which supplies the default) so
    the omission is genuine. A falsy *value* (the empty frozenset) is legal, so
    only true absence is tested."""

    def _completion_kwargs(_self: BaseProvider) -> dict[str, object]:
        return {}

    def _build(cls: type[BaseProvider], _config: object) -> BaseProvider:
        return cls()

    with pytest.raises(TypeError, match=r"_Incomplete.*_accepted_config_fields"):
        _ = type(
            "_Incomplete",
            (BaseProvider,),
            {
                "completion_kwargs": _completion_kwargs,
                "build": classmethod(_build),
                "_provider_name": "X",
                "_mode": instructor.Mode.JSON_SCHEMA,
                "_default_model": "m",
            },
        )


def test_empty_accepted_config_fields_is_a_legal_declaration() -> None:
    """An empty ``frozenset()`` is a valid declaration (a provider reading none of
    the provider-shaped knobs) — the contract checks *presence*, not truthiness."""
    concrete = _define_provider(
        "_NoKnobProvider",
        (BaseProvider,),
        _provider_name="NoKnob",
        _mode=instructor.Mode.JSON_SCHEMA,
        _default_model="m",
        _accepted_config_fields=frozenset[str](),
    )
    assert concrete._accepted_config_fields == frozenset()


def test_intermediate_abstract_base_is_exempt() -> None:
    """An intermediate ABC that omits the hooks is not falsely rejected.

    It leaves ``completion_kwargs`` abstract, so it cannot be constructed
    anyway; the contract is enforced on the concrete subclass that fills it in.
    """
    # No ``completion_kwargs`` and no hooks -> still abstract -> exempt, no raise.
    intermediate = type("_IntermediateBase", (BaseProvider,), {})
    concrete = _define_provider(
        "_ConcreteProvider",
        (intermediate,),
        _provider_name="Concrete",
        _mode=instructor.Mode.JSON_SCHEMA,
        _default_model="model-x",
    )

    provider = concrete()
    assert provider.name == "Concrete"
    assert provider.model == "model-x"
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA


@pytest.mark.parametrize("provider_cls", _SHIPPED_PROVIDERS)
def test_shipped_providers_satisfy_the_contract(provider_cls: type[BaseProvider]) -> None:
    """Every shipped provider names all three hooks (no behavior change)."""
    assert provider_cls._provider_name
    assert provider_cls._default_model
    assert isinstance(provider_cls._mode, instructor.Mode)


# The provider-shaped LLMClientConfig knobs (mirrors ``base._PROVIDER_SHAPED_FIELDS``,
# restated here to keep the test off a module-private import).
_PROVIDER_SHAPED_KNOBS = frozenset(
    {
        "api_key",
        "base_url",
        "aws_region_name",
        "vertex_project",
        "vertex_location",
        "gemini_structured_output",
    }
)


@pytest.mark.parametrize("provider_cls", _SHIPPED_PROVIDERS)
def test_shipped_provider_declares_valid_accepted_config_fields(
    provider_cls: type[BaseProvider],
) -> None:
    """Every shipped provider declares ``_accepted_config_fields`` as a subset of
    the provider-shaped knobs — so ``build_provider`` can reject anything else."""
    accepted = provider_cls._accepted_config_fields
    assert isinstance(accepted, frozenset)
    assert accepted <= _PROVIDER_SHAPED_KNOBS, (
        f"{provider_cls.__name__} lists a non-provider-shaped field: "
        f"{accepted - _PROVIDER_SHAPED_KNOBS}"
    )


@pytest.mark.parametrize("provider_cls", _SHIPPED_PROVIDERS)
def test_shipped_mode_constructs_an_instructor_client(provider_cls: type[BaseProvider]) -> None:
    """Every shipped mode pin constructs an instructor client — offline, keyless.

    ``instructor.from_litellm`` validates the mode against its LiteLLM mode
    registry at *client construction* — instructor 1.15.3 removed
    ``ANTHROPIC_JSON`` and ``OPENROUTER_STRUCTURED_OUTPUTS`` from that
    registry, so three providers raised ``RegistryError`` before any request
    was sent, while the mocked structured-call harness (which patches out
    ``from_litellm`` itself) stayed green. Constructing against the real
    registry here pins the contract, so the next registry drift fails CI
    rather than production. The fake completion is ``async`` because the
    production seam (``_litellm.acompletion_structured``) always hands
    ``from_litellm`` an async completion.
    """

    async def _never_called(**_kwargs: object) -> object:
        raise AssertionError("construction is the contract under test; no call expected")

    client = instructor.from_litellm(_never_called, mode=provider_cls._mode)
    assert isinstance(client, instructor.AsyncInstructor)


@pytest.mark.parametrize("provider_cls", [BaseProvider, *_SHIPPED_PROVIDERS])
def test_routing_class_attrs_are_annotated_classvar(provider_cls: type[BaseProvider]) -> None:
    """All class-level routing config is uniformly annotated ``ClassVar``.

    The three enforced hooks plus ``_model_prefix`` and ``is_local`` are class
    routing config, never per-instance state. A mixed annotation style sends a
    provider author contradictory signals about which attributes are
    class-level, so this pins the convention for the base and every shipped
    provider (each redeclares the attrs it overrides).
    """
    hints: dict[str, object] = typing.get_type_hints(provider_cls)
    routing_attrs = (
        "_provider_name",
        "_mode",
        "_default_model",
        "_model_prefix",
        "is_local",
        "_strict_json_schema",
        "_compose_tools_schema",
        "_supports_tool_choice",
    )
    for attr in routing_attrs:
        assert typing.get_origin(hints[attr]) is typing.ClassVar, (
            f"{provider_cls.__name__}.{attr} must be annotated ClassVar"
        )
