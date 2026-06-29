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

    Supplies ``completion_kwargs`` so the class is concrete (the guard exempts
    abstract classes), then sets whatever hooks the caller passes — so a caller
    can omit one to exercise the loud failure.
    """

    def _completion_kwargs(_self: BaseProvider) -> dict[str, object]:
        return {}

    namespace: dict[str, object] = {"completion_kwargs": _completion_kwargs, **hooks}
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
    routing_attrs = ("_provider_name", "_mode", "_default_model", "_model_prefix", "is_local")
    for attr in routing_attrs:
        assert typing.get_origin(hints[attr]) is typing.ClassVar, (
            f"{provider_cls.__name__}.{attr} must be annotated ClassVar"
        )
