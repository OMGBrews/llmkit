"""Parsing a provider name, and refusing an unparsed one in the config.

A provider name read from a settings file becomes a member through
``Provider(value)``, which ignores case and lists every member when the name
is unknown. ``LLMClientConfig`` takes only a member, so a string fails at
construction naming that parse, never deep inside ``build_provider`` with the
``assert_never`` ``AssertionError`` that guards an unwired member
(``test_registry_exhaustive.py``).
"""

from __future__ import annotations

import re

import pytest

from llmkit import LLMClientConfig, Provider, configure_llm_client, make_provider
from llmkit.providers import describe_llm


@pytest.mark.parametrize("spelling", ["ollama", "Ollama", "OLLAMA", "oLlAmA"])
def test_provider_parses_a_name_in_any_case(spelling: str) -> None:
    assert Provider(spelling) is Provider.OLLAMA


def test_an_unknown_name_raises_value_error_listing_every_member() -> None:
    with pytest.raises(ValueError, match=r"'olama' is not a known provider") as caught:
        _ = Provider("olama")

    message = str(caught.value)
    assert all(repr(member.value) in message for member in Provider)


def test_a_non_string_raises_value_error_too() -> None:
    with pytest.raises(ValueError, match="is not a known provider"):
        _ = Provider(3)


def test_every_member_value_is_unique_under_casefolding() -> None:
    """A future member whose value casefolds onto another's would make
    ``Provider(value)`` ambiguous; this fails first."""
    assert len({member.value.casefold() for member in Provider}) == len(Provider)
    assert all(member.value == member.value.casefold() for member in Provider)


@pytest.mark.parametrize("value", ["ollama", "Ollama", "olama"])
def test_config_refuses_a_string_naming_the_parse(value: str) -> None:
    with pytest.raises(TypeError, match=re.escape(f"Provider({value!r})")):
        _ = LLMClientConfig(provider=value)  # pyright: ignore[reportArgumentType]


def test_make_provider_refuses_a_string_the_same_way() -> None:
    with pytest.raises(TypeError, match=re.escape("Provider('ollama')")):
        _ = make_provider("ollama")  # pyright: ignore[reportArgumentType]


def test_config_accepts_a_parsed_member() -> None:
    config = LLMClientConfig(provider=Provider("Ollama"))
    assert config.provider is Provider.OLLAMA


def test_describe_llm_reports_a_member() -> None:
    previous = configure_llm_client(LLMClientConfig(provider=Provider("OLLAMA")))
    try:
        info = describe_llm()
    finally:
        _ = configure_llm_client(previous)
    assert info.provider is Provider.OLLAMA
