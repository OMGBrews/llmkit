"""The README's per-provider tables match the code they describe.

The configuration section restates two facts per provider that the code owns:
each provider's default model (``_default_model``), with its spelling on
OpenRouter where the same model has one, and each bearer provider's API-key
environment variable. The tables are read from the README itself, so a default
model or key variable that changes in code without the prose fails here.

Three configuration tables have an ``OPENROUTER`` row, so each table is found by
its exact header line rather than by its rows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llmkit.providers import Provider, _provider_class, make_provider

_README = Path(__file__).parents[2] / "README.md"

_DEFAULT_MODEL_HEADER = "| Provider | Default model | On OpenRouter |"
_ENV_VAR_HEADER = "| Provider | Environment variable |"

# The placeholder the "On OpenRouter" column uses for a default with no twin.
_NO_TWIN = "*(none)*"

# OpenRouter's catalogue id for each default model, keyed by the default model
# id itself and checked by hand against OpenRouter's model list (2026-11-02).
# OpenRouter's name can differ from the vendor's own (Claude versions use dots),
# so no rule derives it. Changing any ``_default_model`` makes the lookup below
# fail until someone re-checks the twin and records it here. ``_NO_TWIN`` marks a
# default with no other OpenRouter spelling: OpenRouter's own, and Ollama's
# local model.
_OPENROUTER_TWIN = {
    "google/gemini-3.1-flash-lite": _NO_TWIN,
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "deepseek-chat": "deepseek/deepseek-chat",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "anthropic/claude-haiku-4.5",
    "llama3.2": _NO_TWIN,
}

_MEMBER = re.compile(r"`([A-Z_]+)`")


def _table_rows(header: str) -> dict[Provider, list[str]]:
    """Return the rows of the one README table that opens with ``header``.

    Each row maps its ``Provider`` member to the remaining cells, with the
    surrounding backticks removed. A member listed twice fails, so every
    provider has exactly one row.
    """
    lines = _README.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == header]
    assert len(starts) == 1, f"expected exactly one README table headed {header!r}"

    rows: dict[Provider, list[str]] = {}
    for line in lines[starts[0] + 2 :]:  # skip the header and its |---| rule
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = _MEMBER.fullmatch(cells[0].split(" ")[0])
        assert match is not None, f"row does not start with a Provider member: {line!r}"
        member = Provider[match.group(1)]
        assert member not in rows, f"{member.name} has more than one row"
        rows[member] = [cell.strip("`") for cell in cells[1:]]
    return rows


def test_every_provider_has_one_default_model_row() -> None:
    assert set(_table_rows(_DEFAULT_MODEL_HEADER)) == set(Provider)


@pytest.mark.parametrize("member", list(Provider))
def test_default_model_column_matches_the_provider_class(member: Provider) -> None:
    default_model, _ = _table_rows(_DEFAULT_MODEL_HEADER)[member]

    assert default_model == _provider_class(member)._default_model


@pytest.mark.parametrize("member", list(Provider))
def test_openrouter_column_names_the_recorded_twin_of_the_default(member: Provider) -> None:
    _, on_openrouter = _table_rows(_DEFAULT_MODEL_HEADER)[member]
    default_model = _provider_class(member)._default_model

    assert default_model in _OPENROUTER_TWIN, (
        f"{member.name}'s default {default_model!r} has no recorded OpenRouter twin: "
        + "look it up in OpenRouter's model list and add it to _OPENROUTER_TWIN"
    )
    assert on_openrouter == _OPENROUTER_TWIN[default_model]


def _bearer_env_vars() -> dict[Provider, str]:
    """Each provider class that declares an API-key environment variable."""
    declared = {
        member: getattr(_provider_class(member), "_api_key_env_var", None) for member in Provider
    }
    return {member: var for member, var in declared.items() if isinstance(var, str)}


def test_env_var_table_lists_exactly_the_bearer_providers() -> None:
    documented = {member: cells[0] for member, cells in _table_rows(_ENV_VAR_HEADER).items()}

    assert documented == _bearer_env_vars()


@pytest.mark.parametrize("member", sorted(_bearer_env_vars(), key=str))
def test_env_var_table_names_the_variable_construction_reads(
    member: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building with no ``api_key`` reads the documented variable, and only it.

    Every bearer provider's own variable is cleared first, so a provider that
    read another row's variable fails the first half, and one that ignored the
    documented variable fails the second.
    """
    env_var = _table_rows(_ENV_VAR_HEADER)[member][0]
    for documented in _bearer_env_vars().values():
        monkeypatch.delenv(documented, raising=False)

    with pytest.raises(ValueError, match=env_var):
        _ = make_provider(member)

    monkeypatch.setenv(env_var, "test-key")
    assert make_provider(member).name
