"""Tests for the dict-in / data-out structured call functions.

:func:`structured_data_call` and :func:`structured_data_call_sync` are the
dict→dict siblings of :func:`structured_llm_call`: a caller hands a
JSON-schema *dict*, the call validates the provider's response against a
model built from it, and the validated result comes back as a plain ``dict``
— never the intermediate Pydantic instance. These tests pin that round-trip
through the same faked transport the other structured tests patch
(``llmkit._litellm.acompletion_structured``): the data-out shape, that an
omitted optional is *absent* rather than present-as-``null``, that the same
call-surface knobs (``options``, per-call ``provider=``) thread through, and
that the sync wrapper mirrors the async one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    LocalYamlLogSink,
    configure_llm_logging,
)

# Imported from the submodule rather than the package root: the integration
# phase owns adding ``LLMCallOptions``/``structured_data_call`` to
# ``llmkit.__all__``.
from llmkit.structured_output import (
    LLMCallOptions,
    structured_data_call,
    structured_data_call_sync,
)

# A representative dict schema: a required scalar, a nested object, an array
# of scalars, and an OPTIONAL field whose omission must not surface as null.
_PERSON_SCHEMA: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]  # test-fixture — a literal JSON-schema dict
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "address": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "string"},
            },
            "required": ["city"],
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "nickname": {"type": "string"},
    },
    "required": ["name", "age", "address", "tags"],
}


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse pytest fixture, invoked by pytest's collection machinery
    """Point logging at a no-op sink so these tests don't touch the disk."""
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.model = "gemini-2.5-flash-lite"
    provider.name = "Google AI Studio"
    return provider


type _StructuredFake = Callable[..., Awaitable[tuple[BaseModel, float | None]]]


def _transport_returning(values: dict[str, Any]) -> _StructuredFake:  # pyright: ignore[reportExplicitAny]  # test-helper — arbitrary validated field values
    """A fake ``acompletion_structured`` that instantiates the requested model.

    The transport seam receives the generated model class as its positional
    ``output_schema``; the fake builds an instance of it from *values* so the
    response flows back through ``structured_data_call``'s ``model_dump()``
    exactly as a real validated parse would.
    """

    async def _fake(*args: object, **_kwargs: object) -> tuple[BaseModel, float | None]:
        output_model = args[1]
        assert isinstance(output_model, type) and issubclass(output_model, BaseModel)
        return output_model(**values), 0.0

    return _fake


def test_async_round_trips_dict_in_data_out() -> None:
    """A dict schema round-trips to a plain dict of the right shape."""
    fake = _transport_returning(
        {
            "name": "Ada",
            "age": 36,
            "address": {"city": "London", "zip": "EC1"},
            "tags": ["math", "engines"],
            "nickname": "Countess",
        }
    )
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):
        result = asyncio.run(structured_data_call("hi", _PERSON_SCHEMA, feature="extraction"))

    assert isinstance(result, dict)
    assert result == {
        "name": "Ada",
        "age": 36,
        "address": {"city": "London", "zip": "EC1"},
        "tags": ["math", "engines"],
        "nickname": "Countess",
    }


def test_omitted_optional_is_absent_not_null() -> None:
    """An optional field the model never set is ABSENT from the dump, not null."""
    fake = _transport_returning(
        {
            "name": "Ada",
            "age": 36,
            "address": {"city": "London"},
            "tags": [],
        }
    )
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):
        result = asyncio.run(structured_data_call("hi", _PERSON_SCHEMA, feature="extraction"))

    # The omitted optional must not round-trip as ``"nickname": null``.
    assert "nickname" not in result
    # The nested optional ``zip`` was likewise omitted.
    assert "zip" not in result["address"]
    assert result["address"] == {"city": "London"}


def test_sync_mirrors_async() -> None:
    """``structured_data_call_sync`` drives the same path and returns a dict."""
    fake = _transport_returning(
        {
            "name": "Grace",
            "age": 85,
            "address": {"city": "Arlington"},
            "tags": ["navy"],
        }
    )
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=fake),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):
        result = structured_data_call_sync("hi", _PERSON_SCHEMA, feature="extraction")

    assert isinstance(result, dict)
    assert result == {
        "name": "Grace",
        "age": 85,
        "address": {"city": "Arlington"},
        "tags": ["navy"],
    }


def test_inherits_call_surface_options_and_provider_override() -> None:
    """The data call inherits ``options`` merge AND the per-call ``provider=``.

    Proves it delegates to the shared structured path rather than
    re-implementing the call surface: the transport sees the options-supplied
    ``model``/``max_tokens`` and the explicit per-call provider override.
    """
    seen: list[dict[str, object]] = []
    override = _provider()
    override.name = "OpenRouter"

    async def _fake(*args: object, **kwargs: object) -> tuple[BaseModel, float | None]:
        seen.append(dict(kwargs))
        output_model = args[1]
        assert isinstance(output_model, type) and issubclass(output_model, BaseModel)
        return (
            output_model(name="x", age=1, address={"city": "y"}, tags=[]),
            0.0,
        )

    options = LLMCallOptions(model="shared-model", max_tokens=256)
    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake):
        _ = asyncio.run(
            structured_data_call(
                "hi", _PERSON_SCHEMA, feature="extraction", provider=override, options=options
            )
        )

    assert seen[0]["model"] == "shared-model"
    assert seen[0]["max_tokens"] == 256
    assert seen[0]["provider"] is override
