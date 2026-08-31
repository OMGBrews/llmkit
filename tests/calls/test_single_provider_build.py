"""The default call path builds the provider exactly once.

Regression guard for the transport/log seam: on the default (no per-call
``provider=``) path the transport and the log record share one provider
instance, so ``build_provider`` is called a single time per call — not once
to run the call and again only to read ``.model``/``.name`` for the log. A
per-call ``provider=`` override builds nothing (the passed instance is
reused throughout). These drive the *real* transport with only the LiteLLM /
instructor boundary faked, so the count reflects the production call path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from contextlib import AbstractContextManager, contextmanager
from unittest.mock import MagicMock, patch

from llmkit import calls as llm_calls
from tests._support import OkSchema


def _seam_provider() -> MagicMock:
    """A fake provider satisfying both the transport seam and the log read."""
    provider = MagicMock()
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", "api_base": None})
    provider.instructor_mode = "json"
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None
    provider.model = "gemini-3.1-flash-lite"
    provider.name = "Google AI Studio"
    return provider


@contextmanager
def _count_builds() -> Generator[MagicMock]:
    """Count every ``build_provider`` invocation on the default call path.

    ``_litellm`` binds ``build_provider`` at import while the call/log path
    imports it lazily — in production both are the same function, so the test
    patches both names with one shared mock and the count is the true number
    of provider constructions for the call.
    """
    build = MagicMock(return_value=_seam_provider())
    with (
        patch("llmkit.providers.build_provider", build),
        patch("llmkit._litellm.build_provider", build),
    ):
        yield build


def _patch_structured_seam() -> AbstractContextManager[MagicMock]:
    async def _fake_create_with_completion(**_kwargs: object) -> tuple[OkSchema, MagicMock]:
        return OkSchema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat = MagicMock(
        completions=MagicMock(create_with_completion=_fake_create_with_completion)
    )
    return patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client)


def _patch_text_seam() -> AbstractContextManager[MagicMock]:
    fake_resp = MagicMock(_hidden_params={})
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        return fake_resp

    return patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion)


def _patch_stream_seam() -> AbstractContextManager[MagicMock]:
    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[MagicMock]:
            async def _gen() -> AsyncIterator[MagicMock]:
                for delta in ("he", "llo"):
                    yield MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])

            return _gen()

    async def _fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream()

    return patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion)


def test_structured_default_path_builds_provider_once() -> None:
    with _count_builds() as build, _patch_structured_seam():
        result = asyncio.run(llm_calls.structured_llm_call("hi", OkSchema, feature="x"))
    assert result.ok is True
    assert build.call_count == 1


def test_structured_override_path_builds_nothing() -> None:
    with _count_builds() as build, _patch_structured_seam():
        result = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="x", provider=_seam_provider())
        )
    assert result.ok is True
    assert build.call_count == 0


def test_text_default_path_builds_provider_once() -> None:
    with _count_builds() as build, _patch_text_seam():
        text = asyncio.run(llm_calls.text_llm_call("hi", feature="x"))
    assert text == "ok"
    assert build.call_count == 1


def test_stream_default_path_builds_provider_once() -> None:
    async def _drive() -> list[str]:
        return [chunk async for chunk in llm_calls.text_llm_call_stream("hi", feature="x")]

    with _count_builds() as build, _patch_stream_seam():
        chunks = asyncio.run(_drive())
    assert chunks == ["he", "llo"]
    assert build.call_count == 1
