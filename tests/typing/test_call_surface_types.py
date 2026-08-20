"""Type-level coverage for the public call-surface types.

These fixes (``Message``, ``ReasoningEffort``) are typing-only — runtime is
unchanged — so the real assertions live in the ``_typecheck_*`` bodies below
and are verified by ``basedpyright`` (the ``uv run basedpyright`` gate), not by
executing them. Those helpers are never run: ``test_typecheck_helpers_exist``
only references them (so they count as accessed and are analysed), while the
type checker inspects their bodies. The collected ``test_*`` functions are the
runtime half — the two types are public and a ``Message`` is a plain dict.

The negative cases carry ``# pyright: ignore[reportArgumentType]``. Because
``reportUnnecessaryTypeIgnoreComment`` is a CI-failing warning in this repo's
``recommended`` tier, an ignore that stops being necessary — a future change
that wrongly *widens* the accepted type — fails the gate. That makes each
negative a self-enforcing assertion that the shape is still rejected.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import assert_type

from pydantic import BaseModel

import llmkit
from llmkit import (
    UNSET,
    LLMCallOptions,
    Message,
    ReasoningEffort,
    Unset,
    stream_text_with_log,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_sync,
)


class _Out(BaseModel):
    value: int


async def _typecheck_prompt_accepts_valid_shapes() -> None:
    # A plain string is the simplest prompt; return types stay precise.
    _ = assert_type(await structured_llm_call("hi", _Out, feature="t"), _Out)
    _ = assert_type(structured_llm_call_sync("hi", _Out, feature="t"), _Out)
    _ = assert_type(await text_llm_call("hi", feature="t"), str)
    _ = assert_type(text_llm_call_sync("hi", feature="t"), str)

    # An inline list of message dicts matches Message structurally.
    _ = await structured_llm_call(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        _Out,
        feature="t",
    )

    # A variable annotated with the imported Message symbol — proves Message is
    # public API, not just an inline-literal shape the checker happens to match.
    msgs: list[Message] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    _ = await text_llm_call(msgs, feature="t")

    # Multimodal content: the list-of-parts arm the old str-only content forbade.
    multimodal: list[Message] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example/img.png"}},
            ],
        }
    ]
    _ = await text_llm_call(multimodal, feature="t")

    # Streaming takes the same prompt shapes and keeps its generator return type.
    _ = assert_type(text_llm_call_stream(msgs, feature="t"), AsyncGenerator[str])


async def _typecheck_reasoning_effort_accepts_values() -> None:
    # Canonical members, an arbitrary provider-specific string (the open ``| str``
    # arm — e.g. OpenAI's "minimal"), and None all type-check. There is
    # deliberately NO negative case for reasoning_effort: ``ReasoningEffort``
    # widens to ``str`` under the checker, so every string is accepted by design
    # (see the ReasoningEffort docstring). A typo like "hihg" is NOT a type error.
    for effort in ("disable", "low", "medium", "high", "minimal"):
        eff: ReasoningEffort = effort
        _ = await text_llm_call("hi", feature="t", reasoning_effort=eff)
    _ = await text_llm_call("hi", feature="t", reasoning_effort=None)


async def _typecheck_temperature_accepts_none() -> None:
    # ``temperature=None`` — "use the provider default; send no temperature
    # kwarg" — is accepted on every public call surface, direct and through
    # options. Numeric values (including 0.0) remain accepted.
    _ = await structured_llm_call("hi", _Out, feature="t", temperature=None)
    _ = structured_llm_call_sync("hi", _Out, feature="t", temperature=None)
    _ = await text_llm_call("hi", feature="t", temperature=None)
    _ = text_llm_call_sync("hi", feature="t", temperature=None)
    _ = text_llm_call_stream("hi", feature="t", temperature=None)
    _ = stream_text_with_log("hi", feature="t", temperature=None)
    _ = await structured_llm_call("hi", _Out, feature="t", temperature=0.0)
    _ = await text_llm_call("hi", feature="t", temperature=0.7)
    opts: LLMCallOptions = LLMCallOptions(temperature=None)
    _ = await structured_llm_call("hi", _Out, feature="t", options=opts)
    _ = LLMCallOptions(temperature=0.4)

    # The documented typed-wrapper idiom, now widened: a wrapper typed
    # ``float | None | Unset`` forwards the sentinel unconditionally and
    # lets llmkit resolve "not passed" vs "None" vs a number.
    def wrapper(prompt: str, temperature: float | None | Unset = UNSET) -> None:
        _ = text_llm_call(prompt, feature="t", temperature=temperature)

    _ = wrapper


def _typecheck_prompt_rejects_invalid_shapes() -> None:
    # The raw wire shape the fix retired: dict[str, str] permits typo keys and
    # forbids multimodal content, so it is no longer assignable to the prompt.
    wire: list[dict[str, str]] = [{"role": "user", "content": "hi"}]
    _ = structured_llm_call_sync(wire, _Out, feature="t")  # pyright: ignore[reportArgumentType]

    # A bare list of strings was never a valid message list.
    strings: list[str] = ["hi"]
    _ = text_llm_call_sync(strings, feature="t")  # pyright: ignore[reportArgumentType]

    # The two marquee properties, guarded directly. These MUST be inline dict
    # literals — a pre-typed variable widens to dict[str, str] and would be
    # rejected for list-invariance instead, which is the `wire` case above and
    # says nothing about keys or roles. Written this way, widening `Message`
    # (e.g. `role: str`) makes the corresponding ignore unnecessary and fails
    # the gate via reportUnnecessaryTypeIgnoreComment — so the advertised
    # "unknown keys and mistyped roles are type errors" cannot silently regress.
    _ = structured_llm_call_sync([{"roel": "user", "content": "hi"}], _Out, feature="t")  # pyright: ignore[reportArgumentType]  # unknown key
    _ = structured_llm_call_sync([{"role": "usr", "content": "hi"}], _Out, feature="t")  # pyright: ignore[reportArgumentType]  # mistyped role


def test_typecheck_helpers_exist() -> None:
    # Reference the type-check-only helpers so they count as accessed. They are
    # analysed by basedpyright but never executed (they would need a configured
    # client and real credentials); their value is entirely in the type gate.
    for helper in (
        _typecheck_prompt_accepts_valid_shapes,
        _typecheck_reasoning_effort_accepts_values,
        _typecheck_temperature_accepts_none,
        _typecheck_prompt_rejects_invalid_shapes,
    ):
        assert callable(helper)


def test_public_types_are_exported_and_message_is_a_dict() -> None:
    assert "Message" in llmkit.__all__
    assert "ReasoningEffort" in llmkit.__all__
    # A Message is a plain dict at runtime (TypedDict has no runtime identity).
    message: Message = {"role": "user", "content": "hi"}
    assert message["role"] == "user"
    assert message["content"] == "hi"
