"""The call baseline is private even from its transport entry point."""

from collections.abc import Sequence
from copy import deepcopy
from typing import cast
from unittest.mock import patch

from llmkit import ChatMessage, RetryPolicy, capture_llm_records, text_llm_call
from tests.calls.test_input_snapshots import inputs, provider


async def test_transport_entry_cannot_change_next_attempt() -> None:
    prompt, _ = inputs()
    expected = deepcopy(prompt)
    attempts = 0

    async def transport(
        working: str | Sequence[ChatMessage], **_kwargs: object
    ) -> tuple[str, float | None, tuple[int | None, int | None, int | None]]:
        nonlocal attempts
        attempts += 1
        assert working == expected
        if attempts == 1:
            assert not isinstance(working, str)
            parts = cast("list[dict[str, object]]", working[0]["content"])
            parts[0]["text"] = "transport changed its working prompt"
            raise TimeoutError("retry")
        return "done", None, (None, None, None)

    with (
        patch("llmkit._litellm.acompletion_text", transport),
        capture_llm_records() as records,
    ):
        assert (
            await text_llm_call(
                prompt,
                feature="ownership",
                provider=provider(),
                retry=RetryPolicy(max_attempts=2, backoff_base_seconds=0),
            )
            == "done"
        )
    assert attempts == 2
    assert prompt == expected
    assert [record.prompt for record in records] == [expected, expected]
