"""A retried ``Mode.JSON`` structured call leaves the caller's messages alone.

In ``Mode.JSON`` instructor adds the schema instruction to the request's
system message. instructor before 1.16.0 made that addition to the caller's
own message dicts, which llmkit passes through unchanged, so every re-send of
the same prompt carried one more copy of the schema and the caller was left
holding an edited prompt. 1.16.0 edits a copy instead, and llmkit's floor
requires it. These tests were shown to fail on instructor 1.15.4 (the second
attempt carried two copies of the schema) and pass on 1.16.0 and later.

DeepSeek pins ``Mode.JSON`` by default, so it is the configured provider here.
The prompt starts with a system message, which selects instructor's append
path: the one that stacked copies.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ChatMessage,
    LLMCallOptions,
    LLMClientConfig,
    Provider,
    RetryPolicy,
    configure_llm_client,
    structured_llm_call,
)
from llmkit.retry import with_retries
from llmkit.testing import Reply, TransportRequest, install, service_unavailable_error

# instructor's Mode.JSON schema instruction; it appears once per instruction.
SCHEMA_MARKER = "Make sure to return an instance of the JSON, not the schema itself"


class Verdict(BaseModel):
    answer: str


def _system_content(request: TransportRequest) -> str:
    system = [message for message in request.messages if message["role"] == "system"]
    assert len(system) == 1
    content = system[0]["content"]
    assert isinstance(content, str)
    return content


async def _own_retry(prompt: list[ChatMessage]) -> Verdict:
    return await structured_llm_call(
        prompt,
        Verdict,
        feature="caller-messages",
        options=LLMCallOptions(retry=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)),
    )


async def _outer_with_retries(prompt: list[ChatMessage]) -> Verdict:
    return await with_retries(
        lambda: structured_llm_call(prompt, Verdict, feature="caller-messages", retry=NO_RETRY),
        max_attempts=3,
    )


@pytest.mark.parametrize(
    "call", [_own_retry, _outer_with_retries], ids=["own-retry", "outer-with-retries"]
)
async def test_retried_json_mode_call_sends_the_schema_once_and_leaves_the_prompt(
    call: Callable[[list[ChatMessage]], Awaitable[Verdict]],
) -> None:
    configure_llm_client(
        lambda: LLMClientConfig(
            provider=Provider.DEEPSEEK, api_key="sk-test", model="deepseek-chat"
        )
    )
    system: ChatMessage = {"role": "system", "content": "You are terse."}
    user: ChatMessage = {"role": "user", "content": "Answer yes or no."}
    prompt: list[ChatMessage] = [system, user]
    before = copy.deepcopy(prompt)

    def fake(request: TransportRequest) -> Reply:
        if request.attempt == 1:
            raise service_unavailable_error()
        return Reply.json(Verdict(answer="yes"))

    with install(fake) as installed:
        result = await call(prompt)

    assert result == Verdict(answer="yes")
    assert len(installed.requests) == 2
    assert [_system_content(request).count(SCHEMA_MARKER) for request in installed.requests] == [
        1,
        1,
    ]
    # The caller's own list and dicts, not a copy, are unchanged.
    assert prompt == before
    assert prompt[0] is system
    assert system == {"role": "system", "content": "You are terse."}
