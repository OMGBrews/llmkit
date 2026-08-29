"""The log record itself: the data contract every sink consumes.

Depends on nothing but :mod:`llmkit._types`, which is what makes the package's
internal dependency direction obvious — record <- sink <- local_yaml <-
registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from llmkit._types import ChatMessage, ReasoningEffort

# Compact append-only summary sibling to the per-call YAML files: one JSON
# line per call, so cross-call scans don't have to glob + parse every YAML.
INDEX_FILENAME = "index.jsonl"


@dataclass(frozen=True)
class LLMCallRecord:
    """A single LLM round-trip, as written to the log sink.

    ``model`` is the *resolved effective* model (the provider default
    substituted when the caller passed ``None``) and ``provider`` names
    the active provider, so cost attribution is a ``grep`` over the logs
    rather than a code trace. ``schema`` is the output schema name for a
    structured call; the literal ``"text"`` / ``"stream"`` for a buffered or
    streamed plain-text call; and ``"tools"`` / ``"tools-stream"`` for a
    buffered or streamed tool turn (``"tools+<Name>"`` when a compose schema
    was supplied). Nothing branches on the value — it is a grep key. ``response`` is the Pydantic-dumped result,
    the accumulated text, or ``None``.

    ``prompt`` is the request as the caller passed it: either the plain string
    or the full :data:`~llmkit._types.ChatMessage` list, including any
    assistant-tool-call and tool-result turns a tool loop appended. A sink
    receives it unnormalized — there is no single string form — so a sink that
    renders prompts must handle both shapes.

    ``approximate_cost`` is a best-effort USD estimate for budget
    visibility — NOT a billing figure. It is sourced from LiteLLM's
    per-response cost (no local price table) and is ``None`` when the
    provider does not report it (e.g. streamed calls).

    ``call_id`` is one ``uuid4`` hex per *logical* call and ``attempt`` the
    1-based attempt number within it, so the N records a retried call
    produces join on ``call_id`` instead of feature + timestamp proximity
    (which breaks under concurrent same-feature fan-out). ``queue_wait_ms``
    is the time this attempt spent queued behind llmkit's own rate limiter
    — ``duration_ms`` includes it, so provider latency is approximately
    ``duration_ms - queue_wait_ms``. All four default ``None`` for
    directly-constructed records.

    ``temperature`` is ``None`` exactly when the call requested the
    provider's default sampling (``temperature=None`` — no ``temperature``
    kwarg sent); an unset call resolves llmkit's
    :data:`~llmkit.DEFAULT_TEMPERATURE` (``0.2``) and records that value.
    Custom sinks that read ``record.temperature`` must handle the ``None``
    state (the YAML sink writes it as ``null``).

    ``run_id`` is the *outer* scope ``call_id`` does not provide: the run —
    an eval sweep, a rehearsal, an incident replay — that this call belonged
    to, so a shared log directory can be filtered by run instead of by
    timestamp window (which breaks whenever two runs overlap). The call layer
    stamps it from :func:`~llmkit.run_scope.get_run_id`, which resolves an
    active :func:`~llmkit.run_scope.run_scope`, then a process-wide
    :func:`~llmkit.run_scope.set_run_id`, then ``LLMKIT_RUN_ID``.

    A ``None`` ``run_id`` is the pre-``run_id`` shape of every record, YAML
    body and index line.

    The remaining fields are the call's own identity and request shape:
    ``started_at`` (UTC, when the attempt began), ``feature`` / ``label`` (the
    caller-supplied telemetry pair that scopes the log filename and the
    ``index.jsonl`` grouping), ``error`` (``"<ExceptionType>: <message>"`` for a
    failed attempt, or the bare type name where the exception carries no
    message; ``None`` for a clean one — a streamed call abandoned by its
    consumer records :data:`~llmkit.STREAM_ABANDONED_ERROR` here rather than
    ``None``, and a tool round cancelled mid-flight records ``"CancelledError"``
    rather than reading as a successful round that requested nothing), and
    ``max_tokens`` / ``reasoning_effort`` (the request-shaping knobs as resolved
    for this call, ``None`` when not sent).

    Three fields carry the **tool lane** and are ``None`` on every other
    surface: ``tools`` is the tool list as offered to the provider (each entry
    the ``{"type": "function", "function": {...}}`` wire shape
    :meth:`~llmkit.ToolDefinition.to_litellm` produces), ``tool_calls`` is the
    calls the model requested (:meth:`~llmkit.ToolCall.to_wire` shape), and
    ``usage`` is the turn's token counts
    (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, any of which
    may be ``None`` when the provider did not report it). The YAML body carries
    all three; ``index.jsonl`` deliberately does not, so the compact scan line
    keeps its fixed shape.
    """

    started_at: datetime
    feature: str
    label: str | None
    model: str | None
    provider: str | None
    temperature: float | None
    duration_ms: float
    schema: str
    prompt: str | Sequence[ChatMessage]
    response: Any  # pyright: ignore[reportExplicitAny]  # raw-llm — Pydantic dump or accumulated text
    error: str | None
    approximate_cost: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    call_id: str | None = None
    attempt: int | None = None
    queue_wait_ms: float | None = None
    run_id: str | None = None
    tools: list[dict[str, object]] | None = None
    tool_calls: list[dict[str, object]] | None = None
    usage: dict[str, int | None] | None = None
