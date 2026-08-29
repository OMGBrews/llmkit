"""llmkit's public call surface: one module per call family.

Every call builds an :class:`~llmkit.LLMCallRecord` (prompt, schema, response,
duration, resolved model/provider, approximate cost, any provider error) and
hands it to the configured log sink. Logging is unconditional — a sink failure
is swallowed so the LLM call itself never breaks because logging did.

The provider transport lives in :mod:`llmkit._litellm` (LiteLLM, with
``instructor`` for structured output); these functions own the logging,
retry and cost-recording contract *around* it. Three adjacent concerns are
factored out: the three-layer option merge in :mod:`llmkit.options`, the record
sink seam and capture context managers in :mod:`llmkit.capture`, and the retry
loops in :mod:`llmkit.retry`.

The families:

* :mod:`~llmkit.calls.structured` — a validated Pydantic instance back;
* :mod:`~llmkit.calls.text` — buffered plain text;
* :mod:`~llmkit.calls.stream` — streamed plain text;
* :mod:`~llmkit.calls.tool` — one tool-enabled turn;
* :mod:`~llmkit.calls.tool_stream` — one tool-enabled turn, streamed: text
  deltas, then the same completed result;
* ``_shared`` — what every family does once per call, so the next one added
  does not become another copy.
"""

from llmkit.calls.stream import STREAM_ABANDONED_ERROR, stream_text_with_log, text_llm_call_stream
from llmkit.calls.structured import structured_llm_call, structured_llm_call_sync
from llmkit.calls.text import text_llm_call, text_llm_call_sync
from llmkit.calls.tool import tool_llm_call, tool_llm_call_sync
from llmkit.calls.tool_stream import tool_llm_call_stream

__all__ = [
    "STREAM_ABANDONED_ERROR",
    "stream_text_with_log",
    "structured_llm_call",
    "structured_llm_call_sync",
    "text_llm_call",
    "text_llm_call_stream",
    "text_llm_call_sync",
    "tool_llm_call",
    "tool_llm_call_stream",
    "tool_llm_call_sync",
]
