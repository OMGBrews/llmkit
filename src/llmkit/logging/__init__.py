"""Per-call LLM invocation logging via a pluggable sink.

Every outer call attempt is recorded as an :class:`LLMCallRecord` and handed
to the configured :class:`LogSink`. Its ``requests`` history holds each prepared
llmkit dispatch, including in-call repairs and temperature reissues. The default sink writes one YAML file
per call to a directory resolved lazily at first write (see
:func:`default_log_dir`: ``LLMKIT_LOG_DIR``, else ``data/llm-logs/`` under
the enclosing project root, else a per-user state directory), preserving
the historical log shape so existing analysis tooling keeps working.

Managed logging uses one daemon worker and a bounded queue (256 waiting
records by default). Calls wait at most 100 ms after snapshot preparation;
abandoned streams submit without waiting. Accepted writes may finish later,
but caller capture lists never receive late path appends. Explicit low-level
``write_llm_log`` and sink writes remain synchronous. Configure the sink and
optional :class:`LogDeliveryPolicy` at startup with :func:`configure_llm_logging`.
Module layout
-------------

This module is the package facade. The pieces live in siblings named for what
they own:

* :mod:`~llmkit.logging.record` — :class:`LLMCallRecord`, the data contract;
* :mod:`~llmkit.logging.sink` — the :class:`LogSink` protocol;
* :mod:`~llmkit.logging.local_yaml` — the shipped file sink;
* :mod:`~llmkit.logging.registry` — the configured sink, its
  :func:`get_log_sink` reader, :func:`get_logging_health`, and
  :func:`write_llm_log`;
* ``_health`` — the :class:`LoggingHealth` snapshot, :class:`LoggingFailure`,
  and the lock-guarded counters behind them;
* ``_paths`` — log-directory resolution and filename safety;
* ``_yaml`` — the safe-load-able dumper;
* the warn-once latch both the sink and the registry use lives one level up,
  in ``llmkit._latch``, because the response cache shares it.
"""

from llmkit.logging._delivery import (
    flush_llm_logs,
    flush_llm_logs_async,
)
from llmkit.logging._health import LoggingFailure, LoggingHealth
from llmkit.logging._paths import LOG_DIR_ENV_VAR, default_log_dir
from llmkit.logging._policy import LogDeliveryPolicy
from llmkit.logging.local_yaml import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_RETENTION_DAYS,
    LocalYamlLogSink,
)
from llmkit.logging.record import INDEX_FILENAME, LLMCallRecord, RequestSnapshot
from llmkit.logging.registry import (
    configure_llm_logging,
    get_log_sink,
    get_logging_health,
    write_llm_log,
)
from llmkit.logging.sink import LogSink

__all__ = [
    "LogDeliveryPolicy",
    "flush_llm_logs",
    "flush_llm_logs_async",
    "DEFAULT_MAX_INDEX_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "INDEX_FILENAME",
    "LOG_DIR_ENV_VAR",
    "LLMCallRecord",
    "RequestSnapshot",
    "LocalYamlLogSink",
    "LogSink",
    "LoggingFailure",
    "LoggingHealth",
    "configure_llm_logging",
    "default_log_dir",
    "get_log_sink",
    "get_logging_health",
    "write_llm_log",
]
