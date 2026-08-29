"""Per-call LLM invocation logging via a pluggable sink.

Every LLM round-trip is recorded as an :class:`LLMCallRecord` and handed
to the configured :class:`LogSink`. The default sink writes one YAML file
per call to a directory resolved lazily at first write (see
:func:`default_log_dir`: ``LLMKIT_LOG_DIR``, else ``data/llm-logs/`` under
the enclosing project root, else a per-user state directory), preserving
the historical log shape so existing analysis tooling keeps working.

Logging is unconditional and best-effort — a sink that raises is swallowed
so the LLM call itself never breaks because logging did. The host
application points the sink at its chosen directory once at startup via
:func:`configure_llm_logging`, mirroring the ``configure_rate_limit``
module-level pattern.
Module layout
-------------

This module is the package facade. The pieces live in siblings named for what
they own:

* :mod:`~llmkit.logging.record` — :class:`LLMCallRecord`, the data contract;
* :mod:`~llmkit.logging.sink` — the :class:`LogSink` protocol;
* :mod:`~llmkit.logging.local_yaml` — the shipped file sink;
* :mod:`~llmkit.logging.registry` — the configured sink, its
  :func:`get_log_sink` reader, and :func:`write_llm_log`;
* ``_paths`` — log-directory resolution and filename safety;
* ``_yaml`` — the safe-load-able dumper;
* ``_latch`` — the warn-once latch both the sink and the registry use.
"""

from llmkit.logging._paths import LOG_DIR_ENV_VAR, default_log_dir
from llmkit.logging.local_yaml import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_RETENTION_DAYS,
    LocalYamlLogSink,
)
from llmkit.logging.record import INDEX_FILENAME, LLMCallRecord
from llmkit.logging.registry import configure_llm_logging, get_log_sink, write_llm_log
from llmkit.logging.sink import LogSink

__all__ = [
    "DEFAULT_MAX_INDEX_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "INDEX_FILENAME",
    "LOG_DIR_ENV_VAR",
    "LLMCallRecord",
    "LocalYamlLogSink",
    "LogSink",
    "configure_llm_logging",
    "default_log_dir",
    "get_log_sink",
    "write_llm_log",
]
