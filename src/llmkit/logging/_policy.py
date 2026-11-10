"""Managed delivery configuration, independent of the sink registry."""

import threading
from dataclasses import dataclass

from llmkit._durations import duration


@dataclass(frozen=True)
class LogDeliveryPolicy:
    """Managed delivery settings, configured before the first admission.

    ``queue_capacity`` bounds waiting records (including preparation), plus at
    most one active write. ``delivery_wait`` is seconds from prepared-job
    publication, including queue time; its maximum is 0.1. Snapshot preparation
    has payload-dependent overhead outside that wait. ``drain_timeout`` is the
    finite budget for explicit flushes and the final process-exit drain.
    """

    queue_capacity: int = 256
    delivery_wait: float = 0.1
    drain_timeout: float = 1.0

    def __post_init__(self) -> None:
        if type(self.queue_capacity) is not int or self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if duration(self.delivery_wait, "delivery_wait") > 0.1:
            raise ValueError("delivery_wait must be at most 0.1 seconds")
        _ = duration(self.drain_timeout, "drain_timeout")


_lock = threading.Lock()
_policy = LogDeliveryPolicy()
_frozen = False


def get_delivery_policy() -> LogDeliveryPolicy:
    return _policy


def freeze_delivery_policy() -> None:
    global _frozen
    with _lock:
        _frozen = True


def configure_delivery(policy: LogDeliveryPolicy) -> None:
    if not isinstance(policy, LogDeliveryPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime public boundary
        raise TypeError("delivery_policy must be a LogDeliveryPolicy")  # pyright: ignore[reportUnreachable]  # untyped callers
    global _policy
    with _lock:
        if _frozen:
            raise RuntimeError("configure delivery_policy before the first managed admission")
        _policy = policy
