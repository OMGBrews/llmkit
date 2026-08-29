"""A warn-once latch, shared by the sink and the sink registry."""

from __future__ import annotations


class OnceLatch:
    """Warn-once state for one failure site: WARNING on a new failure
    signature, DEBUG on repeats, re-armed by success.

    A permanently broken sink (unwritable directory, full disk) would
    otherwise emit a warning **with traceback** on every call — flooding
    stderr at exactly the moment the application is busiest. The signature is
    ``(type, errno)``, so a *different* failure (disk full after permission
    denied) still warns loudly instead of hiding behind the first one.
    Instances are per-site and unlocked: a race between two writers costs at
    most one duplicate warning, which is not worth a lock on the hot path.
    """

    def __init__(self) -> None:
        self._signature: tuple[type[BaseException], int | None] | None = None

    def should_warn(self, exc: BaseException) -> bool:
        """Record a failure; ``True`` when it deserves a full WARNING."""
        errno_value: object = getattr(exc, "errno", None)
        signature = (type(exc), errno_value if isinstance(errno_value, int) else None)
        if signature == self._signature:
            return False
        self._signature = signature
        return True

    def succeeded(self) -> None:
        """Re-arm: the site recovered, so the next failure warns again."""
        self._signature = None
