"""The one check a configured duration in seconds passes, shared across packages."""

import math


def duration(value: float, name: str) -> float:
    """Return *value* if it is a finite, nonnegative number of seconds.

    Raises:
        ValueError: if *value* is a ``bool``, nonfinite or negative.
    """
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value
