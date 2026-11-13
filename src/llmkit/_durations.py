"""The one check a configured duration in seconds passes, shared across packages."""

import math


def duration(value: object, name: str) -> float:
    """Return *value* if it is a finite, nonnegative number of seconds.

    Raises:
        ValueError: if *value* is not a number, is a ``bool``, nonfinite or negative.
    """
    message = f"{name} must be finite and nonnegative (a number, not a bool)"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(message)
    return value
