"""The per-provider rate-limit budget value object.

:class:`RateLimitPolicy` lives in its own module because both sides of the
package need it at import time: :mod:`llmkit.rate_limiting._limiter` holds the
shipped default as a class attribute, and the facade re-exports it. Defining it
in the facade would make the limiter import the facade, which already imports
the limiter.
"""

import math
from dataclasses import dataclass

from llmkit.rate_limiting._tuning import BREAKER_COOLDOWN


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """The numbers one provider's calls are limited by.

    Every caller-settable number in the rate limiter lives here; the on/off
    switches (``enabled``, ``adaptive``, ``breaker``) stay keywords on
    :func:`~llmkit.configure_rate_limit`. Pass one policy as the shared default
    (``policy=``) and, where an account's limits differ, one policy per provider
    (``overrides=``). A provider runs under its override if it has one and under
    the default otherwise; the two are never merged field by field, so derive an
    override from the default with :func:`dataclasses.replace`.

    Attributes:
        max_concurrent: Maximum in-flight calls per provider: the ceiling the
            adaptive limit starts at and recovers toward, and the cap on the
            RPM burst. A non-``bool`` ``int`` >= 1; defaults to ``8``.
        rpm: Sustained requests per minute, or ``None`` for no request-rate
            limit (the default). A non-``bool`` ``int`` >= 1 when set.
        tpm: Sustained tokens per minute, or ``None`` for no token-rate limit
            (the default). A non-``bool`` ``int`` >= 1 when set.
        breaker_cooldown: Seconds the circuit breaker fails fast before
            admitting a recovery probe; inert unless the breaker is armed. A
            finite positive number that is not a ``bool``; defaults to ``30``.

    Raises:
        ValueError: from the constructor, when any field breaks its rule above
            (a ``bool``, a float count, NaN and infinity are refused).
    """

    max_concurrent: int = 8
    rpm: int | None = None
    tpm: int | None = None
    breaker_cooldown: float = BREAKER_COOLDOWN

    def __post_init__(self) -> None:
        if type(self.max_concurrent) is not int or self.max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be a positive integer, got {self.max_concurrent!r}"
            )
        if self.rpm is not None and (type(self.rpm) is not int or self.rpm < 1):
            raise ValueError(f"rpm must be a positive integer or None, got {self.rpm!r}")
        if self.tpm is not None and (type(self.tpm) is not int or self.tpm < 1):
            raise ValueError(f"tpm must be a positive integer or None, got {self.tpm!r}")
        cooldown = self.breaker_cooldown
        if isinstance(cooldown, bool) or not math.isfinite(cooldown) or cooldown <= 0:
            raise ValueError(f"breaker_cooldown must be finite and positive, got {cooldown!r}")


DEFAULT_RATE_LIMIT_POLICY = RateLimitPolicy()
"""The shipped policy: concurrency 8, no RPM or TPM limit, 30-second cooldown."""
