"""The YAML dumper that guarantees a ``safe_load``-able log file.

Registration order is load-bearing and therefore local to this module: the
``Enum`` representer must be added before the catch-all ``object`` one, since an
``Enum`` member is also an ``object`` and multi-representers match in
registration order. Both registrations are import-time side effects — anything
that dumps with this Dumper before they run would silently emit
``!!python/object`` tags.
"""

from __future__ import annotations

import enum

import yaml


class LogSafeDumper(yaml.SafeDumper):
    """:class:`yaml.SafeDumper` that degrades unknown objects to plain scalars.

    ``record.response`` is whatever the caller produced — typically a Pydantic
    ``model_dump()`` in python mode, which can carry Enum members, ``Decimal``,
    ``set``, datetime subclasses, and other arbitrary objects. The stock
    (unsafe) ``Dumper`` serializes those as ``!!python/object`` tags, which
    ``yaml.safe_load`` refuses to parse (breaking the documented
    safe-load-able analysis-tooling contract) and which are an
    arbitrary-code-execution hazard for anyone using full ``yaml.load``.

    This dumper keeps the standard types (str/int/float/bool/None/dict/list,
    dates, sets, …) exactly as ``SafeDumper`` renders them, and registers two
    fallbacks for everything else: an :class:`enum.Enum` member is rendered as
    its ``.value`` (the payload, not the Python identity), and any other
    unrepresentable object is rendered as ``str(obj)``. The log therefore
    always contains plain, safe-load-able YAML regardless of what the sink is
    fed.
    """


def _represent_enum(dumper: yaml.SafeDumper, data: enum.Enum) -> yaml.Node:
    """Render an Enum member as its underlying ``.value``."""
    return dumper.represent_data(data.value)  # pyright: ignore[reportAny, reportUnknownMemberType]  # raw-llm — Enum payload is arbitrary; yaml stubs leave represent_data untyped


def _represent_fallback(dumper: yaml.SafeDumper, data: object) -> yaml.Node:
    """Render any otherwise-unrepresentable object as a plain string scalar."""
    try:
        text = str(data)
    except Exception:
        # A hostile/broken __str__ must not break logging; object.__repr__
        # never raises.
        text = object.__repr__(data)
    return dumper.represent_str(text)


# Enum first: multi-representers match in registration order, and an Enum
# member is also an ``object``, so the generic fallback would shadow it.
LogSafeDumper.add_multi_representer(enum.Enum, _represent_enum)
LogSafeDumper.add_multi_representer(object, _represent_fallback)
