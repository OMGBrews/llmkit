"""``structured_llm_call``'s ``T`` is bounded to ``pydantic.BaseModel``.

With ``py.typed`` shipped, the signature *is* the contract: an unbounded
``T`` let a caller pass a dataclass or plain class as ``output_schema``
with zero type errors, deferring the failure to runtime inside instructor.
The ``[T: BaseModel]`` bound makes a non-Pydantic schema a *type* error —
a checker run can't be asserted from pytest, so these tests pin the bound
itself through the PEP 695 type params.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from llmkit import structured_output


def test_structured_call_type_param_is_bounded_to_basemodel() -> None:
    """The async call and the sync wrapper both declare ``[T: BaseModel]``."""
    for func in (
        structured_output.structured_llm_call,
        structured_output.structured_llm_call_sync,
    ):
        (type_param,) = func.__type_params__
        assert isinstance(type_param, TypeVar), f"{func.__name__} has no plain TypeVar"
        assert type_param.__bound__ is BaseModel, f"{func.__name__}'s T is not BaseModel-bounded"
