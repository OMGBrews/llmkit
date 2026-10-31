"""Schema-backed Pydantic models for JSON-schema object contracts.

Build once with :func:`model_from_json_schema`, then reuse the returned class
with ``structured_llm_call`` or as a tool-argument model. Construct instances
with ``Model.model_validate(value)``; access plain JSON containers through
``result.root`` and serialize with ``model_dump`` or ``model_dump_json``.
There are no generated property fields. Explicit nulls, absent keys, allowed
extras, and exact integers survive default serialization; defaults never fill
missing values.

The original deep-copied schema is the validation authority. Draft-07,
2019-09, and 2020-12 are supported, with 2020-12 the default. Nested
resources cannot change the root dialect. Object roots are
required; unavailable or recursive references fail at construction and no
remote schemas are fetched. Unknown keywords are annotations. Formats are
annotations unless ``assert_formats=True``; requested unavailable checkers
fail at construction. Numeric floats use Python double precision and patterns
use Python ``re``.

``model_json_schema`` supplies a fresh adapted copy for provider generation
guidance: acyclic references are inlined and definition containers removed. It
never replaces or mutates the local contract. ``convert`` owns
construction and ``emitted`` owns runtime validation and guidance.
"""

from llmkit.json_schema.convert import model_from_json_schema

__all__ = ["model_from_json_schema"]
