"""Runtime JSON-schema-dict → Pydantic-model conversion.

Consumers that declare their structured-output contracts as **JSON-schema
dicts** — typically because the same contract is shared across a Node
backend, a frontend, and Python — would otherwise have to hand-write a
converter to a Pydantic model before they can call
:func:`~llmkit.structured_llm_call` (which is, and stays,
Pydantic-model-only). :func:`model_from_json_schema` is that converter,
centralised and tested once so each consumer doesn't re-discover the same
footguns.

The intended pattern is **build once, reuse**::

    from llmkit import model_from_json_schema, structured_llm_call

    Invoice = model_from_json_schema(invoice_schema)  # once, at import
    result = await structured_llm_call(prompt, Invoice, feature="billing")

Supported JSON-schema subset (day one)
--------------------------------------
The subset is deliberately the one real dict-based consumers use; anything
outside it raises a clear :class:`ValueError` naming the unsupported
construct, rather than silently producing a wrong model:

* ``object`` with ``properties``; ``required`` vs optional via the
  ``required`` array
* scalar types: ``string``, ``integer``, ``number``, ``boolean`` (and
  ``null`` / nullable via ``["string", "null"]`` or ``anyOf`` with a null
  branch)
* ``array`` (``items``), including arrays of objects
* ``enum`` (on a scalar field) — the annotation is an inline ``Literal[...]``,
  so the emitted schema carries the enum *in place* (``{"type": ..., "enum":
  [...]}``, a legal sibling of ``description``), never a ``$defs`` entry
  behind a ``$ref``
* nested objects, inline or via local ``$defs`` / ``$ref`` references
  (``#/$defs/Name`` or the legacy ``#/definitions/Name``). A ``$ref`` may carry
  siblings: metadata and value bounds (``description``, ``default``, the
  numeric/length bounds) merge over the target with the outer value winning,
  so ``{"$ref": "#/$defs/Count", "minimum": 5}`` keeps the bound and a
  nullable-wrapped ``$ref`` inherits the target's ``description``. A *structural*
  sibling — a type/shape keyword (``type`` / ``enum`` / ``items`` /
  ``properties`` / ``required`` / ``additionalProperties`` / ``title``) or any
  subschema applicator (``anyOf`` / ``oneOf`` / ``allOf`` / ``not`` / ``if`` /
  ``then`` / ``else`` / …) — is a JSON-Schema conjunction a merge cannot express,
  so it is rejected unless it restates the target's own value: a ``$ref``-sibling
  ``enum`` or ``allOf`` is a clear error, never a silently-widened field
* subschema *applicators* — ``allOf`` / ``not`` / ``if`` / ``then`` / ``else`` /
  ``dependentSchemas`` / ``dependentRequired`` / ``propertyNames`` /
  ``patternProperties`` / ``prefixItems`` / ``contains`` /
  ``unevaluatedProperties`` / ``unevaluatedItems`` — are **rejected at every
  site**, not only beside a ``$ref``. Each constrains by composition and a
  generated field is one annotation plus a fixed set of ``Field`` bounds, so an
  applicator has nowhere to land; dropping one is wrong in both directions (a
  dropped ``allOf`` bound accepts what the schema forbids; a dropped
  ``prefixItems`` re-reads the sibling ``items`` as "every element" and rejects
  what the schema permits). ``anyOf`` / ``oneOf`` are exempt — they are the
  nullable spelling, consumed below
* ``object`` with ``properties``; a propertyless object (``properties`` absent
  *or* an explicit empty ``{}``) is rejected unless it opts into open-ended keys
  with ``additionalProperties: true`` — otherwise it would build a zero-field
  model that rejects every real response
* ``additionalProperties``: ``true`` (an open object — extra keys are accepted
  and kept) or ``false`` / absent (strict ``extra="forbid"``, the default); a
  *typed* ``additionalProperties`` map is rejected

Per-field constraints
----------------------
A small, fixed set of per-field constraints is carried through to the
generated Pydantic ``Field`` so the model validates *value bounds*, not just
shape. The supported set is **exactly**:

* numeric: ``minimum`` → ``ge``, ``maximum`` → ``le``,
  ``exclusiveMinimum`` → ``gt``, ``exclusiveMaximum`` → ``lt`` — the **numeric**
  (Draft 2020-12) form only. The Draft-4 *boolean* form
  (``"exclusiveMinimum": true`` qualifying a sibling ``minimum``) is not
  recognised and is dropped, so such a bound is treated as inclusive.
* string: ``minLength`` → ``min_length``, ``maxLength`` → ``max_length``
* array: ``minItems`` → ``min_length``, ``maxItems`` → ``max_length``
* ``description`` → ``Field(description=...)`` (instructor surfaces this as
  per-field guidance to the model)

Any other *leaf* constraint keyword (``pattern``, ``format``, ``multipleOf``,
``uniqueItems``, ``const``, …) is **silently dropped** — deliberately, to
avoid partial enforcement that looks complete. Nothing outside the list above
is enforced; if a schema relies on one of those, validate it elsewhere.

The silent drop is scoped to those per-value keywords. A *structural* construct
outside the supported subset — a subschema applicator, a multi-variant union, a
typed ``additionalProperties`` map — raises instead, because losing one changes
the shape the model validates rather than leaving a single value unchecked.

A schema-level ``default`` on a non-required field is likewise **not** carried
into the model: the field becomes optional with a ``None`` default and, via the
``exclude_none`` dump contract below, is simply omitted when unset. Supply
defaults after parsing if you need them.

Serialization contract
-----------------------
The generated model maps a **non-required** JSON-schema field to an
*optional* Pydantic field whose default is ``None``. To keep an omitted
optional from round-tripping back out as ``"field": null`` — which fails
re-validation against a JSON schema that lists the field as a non-nullable
optional — the generated model's :meth:`~pydantic.BaseModel.model_dump`
and :meth:`~pydantic.BaseModel.model_dump_json` drop such ``None`` values by
default. An optional the model never set is therefore *absent* from the dump,
not present-and-null.

The drop is **scoped to optional fields**: a field that is in the schema's
``required`` array but typed nullable (``["string", "null"]`` or an ``anyOf``
null branch) and legitimately set to ``None`` is **kept**, because dropping a
required field would itself break re-validation. Callers can pass
``exclude_none=False`` to keep every null, or ``exclude_none=True`` for the
native "drop all nulls" behaviour.

Emitted JSON schema (strict structured outputs)
------------------------------------------------
instructor serialises the generated model with a zero-argument
``model_json_schema()`` call, and OpenAI's strict ``response_format``
validator rejects any ``$ref`` node carrying sibling keywords (``$ref cannot
have keywords {'description'}``) as well as ``allOf`` wholesale. The
generated models therefore guarantee neither shape appears in the emitted
document: enum fields are inline ``Literal`` annotations (no ``$ref`` at
all), and a *described object-typed property* — which pydantic factors into
``$defs`` and references with the ``description`` beside the ``$ref`` — is
inlined at the use site with the siblings merged over the def (outer wins),
after which ``$defs`` entries nothing references any more are pruned. Bare
``$ref`` nodes, which the validator allows, stay shared. Out of scope here:
strict mode also demands every property be listed in ``required``, so a
schema with *optional* fields still needs relaxing before a ``strict: true``
call can accept it.
Module layout
-------------

Two halves, split where the concerns actually part:

* :mod:`~llmkit.json_schema.convert` — the *build* side: schema dict in, model
  class out, plus every rejection this converter makes deliberately;
* :mod:`~llmkit.json_schema.emitted` — the *runtime* side: what a generated
  model does when something dumps it or asks it for its JSON schema.

`emitted` is a strict leaf; `convert` imports it, never the reverse.
"""

from llmkit.json_schema.convert import model_from_json_schema

__all__ = ["model_from_json_schema"]
