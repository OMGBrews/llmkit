"""``LLMClientConfig.__repr__`` masks the credential and shows only what is set.

The dataclass auto-repr would render ``api_key='sk-...'`` verbatim, so any
host-side ``print(config)``, log line, or exception reporter's locals capture
would exfiltrate the credential. The hand-written repr masks a set key while
keeping its *presence* visible for debugging, and — like
``LLMCallOptions.__repr__`` — prints only fields that differ from their default
so the one value a maintainer is looking for isn't drowned in defaults.
"""

from __future__ import annotations

from llmkit import LLMClientConfig, Provider

_SECRET = "sk-proj-abcdef0123456789-super-secret"


def test_config_repr_with_api_key_masks_the_value() -> None:
    """A set ``api_key`` shows as ``<redacted>`` and the raw value never appears."""
    config = LLMClientConfig(provider=Provider.ANTHROPIC, api_key=_SECRET)
    rendered = repr(config)
    assert "api_key=<redacted>" in rendered
    assert _SECRET not in rendered


def test_config_str_never_contains_api_key() -> None:
    """``str()`` (which falls back to ``__repr__``) is safe too — the path an
    f-string or ``logging`` ``%s`` takes."""
    config = LLMClientConfig(provider=Provider.OPENAI, api_key=_SECRET)
    assert _SECRET not in str(config)


def test_config_repr_without_api_key_omits_the_field() -> None:
    """An unset ``api_key`` (the default) is omitted entirely — no ``api_key=None``
    noise, and no false suggestion that a key is set."""
    config = LLMClientConfig(provider=Provider.BEDROCK, aws_region_name="us-east-1")
    rendered = repr(config)
    assert "api_key" not in rendered
    assert "aws_region_name='us-east-1'" in rendered
    assert "provider=" in rendered


def test_config_repr_shows_only_non_default_fields() -> None:
    """Only fields differing from their default are rendered; ``provider`` (which
    has no default) always is. Defaults like ``model=None`` stay out."""
    config = LLMClientConfig(provider=Provider.OLLAMA, base_url="http://host:11434")
    rendered = repr(config)
    assert "base_url='http://host:11434'" in rendered
    assert "model=" not in rendered
    assert "gemini_structured_output=" not in rendered
    assert "reasoning_effort=" not in rendered


def test_config_repr_empty_string_api_key_stays_visible() -> None:
    """An explicit empty ``api_key=""`` is not a secret and is shown as itself,
    so an accidental empty key is visible rather than masquerading as a set one."""
    config = LLMClientConfig(provider=Provider.OPENAI, api_key="")
    assert "api_key=''" in repr(config)


def test_config_repr_round_trips_the_non_secret_fields() -> None:
    """The repr is still a faithful, readable summary of the routing fields — the
    masking is scoped to the credential, nothing else."""
    config = LLMClientConfig(
        provider=Provider.VERTEX,
        model="gemini-2.5-flash",
        vertex_project="proj-123",
        vertex_location="europe-west4",
    )
    rendered = repr(config)
    assert rendered.startswith("LLMClientConfig(")
    assert "model='gemini-2.5-flash'" in rendered
    assert "vertex_project='proj-123'" in rendered
    assert "vertex_location='europe-west4'" in rendered
