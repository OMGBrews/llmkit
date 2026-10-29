"""The README's ``StructuredStdoutSink`` example prints what it illustrates.

The example is read from the README itself rather than copied, so this fails
when the published block drifts. It runs in a fresh, isolated interpreter
(``python -I``) with no logging configuration at all, the state a reader
pasting the example starts from, and asserts on the actual stdout.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

_README = Path(__file__).parents[2] / "README.md"

_FIELDS = {
    "feature",
    "label",
    "model",
    "provider",
    "schema",
    "duration_ms",
    "approximate_cost",
    "error",
}

_DRIVER = """
from llmkit import LLMClientConfig, Provider, configure_llm_client, text_llm_call_sync
from llmkit.testing import Reply, install

configure_llm_client(
    lambda: LLMClientConfig(
        provider=Provider.OPENROUTER, model="google/gemini-2.5-flash", api_key="sk-or-x"
    )
)
with install(lambda req: Reply.text("hi")):
    text_llm_call_sync("say hi", feature="demo", label="x")
"""


def _example_block() -> str:
    blocks = cast(
        "list[str]",
        re.findall(r"```python\n(.*?)```", _README.read_text(encoding="utf-8"), re.DOTALL),
    )
    matches = [b for b in blocks if "class StructuredStdoutSink" in b]
    assert len(matches) == 1, "expected exactly one README block defining StructuredStdoutSink"
    return matches[0]


def test_readme_stdout_sink_prints_one_json_line_with_the_illustrated_fields() -> None:
    script = _example_block() + _DRIVER

    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, proc.stdout
    payload = cast("object", json.loads(lines[0]))
    assert isinstance(payload, dict)
    assert set(payload) == _FIELDS  # pyright: ignore[reportUnknownArgumentType]  # json payload
    assert payload["feature"] == "demo"
    assert payload["label"] == "x"
    assert payload["model"] == "google/gemini-2.5-flash"
    assert payload["error"] is None
    assert "WARNING" not in proc.stderr
