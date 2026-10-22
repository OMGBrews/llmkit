"""Tool calls run on a plain LiteLLM install, which has no ``mcp`` package.

LiteLLM 1.106.0 through 1.106.2 import the ``mcp`` package (shipped only in
LiteLLM's ``mcp``/``proxy`` extras) on every completion that carries ``tools``, before
anything is sent: ``completion`` enters its proxy-MCP-gateway check unless the
per-call ``_skip_mcp_handler`` is set, and that check's module now imports the
proxy MCP catalog at module level (BerriAI/litellm#46045, fixed on LiteLLM's
``main`` by #45961). The transport funnel (``llmkit._litellm._acompletion``)
sets the switch on every request carrying tools.

These tests call the *real* ``litellm.acompletion`` against a mock response, so
no network is needed, in fresh interpreters that make ``mcp`` unimportable
before LiteLLM loads — the parent session's own imports, or an ``mcp`` that
happens to be installed, cannot mask the simulation. LiteLLM's mock path runs
after the gateway check, so a request that reaches the check fails here exactly
as it does against a live provider.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.metadata import version

import pytest

# Runs first in each child: ``mcp`` and its submodules raise the exact error a
# plain install raises, whether or not the package is present on this machine.
_WITHOUT_MCP = textwrap.dedent(
    """
    import sys

    class _NoMcp:
        def find_spec(self, name, path=None, target=None):
            if name == "mcp" or name.startswith("mcp."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _NoMcp())

    import asyncio

    TOOL_REQUEST = dict(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        mock_response="ok",
        api_key="sk-test",
    )
    """
)


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``_WITHOUT_MCP`` then *code* in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_MCP + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def _litellm_release() -> tuple[int, ...]:
    return tuple(int(part) for part in version("litellm").split(".")[:3])


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
def test_a_tool_call_does_not_import_mcp(stream: bool) -> None:
    """The funnel's tool request completes with ``mcp`` unimportable.

    Both tool lanes (``tool_llm_call`` and ``tool_llm_call_stream``) reach
    LiteLLM only through ``_acompletion``. A LiteLLM that stops reading
    ``_skip_mcp_handler`` would also fail here: the key would then reach
    parameter mapping, or the gateway check would run again.
    """
    result = _run_python(
        f"""
        from litellm import CustomStreamWrapper, ModelResponse
        from llmkit._litellm import _acompletion

        async def main():
            resp = await _acompletion(**TOOL_REQUEST, stream={stream})
            if {stream}:
                assert isinstance(resp, CustomStreamWrapper), type(resp)
                text = "".join([c.choices[0].delta.content or "" async for c in resp])
            else:
                assert isinstance(resp, ModelResponse), type(resp)
                text = resp.choices[0].message.content
            assert text == "ok", text

        asyncio.run(main())
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    _litellm_release() < (1, 106, 0),
    reason="only LiteLLM 1.106.0 and later import mcp on the tool path",
)
def test_control_raw_litellm_tool_call_needs_mcp() -> None:
    """Without the switch, the same request fails for want of ``mcp``.

    This proves the simulation above is live, so the primary test's pass is
    evidence rather than decoration. It is expected to start failing once the
    installed LiteLLM contains upstream commit ``783cf06`` (BerriAI/litellm#45961).
    Then record the first fixed release in the comment on the ``litellm``
    requirement in ``pyproject.toml`` and retire this control; keep the
    primary test.
    """
    result = _run_python(
        """
        import litellm

        async def main():
            try:
                await litellm.acompletion(**TOOL_REQUEST)
            except Exception as exc:
                print(exc)
            else:
                raise SystemExit("the raw tool call succeeded without mcp")

        asyncio.run(main())
        """
    )
    assert result.returncode == 0, result.stderr
    assert "No module named 'mcp'" in result.stdout, result.stdout
