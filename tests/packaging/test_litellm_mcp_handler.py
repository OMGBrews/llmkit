"""Tool calls run on a plain LiteLLM install, which has no ``mcp`` package.

LiteLLM 1.106.0 through 1.108.3 import the ``mcp`` package (shipped only in
LiteLLM's ``mcp``/``proxy`` extras) on every completion that carries ``tools``, before
anything is sent: ``completion`` enters its proxy-MCP-gateway check unless the
per-call ``_skip_mcp_handler`` is set, and that check's module now imports the
proxy MCP catalog at module level (BerriAI/litellm#46045, fixed by #45961 and
first released in 1.108.4). The transport funnel (``llmkit._litellm._acompletion``)
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
