"""Live MCP integration test — connects to a real MCP server over stdio.

Set MORAINET_MCP_COMMAND to the server launch command, e.g.:

    MORAINET_MCP_COMMAND="uvx mcp-server-time" pytest -m live -k mcp

Skipped when the variable is unset.
"""

from __future__ import annotations

import os
import shlex

import pytest

pytestmark = pytest.mark.live

_CMD = os.getenv("MORAINET_MCP_COMMAND")


@pytest.mark.skipif(not _CMD, reason="set MORAINET_MCP_COMMAND to run")
async def test_mcp_stdio_lists_tools():
    from morainet.mcp import stdio_session

    parts = shlex.split(_CMD or "")
    async with stdio_session(parts[0], parts[1:]) as client:
        tools = await client.list_tools()
        assert isinstance(tools, list)
        # A server should expose at least one tool with a name + schema.
        for t in tools:
            assert t.name
            assert "parameters" in t.schema
