"""Offline tests for the MCP connection pool.

The stdio transport is replaced by a fake session, so no ``mcp`` SDK or live
server is needed. Covers server registration, connect/disconnect, tool
aggregation, routing, health checks, and the pooled session adapter.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from morainet.exceptions import MorainetError
from morainet.mcp.client import MCPClient
from morainet.mcp.pool import MCPConnectionPool, _PooledSession


class FakeSession:
    def __init__(self, tools=None, call_result: str = "ok") -> None:
        self._tools = tools if tools is not None else [
            {"name": "t1", "description": "d", "inputSchema": {}}
        ]
        self._call_result = call_result
        self.calls: list[tuple] = []

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._call_result

    async def list_resources(self):
        return [{"uri": "u", "name": "r", "description": "d"}]

    async def read_resource(self, uri):
        return f"resource:{uri}"

    async def list_prompts(self):
        return [{"name": "p", "description": "d"}]

    async def get_prompt(self, name, arguments):
        return f"prompt:{name}"


def _patch_connect(monkeypatch, pool, client) -> None:
    async def fake(config):  # noqa: ANN001
        return client

    monkeypatch.setattr(pool, "_connect_one", fake)


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------


def test_add_and_remove_server():
    pool = MCPConnectionPool()
    assert pool.add_server("a", "cmd") is pool
    pool.add_server("b", "cmd")
    assert pool.server_names == ["a", "b"]
    pool.remove_server("a")
    assert pool.server_names == ["b"]
    pool.remove_server("missing")  # no error


def test_stats_empty():
    pool = MCPConnectionPool()
    stats = pool.stats()
    assert stats["total_servers"] == 0
    assert stats["connected"] == 0
    assert stats["tools"] == 0


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


async def test_connect_server_success(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    assert await pool.connect_server("a") is True
    assert pool.connected_servers == ["a"]
    assert pool.tool_count == 1
    assert pool.stats()["connected"] == 1


async def test_connect_server_unknown_raises():
    pool = MCPConnectionPool()
    with pytest.raises(MorainetError):
        await pool.connect_server("nope")


async def test_connect_server_failure(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")

    async def boom(config):  # noqa: ANN001
        raise RuntimeError("no server")

    monkeypatch.setattr(pool, "_connect_one", boom)
    assert await pool.connect_server("a") is False
    assert pool.connected_servers == []
    assert pool.stats()["servers"]["a"]["error"] == "no server"


async def test_connect_all(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    pool.add_server("b", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    results = await pool.connect_all()
    assert results == {"a": True, "b": True}


async def test_disconnect_server_and_all(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    await pool.connect_server("a")
    await pool.disconnect_server("a")
    assert pool.connected_servers == []
    assert pool.tool_count == 0

    await pool.connect_server("a")
    await pool.disconnect_all()
    assert pool.connected_servers == []


# ---------------------------------------------------------------------------
# Tool aggregation / routing
# ---------------------------------------------------------------------------


async def test_list_all_tools_skips_unconnected(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    await pool.connect_server("a")
    pool.add_server("b", "cmd")  # never connected
    tools = await pool.list_all_tools()
    assert len(tools) == 1
    assert tools[0].name == "t1"


async def test_call_tool_routes_to_server(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    session = FakeSession()
    _patch_connect(monkeypatch, pool, MCPClient(session))
    await pool.connect_server("a")
    out = await pool.call_tool("t1", {"x": 1})
    assert out == "ok"
    assert session.calls == [("t1", {"x": 1})]


async def test_call_tool_unknown_raises():
    pool = MCPConnectionPool()
    with pytest.raises(MorainetError):
        await pool.call_tool("nope", {})


async def test_call_tool_disconnected_server_raises(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    await pool.connect_server("a")
    pool._servers["a"].connected = False  # tool_map still has t1
    with pytest.raises(MorainetError):
        await pool.call_tool("t1", {})


# ---------------------------------------------------------------------------
# Resource / prompt aggregation
# ---------------------------------------------------------------------------


async def test_aggregate_resources_and_prompts(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    await pool.connect_server("a")

    resources = await pool.list_all_resources()
    assert resources[0]["_server"] == "a"

    prompts = await pool.list_all_prompts()
    assert prompts[0]["_server"] == "a"

    assert await pool.get_prompt("p") == "prompt:p"


async def test_get_prompt_without_servers():
    pool = MCPConnectionPool()
    assert await pool.get_prompt("p") == ""


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


async def test_health_check_success(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    _patch_connect(monkeypatch, pool, MCPClient(FakeSession()))
    await pool.connect_server("a")
    assert await pool.health_check() == {"a": True}


async def test_health_check_no_client():
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    assert await pool.health_check() == {"a": False}


async def test_health_check_failure(monkeypatch):
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")

    class BadSession:
        async def list_tools(self):
            raise RuntimeError("dead")

        async def call_tool(self, name, arguments):
            return ""

    _patch_connect(monkeypatch, pool, MCPClient(BadSession()))
    await pool.connect_server("a")  # connection fails but client is retained
    assert await pool.health_check() == {"a": False}


async def test_health_loop_start_stop():
    pool = MCPConnectionPool()
    pool.add_server("a", "cmd")
    await pool.start_health_loop(interval=10)
    assert pool._health_task is not None
    await pool.stop_health_loop()
    assert pool._health_task is None


# ---------------------------------------------------------------------------
# _PooledSession adapter
# ---------------------------------------------------------------------------


class _FakeSDK:
    async def list_tools(self):
        return SimpleNamespace(
            tools=[SimpleNamespace(name="t", description="d", inputSchema={})]
        )

    async def call_tool(self, name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])

    async def list_resources(self):
        return SimpleNamespace(resources=[SimpleNamespace(uri="u", name="r", description="d")])

    async def read_resource(self, uri):
        return SimpleNamespace(contents=[SimpleNamespace(text="body")])

    async def list_prompts(self):
        return SimpleNamespace(prompts=[SimpleNamespace(name="p", description="d")])

    async def get_prompt(self, name, arguments):
        return SimpleNamespace(messages=[SimpleNamespace(content=SimpleNamespace(text="pm"))])

    async def __aexit__(self, *exc):
        return False


async def test_pooled_session_adapter():
    session = _PooledSession(_FakeSDK(), None, None)
    tools = await session.list_tools()
    assert tools[0]["name"] == "t"
    assert await session.call_tool("t", {}) == "hello"
    resources = await session.list_resources()
    assert resources[0]["uri"] == "u"
    assert await session.read_resource("u") == "body"
    prompts = await session.list_prompts()
    assert prompts[0]["name"] == "p"
    assert await session.get_prompt("p", {}) == "pm"
    await session.close()
