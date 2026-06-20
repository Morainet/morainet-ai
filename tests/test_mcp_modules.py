"""Tests for morainet.mcp modules — convert and cache."""

from __future__ import annotations

import os
import tempfile


from morainet.mcp.cache import MCPResourceCache
from morainet.mcp.convert import mcp_tool_to_tool
from morainet.tools import Tool


# ---------------------------------------------------------------------------
# mcp_tool_to_tool
# ---------------------------------------------------------------------------

async def test_mcp_tool_to_tool_simple():
    async def caller(name, kwargs):
        return f"called {name} with {kwargs}"

    descriptor = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "first arg"},
                "arg2": {"type": "integer", "description": "second arg"},
            },
            "required": ["arg1"],
        },
    }

    t = mcp_tool_to_tool(descriptor, caller)
    assert isinstance(t, Tool)
    assert t.schema["name"] == "test_tool"
    assert t.schema["description"] == "A test tool"

    result = await t.invoke({"arg1": "hello", "arg2": 42})
    assert "test_tool" in result
    assert "hello" in result
    assert "42" in result


async def test_mcp_tool_to_tool_minimal():
    async def caller(name, kwargs):
        return f"done: {name}"

    descriptor = {
        "name": "minimal",
        "inputSchema": {"type": "object", "properties": {}},
    }

    t = mcp_tool_to_tool(descriptor, caller)
    assert t.schema["name"] == "minimal"
    assert t.schema["description"] == "minimal"  # falls back to name when no description


async def test_mcp_tool_to_tool_no_input_schema():
    async def caller(name, kwargs):
        return "ok"

    descriptor = {"name": "no_schema"}
    t = mcp_tool_to_tool(descriptor, caller)
    assert t.schema["parameters"]["type"] == "object"
    assert t.schema["parameters"]["properties"] == {}


# ---------------------------------------------------------------------------
# MCPResourceCache: construction
# ---------------------------------------------------------------------------

def test_mcp_cache_defaults():
    cache = MCPResourceCache()
    assert cache.ttl == 300.0
    assert cache.max_size == 1000
    assert cache.persist_path == ""


def test_mcp_cache_custom():
    cache = MCPResourceCache(ttl=60.0, max_size=100, persist_path="")
    assert cache.ttl == 60.0
    assert cache.max_size == 100


def test_mcp_cache_stats_empty():
    cache = MCPResourceCache()
    s = cache.stats()
    assert s["tools"] == 0
    assert s["prompts"] == 0
    assert s["resources"] == 0
    assert s["resource_content"] == 0


# ---------------------------------------------------------------------------
# MCPResourceCache: get_tools
# ---------------------------------------------------------------------------

class _FakeMCPClient:
    def __init__(self, tools=None, prompts=None, resources=None, resource_content=""):
        self._tools = tools or []
        self._prompts = prompts or []
        self._resources = resources or []
        self._content = resource_content

    async def list_tools(self):
        return self._tools

    async def list_prompts(self):
        return self._prompts

    async def list_resources(self):
        return self._resources

    async def read_resource(self, uri):
        return self._content


async def test_mcp_cache_get_tools_miss_then_hit():
    cache = MCPResourceCache(ttl=0)  # ttl=0 means no expiry
    client = _FakeMCPClient(tools=["tool_a", "tool_b"])

    result1 = await cache.get_tools(client)
    assert result1 == ["tool_a", "tool_b"]

    # Change client but cache should return cached value
    client._tools = ["changed"]
    result2 = await cache.get_tools(client)
    assert result2 == ["tool_a", "tool_b"]  # still cached


async def test_mcp_cache_get_tools_ttl_expiry():
    cache = MCPResourceCache(ttl=0.001)  # very short TTL
    client = _FakeMCPClient(tools=["v1"])
    result1 = await cache.get_tools(client)
    assert result1 == ["v1"]

    # Wait for TTL to expire
    import time
    time.sleep(0.01)

    client._tools = ["v2"]
    result2 = await cache.get_tools(client)
    assert result2 == ["v2"]


async def test_mcp_cache_invalidate_tools():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(tools=["old"])
    await cache.get_tools(client)
    cache.invalidate_tools()
    client._tools = ["new"]
    result = await cache.get_tools(client)
    assert result == ["new"]


# ---------------------------------------------------------------------------
# MCPResourceCache: get_prompts
# ---------------------------------------------------------------------------

async def test_mcp_cache_get_prompts():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(prompts=[{"name": "greet", "template": "Hello {name}"}])
    result = await cache.get_prompts(client)
    assert len(result) == 1
    assert result[0]["name"] == "greet"


async def test_mcp_cache_invalidate_prompts():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(prompts=[{"name": "old"}])
    await cache.get_prompts(client)
    cache.invalidate_prompts()
    client._prompts = [{"name": "new"}]
    result = await cache.get_prompts(client)
    assert result[0]["name"] == "new"


# ---------------------------------------------------------------------------
# MCPResourceCache: get_resources
# ---------------------------------------------------------------------------

async def test_mcp_cache_get_resources():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(resources=[{"uri": "file:///x.txt", "name": "doc"}])
    result = await cache.get_resources(client)
    assert len(result) == 1
    assert result[0]["name"] == "doc"


async def test_mcp_cache_invalidate_resources():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(resources=[{"name": "old"}])
    await cache.get_resources(client)
    cache.invalidate_resources()
    client._resources = [{"name": "new"}]
    result = await cache.get_resources(client)
    assert result[0]["name"] == "new"


# ---------------------------------------------------------------------------
# MCPResourceCache: get_resource_content
# ---------------------------------------------------------------------------

async def test_mcp_cache_get_resource_content():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(resource_content="file content here")
    result = await cache.get_resource_content(client, "file:///test.txt")
    assert result == "file content here"

    # Second call should be cached
    client._content = "changed"
    result2 = await cache.get_resource_content(client, "file:///test.txt")
    assert result2 == "file content here"  # cached


# ---------------------------------------------------------------------------
# MCPResourceCache: invalidate_all
# ---------------------------------------------------------------------------

async def test_mcp_cache_invalidate_all():
    cache = MCPResourceCache(ttl=0)
    client = _FakeMCPClient(tools=["t"], prompts=[{"name": "p"}], resources=[{"name": "r"}])

    await cache.get_tools(client)
    await cache.get_prompts(client)
    await cache.get_resources(client)
    await cache.get_resource_content(client, "uri:x")

    s = cache.stats()
    assert s["tools"] == 1
    assert s["prompts"] == 1
    assert s["resources"] == 1
    assert s["resource_content"] == 1

    cache.invalidate_all()
    s = cache.stats()
    assert s["tools"] == 0
    assert s["prompts"] == 0
    assert s["resources"] == 0
    assert s["resource_content"] == 0


# ---------------------------------------------------------------------------
# MCPResourceCache: persistence
# ---------------------------------------------------------------------------

def test_mcp_cache_persist_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate cache
        cache = MCPResourceCache(ttl=60, persist_path=tmpdir)
        cache._prompts_cache["default"] = (100.0, [{"name": "test_prompt"}])
        cache._save_disk()

        # Load into new cache
        cache2 = MCPResourceCache(ttl=60, persist_path=tmpdir)
        assert "default" in cache2._prompts_cache
        assert cache2._prompts_cache["default"][1] == [{"name": "test_prompt"}]


def test_mcp_cache_load_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent_subdir")
    cache = MCPResourceCache(persist_path=missing)
    assert cache.stats()["prompts"] == 0


def test_mcp_cache_load_corrupt():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "mcp_cache.json")
        with open(corrupt_path, "w") as f:
            f.write("not valid json {{{")

        cache = MCPResourceCache(persist_path=tmpdir)
        assert cache.stats()["prompts"] == 0  # gracefully handled


# ---------------------------------------------------------------------------
# MCPResourceCache: max_size enforcement
# ---------------------------------------------------------------------------

async def test_mcp_cache_max_size():
    cache = MCPResourceCache(ttl=0, max_size=2)
    client = _FakeMCPClient(tools=["a"])

    await cache.get_tools(client, key="k1")
    await cache.get_tools(client, key="k2")
    await cache.get_tools(client, key="k3")

    assert cache.stats()["tools"] == 2  # oldest evicted
