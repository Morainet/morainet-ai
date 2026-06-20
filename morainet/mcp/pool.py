"""MCP connection pool: batch-connect to multiple MCP servers.

Manages multiple MCP sessions, supports health-checking, reconnection,
tool aggregation across servers, and load-balanced tool execution.

Usage::

    pool = MCPConnectionPool()
    await pool.add_server("weather", command="python", args=["-m", "weather_mcp"])
    await pool.add_server("search", command="node", args=["search_mcp.js"])
    await pool.connect_all()

    # Aggregate tools from all servers
    tools = await pool.list_all_tools()

    # Execute any tool by name across the pool
    result = await pool.call_tool("get_weather", {"city": "Beijing"})
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from morainet.exceptions import MorainetError
from morainet.mcp.client import MCPClient
from morainet.tools import Tool


@dataclass
class ServerConfig:
    """Configuration for one MCP server connection."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 3
    health_check_interval: float = 30.0


@dataclass
class ServerState:
    """Runtime state of a connected MCP server."""

    name: str
    config: ServerConfig
    client: MCPClient | None = None
    connected: bool = False
    last_health_check: float = 0.0
    reconnect_count: int = 0
    tool_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


class MCPConnectionPool:
    """Batch-connect to multiple MCP servers and expose all tools uniformly.

    Features:
    - Connect/disconnect servers in batch
    - Health-check running connections
    - Auto-reconnect on failure
    - Aggregate tools from every server
    - Route tool calls to the right server by tool name
    - Resource and prompt aggregation
    """

    def __init__(self) -> None:
        self._servers: dict[str, ServerState] = {}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_name
        self._health_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

    # -- server management ---------------------------------------------------

    def add_server(self, name: str, command: str, args: list[str] | None = None,
                   env: dict[str, str] | None = None, auto_reconnect: bool = True,
                   reconnect_delay: float = 5.0, max_reconnect_attempts: int = 3,
                   health_check_interval: float = 30.0) -> "MCPConnectionPool":
        """Register a server configuration (does not connect yet)."""
        config = ServerConfig(
            name=name, command=command, args=args or [], env=env,
            auto_reconnect=auto_reconnect, reconnect_delay=reconnect_delay,
            max_reconnect_attempts=max_reconnect_attempts,
            health_check_interval=health_check_interval,
        )
        self._servers[name] = ServerState(name=name, config=config)
        return self

    def remove_server(self, name: str) -> None:
        """Remove a server from the pool (does not disconnect)."""
        self._servers.pop(name, None)
        self._tool_map = {tn: sn for tn, sn in self._tool_map.items() if sn != name}

    # -- connection lifecycle ------------------------------------------------

    async def connect_server(self, name: str) -> bool:
        """Connect a single server by name. Returns True on success."""
        state = self._servers.get(name)
        if state is None:
            raise MorainetError(f"Server '{name}' not registered")

        try:
            state.client = await self._connect_one(state.config)
            state.connected = True
            state.error = None

            # Refresh tool map
            tools = await state.client.list_tools()
            state.tool_count = len(tools)
            state.tool_names = [t.name for t in tools]
            for t in tools:
                self._tool_map[t.name] = name
            return True
        except Exception as exc:
            state.error = str(exc)
            state.connected = False
            return False

    async def connect_all(self) -> dict[str, bool]:
        """Connect all registered servers in parallel. Returns per-server results."""
        results = await asyncio.gather(
            *(self.connect_server(name) for name in self._servers),
            return_exceptions=True,
        )
        return {
            name: (False if isinstance(r, BaseException) else r)
            for name, r in zip(self._servers, results, strict=True)
        }

    async def disconnect_server(self, name: str) -> None:
        """Disconnect a single server."""
        state = self._servers.get(name)
        if state is not None:
            # Stdio sessions are context-managed; we only clear the reference
            state.client = None
            state.connected = False
        self._tool_map = {tn: sn for tn, sn in self._tool_map.items() if sn != name}

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in self._servers:
            self._servers[name].client = None
            self._servers[name].connected = False
        self._tool_map.clear()

    async def _connect_one(self, config: ServerConfig) -> MCPClient:
        """Establish a single stdio MCP connection (non-context-managed for pool)."""
        # We create the session context but keep the client reference active
        # by not exiting the async context manager
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover
            raise MorainetError("MCP support requires 'mcp'. pip install morainet-ai[mcp]") from exc

        params = StdioServerParameters(command=config.command, args=config.args, env=config.env)
        # For pooled connections we use a managed context
        read, write = await stdio_client(params).__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        # Wrap in _PooledSession adapter
        return MCPClient(_PooledSession(session, read, write))

    # -- tool aggregation ----------------------------------------------------

    async def list_all_tools(self) -> list[Tool]:
        """Return all tools from all connected servers."""
        tools: list[Tool] = []
        for name, state in self._servers.items():
            if state.connected and state.client is not None:
                try:
                    server_tools = await state.client.list_tools()
                    tools.extend(server_tools)
                except Exception:
                    pass
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the server that owns the tool."""
        server_name = self._tool_map.get(tool_name)
        if server_name is None:
            raise MorainetError(f"Tool '{tool_name}' not found in any connected server")
        state = self._servers[server_name]
        if not state.connected or state.client is None:
            raise MorainetError(f"Server '{server_name}' for tool '{tool_name}' is not connected")
        return await state.client.session.call_tool(tool_name, arguments)

    # -- resource / prompt aggregation ---------------------------------------

    async def list_all_resources(self) -> list[dict[str, Any]]:
        """Aggregate resources from all connected servers."""
        results: list[dict[str, Any]] = []
        for state in self._servers.values():
            if state.connected and state.client is not None:
                try:
                    resources = await state.client.list_resources()
                    for r in resources:
                        r["_server"] = state.name
                    results.extend(resources)
                except Exception:
                    pass
        return results

    async def list_all_prompts(self) -> list[dict[str, Any]]:
        """Aggregate prompts from all connected servers."""
        results: list[dict[str, Any]] = []
        for state in self._servers.values():
            if state.connected and state.client is not None:
                try:
                    prompts = await state.client.list_prompts()
                    for p in prompts:
                        p["_server"] = state.name
                    results.extend(prompts)
                except Exception:
                    pass
        return results

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Get a rendered prompt by name across all servers."""
        for state in self._servers.values():
            if state.connected and state.client is not None:
                try:
                    result = await state.client.get_prompt(name, arguments)
                    if result:
                        return result
                except Exception:
                    pass
        return ""

    # -- health checking -----------------------------------------------------

    async def health_check(self) -> dict[str, bool]:
        """Run a health check on all servers. Returns server_name -> healthy."""
        results: dict[str, bool] = {}
        for name, state in self._servers.items():
            if state.client is not None:
                try:
                    tools = await asyncio.wait_for(
                        state.client.list_tools(), timeout=5.0
                    )
                    state.connected = True
                    state.tool_names = [t.name for t in tools]
                    state.error = None
                    results[name] = True
                except Exception as exc:
                    state.connected = False
                    state.error = str(exc)
                    results[name] = False
            else:
                results[name] = False
            state.last_health_check = time.time()
        return results

    async def start_health_loop(self, interval: float = 30.0) -> None:
        """Start a background health-check loop."""
        self._health_task = asyncio.create_task(self._health_loop(interval))

    async def stop_health_loop(self) -> None:
        """Stop the background health-check loop."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _health_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.health_check()
            # Auto-reconnect failed servers
            for name, state in self._servers.items():
                if not state.connected and state.config.auto_reconnect:
                    if state.reconnect_count < state.config.max_reconnect_attempts:
                        state.reconnect_count += 1
                        await asyncio.sleep(state.config.reconnect_delay)
                        await self.connect_server(name)

    # -- properties ----------------------------------------------------------

    @property
    def server_names(self) -> list[str]:
        return list(self._servers)

    @property
    def connected_servers(self) -> list[str]:
        return [n for n, s in self._servers.items() if s.connected]

    @property
    def tool_count(self) -> int:
        return len(self._tool_map)

    def stats(self) -> dict[str, Any]:
        return {
            "total_servers": len(self._servers),
            "connected": len(self.connected_servers),
            "tools": self.tool_count,
            "servers": {
                name: {
                    "connected": s.connected,
                    "tool_count": s.tool_count,
                    "error": s.error,
                    "reconnects": s.reconnect_count,
                }
                for name, s in self._servers.items()
            },
        }


class _PooledSession:
    """Session wrapper that tracks stdio transport for proper cleanup."""

    def __init__(self, session: Any, read: Any, write: Any) -> None:
        self._session = session
        self._read = read
        self._write = write

    async def list_tools(self) -> list[dict[str, Any]]:
        resp = await self._session.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema}
            for t in resp.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self._session.call_tool(name, arguments)
        parts = [getattr(c, "text", "") for c in resp.content if getattr(c, "type", "") == "text"]
        return "\n".join(p for p in parts if p)

    async def list_resources(self) -> list[dict[str, Any]]:
        resp = await self._session.list_resources()
        return [
            {"uri": str(r.uri), "name": r.name or "", "description": r.description or ""}
            for r in resp.resources
        ]

    async def read_resource(self, uri: str) -> str:
        resp = await self._session.read_resource(uri)
        parts = [getattr(c, "text", "") for c in resp.contents if getattr(c, "text", None)]
        return "\n".join(parts)

    async def list_prompts(self) -> list[dict[str, Any]]:
        resp = await self._session.list_prompts()
        return [{"name": p.name, "description": p.description or ""} for p in resp.prompts]

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self._session.get_prompt(name, arguments)
        parts = [
            getattr(m.content, "text", "")
            for m in resp.messages
            if getattr(m.content, "text", None)
        ]
        return "\n".join(parts)

    async def close(self) -> None:
        """Close the underlying session and transport."""
        try:
            await self._session.__aexit__(None, None, None)
        except Exception:
            pass
