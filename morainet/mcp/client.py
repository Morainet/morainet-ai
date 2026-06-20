"""MCP client: expose an MCP server's tools as Morainet Tools.

The client works against any *session* implementing :class:`MCPSession`
(``list_tools`` + ``call_tool``). This keeps the core testable. A real stdio
connection is provided by :func:`stdio_session`, which uses the optional
``mcp`` SDK and needs live testing against a server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from morainet.core.models import Message
from morainet.exceptions import MorainetError
from morainet.mcp.convert import mcp_tool_to_tool
from morainet.tools import Tool


@runtime_checkable
class MCPSession(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...
    # Optional capabilities (resources / prompts) — checked via hasattr.


class MCPClient:
    def __init__(self, session: MCPSession) -> None:
        self.session = session

    async def list_tools(self) -> list[Tool]:
        descriptors = await self.session.list_tools()
        return [mcp_tool_to_tool(d, self.session.call_tool) for d in descriptors]

    # --- resources (mapped into context / Long Memory) --------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        fn = getattr(self.session, "list_resources", None)
        return await fn() if fn else []

    async def read_resource(self, uri: str) -> str:
        fn = getattr(self.session, "read_resource", None)
        return await fn(uri) if fn else ""

    async def resource_messages(self) -> list[Message]:
        """Read all resources as system messages, ready to inject as context."""
        messages: list[Message] = []
        for res in await self.list_resources():
            uri = res.get("uri", "")
            text = await self.read_resource(uri)
            if text:
                messages.append(Message.system(f"[resource:{res.get('name', uri)}] {text}"))
        return messages

    # --- prompts (rendered server-side) -----------------------------------

    async def list_prompts(self) -> list[dict[str, Any]]:
        fn = getattr(self.session, "list_prompts", None)
        return await fn() if fn else []

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        fn = getattr(self.session, "get_prompt", None)
        return await fn(name, arguments or {}) if fn else ""


class _SDKSession:
    """Adapts a live ``mcp.ClientSession`` to the :class:`MCPSession` protocol."""

    def __init__(self, session: Any) -> None:
        self._s = session

    async def list_tools(self) -> list[dict[str, Any]]:
        resp = await self._s.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema}
            for t in resp.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self._s.call_tool(name, arguments)
        parts = [getattr(c, "text", "") for c in resp.content if getattr(c, "type", "") == "text"]
        return "\n".join(p for p in parts if p)

    async def list_resources(self) -> list[dict[str, Any]]:
        resp = await self._s.list_resources()
        return [
            {"uri": str(r.uri), "name": r.name or "", "description": r.description or ""}
            for r in resp.resources
        ]

    async def read_resource(self, uri: str) -> str:
        resp = await self._s.read_resource(uri)
        parts = [getattr(c, "text", "") for c in resp.contents if getattr(c, "text", None)]
        return "\n".join(parts)

    async def list_prompts(self) -> list[dict[str, Any]]:
        resp = await self._s.list_prompts()
        return [{"name": p.name, "description": p.description or ""} for p in resp.prompts]

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self._s.get_prompt(name, arguments)
        parts = [
            getattr(m.content, "text", "")
            for m in resp.messages
            if getattr(m.content, "text", None)
        ]
        return "\n".join(parts)


@asynccontextmanager
async def stdio_session(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[MCPClient]:
    """Connect to an MCP server over stdio. Requires ``pip install morainet-ai[mcp]``.

    Note: not integration-tested in CI — verify against your server.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - optional dep
        raise MorainetError(
            "MCP support requires the 'mcp' package. Install with: pip install morainet-ai[mcp]"
        ) from exc

    params = StdioServerParameters(command=command, args=args or [], env=env)
    async with stdio_client(params) as (read, write):  # pragma: no cover - needs live server
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPClient(_SDKSession(session))
