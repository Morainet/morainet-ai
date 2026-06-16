"""Convert MCP tool descriptors into Morainet Tools."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from morainet.tools import Tool

# A caller runs a remote MCP tool: (name, arguments) -> text result.
ToolCaller = Callable[[str, dict[str, Any]], Awaitable[str]]


def mcp_tool_to_tool(descriptor: dict[str, Any], caller: ToolCaller) -> Tool:
    """Map one MCP tool descriptor to a Tool whose invoke calls the server.

    ``descriptor`` follows the MCP shape: ``name``, ``description``,
    ``inputSchema`` (a JSON Schema object).
    """
    name = descriptor["name"]

    async def invoke(**kwargs: Any) -> str:
        return await caller(name, kwargs)

    return Tool.from_schema(
        name=name,
        description=descriptor.get("description", ""),
        parameters=descriptor.get("inputSchema") or {"type": "object", "properties": {}},
        invoke=invoke,
    )
