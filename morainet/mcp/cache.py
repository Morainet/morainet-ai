"""MCP resource cache: cache prompts and tool schemas from MCP servers.

Reduces repeated MCP discovery calls (list_tools / list_prompts / list_resources)
by caching results with TTL. Supports in-memory and disk-based caches.

Usage::

    cache = MCPResourceCache(ttl=300, max_size=1000)
    client = MCPClient(session)

    # Cache tools — only fetches from server on cache miss
    tools = await cache.get_tools(client)

    # Cache prompts
    prompts = await cache.get_prompts(client)

    # Cache resources
    resources = await cache.get_resources(client)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from morainet.core.models import Message
from morainet.tools import Tool


class MCPResourceCache:
    """Caches MCP tool lists, prompts, and resources with TTL.

    Attributes:
        ttl: Time-to-live in seconds (0 = no expiry).
        max_size: Maximum entries per category.
        persist_path: Optional disk path for persistent caching.
    """

    def __init__(self, ttl: float = 300.0, max_size: int = 1000,
                 persist_path: str = "") -> None:
        self.ttl = ttl
        self.max_size = max_size
        self.persist_path = persist_path

        self._tools_cache: dict[str, tuple[float, list[Tool]]] = {}
        self._prompts_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._resources_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._resource_content: dict[str, tuple[float, str]] = {}

        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            self._load_disk()

    # -- tools ---------------------------------------------------------------

    async def get_tools(self, client: Any, key: str = "default") -> list[Tool]:
        """Get cached tools for a client, refreshing on miss or expiry."""
        entry = self._tools_cache.get(key)
        if entry is not None:
            ts, tools = entry
            if self.ttl == 0 or (time.time() - ts) < self.ttl:
                return tools

        tools = await client.list_tools()
        self._tools_cache[key] = (time.time(), tools)
        self._enforce_max_size(self._tools_cache)
        self._save_disk()
        return tools

    def invalidate_tools(self, key: str = "default") -> None:
        """Force a refresh on next access."""
        self._tools_cache.pop(key, None)

    # -- prompts -------------------------------------------------------------

    async def get_prompts(self, client: Any, key: str = "default") -> list[dict[str, Any]]:
        """Get cached prompts for a client."""
        entry = self._prompts_cache.get(key)
        if entry is not None:
            ts, prompts = entry
            if self.ttl == 0 or (time.time() - ts) < self.ttl:
                return prompts

        prompts = await client.list_prompts()
        self._prompts_cache[key] = (time.time(), prompts)
        self._enforce_max_size(self._prompts_cache)
        self._save_disk()
        return prompts

    def invalidate_prompts(self, key: str = "default") -> None:
        self._prompts_cache.pop(key, None)

    # -- resources -----------------------------------------------------------

    async def get_resources(self, client: Any, key: str = "default") -> list[dict[str, Any]]:
        """Get cached resources for a client."""
        entry = self._resources_cache.get(key)
        if entry is not None:
            ts, resources = entry
            if self.ttl == 0 or (time.time() - ts) < self.ttl:
                return resources

        resources = await client.list_resources()
        self._resources_cache[key] = (time.time(), resources)
        self._enforce_max_size(self._resources_cache)
        self._save_disk()
        return resources

    async def get_resource_content(self, client: Any, uri: str) -> str:
        """Get cached resource content by URI."""
        entry = self._resource_content.get(uri)
        if entry is not None:
            ts, content = entry
            if self.ttl == 0 or (time.time() - ts) < self.ttl:
                return content

        content = await client.read_resource(uri)
        self._resource_content[uri] = (time.time(), content)
        self._enforce_max_size(self._resource_content)
        return content

    def invalidate_resources(self, key: str = "default") -> None:
        self._resources_cache.pop(key, None)

    # -- cache-wide operations -----------------------------------------------

    def invalidate_all(self) -> None:
        """Clear all caches."""
        self._tools_cache.clear()
        self._prompts_cache.clear()
        self._resources_cache.clear()
        self._resource_content.clear()

    def stats(self) -> dict[str, int]:
        """Return cache hit/entry counts."""
        return {
            "tools": len(self._tools_cache),
            "prompts": len(self._prompts_cache),
            "resources": len(self._resources_cache),
            "resource_content": len(self._resource_content),
        }

    # -- persistence ---------------------------------------------------------

    def _enforce_max_size(self, cache: dict[Any, Any]) -> None:
        while len(cache) > self.max_size:
            cache.pop(next(iter(cache)))

    def _save_disk(self) -> None:
        if not self.persist_path:
            return
        data: dict[str, Any] = {"ttl": self.ttl, "ts": time.time()}
        # Only save prompt/resource shapes (tools are complex objects)
        prompts = {k: v for k, (ts, v) in self._prompts_cache.items()}
        resources = {k: v for k, (ts, v) in self._resources_cache.items()}
        data["prompts"] = prompts
        data["resources"] = resources
        file_path = os.path.join(self.persist_path, "mcp_cache.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)

    def _load_disk(self) -> None:
        if not self.persist_path:
            return
        file_path = os.path.join(self.persist_path, "mcp_cache.json")
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("ts", 0)
            self._prompts_cache = {k: (ts, v) for k, v in data.get("prompts", {}).items()}
            self._resources_cache = {k: (ts, v) for k, v in data.get("resources", {}).items()}
        except (json.JSONDecodeError, OSError, KeyError):
            pass
