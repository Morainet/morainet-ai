"""Tool call caching: avoids repeat API calls for duplicate queries.

Caches tool results keyed by (tool_name, arguments_hash). Supports TTL-based
expiration, max-size eviction, and optional persistent storage via JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from morainet.observability.tracing import logger


@dataclass
class CacheEntry:
    key: str
    result: Any
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class ToolCache:
    """Caches tool call results to avoid redundant external API / heavy computation.

    Keys are deterministic: ``sha256(tool_name + sorted_json(args))``.
    Supports TTL, max-size LRU eviction, and optional file persistence.

    Usage::

        cache = ToolCache(ttl=300, max_size=1000)
        agent = Agent(provider=..., tools=[...], tool_cache=cache)
    """

    def __init__(
        self,
        ttl: float | None = 300.0,
        max_size: int = 1000,
        persist_path: str | None = None,
        disabled: bool = False,
    ) -> None:
        self.ttl = ttl  # None = never expire
        self.max_size = max_size
        self.persist_path = persist_path
        self.disabled = disabled
        self._store: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

        if persist_path and os.path.exists(persist_path):
            self._load()

    # -- public API ------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "disabled": self.disabled,
        }

    def make_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Produce a stable cache key for a tool call."""
        payload = json.dumps([tool_name, arguments], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, tool_name: str, arguments: dict[str, Any]) -> tuple[Any, str | None] | None:
        """Return ``(result, error)`` if cached, or ``None`` on miss."""
        if self.disabled:
            self.misses += 1
            return None

        key = self.make_key(tool_name, arguments)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        if self.ttl is not None and time.monotonic() - entry.created_at > self.ttl:
            del self._store[key]
            self.misses += 1
            return None

        entry.hit_count += 1
        self.hits += 1
        logger.debug(f"Tool cache HIT: {tool_name}#{key}")
        return entry.result, entry.error

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        error: str | None = None,
    ) -> None:
        """Cache a tool call result (including errors, to avoid retry loops)."""
        if self.disabled:
            return

        key = self.make_key(tool_name, arguments)
        entry = CacheEntry(key=key, result=result, error=error)

        # Evict oldest entry if at capacity (simple FIFO eviction)
        if len(self._store) >= self.max_size and key not in self._store:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

        self._store[key] = entry
        logger.debug(f"Tool cache SET: {tool_name}#{key} (size={len(self._store)})")

    def invalidate(self, tool_name: str | None = None) -> int:
        """Remove entries. If ``tool_name`` is None, clear the entire cache.

        Returns the number of entries removed.
        """
        if tool_name is None:
            count = len(self._store)
            self._store.clear()
            if self.persist_path and os.path.exists(self.persist_path):
                try:
                    os.remove(self.persist_path)
                except OSError:
                    pass
            logger.debug(f"Tool cache: cleared {count} entries")
            return count

        to_remove = []
        prefix = hashlib.sha256(
            json.dumps([tool_name, {}], sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        for key in list(self._store):
            if key.startswith(prefix):
                to_remove.append(key)
        for key in to_remove:
            del self._store[key]
        logger.debug(f"Tool cache: invalidated {len(to_remove)} entries for tool '{tool_name}'")
        return len(to_remove)

    def save(self, path: str | None = None) -> None:
        """Persist cache to a JSON file."""
        target = path or self.persist_path
        if not target:
            return
        data = {
            k: {"key": v.key, "result": v.result, "error": v.error, "created_at": v.created_at}
            for k, v in self._store.items()
        }
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, ensure_ascii=False, indent=2)
        logger.debug(f"Tool cache: saved {len(data)} entries to {target}")

    def _load(self) -> None:
        """Restore cache from persist_path."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                raw: dict[str, dict[str, Any]] = json.load(f)
            for key, entry_data in raw.items():
                self._store[key] = CacheEntry(
                    key=entry_data["key"],
                    result=entry_data["result"],
                    error=entry_data.get("error"),
                    created_at=entry_data.get("created_at", time.monotonic()),
                )
            logger.debug(f"Tool cache: loaded {len(self._store)} entries from {self.persist_path}")
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning(f"Tool cache: failed to load {self.persist_path}: {exc}")
