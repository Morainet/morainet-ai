"""Redis-backed checkpoint store for distributed breakpoint resume.

Requires ``redis`` (``pip install redis``).

Usage::

    store = RedisCheckpointStore(redis_url="redis://localhost:6379/0")
    agent = Agent(provider=..., checkpoint_store=store)
    # After a crash, resume on any node:
    checkpoint = await store.load(trace_id)
    agent.resume(checkpoint)
"""

from __future__ import annotations

from typing import Any

from morainet.persistence.checkpoint import Checkpoint, CheckpointStore


class RedisCheckpointStore(CheckpointStore):
    """Redis-backed checkpoint persistence with optional TTL.

    ``redis_url``    — Redis connection URL (e.g. ``redis://localhost:6379/0``).
    ``ttl_seconds``  — auto-expire checkpoints after this duration (0 = no expiry).
    ``key_prefix``   — prefix for Redis keys (default: ``morainet:checkpoint:``).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 0,
        key_prefix: str = "morainet:checkpoint:",
    ) -> None:
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-init Redis client (avoids import errors when redis not installed)."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis  # type: ignore[import-untyped,unused-ignore]
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisCheckpointStore. "
                    "Install with: pip install morainet-ai[redis]"
                ) from None
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, trace_id: str) -> str:
        return f"{self._key_prefix}{trace_id}"

    async def save(self, checkpoint: Checkpoint) -> None:
        key = self._key(checkpoint.trace_id)
        data = checkpoint.model_dump_json()
        if self._ttl_seconds > 0:
            await self.client.setex(key, self._ttl_seconds, data)
        else:
            await self.client.set(key, data)

    async def load(self, trace_id: str) -> Checkpoint | None:
        key = self._key(trace_id)
        data = await self.client.get(key)
        if data is None:
            return None
        return Checkpoint.model_validate_json(data)

    async def delete(self, trace_id: str) -> bool:
        """Delete a checkpoint. Returns ``True`` if it existed."""
        key = self._key(trace_id)
        deleted = await self.client.delete(key)
        return bool(deleted)

    async def list_trace_ids(self, pattern: str = "*") -> list[str]:
        """List all trace_ids matching the glob pattern."""
        keys: list[str] = await self.client.keys(f"{self._key_prefix}{pattern}")
        prefix_len = len(self._key_prefix)
        return [k[prefix_len:] for k in keys]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
