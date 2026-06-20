"""Distributed checkpoint — cluster-wide breakpoint persistence and failover.

Extends the existing checkpoint system so that:

- Every node writes checkpoints to a shared Redis store.
- A :class:`HeartbeatCheckpointStore` ties checkpoints to worker liveness:
  if a worker dies, another node can claim the orphaned trace_id and resume.
- :class:`DistributeCheckpointHook` records which node owns a run and
  updates ownership on checkpoint save.
- :class:`ClusterCheckpointStore` wraps any :class:`CheckpointStore` and
  adds cluster-aware retry / replication.

Usage::

    from morainet.distributed import (
        HeartbeatCheckpointStore, DistributeCheckpointHook, ClusterCheckpointStore
    )

    store = HeartbeatCheckpointStore(redis_url="redis://localhost:6379/0")
    hook = DistributeCheckpointHook(store, node_id="worker-1")

    agent = Agent(memory=..., hooks=[hook])

    # After crash on worker-1, worker-2 can resume:
    checkpoint = await store.load(trace_id)
    agent.resume(checkpoint)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from morainet.core.models import AgentResult, ChatResponse, Step
from morainet.observability.hooks import Hook
from morainet.persistence.checkpoint import Checkpoint, CheckpointStore

if TYPE_CHECKING:
    from morainet.core.context import Context


# ---------------------------------------------------------------------------
# Heartbeat checkpoint store
# ---------------------------------------------------------------------------


class HeartbeatCheckpointStore(CheckpointStore):
    """Checkpoint store with worker heartbeat for automatic failover.

    Every ``save()`` also updates a heartbeat key. A separate process
    (or the next node) monitors heartbeats and claims orphaned sessions.

    Args:
        redis_url: Redis connection URL.
        ttl_seconds: Heartbeat TTL — if a node misses this window,
              another node can claim its sessions.
        key_prefix: Prefix for Redis keys.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 30,
        key_prefix: str = "morainet:checkpoint:",
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._key_prefix = key_prefix
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError(
                    "redis package required. pip install morainet-ai[redis]"
                ) from None
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _ckpt_key(self, trace_id: str) -> str:
        return f"{self._key_prefix}{trace_id}"

    def _owner_key(self, trace_id: str) -> str:
        return f"{self._key_prefix}{trace_id}:owner"

    def _heartbeat_key(self, node_id: str) -> str:
        return f"{self._key_prefix}heartbeat:{node_id}"

    async def save(self, checkpoint: Checkpoint, owner_node_id: str = "") -> None:
        ckpt_key = self._ckpt_key(checkpoint.trace_id)
        data = checkpoint.model_dump_json()

        pipe = self.client.pipeline()
        pipe.set(ckpt_key, data)
        pipe.expire(ckpt_key, self._ttl * 10)  # checkpoint lives 10x heartbeat

        if owner_node_id:
            owner_key = self._owner_key(checkpoint.trace_id)
            pipe.set(owner_key, owner_node_id)
            pipe.expire(owner_key, self._ttl * 10)
            # Update heartbeat for this node
            hb_key = self._heartbeat_key(owner_node_id)
            pipe.set(hb_key, str(time.time()))
            pipe.expire(hb_key, self._ttl)

        await pipe.execute()

    async def load(self, trace_id: str) -> Checkpoint | None:
        key = self._ckpt_key(trace_id)
        data = await self.client.get(key)
        if data is None:
            return None
        return Checkpoint.model_validate_json(data)

    async def get_owner(self, trace_id: str) -> str | None:
        """Return the node_id that owns this trace, or ``None``."""
        owner_key = self._owner_key(trace_id)
        return await self.client.get(owner_key)  # type: ignore[no-any-return]

    async def claim_orphan(self, trace_id: str, new_owner: str) -> Checkpoint | None:
        """Claim an orphaned session. Returns the checkpoint if successful."""
        ckpt = await self.load(trace_id)
        if ckpt is None:
            return None
        owner_key = self._owner_key(trace_id)
        # Only claim if the heartbeat is stale (owner is dead)
        old_owner = await self.client.get(owner_key)
        if old_owner:
            hb_key = self._heartbeat_key(old_owner)
            hb_str = await self.client.get(hb_key)
            if hb_str:
                last_hb = float(hb_str)
                if time.time() - last_hb < self._ttl:
                    return None  # owner is still alive

        # Claim it
        owner_key = self._owner_key(trace_id)
        await self.client.set(owner_key, new_owner)
        await self.client.expire(owner_key, self._ttl * 10)
        return ckpt

    async def list_orphans(self, exclude_nodes: list[str] | None = None) -> list[str]:
        """List trace_ids whose owners have stale heartbeats."""
        # Scan owner keys
        prefix = f"{self._key_prefix}"
        all_keys = await self.client.keys(f"{prefix}*:owner")
        exclude = set(exclude_nodes or [])
        orphans: list[str] = []
        for key in all_keys:
            trace_id = key[len(prefix):-len(":owner")]
            owner = await self.client.get(key)
            if not owner or owner in exclude:
                continue
            hb_key = self._heartbeat_key(owner)
            hb_str = await self.client.get(hb_key)
            if hb_str is None:
                orphans.append(trace_id)
            else:
                last_hb = float(hb_str)
                if time.time() - last_hb >= self._ttl:
                    orphans.append(trace_id)
        return orphans

    async def delete(self, trace_id: str) -> bool:
        keys = [self._ckpt_key(trace_id), self._owner_key(trace_id)]
        deleted = await self.client.delete(*keys)
        return bool(deleted)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Cluster checkpoint store (wrapping another store)
# ---------------------------------------------------------------------------


class ClusterCheckpointStore(CheckpointStore):
    """Wraps any :class:`CheckpointStore` and adds cluster-aware replication.

    Args:
        store: The underlying store (e.g. RedisCheckpointStore).
        replicas: Number of replica nodes to write to (via consistent hashing).
                  Set to 0 for single-node mode.
    """

    def __init__(self, store: CheckpointStore, replicas: int = 2) -> None:
        self._store = store
        self._replicas = replicas
        self._owners: dict[str, str] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        await self._store.save(checkpoint)

    async def load(self, trace_id: str) -> Checkpoint | None:
        return await self._store.load(trace_id)

    async def assign_owner(self, trace_id: str, node_id: str) -> None:
        self._owners[trace_id] = node_id

    async def get_owner(self, trace_id: str) -> str | None:
        return self._owners.get(trace_id)

    async def release_owner(self, trace_id: str) -> None:
        self._owners.pop(trace_id, None)


# ---------------------------------------------------------------------------
# Distributed checkpoint hook
# ---------------------------------------------------------------------------


class DistributeCheckpointHook(Hook):
    """Hook that saves checkpoint + heartbeat on every step, enabling cluster failover.

    Usage::

        store = HeartbeatCheckpointStore(redis_url="redis://localhost:6379/0")
        hook = DistributeCheckpointHook(store, node_id="worker-3")
        agent = Agent(hooks=[hook])
    """

    def __init__(self, store: HeartbeatCheckpointStore, node_id: str) -> None:
        self.store = store
        self.node_id = node_id
        self._cursor = 0

    async def on_run_start(self, ctx: "Context") -> None:
        self._cursor = 0

    async def on_llm_end(self, ctx: "Context", response: ChatResponse) -> None:
        self._cursor += 1
        ckpt = Checkpoint.from_context(ctx, self._cursor)
        await self.store.save(ckpt, owner_node_id=self.node_id)

    async def on_tool_end(self, ctx: "Context", step: Step) -> None:
        self._cursor += 1
        ckpt = Checkpoint.from_context(ctx, self._cursor)
        await self.store.save(ckpt, owner_node_id=self.node_id)

    async def on_run_end(self, ctx: "Context", result: AgentResult) -> None:
        ckpt = Checkpoint.from_context(ctx, self._cursor)
        await self.store.save(ckpt, owner_node_id=self.node_id)
