"""Offline tests for distributed checkpoints (fake redis, no server).

Covers HeartbeatCheckpointStore (save/load/owner/claim_orphan/list_orphans),
ClusterCheckpointStore delegation, and DistributeCheckpointHook event routing.
"""

from __future__ import annotations

import fnmatch
import time
from types import SimpleNamespace

import pytest

from morainet.core.models import ChatResponse, Message, Usage
from morainet.distributed.checkpoint_distributed import (
    ClusterCheckpointStore,
    DistributeCheckpointHook,
    HeartbeatCheckpointStore,
)
from morainet.persistence.checkpoint import Checkpoint, InMemoryCheckpointStore


def _checkpoint(trace_id: str = "t1") -> Checkpoint:
    return Checkpoint(trace_id=trace_id, query="q")


def _resp(content: str) -> ChatResponse:
    return ChatResponse(message=Message.assistant(content=content), usage=Usage(), model="mock")


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(trace_id="t", query="q", messages=[], steps=[], usage=Usage())


class FakePipeline:
    def __init__(self, redis) -> None:
        self._redis = redis
        self._ops: list = []

    def set(self, key, value):
        self._ops.append(("set", key, value))

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))

    async def execute(self):
        for op in self._ops:
            if op[0] == "set":
                self._redis.data[op[1]] = op[2]
        return [True] * len(self._ops)


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict = {}
        self.closed = False

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def set(self, key, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def expire(self, key, ttl):
        return True

    async def delete(self, *keys):
        n = 0
        for key in keys:
            if self.data.pop(key, None) is not None:
                n += 1
        return n

    async def keys(self, pattern):
        return [k for k in self.data if fnmatch.fnmatch(k, pattern)]

    async def aclose(self):
        self.closed = True


def _hb(**kw) -> tuple[HeartbeatCheckpointStore, FakeRedis]:
    store = HeartbeatCheckpointStore(**kw)
    client = FakeRedis()
    store._client = client
    return store, client


# ---------------------------------------------------------------------------
# HeartbeatCheckpointStore
# ---------------------------------------------------------------------------


def test_heartbeat_key_helpers():
    store = HeartbeatCheckpointStore(key_prefix="pfx:")
    assert store._ckpt_key("t") == "pfx:t"
    assert store._owner_key("t") == "pfx:t:owner"
    assert store._heartbeat_key("n") == "pfx:heartbeat:n"


async def test_heartbeat_save_and_load_with_owner():
    store, client = _hb()
    await store.save(_checkpoint("t1"), owner_node_id="node-1")
    assert "morainet:checkpoint:t1" in client.data
    assert client.data["morainet:checkpoint:t1:owner"] == "node-1"
    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"


async def test_heartbeat_save_without_owner():
    store, client = _hb()
    await store.save(_checkpoint("t1"))
    assert "morainet:checkpoint:t1:owner" not in client.data


async def test_heartbeat_get_owner():
    store, _ = _hb()
    await store.save(_checkpoint("t1"), owner_node_id="node-1")
    assert await store.get_owner("t1") == "node-1"


async def test_heartbeat_load_missing():
    store, _ = _hb()
    assert await store.load("nope") is None


async def test_claim_orphan_owner_alive_returns_none():
    store, _ = _hb(ttl_seconds=30)
    await store.save(_checkpoint("t1"), owner_node_id="node-1")
    assert await store.claim_orphan("t1", "node-2") is None


async def test_claim_orphan_no_heartbeat_succeeds():
    store, client = _hb()
    client.data["morainet:checkpoint:t1"] = _checkpoint("t1").model_dump_json()
    client.data["morainet:checkpoint:t1:owner"] = "node-1"
    ckpt = await store.claim_orphan("t1", "node-2")
    assert ckpt is not None
    assert await store.get_owner("t1") == "node-2"


async def test_claim_orphan_stale_heartbeat_succeeds():
    store, client = _hb(ttl_seconds=0)
    client.data["morainet:checkpoint:t1"] = _checkpoint("t1").model_dump_json()
    client.data["morainet:checkpoint:t1:owner"] = "node-1"
    client.data["morainet:checkpoint:heartbeat:node-1"] = str(time.time())
    assert await store.claim_orphan("t1", "node-2") is not None


async def test_claim_orphan_missing_checkpoint():
    store, _ = _hb()
    assert await store.claim_orphan("nope", "n2") is None


async def test_list_orphans_stale_and_missing_heartbeat():
    store, client = _hb(ttl_seconds=0)
    client.data["morainet:checkpoint:t1:owner"] = "node-1"
    client.data["morainet:checkpoint:heartbeat:node-1"] = str(time.time())
    client.data["morainet:checkpoint:t2:owner"] = "node-2"
    orphans = await store.list_orphans()
    assert set(orphans) == {"t1", "t2"}


async def test_list_orphans_excludes_alive_and_excluded():
    store, client = _hb(ttl_seconds=9999)
    client.data["morainet:checkpoint:t1:owner"] = "node-1"
    client.data["morainet:checkpoint:heartbeat:node-1"] = str(time.time())
    assert await store.list_orphans() == []

    client.data["morainet:checkpoint:t2:owner"] = "node-2"
    assert await store.list_orphans(exclude_nodes=["node-2"]) == []


async def test_heartbeat_delete():
    store, _ = _hb()
    await store.save(_checkpoint("t1"), owner_node_id="n1")
    assert await store.delete("t1") is True


async def test_heartbeat_close():
    store, client = _hb()
    await store.close()
    assert client.closed is True
    assert store._client is None


def test_heartbeat_client_requires_redis():
    try:
        import redis  # noqa: F401
    except ImportError:
        store = HeartbeatCheckpointStore()
        with pytest.raises(ImportError):
            _ = store.client
        return
    pytest.skip("redis installed")


# ---------------------------------------------------------------------------
# ClusterCheckpointStore
# ---------------------------------------------------------------------------


async def test_cluster_store_delegates():
    store = ClusterCheckpointStore(InMemoryCheckpointStore(), replicas=2)
    await store.save(_checkpoint("t1"))
    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"


async def test_cluster_owner_tracking():
    store = ClusterCheckpointStore(InMemoryCheckpointStore())
    await store.assign_owner("t1", "n1")
    assert await store.get_owner("t1") == "n1"
    await store.release_owner("t1")
    assert await store.get_owner("t1") is None


# ---------------------------------------------------------------------------
# DistributeCheckpointHook
# ---------------------------------------------------------------------------


class FakeHeartbeatStore:
    def __init__(self) -> None:
        self.saved: list = []

    async def save(self, checkpoint, owner_node_id: str = "") -> None:
        self.saved.append((checkpoint.trace_id, owner_node_id))


async def test_distribute_hook_records_owner_on_events():
    store = FakeHeartbeatStore()
    hook = DistributeCheckpointHook(store, node_id="worker-1")
    ctx = _ctx()
    await hook.on_run_start(ctx)
    await hook.on_llm_end(ctx, _resp("x"))
    await hook.on_tool_end(ctx, None)  # type: ignore[arg-type]
    await hook.on_run_end(ctx, None)  # type: ignore[arg-type]
    assert len(store.saved) == 3
    assert all(owner == "worker-1" for _, owner in store.saved)
