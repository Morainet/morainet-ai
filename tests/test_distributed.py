"""Tests for the `morainet.distributed` package.

Hermetic: no real Redis / RabbitMQ / network calls. Backends are exercised
with an in-memory fake (FakeBackend) and RedisBackend/HeartbeatCheckpointStore
are driven through a FakeRedisClient, so the suite runs without external
services.
"""

import asyncio
import hashlib
from collections import Counter
from typing import cast
from types import SimpleNamespace

import pytest

from morainet.core.context import Context
from morainet.core.models import AgentResult, ChatResponse, Step, Usage
from morainet.distributed import (
    AgentCluster,
    ClusterCheckpointStore,
    ClusterMember,
    ClusterRole,
    ConsistentHashRing,
    DistributeCheckpointHook,
    DistributedNodeExecutor,
    DistributedParallelScheduler,
    DistributedProgressScheduler,
    EdgeNode,
    Endpoint,
    HeartbeatCheckpointStore,
    HybridRouter,
    MemberStatus,
    ModelRouter,
    ProviderShard,
    RabbitMQBackend,
    RedisBackend,
    RoundRobinBalancer,
    SessionShard,
    SessionShardRouter,
    Task,
    TaskConsumer,
    TaskEnvelope,
    TaskProducer,
    TaskResult,
    TaskStatus,
    Tier,
    WeightedRoundRobinBalancer,
    cloud_or_edge,
)
from morainet.distributed.cluster import _ClusterEventRegistry
from morainet.distributed.dag_distributed import _serialize_context
from morainet.distributed.task_queue import TaskBackend
from morainet.exceptions import MorainetError
from morainet.persistence.checkpoint import Checkpoint, InMemoryCheckpointStore
from morainet.workflow import Workflow
from morainet.workflow.dag import Node


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBackend(TaskBackend):
    """In-memory TaskBackend: FIFO queues + results dict."""

    def __init__(self) -> None:
        self.queues: dict[str, list[Task]] = {}
        self.results: dict[str, TaskResult] = {}

    async def enqueue(self, task: Task) -> str:
        self.queues.setdefault(task.queue, []).append(task)
        return task.task_id

    async def dequeue(self, queue: str, timeout: float = 0.0) -> Task | None:
        q = self.queues.get(queue, [])
        return q.pop(0) if q else None

    async def ack(self, task_id: str, result: TaskResult) -> None:
        self.results[task_id] = result

    async def nack(self, task_id: str, queue: str) -> None:
        pass

    async def get_result(self, task_id: str) -> TaskResult | None:
        return self.results.get(task_id)

    async def queue_length(self, queue: str) -> int:
        return len(self.queues.get(queue, []))

    async def close(self) -> None:
        pass


class FakePipeline:
    def __init__(self, client: "FakeRedisClient") -> None:
        self.client = client
        self.ops: list[tuple[object, ...]] = []

    def set(self, key: str, value: str) -> "FakePipeline":
        self.ops.append(("set", key, value))
        return self

    def expire(self, key: str, ttl: int) -> "FakePipeline":
        self.ops.append(("expire", key, ttl))
        return self

    async def execute(self) -> list[object]:
        for op in self.ops:
            if op[0] == "set":
                self.client.values[cast(str, op[1])] = cast(str, op[2])
        self.ops = []
        return []


class FakeRedisClient:
    """Minimal in-memory stand-in for redis.asyncio used by RedisBackend."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}  # LPUSH head / RPOP tail
        self.zsets: dict[str, dict[str, float]] = {}
        self.values: dict[str, str] = {}

    async def lpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).insert(0, value)  # push to head
        return 1

    async def rpop(self, key: str) -> str | None:
        lst = self.lists.get(key, [])
        return lst.pop() if lst else None  # pop from tail -> FIFO

    async def brpop(self, keys: list[str], timeout: int = 0) -> list[str] | None:
        for key in keys:
            lst = self.lists.get(key, [])
            if lst:
                return [key, lst.pop()]
        return None

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def zpopmax(self, key: str, count: int = 1) -> list[tuple[str, float]]:
        zset = self.zsets.get(key, {})
        if not zset:
            return []
        picked = sorted(zset.items(), key=lambda kv: kv[1], reverse=True)[:count]
        for member, _ in picked:
            del zset[member]
        return [(m, s) for m, s in picked]

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str) -> None:
        self.values[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            for store in (self.values, self.lists, self.zsets):
                if key in store:
                    del store[key]
                    deleted += 1
        return deleted

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def aclose(self) -> None:
        pass


class _CapturingStore(InMemoryCheckpointStore):
    """InMemoryCheckpointStore that records owner_node_id on save."""

    def __init__(self) -> None:
        super().__init__()
        self.owners: dict[str, str] = {}
        self.saved: list[Checkpoint] = []

    async def save(self, checkpoint: Checkpoint, owner_node_id: str = "") -> None:
        await super().save(checkpoint)
        self.owners[checkpoint.trace_id] = owner_node_id
        self.saved.append(checkpoint)


def _hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# cluster.py — ClusterMember
# ---------------------------------------------------------------------------


class TestClusterMember:
    def test_defaults_and_available(self) -> None:
        m = ClusterMember()
        assert m.node_id
        assert m.role == ClusterRole.WORKER
        assert m.status == MemberStatus.ONLINE
        assert m.weight == 1.0
        assert m.available is True

    def test_roundtrip(self) -> None:
        m = ClusterMember(
            node_id="n1",
            role=ClusterRole.EDGE,
            host="10.0.0.1",
            port=9000,
            labels={"region": "cn"},
            weight=2.0,
        )
        m2 = ClusterMember.from_dict(m.to_dict())
        assert m2.node_id == "n1"
        assert m2.role == ClusterRole.EDGE
        assert m2.host == "10.0.0.1"
        assert m2.port == 9000
        assert m2.labels == {"region": "cn"}
        assert m2.weight == 2.0

    def test_available_respects_status_and_capacity(self) -> None:
        m = ClusterMember(capacity=1, active_sessions=1)
        assert m.available is False
        m.active_sessions = 0
        assert m.available is True
        m.status = MemberStatus.DRAINING
        assert m.available is False


# ---------------------------------------------------------------------------
# cluster.py — ConsistentHashRing / shards / edge
# ---------------------------------------------------------------------------


class TestConsistentHashRing:
    def test_empty(self) -> None:
        ring = ConsistentHashRing()
        assert ring.get_node("k") is None
        assert ring.get_nodes("k") == []
        assert not ring
        assert len(ring) == 0

    def test_get_node_deterministic(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=32)
        ring.add_member("n1")
        ring.add_member("n2")
        node = ring.get_node("session-1")
        assert node in {"n1", "n2"}
        assert ring.get_node("session-1") == node

    def test_remove_member(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=32)
        for m in ("n1", "n2", "n3"):
            ring.add_member(m)
        ring.remove_member("n2")
        assert "n2" not in ring.members
        assert all(ring.get_node(f"k{i}") != "n2" for i in range(50))

    def test_update_weight(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=32)
        ring.add_member("a", weight=1.0)
        ring.update_weight("a", 5.0)
        assert ring._members["a"] == 5.0
        ring.update_weight("missing", 5.0)  # no-op
        assert ring.members == ["a"]

    def test_get_nodes_distinct_and_limited(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=32)
        for m in ("n1", "n2", "n3"):
            ring.add_member(m)
        nodes = ring.get_nodes("k", count=2)
        assert len(nodes) == 2 and len(set(nodes)) == 2
        # count larger than member set -> capped at distinct members
        assert len(ring.get_nodes("k", count=10)) == 3
        assert ring.get_nodes("k", count=0) == []
        assert ring.get_nodes("k", count=-1) == []

    def test_removal_keeps_remaining_members(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=64)
        for m in ("a", "b", "c"):
            ring.add_member(m)
        keys = [f"sess-{i}" for i in range(300)]
        ring.remove_member("c")
        after = {k: ring.get_node(k) for k in keys}
        # No key maps to the removed member
        assert set(after.values()) <= {"a", "b"}
        # Load stays balanced across remaining members
        a_share = sum(1 for v in after.values() if v == "a") / len(keys)
        assert 0.4 < a_share < 0.6

    def test_reshard_keeps_majority_of_assignments(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=64)
        for m in ("a", "b", "c"):
            ring.add_member(m)
        keys = [f"sess-{i}" for i in range(300)]
        before = {k: ring.get_node(k) for k in keys}
        ring.remove_member("c")
        after = {k: ring.get_node(k) for k in keys}
        stable = sum(1 for k in keys if before[k] == after[k])
        # vnode count is weight-normalized (int(vnodes*w/total)), so removing a
        # member grows the surviving members' vnode sets and causes more
        # reshuffling than a textbook fixed-vnode ring; assert the observed
        # behaviour rather than the ideal ~2/3.
        assert stable / len(keys) > 0.4


class TestSessionShard:
    def test_contains_normal_range(self) -> None:
        h = _hash("trace-a")
        shard = SessionShard(shard_id="s1", member_id="m1", range_start=h, range_end=h)
        assert shard.contains("trace-a")

    def test_not_contains_outside_range(self) -> None:
        h = _hash("target")
        # Widen to a known disjoint range: (h+1, h-1) wraps and excludes h
        assert not SessionShard("s2", "m2", h + 1, h - 1).contains("target")

    def test_wraparound(self) -> None:
        h = _hash("target")
        # start > end -> wrap-around range that still includes h
        wrapped = SessionShard(shard_id="s1", member_id="m1", range_start=h + 1, range_end=h)
        assert wrapped.contains("target")

    def test_router(self) -> None:
        ring = ConsistentHashRing(virtual_nodes=64)
        for m in ("n1", "n2", "n3"):
            ring.add_member(m)
        router = SessionShardRouter(ring)
        sid = router.get_shard_id("trace-1")
        assert sid in {"n1", "n2", "n3"}
        assert router.get_shard_id("trace-1") == sid  # deterministic
        ids = router.get_shard_ids("trace-1", replicas=2)
        assert len(ids) == 2 and len(set(ids)) == 2


class TestEdgeNode:
    @pytest.mark.asyncio
    async def test_routing(self) -> None:
        edge = EdgeNode(complexity_threshold=0.5)
        assert edge.should_route_to_cloud(0.1) is False
        assert edge.should_route_to_cloud(0.9) is True
        assert await edge.route("simple", 0.1) == "local"
        assert await edge.route("complex", 0.9) == "cloud"

    def test_cloud_or_edge(self) -> None:
        assert cloud_or_edge(0.1) == "edge"
        assert cloud_or_edge(0.9) == "cloud"
        assert cloud_or_edge(0.5) == "edge"  # threshold is not exceeded


class TestClusterEvents:
    @pytest.mark.asyncio
    async def test_registry_emit_and_error_isolation(self) -> None:
        registry = _ClusterEventRegistry()
        calls: list[object] = []

        async def h1(event, data):
            calls.append(("h1", data["x"]))

        async def h2(event, data):
            raise RuntimeError("boom")

        async def h3(event, data):
            calls.append("h3")

        registry.on("evt", h1)
        registry.on("evt", h2)
        registry.on("evt", h3)
        await registry.emit("evt", {"x": 1})  # h2 exception must be swallowed
        assert calls == [("h1", 1), "h3"]

    @pytest.mark.asyncio
    async def test_decorator_registers_handler(self) -> None:
        from morainet.distributed.cluster import cluster_event_handler

        events: list[str] = []

        @cluster_event_handler("test_member_joined_unique")
        async def on_join(event, data):
            events.append(data["node_id"])

        await on_join("test_member_joined_unique", {"node_id": "n-7"})
        assert events == ["n-7"]


class TestAgentCluster:
    def test_session_routing_local(self) -> None:
        cluster = AgentCluster(node_id="node-local")
        cluster._members["node-local"] = cluster.self_member
        cluster._ring.add_member("node-local", weight=1.0)

        assert cluster.ring.get_node("anything") == "node-local"
        assert cluster.is_local_session("anything") is True
        member = cluster.route_session("anything")
        assert member is not None
        assert member.node_id == "node-local"
        assert cluster.acquire_session("anything") is True
        cluster.release_session("anything")
        assert cluster.self_member.active_sessions == 0

    def test_acquire_respects_capacity(self) -> None:
        cluster = AgentCluster(node_id="node-cap")
        cluster._members["node-cap"] = cluster.self_member
        cluster._ring.add_member("node-cap", weight=1.0)
        cluster.self_member.capacity = 1

        assert cluster.acquire_session("s1") is True
        assert cluster.acquire_session("s2") is False  # local but full

    def test_rebuild_ring_excludes_offline(self) -> None:
        cluster = AgentCluster(node_id="node-local")
        cluster._members["node-local"] = cluster.self_member
        cluster._members["node-1"] = ClusterMember(node_id="node-1")
        cluster._rebuild_ring()
        assert set(cluster.ring.members) == {"node-local", "node-1"}
        assert cluster.online_count == 2

        cluster._members["node-1"].status = MemberStatus.OFFLINE
        cluster._rebuild_ring()
        assert "node-1" not in cluster.ring.members
        assert cluster.online_count == 1


# ---------------------------------------------------------------------------
# load_balancer.py
# ---------------------------------------------------------------------------


class TestEndpoint:
    def test_available_and_circuit_breaker(self) -> None:
        ep = Endpoint(url="http://a", max_errors=3)
        assert ep.available
        for _ in range(3):
            ep.record_error()
        assert not ep.available
        assert not ep.healthy
        ep.reset()
        assert ep.available and ep.healthy and ep.error_count == 0

    def test_record_success_resets_errors(self) -> None:
        ep = Endpoint(url="http://a")
        ep.record_error()
        assert ep.error_count == 1
        ep.record_success(elapsed_ms=12.0)
        assert ep.error_count == 0
        assert ep.latency_ms == 12.0


class TestProviderShard:
    def test_healthy_endpoints(self) -> None:
        shard = ProviderShard(name="openai", endpoints=[
            Endpoint(url="a"),
            Endpoint(url="b", healthy=False),
        ])
        assert [ep.url for ep in shard.healthy_endpoints] == ["a"]

    def test_add_remove(self) -> None:
        shard = ProviderShard(name="openai", endpoints=[Endpoint(url="a")])
        shard.add_endpoint(Endpoint(url="b"))
        assert len(shard.endpoints) == 2
        shard.remove_endpoint("a")
        assert [ep.url for ep in shard.endpoints] == ["b"]


def _next_url(lb) -> str:
    ep = lb.next()
    assert ep is not None
    return ep.url


class TestRoundRobinBalancer:
    def test_cycles(self) -> None:
        lb = RoundRobinBalancer([Endpoint(url="a"), Endpoint(url="b")])
        assert [_next_url(lb) for _ in range(4)] == ["a", "b", "a", "b"]
        lb.reset()
        assert _next_url(lb) == "a"

    def test_skips_unhealthy(self) -> None:
        lb = RoundRobinBalancer([
            Endpoint(url="a"),
            Endpoint(url="b"),
            Endpoint(url="c", healthy=False),
        ])
        assert [_next_url(lb) for _ in range(4)] == ["a", "b", "a", "b"]

    def test_empty(self) -> None:
        assert RoundRobinBalancer([]).next() is None


class TestWeightedRoundRobinBalancer:
    def test_distribution(self) -> None:
        lb = WeightedRoundRobinBalancer([
            Endpoint(url="a", weight=3),
            Endpoint(url="b", weight=1),
        ])
        counts = Counter(_next_url(lb) for _ in range(200))
        assert counts["a"] > counts["b"]
        assert counts["a"] > 100

    def test_all_down(self) -> None:
        lb = WeightedRoundRobinBalancer([Endpoint(url="a", healthy=False)])
        assert lb.next() is None


def _make_shard(name: str, tier: Tier, urls: list[str], healthy: bool = True) -> ProviderShard:
    return ProviderShard(name=name, tier=tier, endpoints=[
        Endpoint(url=u, tier=tier, healthy=healthy) for u in urls
    ])


class TestModelRouter:
    def test_route_by_provider(self) -> None:
        router = ModelRouter()
        router.register("openai", _make_shard("openai", Tier.LARGE, ["https://openai"]))
        router.register("ollama", _make_shard("ollama", Tier.SMALL, ["http://ollama"]))
        ep = router.route(tier="large", provider="openai")
        assert ep is not None and ep.url == "https://openai"

    def test_route_any_provider(self) -> None:
        router = ModelRouter()
        router.register("ollama", _make_shard("ollama", Tier.MEDIUM, ["http://ollama"]))
        ep = router.route(tier="medium")
        assert ep is not None and ep.url == "http://ollama"

    def test_preferred_labels(self) -> None:
        router = ModelRouter()
        shard = ProviderShard(name="openai", tier=Tier.LARGE, endpoints=[
            Endpoint(url="us", labels={"region": "us"}),
            Endpoint(url="eu", labels={"region": "eu"}),
        ])
        router.register("openai", shard)
        ep = router.route(tier="large", preferred_labels={"region": "eu"})
        assert ep is not None and ep.url == "eu"

    def test_fallback_to_next_tier(self) -> None:
        router = ModelRouter()
        router.register("openai", _make_shard("openai", Tier.LARGE, ["https://openai"], healthy=False))
        router.register("ollama", _make_shard("ollama", Tier.MEDIUM, ["http://ollama"]))
        ep = router.route(tier="large")
        assert ep is not None and ep.url == "http://ollama"

    def test_no_candidates(self) -> None:
        router = ModelRouter()
        router.register("openai", _make_shard("openai", Tier.LARGE, ["x"], healthy=False))
        assert router.route(tier="large") is None
        assert router.shard_names == ["openai"]
        assert router.get_shard("openai") is not None


class TestHybridRouter:
    def test_estimate_complexity(self) -> None:
        hybrid = HybridRouter()
        assert hybrid.estimate_complexity("hi") < 0.5
        assert hybrid.estimate_complexity(
            "please explain analyze optimize this architecture design deployment"
        ) > 0.5

    def test_decide_edge_and_cloud(self) -> None:
        hybrid = HybridRouter()
        route, ep = hybrid.decide("hi")
        assert route == "edge"
        assert ep is not None and ep.tier == Tier.EDGE

        route, ep = hybrid.decide("please explain and analyze this complex architecture design")
        assert route == "cloud"
        assert ep is not None and ep.tier == Tier.LARGE


# ---------------------------------------------------------------------------
# task_queue.py
# ---------------------------------------------------------------------------


class TestTask:
    def test_roundtrip(self) -> None:
        t = Task(task_id="t1", payload={"a": 1}, queue="q", priority=5, retries=2)
        t2 = Task.from_dict(t.to_dict())
        assert t2 == t
        assert t2.priority == 5

    def test_defaults(self) -> None:
        t = Task()
        assert t.task_id and t.queue == "default"
        assert t.max_retries == 3 and t.priority == 0 and t.timeout == 0.0


class TestTaskResult:
    def test_defaults(self) -> None:
        r = TaskResult(task_id="t1", status="success")
        assert r.result is None
        assert r.error == "" and r.worker_id == "" and r.elapsed_ms == 0.0


class TestRedisBackend:
    def test_client_import_error_without_redis(self) -> None:
        backend = RedisBackend()
        with pytest.raises(ImportError):
            _ = backend.client

    @pytest.mark.asyncio
    async def test_fifo(self) -> None:
        backend = RedisBackend()
        backend._client = FakeRedisClient()
        await backend.enqueue(Task(task_id="t1", queue="q"))
        await backend.enqueue(Task(task_id="t2", queue="q"))
        t = await backend.dequeue("q")
        assert t is not None and t.task_id == "t1"
        t = await backend.dequeue("q")
        assert t is not None and t.task_id == "t2"
        assert await backend.dequeue("q") is None

    @pytest.mark.asyncio
    async def test_priority_dequeue(self) -> None:
        backend = RedisBackend()
        backend._client = FakeRedisClient()
        await backend.enqueue(Task(task_id="low", queue="q", priority=1))
        await backend.enqueue(Task(task_id="high", queue="q", priority=10))
        t = await backend.dequeue("q")
        assert t is not None and t.task_id == "high"

    @pytest.mark.asyncio
    async def test_ack_get_result(self) -> None:
        backend = RedisBackend(result_ttl=60)
        backend._client = FakeRedisClient()
        await backend.ack("t1", TaskResult(task_id="t1", status="success", result=42, worker_id="w1"))
        r = await backend.get_result("t1")
        assert r is not None
        assert r.status == "success" and r.result == 42 and r.worker_id == "w1"
        assert await backend.get_result("missing") is None

    @pytest.mark.asyncio
    async def test_queue_length(self) -> None:
        backend = RedisBackend()
        backend._client = FakeRedisClient()
        await backend.enqueue(Task(task_id="t1", queue="q"))
        await backend.enqueue(Task(task_id="t2", queue="q"))
        assert await backend.queue_length("q") == 2

    @pytest.mark.asyncio
    async def test_nack_marks_retrying(self) -> None:
        backend = RedisBackend()
        backend._client = FakeRedisClient()
        await backend.ack("t1", TaskResult(task_id="t1", status="failed"))
        await backend.nack("t1", "q")
        r = await backend.get_result("t1")
        assert r is not None and r.status == "retrying"


class TestRabbitMQBackend:
    @pytest.mark.asyncio
    async def test_results_in_memory(self) -> None:
        backend = RabbitMQBackend()
        await backend.ack("t1", TaskResult(task_id="t1", status="success", result=[1]))
        r = await backend.get_result("t1")
        assert r is not None and r.result == [1]
        assert await backend.get_result("nope") is None
        await backend.nack("t1", "q")  # no-op


class TestTaskProducer:
    @pytest.mark.asyncio
    async def test_enqueue_sets_queue(self) -> None:
        backend = FakeBackend()
        producer = TaskProducer(backend, "my-queue")
        task = Task(payload={})
        tid = await producer.enqueue(task)
        assert task.queue == "my-queue"
        assert tid == task.task_id

    @pytest.mark.asyncio
    async def test_enqueue_bulk(self) -> None:
        backend = FakeBackend()
        producer = TaskProducer(backend, "bulk")
        ids = await producer.enqueue_bulk([Task(payload={"i": i}) for i in range(3)])
        assert len(ids) == 3
        assert await backend.queue_length("bulk") == 3


class TestTaskConsumer:
    @pytest.mark.asyncio
    async def test_full_flow(self) -> None:
        backend = FakeBackend()
        producer = TaskProducer(backend, "q1")
        await producer.enqueue(Task(payload={"n": 1}))
        await producer.enqueue(Task(payload={"n": 2}))

        consumer = TaskConsumer(backend, "q1", worker_id="w1", poll_interval=0.01)
        seen: list[int] = []
        task_ids: list[str] = []
        async for task in consumer.consume():
            seen.append(task.payload["n"])
            task_ids.append(task.task_id)
            await consumer.ack(task, result=task.payload["n"] * 10)
            if len(seen) == 2:
                consumer.stop()

        assert seen == [1, 2]
        r1 = await backend.get_result(task_ids[0])
        assert r1 is not None
        assert r1.status == "success" and r1.result == 10 and r1.worker_id == "w1"

    @pytest.mark.asyncio
    async def test_ack_with_error(self) -> None:
        backend = FakeBackend()
        producer = TaskProducer(backend, "q")
        tid = await producer.enqueue(Task(payload={}))
        consumer = TaskConsumer(backend, "q", worker_id="w1")
        async for task in consumer.consume():
            await consumer.ack(task, error="boom")
            consumer.stop()
        r = await backend.get_result(tid)
        assert r is not None and r.status == "failed" and r.error == "boom"

    @pytest.mark.asyncio
    async def test_stop_breaks_consume_loop(self) -> None:
        backend = FakeBackend()
        producer = TaskProducer(backend, "q")
        await producer.enqueue(Task(payload={"n": 1}))
        await producer.enqueue(Task(payload={"n": 2}))
        consumer = TaskConsumer(backend, "q", poll_interval=0.01)
        received: list[int] = []
        async for task in consumer.consume():
            received.append(task.payload["n"])
            consumer.stop()  # stop() must interrupt the iteration
        assert received == [1]


# ---------------------------------------------------------------------------
# dag_distributed.py
# ---------------------------------------------------------------------------


def _module_func(ctx):
    return ctx["v"] * 2


class TestTaskEnvelope:
    def test_roundtrip(self) -> None:
        env = TaskEnvelope(
            node_name="n", workflow_id="w", context={"k": 1},
            deps=["a"], workflow_level=2, retries=1,
        )
        env2 = TaskEnvelope.from_dict(env.to_dict())
        assert env2.node_name == "n"
        assert env2.workflow_id == "w"
        assert env2.workflow_level == 2
        assert env2.retries == 1
        assert env2.status == TaskStatus.PENDING

    def test_from_node_lambda(self) -> None:
        node = Node(name="f", func=lambda ctx: ctx["x"])
        env = TaskEnvelope.from_node("wf", node, {"x": 1})
        assert env.node_name == "f"
        assert env.context == {"x": 1}
        assert env.function_code  # cloudpickle fallback serialized

    def test_restore_lambda(self) -> None:
        node = Node(name="f", func=lambda ctx: ctx["x"] * 10)
        env = TaskEnvelope.from_node("wf", node, {"x": 1})
        assert env.restore_function()({"x": 5}) == 50

    def test_restore_import_path(self) -> None:
        node = Node(name="f", func=_module_func)
        env = TaskEnvelope.from_node("wf", node, {"v": 3})
        assert env.restore_function()({"v": 3}) == 6

    def test_restore_failure(self) -> None:
        env = TaskEnvelope(node_name="n", function_module="", function_code="")
        with pytest.raises(MorainetError):
            env.restore_function()


class TestSerializeContext:
    def test_serializable_and_not(self) -> None:
        ctx = {"ok": 1, "obj": object()}
        safe = _serialize_context(ctx)
        assert safe["ok"] == 1
        assert safe["obj"].startswith("<non-serializable: object>")


class TestDistributedNodeExecutor:
    @pytest.mark.asyncio
    async def test_executes_sync_and_async(self) -> None:
        executor = DistributedNodeExecutor(FakeBackend(), "wf", worker_id="w1")

        async def afn(ctx):
            return ctx["v"] + 1

        sync_env = TaskEnvelope.from_node("w", Node(name="a", func=lambda ctx: ctx["v"] + 1), {"v": 1})
        async_env = TaskEnvelope.from_node("w", Node(name="b", func=afn), {"v": 1})
        assert await executor._execute(Task(task_id=sync_env.task_id, payload={"envelope": sync_env.to_dict()})) == 2
        assert await executor._execute(Task(task_id=async_env.task_id, payload={"envelope": async_env.to_dict()})) == 2

    @pytest.mark.asyncio
    async def test_execute_propagates_error(self) -> None:
        executor = DistributedNodeExecutor(FakeBackend(), "wf", worker_id="w1")
        env = TaskEnvelope.from_node("w", Node(name="bad", func=lambda ctx: 1 / 0), {})
        task = Task(task_id=env.task_id, payload={"envelope": env.to_dict()})
        with pytest.raises(ZeroDivisionError):
            await executor._execute(task)


class TestDistributedParallelScheduler:
    @pytest.mark.asyncio
    async def test_end_to_end(self) -> None:
        backend = FakeBackend()
        wf = Workflow()
        wf.add_node("fetch", lambda ctx: {"url": ctx["url"]})
        wf.add_node("parse", lambda ctx: ctx["fetch"]["url"].upper())
        wf.connect("fetch", "parse")

        scheduler = DistributedParallelScheduler(
            backend, queue="wf", workflow_id="wf-e2e",
            worker_timeout=5.0, poll_interval=0.01,
        )
        executor = DistributedNodeExecutor(backend, queue="wf", worker_id="w1")
        worker = asyncio.create_task(executor.serve())
        try:
            out = await asyncio.wait_for(
                scheduler.run(wf, {"url": "https://example.com"}), timeout=5
            )
        finally:
            executor.stop()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        assert out["fetch"] == {"url": "https://example.com"}
        assert out["parse"] == "HTTPS://EXAMPLE.COM"

    @pytest.mark.asyncio
    async def test_node_failure_raises(self) -> None:
        backend = FakeBackend()
        wf = Workflow()
        wf.add_node("bad", lambda ctx: 1 / 0)

        scheduler = DistributedParallelScheduler(
            backend, queue="wf", workflow_id="wf-fail",
            worker_timeout=5.0, poll_interval=0.01,
        )
        executor = DistributedNodeExecutor(backend, queue="wf", worker_id="w1")
        worker = asyncio.create_task(executor.serve())
        try:
            with pytest.raises(MorainetError, match="failed"):
                await asyncio.wait_for(scheduler.run(wf, {}), timeout=5)
        finally:
            executor.stop()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)


class TestDistributedProgressScheduler:
    @pytest.mark.asyncio
    async def test_progress_tracking(self) -> None:
        backend = FakeBackend()
        wf = Workflow()
        wf.add_node("a", lambda ctx: 1)
        wf.add_node("b", lambda ctx: ctx["a"] + 1)
        wf.connect("a", "b")

        updates: list[object] = []
        scheduler = DistributedProgressScheduler(
            backend, queue="wf", workflow_id="wf-prog",
            worker_timeout=5.0, poll_interval=0.01,
            on_progress=updates.append,
        )
        executor = DistributedNodeExecutor(backend, queue="wf", worker_id="w1")
        worker = asyncio.create_task(executor.serve())
        try:
            out = await asyncio.wait_for(scheduler.run(wf, {}), timeout=5)
        finally:
            executor.stop()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        assert out["a"] == 1 and out["b"] == 2
        assert scheduler.progress.total == 2
        assert scheduler.progress.completed == 2
        assert scheduler.progress.failed == 0
        assert all(n.status == "success" for n in scheduler.progress.nodes.values())
        assert updates  # on_progress was invoked


# ---------------------------------------------------------------------------
# checkpoint_distributed.py
# ---------------------------------------------------------------------------


class TestHeartbeatCheckpointStore:
    def test_client_import_error_without_redis(self) -> None:
        store = HeartbeatCheckpointStore()
        with pytest.raises(ImportError):
            _ = store.client


class TestClusterCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_load_owner(self) -> None:
        inner = InMemoryCheckpointStore()
        store = ClusterCheckpointStore(inner, replicas=2)
        ckpt = Checkpoint(trace_id="t1", query="q")
        await store.save(ckpt)
        loaded = await store.load("t1")
        assert loaded is not None and loaded.trace_id == "t1"
        assert await store.load("missing") is None

        await store.assign_owner("t1", "node-1")
        assert await store.get_owner("t1") == "node-1"
        await store.release_owner("t1")
        assert await store.get_owner("t1") is None


class TestDistributeCheckpointHook:
    @pytest.mark.asyncio
    async def test_saves_checkpoints_with_owner(self) -> None:
        store = _CapturingStore()
        hook = DistributeCheckpointHook(
            cast(HeartbeatCheckpointStore, cast(object, store)), node_id="worker-1"
        )
        ctx = cast(Context, cast(object, SimpleNamespace(
            trace_id="t1", query="hello",
            messages=[], steps=[], usage=Usage(),
        )))

        await hook.on_run_start(ctx)
        await hook.on_llm_end(ctx, cast(ChatResponse, cast(object, SimpleNamespace())))
        await hook.on_tool_end(ctx, cast(Step, cast(object, SimpleNamespace())))
        await hook.on_run_end(ctx, cast(AgentResult, cast(object, SimpleNamespace())))

        assert store.owners["t1"] == "worker-1"
        assert len(store.saved) == 3  # llm_end + tool_end + run_end
        assert store.saved[-1].trace_id == "t1"
        assert store.saved[-1].query == "hello"
