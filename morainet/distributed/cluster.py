"""Agent cluster management — session sharding, consistent hashing,
distributed memory, and cluster-wide Pub/Sub.

Design:

- **Consistent hash ring**: Maps trace IDs / session keys to cluster members
  so each session is handled by a deterministic node, even as cluster
  membership changes (minimal re-sharding).

- **Session shards**: A shard = a range of the hash ring assigned to one
  member. Distributed memory writes go to the responsible shard.

- **Cluster registry**: Members heartbeat via Redis Pub/Sub. Membership
  changes trigger rebalancing events.

- **Edge + Cloud**: :class:`EdgeNode` represents a pre-processing layer
  that can route simple queries locally and forward complex ones to the
  cloud cluster.

Usage::

    from morainet.distributed import AgentCluster, ClusterMember, MemberStatus

    cluster = AgentCluster(redis_url="redis://localhost:6379/0")
    member = ClusterMember(node_id="node-1", role="worker")
    await cluster.join(member)

    # Route a session to the responsible node
    node_id = cluster.ring.get_node("session-abc")
    # -> "node-3" (deterministic)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MemberStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DRAINING = "draining"  # stopping but finishing in-flight work


class ClusterRole(str, Enum):
    WORKER = "worker"       # executes agent runs
    COORDINATOR = "coordinator"  # routes work, no execution
    EDGE = "edge"          # edge pre-processing node
    OBSERVER = "observer"  # read-only trace consumer


@dataclass
class ClusterMember:
    """A node in the Agent cluster."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: ClusterRole = ClusterRole.WORKER
    host: str = "localhost"
    port: int = 8090
    status: MemberStatus = MemberStatus.ONLINE
    weight: float = 1.0  # used for weighted load balancing
    labels: dict[str, str] = field(default_factory=dict)  # e.g. {"region": "us-east", "gpu": "a100"}
    last_heartbeat: float = field(default_factory=time.time)
    capacity: int = 10       # max concurrent sessions
    active_sessions: int = 0

    @property
    def available(self) -> bool:
        return self.status == MemberStatus.ONLINE and self.active_sessions < self.capacity

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "weight": self.weight,
            "labels": self.labels,
            "last_heartbeat": self.last_heartbeat,
            "capacity": self.capacity,
            "active_sessions": self.active_sessions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterMember":
        return cls(
            node_id=data["node_id"],
            role=ClusterRole(data.get("role", "worker")),
            host=data.get("host", "localhost"),
            port=data.get("port", 8090),
            status=MemberStatus(data.get("status", "online")),
            weight=data.get("weight", 1.0),
            labels=data.get("labels", {}),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            capacity=data.get("capacity", 10),
            active_sessions=data.get("active_sessions", 0),
        )


@dataclass
class EdgeNode:
    """Pre-processing edge node: routes simple queries locally,
    forwards complex ones to the cloud cluster."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    local_model: str = "llama3.2:3b"  # lightweight local model
    cloud_endpoint: str = "http://cloud-cluster:8090"
    complexity_threshold: float = 0.5  # 0-1, above which → cloud

    def should_route_to_cloud(self, complexity_score: float) -> bool:
        return complexity_score > self.complexity_threshold

    async def route(self, query: str, complexity_score: float) -> str:
        """Route query to local or cloud based on complexity."""
        if self.should_route_to_cloud(complexity_score):
            return "cloud"
        return "local"


def cloud_or_edge(complexity: float, threshold: float = 0.5) -> str:
    """Simple decision helper: return ``"cloud"`` or ``"edge"``."""
    return "cloud" if complexity > threshold else "edge"


# ---------------------------------------------------------------------------
# Consistent hash ring
# ---------------------------------------------------------------------------


@dataclass
class VirtualNode:
    """A virtual node on the hash ring pointing to a real member."""

    hash_key: int
    member_id: str


class ConsistentHashRing:
    """Consistent hash ring for session-to-node assignment.

    Each real member is mapped to ``virtual_nodes`` points on the ring
    (default 128), ensuring balanced distribution even with small clusters.

    Usage::

        ring = ConsistentHashRing(virtual_nodes=128)
        ring.add_member("node-1")
        ring.add_member("node-2", weight=2.0)  # 2x the virtual nodes
        node = ring.get_node("session-abc123")
    """

    def __init__(self, virtual_nodes: int = 128) -> None:
        self._vnodes: int = virtual_nodes
        self._ring: list[VirtualNode] = []  # sorted by hash_key
        self._members: dict[str, float] = {}  # member_id -> weight

    def _hash(self, key: str) -> int:
        """MD5-based hash, returns 32-bit int."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) & 0xFFFFFFFF

    def add_member(self, member_id: str, weight: float = 1.0) -> None:
        """Add a member to the ring with optional weight."""
        self._members[member_id] = weight
        self._rebuild()

    def remove_member(self, member_id: str) -> None:
        """Remove a member from the ring."""
        self._members.pop(member_id, None)
        self._rebuild()

    def update_weight(self, member_id: str, weight: float) -> None:
        if member_id in self._members:
            self._members[member_id] = weight
            self._rebuild()

    def _rebuild(self) -> None:
        self._ring.clear()
        if not self._members:
            return
        total_weight = sum(self._members.values())
        for mid, w in self._members.items():
            vnode_count = max(1, int(self._vnodes * w / total_weight)) if total_weight > 0 else self._vnodes
            for i in range(vnode_count):
                h = self._hash(f"{mid}:vnode:{i}")
                self._ring.append(VirtualNode(hash_key=h, member_id=mid))
        self._ring.sort(key=lambda vn: vn.hash_key)

    def get_node(self, key: str) -> str | None:
        """Find the responsible member for a key."""
        if not self._ring:
            return None
        h = self._hash(key)
        # Binary search for first vnode with hash >= h
        hashes = [vn.hash_key for vn in self._ring]
        idx = bisect_left(hashes, h)
        if idx >= len(self._ring):
            idx = 0  # wrap around
        return self._ring[idx].member_id

    def get_nodes(self, key: str, count: int = 3) -> list[str]:
        """Get ``count`` distinct members for a key (for replication)."""
        if not self._ring or count <= 0:
            return []
        h = self._hash(key)
        hashes = [vn.hash_key for vn in self._ring]
        idx = bisect_left(hashes, h)
        seen: set[str] = set()
        result: list[str] = []
        for i in range(len(self._ring)):
            vn = self._ring[(idx + i) % len(self._ring)]
            if vn.member_id not in seen:
                seen.add(vn.member_id)
                result.append(vn.member_id)
                if len(result) >= count:
                    break
        return result

    @property
    def members(self) -> list[str]:
        return sorted(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __bool__(self) -> bool:
        return bool(self._members)


# ---------------------------------------------------------------------------
# Session shard
# ---------------------------------------------------------------------------


@dataclass
class SessionShard:
    """Represents a range of the hash space assigned to a member.

    Used for distributed memory: each shard holds the long-term memory
    for sessions whose trace_id hashes to this shard's range.
    """

    shard_id: str
    member_id: str
    range_start: int
    range_end: int

    def contains(self, trace_id: str) -> bool:
        h = int(hashlib.md5(trace_id.encode()).hexdigest(), 16) & 0xFFFFFFFF
        if self.range_start <= self.range_end:
            return self.range_start <= h <= self.range_end
        # Wrap-around range
        return h >= self.range_start or h <= self.range_end


class SessionShardRouter:
    """Routes memory operations to the correct shard.

    Usage::

        router = SessionShardRouter(ring, members)
        shard_id = router.get_shard("trace-abc")
        # write to shard_id's memory store
    """

    def __init__(self, ring: ConsistentHashRing) -> None:
        self.ring = ring

    def get_shard_id(self, trace_id: str) -> str | None:
        return self.ring.get_node(trace_id)

    def get_shard_ids(self, trace_id: str, replicas: int = 2) -> list[str]:
        return self.ring.get_nodes(trace_id, count=replicas)


# ---------------------------------------------------------------------------
# Cluster event system
# ---------------------------------------------------------------------------

ClusterEventCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class _ClusterEventRegistry:
    """Registry for cluster lifecycle event handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[ClusterEventCallback]] = {}

    def on(self, event: str, handler: ClusterEventCallback) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        for h in self._handlers.get(event, []):
            try:
                await h(event, data)
            except Exception:
                pass  # handlers should not crash the cluster


cluster_registry = _ClusterEventRegistry()


def cluster_event_handler(event: str) -> Callable:  # type: ignore[type-arg]
    """Decorator: register a cluster event handler.

    ::

        @cluster_event_handler("member_joined")
        async def on_join(event: str, data: dict) -> None:
            print(f"Node {data['node_id']} joined")
    """
    def decorator(fn: ClusterEventCallback) -> ClusterEventCallback:
        cluster_registry.on(event, fn)
        return fn
    return decorator


def register_cluster_event(event: str, handler: ClusterEventCallback) -> None:
    cluster_registry.on(event, handler)


# ---------------------------------------------------------------------------
# Agent cluster
# ---------------------------------------------------------------------------

CHANNEL_MEMBERSHIP = "morainet:cluster:membership"
CHANNEL_HEARTBEAT = "morainet:cluster:heartbeat"
CHANNEL_SESSION = "morainet:cluster:session"
HEARTBEAT_INTERVAL = 5  # seconds
MEMBER_TTL = 15  # seconds before a member is considered dead


class AgentCluster:
    """Distributed Agent cluster backed by Redis Pub/Sub.

    - Members announce themselves and heartbeat on a shared channel.
    - The consistent hash ring is rebuilt whenever membership changes.
    - Session shards are deterministic based on the ring.

    Args:
        redis_url: Redis connection URL.
        node_id: This node's ID (auto-generated if not given).
        role: This node's role.
        host: Advertised host.
        port: Advertised port.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        node_id: str = "",
        role: ClusterRole = ClusterRole.WORKER,
        host: str = "localhost",
        port: int = 8090,
        virtual_nodes: int = 128,
    ) -> None:
        self._redis_url = redis_url
        self._virtual_nodes = virtual_nodes
        self._client: Any = None
        self._pubsub: Any = None

        # This node
        self._self = ClusterMember(
            node_id=node_id or uuid.uuid4().hex[:12],
            role=role,
            host=host,
            port=port,
        )

        # Cluster state
        self._members: dict[str, ClusterMember] = {}
        self._ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        self._router = SessionShardRouter(self._ring)
        self._running = False
        self._heartbeat_task: asyncio.Task[Any] | None = None
        self._listen_task: asyncio.Task[Any] | None = None

    # -- properties -----------------------------------------------------

    @property
    def node_id(self) -> str:
        return self._self.node_id

    @property
    def self_member(self) -> ClusterMember:
        return self._self

    @property
    def members(self) -> dict[str, ClusterMember]:
        return dict(self._members)

    @property
    def ring(self) -> ConsistentHashRing:
        return self._ring

    @property
    def router(self) -> SessionShardRouter:
        return self._router

    @property
    def online_count(self) -> int:
        return sum(1 for m in self._members.values() if m.status == MemberStatus.ONLINE)

    # -- lifecycle ------------------------------------------------------

    async def join(self) -> None:
        """Join the cluster and start heartbeat."""
        self._running = True
        await self._get_client()
        await self._announce("join")
        await self._rebuild_from_redis()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._listen_task = asyncio.create_task(self._listen_loop())
        await cluster_registry.emit("member_joined", self._self.to_dict())

    async def leave(self) -> None:
        """Gracefully leave the cluster."""
        self._self.status = MemberStatus.OFFLINE
        await self._announce("leave")
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._listen_task:
            self._listen_task.cancel()
        await cluster_registry.emit("member_left", {"node_id": self.node_id})
        await self._close_client()

    # -- session routing ------------------------------------------------

    def route_session(self, trace_id: str) -> ClusterMember | None:
        """Get the member responsible for a session."""
        node_id = self._ring.get_node(trace_id)
        if node_id is None:
            return None
        return self._members.get(node_id)

    def is_local_session(self, trace_id: str) -> bool:
        """Check if this node owns the session."""
        node_id = self._ring.get_node(trace_id)
        return node_id == self.node_id

    def acquire_session(self, trace_id: str) -> bool:
        """Try to acquire a session slot on the responsible node."""
        member = self.route_session(trace_id)
        if member is None:
            return False
        if member.node_id == self.node_id:
            if self._self.active_sessions < self._self.capacity:
                self._self.active_sessions += 1
                return True
        return False

    def release_session(self, trace_id: str) -> None:
        """Release a session slot."""
        if self.is_local_session(trace_id):
            self._self.active_sessions = max(0, self._self.active_sessions - 1)

    # -- internal -------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                raise ImportError("redis package required. pip install morainet-ai[redis]") from None
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(CHANNEL_MEMBERSHIP, CHANNEL_HEARTBEAT)
        return self._client

    async def _close_client(self) -> None:
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._pubsub = None

    async def _announce(self, event: str) -> None:
        client = await self._get_client()
        msg = json.dumps({"event": event, "member": self._self.to_dict()}, ensure_ascii=False)
        await client.publish(CHANNEL_MEMBERSHIP, msg)

    async def _heartbeat(self) -> None:
        client = await self._get_client()
        self._self.last_heartbeat = time.time()
        msg = json.dumps(self._self.to_dict(), ensure_ascii=False)
        await client.publish(CHANNEL_HEARTBEAT, msg)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await self._heartbeat()
            except Exception:
                pass
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _listen_loop(self) -> None:
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("type") == "message":
                    data = json.loads(message["data"])
                    channel = message["channel"]

                    if channel == CHANNEL_MEMBERSHIP:
                        await self._handle_membership(data)
                    elif channel == CHANNEL_HEARTBEAT:
                        await self._handle_heartbeat(data)
            except Exception:
                await asyncio.sleep(1)

    async def _handle_membership(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        member = ClusterMember.from_dict(data["member"])
        if member.node_id == self.node_id:
            return
        if event == "join":
            self._members[member.node_id] = member
            self._rebuild_ring()
            await cluster_registry.emit("member_joined", member.to_dict())
        elif event == "leave":
            self._members.pop(member.node_id, None)
            self._rebuild_ring()
            await cluster_registry.emit("member_left", {"node_id": member.node_id})

    async def _handle_heartbeat(self, data: dict[str, Any]) -> None:
        member = ClusterMember.from_dict(data)
        if member.node_id == self.node_id:
            return
        member.last_heartbeat = time.time()
        is_new = member.node_id not in self._members
        self._members[member.node_id] = member
        if is_new:
            self._rebuild_ring()

    def _rebuild_ring(self) -> None:
        active = {mid: m.weight for mid, m in self._members.items()
                  if m.status == MemberStatus.ONLINE}
        # Always include self
        active[self.node_id] = self._self.weight
        self._ring._members = active
        self._ring._rebuild()

    async def _rebuild_from_redis(self) -> None:
        """Scan Redis for existing heartbeat data to rehydrate cluster state."""
        try:
            client = await self._get_client()
            keys = await client.keys("morainet:cluster:member:*")
            for key in keys:
                raw = await client.get(key)
                if raw:
                    member = ClusterMember.from_dict(json.loads(raw))
                    if member.node_id != self.node_id:
                        self._members[member.node_id] = member
            self._rebuild_ring()
        except Exception:
            pass  # best-effort
