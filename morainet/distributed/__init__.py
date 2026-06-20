"""Distributed Agent scheduling — cluster-aware execution, task queues,
load balancing, and global trace linking.

- :mod:`morainet.distributed.task_queue` — Redis / RabbitMQ task queue backends
- :mod:`morainet.distributed.dag_distributed` — Distributed DAG scheduler across machines
- :mod:`morainet.distributed.cluster` — Session sharding, cluster membership, pub/sub
- :mod:`morainet.distributed.load_balancer` — Model inference routing across nodes
- :mod:`morainet.distributed.checkpoint_distributed` — Distributed breakpoint recovery
"""

from morainet.distributed.checkpoint_distributed import (
    ClusterCheckpointStore,
    DistributeCheckpointHook,
    HeartbeatCheckpointStore,
)
from morainet.distributed.cluster import (
    AgentCluster,
    ClusterMember,
    ClusterRole,
    ConsistentHashRing,
    EdgeNode,
    MemberStatus,
    SessionShard,
    SessionShardRouter,
    cloud_or_edge,
    cluster_event_handler,
    cluster_registry,
    register_cluster_event,
)
from morainet.distributed.dag_distributed import (
    DistributedNodeExecutor,
    DistributedParallelScheduler,
    DistributedProgressScheduler,
    DistributedScheduler,
    TaskEnvelope,
    TaskStatus,
)
from morainet.distributed.load_balancer import (
    Endpoint,
    HybridRouter,
    LoadBalancer,
    ModelRouter,
    ProviderShard,
    RoundRobinBalancer,
    Tier,
    WeightedRoundRobinBalancer,
)
from morainet.distributed.task_queue import (
    RabbitMQBackend,
    RedisBackend,
    Task,
    TaskBackend,
    TaskConsumer,
    TaskProducer,
    TaskResult,
)

__all__ = [
    # Task queue
    "Task",
    "TaskResult",
    "TaskBackend",
    "RedisBackend",
    "RabbitMQBackend",
    "TaskProducer",
    "TaskConsumer",
    # Distributed DAG
    "TaskEnvelope",
    "TaskStatus",
    "DistributedScheduler",
    "DistributedParallelScheduler",
    "DistributedProgressScheduler",
    "DistributedNodeExecutor",
    # Cluster
    "AgentCluster",
    "ClusterMember",
    "MemberStatus",
    "ClusterRole",
    "EdgeNode",
    "ConsistentHashRing",
    "SessionShard",
    "SessionShardRouter",
    "cloud_or_edge",
    "cluster_registry",
    "cluster_event_handler",
    "register_cluster_event",
    # Load balancer
    "Endpoint",
    "Tier",
    "ProviderShard",
    "ModelRouter",
    "LoadBalancer",
    "RoundRobinBalancer",
    "WeightedRoundRobinBalancer",
    "HybridRouter",
    # Distributed checkpoint
    "ClusterCheckpointStore",
    "HeartbeatCheckpointStore",
    "DistributeCheckpointHook",
]
