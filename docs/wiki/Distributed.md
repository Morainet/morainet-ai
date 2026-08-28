# 分布式（Distributed）

> 模块：`morainet/distributed/` · 版本：v1.4+ · 示例：`examples/distributed_cluster.py`、`examples/distributed_workflow.py`

将 Agent 从单进程扩展到多节点，四大件：**任务队列 → 分布式调度 → 集群分片 → 负载均衡**，外加分布式 Checkpoint。

## 1. 任务队列

```python
from morainet import Task, RedisBackend, RabbitMQBackend, TaskProducer, TaskConsumer

task = Task(name="embed_doc", payload={"doc_id": "123"})
backend = RedisBackend(url="redis://localhost:6379/0")   # 或 RabbitMQBackend(...)
producer = TaskProducer(backend)
await producer.publish(task)

consumer = TaskConsumer(backend)
while task := await consumer.poll():
    await process(task)          # 自定义处理
```

`TaskBackend` 为抽象层，默认路径不依赖真实服务（可注入内存实现）。

## 2. DAG 分布式调度

复用 v1.0 Workflow 的 DAG 语义做跨节点执行：

```python
from morainet import (
    DistributedScheduler, DistributedParallelScheduler,
    DistributedNodeExecutor, TaskEnvelope,
)

executor = DistributedNodeExecutor()
scheduler = DistributedScheduler(executor=executor)
await scheduler.run(workflow)
```

- `TaskEnvelope` 携带任务元数据，`TaskStatus` 记录状态迁移
- `DistributedParallelScheduler` 分层并行、`DistributedProgressScheduler` 进度上报

## 3. 集群与分片

```python
from morainet import (
    AgentCluster, ClusterMember, ConsistentHashRing,
    SessionShardRouter, cloud_or_edge,
)

ring = ConsistentHashRing(nodes=["n1", "n2", "n3"], replicas=100)
node = ring.get_node("session-42")              # 稳定路由
cluster = AgentCluster(members=[ClusterMember(id="n1"), ...])

router = SessionShardRouter()                   # 会话级分片路由
target = router.route("session-42", ["n1", "n2", "n3"])
```

- `ConsistentHashRing`：一致性哈希，节点增减影响最小
- `SessionShard` / `SessionShardRouter`：会话稳定分片
- `AgentCluster` / `ClusterMember` / `EdgeNode` + `MemberStatus` / `ClusterRole`：集群拓扑
- `cloud_or_edge(...)`：云边协同启发式决策

## 4. 负载均衡

```python
from morainet import (
    RoundRobinBalancer, WeightedRoundRobinBalancer,
    HybridRouter, ProviderShard, Endpoint,
)

endpoints = [Endpoint(id="a", weight=3, healthy=True), Endpoint(id="b", weight=1)]
rr = RoundRobinBalancer()
wrr = WeightedRoundRobinBalancer()
hybrid = HybridRouter()                          # 综合权重 / 健康度 / 延迟

shard = ProviderShard(provider="openai", balancer=wrr)
ep = shard.select()
```

## 5. 分布式 Checkpoint

```python
from morainet import (
    ClusterCheckpointStore, HeartbeatCheckpointStore, DistributeCheckpointHook,
)

store = ClusterCheckpointStore()                 # 集群共享状态
hook = DistributeCheckpointHook(store=store)
agent = Agent(tools=[...], hooks=[hook])         # 自动挂钩
```

`HeartbeatCheckpointStore` 心跳续租，防止节点失联后状态丢失。

> 提示：Redis / RabbitMQ / 集群拓扑需真实服务或容器编排验证；无外部服务时可用内存实现离线跑通调度与负载均衡路径。
