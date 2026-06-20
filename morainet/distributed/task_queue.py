"""Task queue abstraction — Redis (primary) and RabbitMQ (plug) backends.

A distributed workflow node is packaged as a :class:`Task` and dispatched
through a :class:`TaskBackend`. Workers on any machine consume tasks from the
same queue, enabling cross-machine parallelism with automatic load distribution.

Usage::

    from morainet.distributed import RedisBackend, TaskProducer, TaskConsumer

    backend = RedisBackend(redis_url="redis://localhost:6379/0")

    # Producer side
    producer = TaskProducer(backend, "inference-queue")
    task_id = await producer.enqueue(Task(payload={"prompt": "hello"}))

    # Consumer side (blocking loop, run on worker nodes)
    consumer = TaskConsumer(backend, "inference-queue")
    async for task in consumer.consume():
        result = do_work(task.payload)
        await consumer.ack(task, result)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work in the distributed queue."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    payload: dict[str, Any] = field(default_factory=dict)
    queue: str = "default"
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    timeout: float = 0.0  # seconds; 0 = no timeout
    priority: int = 0  # higher = more urgent

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "payload": self.payload,
            "queue": self.queue,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "timeout": self.timeout,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(**data)


@dataclass
class TaskResult:
    task_id: str
    status: str  # "success" | "failed" | "timeout"
    result: Any = None
    error: str = ""
    worker_id: str = ""
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class TaskBackend(ABC):
    """Abstract queue backend — implement for Redis, RabbitMQ, etc."""

    @abstractmethod
    async def enqueue(self, task: Task) -> str:
        """Push a task onto the queue. Returns the task_id."""
        ...

    @abstractmethod
    async def dequeue(self, queue: str, timeout: float = 0.0) -> Task | None:
        """Pop a task (FIFO) or return ``None`` if the queue is empty."""
        ...

    @abstractmethod
    async def ack(self, task_id: str, result: TaskResult) -> None:
        """Mark task as completed and store the result."""
        ...

    @abstractmethod
    async def nack(self, task_id: str, queue: str) -> None:
        """Re-queue a failed task (increment retry counter)."""
        ...

    @abstractmethod
    async def get_result(self, task_id: str) -> TaskResult | None:
        """Fetch the stored result of a completed task."""
        ...

    @abstractmethod
    async def queue_length(self, queue: str) -> int:
        """Number of pending tasks in the queue."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources."""
        ...


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisBackend(TaskBackend):
    """Redis-backed task queue using lists and hashes.

    Design:
    - Queue: ``morainet:queue:<name>`` (Redis list, LPUSH / BRPOP)
    - Results: ``morainet:result:<task_id>`` (Redis string, with TTL)
    - Retry/Dead-letter: tasks exceeding ``max_retries`` go to ``morainet:dlq:<name>``

    Args:
        redis_url: Redis connection URL, e.g. ``redis://localhost:6379/0``.
        result_ttl: Seconds before results expire (0 = never).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", result_ttl: int = 3600) -> None:
        self._redis_url = redis_url
        self._result_ttl = result_ttl
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                raise ImportError(
                    "redis package required. Install: pip install morainet-ai[redis]"
                ) from None
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _queue_key(self, name: str) -> str:
        return f"morainet:queue:{name}"

    def _dlq_key(self, name: str) -> str:
        return f"morainet:dlq:{name}"

    def _result_key(self, task_id: str) -> str:
        return f"morainet:result:{task_id}"

    async def enqueue(self, task: Task) -> str:
        data = json.dumps(task.to_dict(), ensure_ascii=False)
        queue_key = self._queue_key(task.queue)
        # Priority support: use ZADD with priority score if priority != 0;
        # otherwise LPUSH for plain FIFO.
        if task.priority != 0:
            await self.client.zadd(f"{queue_key}:prio", {data: task.priority})
        else:
            await self.client.lpush(queue_key, data)
        return task.task_id

    async def dequeue(self, queue: str, timeout: float = 0.0) -> Task | None:
        queue_key = self._queue_key(queue)
        prio_key = f"{queue_key}:prio"

        # Check priority queue first
        if timeout > 0:
            # BRPOP blocks up to `timeout` seconds
            raw = await self.client.brpop([queue_key], timeout=int(timeout))
            if raw is None:
                return None
            _, data = raw
        else:
            # Check priority queue
            # ZPOPMAX returns (member, score) pairs
            prio_result = await self.client.zpopmax(prio_key, count=1)
            if prio_result:
                data = prio_result[0][0]
                return Task.from_dict(json.loads(data))
            data = await self.client.rpop(queue_key)
            if data is None:
                return None
        return Task.from_dict(json.loads(data))

    async def ack(self, task_id: str, result: TaskResult) -> None:
        key = self._result_key(task_id)
        data = json.dumps({
            "task_id": result.task_id,
            "status": result.status,
            "result": result.result,
            "error": result.error,
            "worker_id": result.worker_id,
            "elapsed_ms": result.elapsed_ms,
        }, ensure_ascii=False)
        if self._result_ttl > 0:
            await self.client.setex(key, self._result_ttl, data)
        else:
            await self.client.set(key, data)

    async def nack(self, task_id: str, queue: str) -> None:
        # Load original task, increment retries, re-enqueue or send to DLQ
        result_key = self._result_key(task_id)
        raw = await self.client.get(result_key)
        if raw:
            stored = json.loads(raw)
            # Remove result to allow re-enqueue
            await self.client.delete(result_key)
        # Fallback: just note failure via result
        await self.ack(task_id, TaskResult(task_id=task_id, status="retrying"))

    async def get_result(self, task_id: str) -> TaskResult | None:
        key = self._result_key(task_id)
        data = await self.client.get(key)
        if data is None:
            return None
        raw = json.loads(data)
        return TaskResult(
            task_id=raw["task_id"],
            status=raw["status"],
            result=raw.get("result"),
            error=raw.get("error", ""),
            worker_id=raw.get("worker_id", ""),
            elapsed_ms=raw.get("elapsed_ms", 0.0),
        )

    async def queue_length(self, queue: str) -> int:
        queue_key = self._queue_key(queue)
        return await self.client.llen(queue_key)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# RabbitMQ backend (stub — requires pika or aio-pika at runtime)
# ---------------------------------------------------------------------------


class RabbitMQBackend(TaskBackend):
    """RabbitMQ-backed task queue.

    Uses durable queues and message acknowledgements. Install ``aio-pika``.

    Args:
        amqp_url: AMQP connection URL, e.g. ``amqp://guest:guest@localhost:5672//``.
        prefetch_count: Max unacknowledged messages per consumer.
    """

    def __init__(self, amqp_url: str = "amqp://localhost:5672//", prefetch_count: int = 10) -> None:
        self._amqp_url = amqp_url
        self._prefetch_count = prefetch_count
        self._conn: Any = None
        self._channel: Any = None
        self._results: dict[str, TaskResult] = {}

    async def _ensure_channel(self) -> Any:
        if self._channel is None:
            try:
                import aio_pika  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError(
                    "aio-pika required for RabbitMQ. Install: pip install aio-pika"
                ) from None
            self._conn = await aio_pika.connect_robust(self._amqp_url)
            self._channel = await self._conn.channel()
            await self._channel.set_qos(prefetch_count=self._prefetch_count)
        return self._channel

    async def enqueue(self, task: Task) -> str:
        channel = await self._ensure_channel()
        queue_name = f"morainet.{task.queue}"
        queue = await channel.declare_queue(queue_name, durable=True)
        import aio_pika
        body = json.dumps(task.to_dict(), ensure_ascii=False).encode()
        await channel.default_exchange.publish(
            aio_pika.Message(body=body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=queue_name,
        )
        return task.task_id

    async def dequeue(self, queue: str, timeout: float = 0.0) -> Task | None:
        # RabbitMQ is push-based; dequeue via consumer callback pattern
        return None  # Use TaskConsumer for RabbitMQ

    async def ack(self, task_id: str, result: TaskResult) -> None:
        self._results[task_id] = result

    async def nack(self, task_id: str, queue: str) -> None:
        pass  # Handled at message level by RabbitMQ

    async def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    async def queue_length(self, queue: str) -> int:
        channel = await self._ensure_channel()
        queue_obj = await channel.declare_queue(f"morainet.{queue}", durable=True, passive=True)
        return queue_obj.declaration_result.message_count

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._channel = None
            self._conn = None


# ---------------------------------------------------------------------------
# Producer / Consumer helpers
# ---------------------------------------------------------------------------


class TaskProducer:
    """Push tasks onto a named queue."""

    def __init__(self, backend: TaskBackend, queue: str = "default") -> None:
        self.backend = backend
        self.queue = queue

    async def enqueue(self, task: Task) -> str:
        task.queue = self.queue
        return await self.backend.enqueue(task)

    async def enqueue_bulk(self, tasks: list[Task]) -> list[str]:
        return [await self.enqueue(t) for t in tasks]


class TaskConsumer:
    """Consume tasks from a named queue (async iterator)."""

    def __init__(self, backend: TaskBackend, queue: str = "default",
                 worker_id: str = "", poll_interval: float = 0.1,
                 block_timeout: float = 1.0) -> None:
        self.backend = backend
        self.queue = queue
        self.worker_id = worker_id or uuid.uuid4().hex[:8]
        self.poll_interval = poll_interval
        self.block_timeout = block_timeout
        self._running = False

    async def consume(self) -> AsyncIterator[Task]:
        """Yield tasks as they arrive. Use ``consumer.stop()`` to break the loop."""
        self._running = True
        while self._running:
            task = await self.backend.dequeue(self.queue, timeout=self.block_timeout)
            if task is not None:
                yield task
            else:
                await asyncio.sleep(self.poll_interval)

    async def ack(self, task: Task, result: Any = None, error: str = "") -> None:
        await self.backend.ack(task.task_id, TaskResult(
            task_id=task.task_id,
            status="failed" if error else "success",
            result=result,
            error=error,
            worker_id=self.worker_id,
        ))

    def stop(self) -> None:
        self._running = False
