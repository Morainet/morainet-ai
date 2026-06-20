"""Distributed DAG workflow scheduler — cross-machine parallel execution.

Extends the existing workflow scheduler system so that DAG nodes can be
dispatched to remote workers via a :class:`TaskBackend`. Each node callable
is serialised as a :class:`TaskEnvelope`; workers dequeue, execute, and
report results. The coordinator assembles the final context.

Usage::

    from morainet.distributed import (
        DistributedParallelScheduler, RedisBackend,
    )
    from morainet.workflow import Workflow

    backend = RedisBackend(redis_url="redis://localhost:6379/0")
    wf = Workflow()
    wf.add_node("fetch", fetch_data).add_node("parse", parse_data)
    wf.connect("fetch", "parse")

    scheduler = DistributedParallelScheduler(backend, queue="wf-tasks")
    result = await scheduler.run(wf, {"url": "https://example.com"})
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from morainet.distributed.task_queue import TaskBackend, TaskProducer, TaskConsumer, Task, TaskResult
from morainet.exceptions import MorainetError
from morainet.workflow.dag_scheduler import (
    NodeProgress,
    Scheduler,
    SchedulerProgress,
)


# ---------------------------------------------------------------------------
# Task envelope
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TaskEnvelope:
    """Serializable wrapper for a workflow node execution."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    node_name: str = ""
    workflow_id: str = ""
    function_module: str = ""   # fully-qualified import path + qualname
    function_code: str = ""     # base64(pickle) fallback for non-importable funcs
    context: dict[str, Any] = field(default_factory=dict)  # serializable subset
    deps: list[str] = field(default_factory=list)
    workflow_level: int = 0
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_name": self.node_name,
            "workflow_id": self.workflow_id,
            "function_module": self.function_module,
            "function_code": self.function_code,
            "context": self.context,
            "deps": self.deps,
            "workflow_level": self.workflow_level,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskEnvelope":
        return cls(
            task_id=data["task_id"],
            node_name=data.get("node_name", ""),
            workflow_id=data.get("workflow_id", ""),
            function_module=data.get("function_module", ""),
            function_code=data.get("function_code", ""),
            context=data.get("context", {}),
            deps=data.get("deps", []),
            workflow_level=data.get("workflow_level", 0),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            created_at=data.get("created_at", time.time()),
            status=TaskStatus(data.get("status", "pending")),
        )

    @classmethod
    def from_node(cls, workflow_id: str, node: Any, context: dict[str, Any],
                  level: int = 0) -> "TaskEnvelope":
        """Create an envelope from a workflow node and current context."""
        func = node.func
        qualname = getattr(func, "__qualname__", func.__name__)
        module = getattr(func, "__module__", "")

        # Try to serialise via import path; fallback to pickle
        func_code = ""
        try:
            import cloudpickle  # type: ignore[import-untyped]
            func_code = base64.b64encode(cloudpickle.dumps(func)).decode()
        except ImportError:
            pass

        return cls(
            node_name=node.name,
            workflow_id=workflow_id,
            function_module=f"{module}.{qualname}" if module else qualname,
            function_code=func_code,
            context=_serialize_context(context),
            deps=list(node.deps),
            workflow_level=level,
        )

    def restore_function(self) -> Any:
        """Try to restore the callable from its serialised form."""
        # Prefer import
        if self.function_module:
            parts = self.function_module.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    import importlib
                    mod = importlib.import_module(parts[0])
                    return getattr(mod, parts[1])
                except Exception:
                    pass
        # Fallback: pickle
        if self.function_code:
            try:
                import cloudpickle
                return cloudpickle.loads(base64.b64decode(self.function_code))
            except Exception:
                pass
        raise MorainetError(
            f"Cannot restore function for node '{self.node_name}'. "
            f"Ensure cloudpickle is installed or the function is importable."
        )


def _serialize_context(context: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-safe subset of the context. Non-serializable values
    are represented as ``"<non-serializable: TypeName>"``."""
    safe: dict[str, Any] = {}
    for k, v in context.items():
        try:
            json.dumps(v)  # quick check
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = f"<non-serializable: {type(v).__name__}>"
    return safe


# ---------------------------------------------------------------------------
# Distributed scheduler
# ---------------------------------------------------------------------------


class DistributedScheduler(Scheduler):
    """Base class for distributed DAG schedulers.

    Args:
        backend: Task queue backend (Redis or RabbitMQ).
        queue: Queue name for task dispatch.
        workflow_id: Unique ID for this workflow run.
        worker_timeout: Seconds before a task is considered timed out.
    """

    def __init__(
        self,
        backend: TaskBackend,
        queue: str = "morainet:wf",
        workflow_id: str = "",
        worker_timeout: float = 300.0,
    ) -> None:
        self.backend = backend
        self.queue = queue
        self.workflow_id = workflow_id or uuid.uuid4().hex
        self.worker_timeout = worker_timeout
        self._producer = TaskProducer(backend, queue)

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a workflow DAG across distributed workers.

        Subclasses must implement `_dispatch_level`.
        """
        raise NotImplementedError


class DistributedParallelScheduler(DistributedScheduler):
    """Dispatch each DAG level to workers in parallel, collect results.

    Same semantics as :class:`ParallelScheduler`, but each node runs on
    a *different worker* (potentially on a different machine).

    Args:
        backend: Task queue backend.
        queue: Queue name.
        workflow_id: Unique workflow run ID.
        worker_timeout: Per-task timeout.
        poll_interval: Interval to check for completed tasks.
    """

    def __init__(
        self,
        backend: TaskBackend,
        queue: str = "morainet:wf",
        workflow_id: str = "",
        worker_timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> None:
        super().__init__(backend, queue, workflow_id, worker_timeout)
        self.poll_interval = poll_interval

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = dict(inputs)

        for level_idx, level in enumerate(workflow.topological_levels()):
            # Filter out nodes whose dependencies are already satisfied by context
            tasks = []
            for name in level:
                node = workflow.nodes[name]
                # Skip if all deps already in context (for mixed local/remote)
                if all(d in context for d in node.deps):
                    envelope = TaskEnvelope.from_node(
                        self.workflow_id, node, context, level=level_idx
                    )
                    tasks.append((name, envelope))

            if not tasks:
                continue

            # Enqueue all tasks for this level
            task_ids: dict[str, str] = {}
            for name, envelope in tasks:
                t = Task(
                    task_id=envelope.task_id,
                    payload={"envelope": envelope.to_dict()},
                    queue=self.queue,
                    max_retries=envelope.max_retries,
                    timeout=self.worker_timeout,
                )
                task_ids[name] = await self._producer.enqueue(t)

            # Wait for all tasks in this level
            results = await asyncio.gather(*(
                self._wait_for_task(tid, name)
                for name, tid in task_ids.items()
            ), return_exceptions=True)

            for (name, _), result in zip(tasks, results, strict=True):
                if isinstance(result, Exception):
                    raise MorainetError(f"Node '{name}' failed: {result}") from result
                if result is not None:
                    context[name] = result

        return context

    async def _wait_for_task(self, task_id: str, node_name: str) -> Any:
        """Poll until a task result is available, then return the value."""
        deadline = time.time() + self.worker_timeout
        while time.time() < deadline:
            result = await self.backend.get_result(task_id)
            if result is not None:
                if result.status == "success":
                    return result.result
                raise MorainetError(
                    f"Node '{node_name}' task failed: {result.error}"
                )
            await asyncio.sleep(self.poll_interval)
        raise MorainetError(f"Node '{node_name}' timed out after {self.worker_timeout}s")


class DistributedProgressScheduler(DistributedParallelScheduler):
    """Distributed parallel scheduler with per-node progress tracking."""

    def __init__(
        self,
        backend: TaskBackend,
        queue: str = "morainet:wf",
        workflow_id: str = "",
        worker_timeout: float = 300.0,
        poll_interval: float = 0.5,
        on_progress: Any = None,
    ) -> None:
        super().__init__(backend, queue, workflow_id, worker_timeout, poll_interval)
        self.on_progress = on_progress
        self.progress = SchedulerProgress()

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        self.progress = SchedulerProgress()
        total = sum(len(level) for level in workflow.topological_levels())
        self.progress.total = total

        context: dict[str, Any] = dict(inputs)

        for level_idx, level in enumerate(workflow.topological_levels()):
            tasks = []
            for name in level:
                node = workflow.nodes[name]
                if all(d in context for d in node.deps):
                    envelope = TaskEnvelope.from_node(
                        self.workflow_id, node, context, level=level_idx
                    )
                    tasks.append((name, envelope))
                    self.progress.nodes[name] = NodeProgress(name=name, status="pending")

            if not tasks:
                continue

            task_ids: dict[str, str] = {}
            for name, envelope in tasks:
                self.progress.nodes[name].status = "running"
                self.progress.nodes[name].started_at = time.time()
                self.progress.running += 1
                self._notify()
                t = Task(
                    task_id=envelope.task_id,
                    payload={"envelope": envelope.to_dict()},
                    queue=self.queue,
                    max_retries=envelope.max_retries,
                    timeout=self.worker_timeout,
                )
                task_ids[name] = await self._producer.enqueue(t)

            results = await asyncio.gather(*(
                self._wait_and_track(tid, name)
                for name, tid in task_ids.items()
            ), return_exceptions=True)

            for (name, _), result in zip(tasks, results, strict=True):
                np = self.progress.nodes[name]
                if isinstance(result, Exception):
                    np.status = "failed"
                    np.error = str(result)
                    np.finished_at = time.time()
                    self.progress.failed += 1
                    self.progress.running -= 1
                    self._notify()
                    raise MorainetError(f"Node '{name}' failed: {result}") from result

                np.status = "success"
                np.result = result
                np.finished_at = time.time()
                self.progress.completed += 1
                self.progress.running -= 1
                context[name] = result
                self._notify()

        return context

    async def _wait_and_track(self, task_id: str, node_name: str) -> Any:
        result = await self._wait_for_task(task_id, node_name)
        return result

    def _notify(self) -> None:
        if self.on_progress and callable(self.on_progress):
            self.on_progress(self.progress)


# ---------------------------------------------------------------------------
# Worker-side executor
# ---------------------------------------------------------------------------


class DistributedNodeExecutor:
    """Runs on worker nodes: dequeues tasks, executes node functions, reports results.

    Usage (worker process)::

        backend = RedisBackend(redis_url="redis://localhost:6379/0")
        executor = DistributedNodeExecutor(backend, "morainet:wf", worker_id="worker-1")
        await executor.serve()  # blocking
    """

    def __init__(
        self,
        backend: TaskBackend,
        queue: str = "morainet:wf",
        worker_id: str = "",
    ) -> None:
        self.backend = backend
        self.queue = queue
        self.worker_id = worker_id or uuid.uuid4().hex[:8]
        self._consumer = TaskConsumer(backend, queue, worker_id=self.worker_id)
        self._running = False

    async def serve(self) -> None:
        """Start the worker loop — blocks until stopped."""
        self._running = True
        async for task in self._consumer.consume():
            if not self._running:
                break
            try:
                result = await self._execute(task)
                await self._consumer.ack(task, result=result)
            except Exception as exc:
                await self._consumer.ack(task, error=str(exc))

    async def _execute(self, task: Task) -> Any:
        envelope_dict = task.payload.get("envelope", {})
        envelope = TaskEnvelope.from_dict(envelope_dict)
        func = envelope.restore_function()
        result = func(envelope.context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def stop(self) -> None:
        self._running = False
        self._consumer.stop()
