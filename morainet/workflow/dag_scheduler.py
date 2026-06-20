"""DAG scheduler with plugin-based node execution strategies.

Extends the Workflow DAG with pluggable schedulers that control:
- Execution strategy (serial / parallel / priority-based).
- Node-level concurrency limits and retry policies.
- Progress tracking and cancellation.
- Plugin registration so third-party schedulers can be discovered.

Usage::

    from morainet.workflow import Workflow, ParallelScheduler, register_scheduler

    wf = Workflow().add_node("fetch", fetch_func).add_node("parse", parse_func)
    wf.connect("fetch", "parse")

    scheduler = ParallelScheduler(max_workers=4)
    result = await scheduler.run(wf, {"url": "https://example.com"})

    # Register a custom scheduler as a plugin
    register_scheduler("my-scheduler", MyCustomScheduler)
"""

from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from morainet.exceptions import MorainetError


# -- Scheduler abstraction ---------------------------------------------------


class Scheduler(ABC):
    """Abstract DAG scheduler — pluggable execution strategy for workflows."""

    @abstractmethod
    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a workflow DAG with the given inputs."""
        ...


# -- Progress tracking -------------------------------------------------------


@dataclass
class NodeProgress:
    """Execution state for a single node."""

    name: str
    status: str = "pending"  # pending / running / success / failed / skipped
    started_at: float = 0.0
    finished_at: float = 0.0
    result: Any = None
    error: str | None = None
    retries: int = 0


@dataclass
class SchedulerProgress:
    """Aggregated progress for a workflow run."""

    total: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0
    nodes: dict[str, NodeProgress] = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100


# -- Built-in schedulers -----------------------------------------------------


class SerialScheduler(Scheduler):
    """Execute DAG nodes one at a time in topological order."""

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = dict(inputs)
        for level in workflow.topological_levels():
            for name in level:
                node = workflow.nodes[name]
                result = node.func(context)
                if inspect.isawaitable(result):
                    result = await result
                context[name] = result
        return context


class ParallelScheduler(Scheduler):
    """Execute independent DAG nodes in parallel within each level.

    Args:
        max_workers: Maximum concurrent node executions (0 = unlimited).
        timeout: Per-level timeout in seconds (0 = no timeout).
        retry_count: Number of retries on node failure.
        retry_delay: Delay between retries in seconds.
    """

    def __init__(self, max_workers: int = 0, timeout: float = 0.0,
                 retry_count: int = 0, retry_delay: float = 1.0) -> None:
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = dict(inputs)
        semaphore = asyncio.Semaphore(self.max_workers) if self.max_workers > 0 else None

        for level in workflow.topological_levels():
            tasks = []
            for name in level:
                node = workflow.nodes[name]
                task = self._execute_node(node, context, semaphore)
                tasks.append((name, task))

            if self.timeout > 0:
                results = await asyncio.wait_for(
                    asyncio.gather(*(t for _, t in tasks), return_exceptions=True),
                    timeout=self.timeout,
                )
            else:
                results = await asyncio.gather(
                    *(t for _, t in tasks), return_exceptions=True
                )

            for (name, _), result in zip(tasks, results, strict=True):
                if isinstance(result, Exception):
                    raise MorainetError(f"Node '{name}' failed: {result}") from result
                context[name] = result

        return context

    async def _execute_node(self, node: Any, context: dict[str, Any],
                            semaphore: asyncio.Semaphore | None) -> Any:
        if semaphore:
            async with semaphore:
                return await self._invoke_with_retry(node, context)
        return await self._invoke_with_retry(node, context)

    async def _invoke_with_retry(self, node: Any, context: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                result = node.func(context)
                if inspect.isawaitable(result):
                    return await result
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay)
        raise last_error  # type: ignore[misc]


class ProgressScheduler(ParallelScheduler):
    """Parallel scheduler that tracks progress for each node."""

    def __init__(self, max_workers: int = 0, timeout: float = 0.0,
                 retry_count: int = 0, retry_delay: float = 1.0,
                 on_progress: Callable[[SchedulerProgress], Any] | None = None) -> None:
        super().__init__(max_workers, timeout, retry_count, retry_delay)
        self.on_progress = on_progress
        self.progress = SchedulerProgress()

    async def run(self, workflow: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        self.progress = SchedulerProgress()
        node_count = 0
        for level in workflow.topological_levels():
            node_count += len(level)
        self.progress.total = node_count

        context: dict[str, Any] = dict(inputs)
        semaphore = asyncio.Semaphore(self.max_workers) if self.max_workers > 0 else None

        for level in workflow.topological_levels():
            tasks = []
            for name in level:
                node = workflow.nodes[name]
                self.progress.nodes[name] = NodeProgress(name=name, status="running")
                self.progress.running += 1
                self._notify()

                task = self._execute_with_progress(node, context, name, semaphore)
                tasks.append((name, task))

            if self.timeout > 0:
                results = await asyncio.wait_for(
                    asyncio.gather(*(t for _, t in tasks), return_exceptions=True),
                    timeout=self.timeout,
                )
            else:
                results = await asyncio.gather(
                    *(t for _, t in tasks), return_exceptions=True
                )

            for (name, _), result in zip(tasks, results, strict=True):
                node_progress = self.progress.nodes[name]
                if isinstance(result, Exception):
                    node_progress.status = "failed"
                    node_progress.error = str(result)
                    node_progress.finished_at = time.time()
                    self.progress.failed += 1
                    self.progress.running -= 1
                    self._notify()
                    raise MorainetError(f"Node '{name}' failed: {result}") from result

                node_progress.status = "success"
                node_progress.result = result
                node_progress.finished_at = time.time()
                self.progress.completed += 1
                self.progress.running -= 1
                context[name] = result
                self._notify()

        return context

    async def _execute_with_progress(self, node: Any, context: dict[str, Any],
                                     name: str, semaphore: asyncio.Semaphore | None) -> Any:
        self.progress.nodes[name].started_at = time.time()
        try:
            if semaphore:
                async with semaphore:
                    return await self._invoke_with_retry(node, context)
            return await self._invoke_with_retry(node, context)
        except Exception:
            self.progress.nodes[name].status = "failed"
            raise

    def _notify(self) -> None:
        if self.on_progress and callable(self.on_progress):
            self.on_progress(self.progress)


# -- Scheduler plugin registry -----------------------------------------------


class SchedulerRegistry:
    """Global registry for DAG scheduler plugins.

    Usage::

        registry = SchedulerRegistry()
        registry.register("parallel", ParallelScheduler)
        scheduler_cls = registry.get("parallel")
        scheduler = scheduler_cls(max_workers=8)
    """

    def __init__(self) -> None:
        self._schedulers: dict[str, type[Scheduler] | Callable[..., Scheduler]] = {}

    def register(self, name: str, scheduler: type[Scheduler] | Callable[..., Scheduler]) -> None:
        """Register a scheduler by name.

        ``scheduler`` can be a class (subclass of Scheduler) or a factory callable.
        """
        self._schedulers[name] = scheduler

    def get(self, name: str) -> type[Scheduler] | Callable[..., Scheduler]:
        """Get a registered scheduler by name."""
        if name not in self._schedulers:
            raise MorainetError(f"Scheduler '{name}' not registered. Available: {self.names()}")
        return self._schedulers[name]

    def create(self, name: str, **kwargs: Any) -> Scheduler:
        """Create a scheduler instance by name with keyword arguments."""
        sched = self.get(name)
        if isinstance(sched, type) and issubclass(sched, Scheduler):
            return sched(**kwargs)
        if callable(sched):
            result = sched(**kwargs)
            if isinstance(result, Scheduler):
                return result
            raise MorainetError(f"Scheduler factory for '{name}' did not return a Scheduler")
        raise MorainetError(f"Scheduler '{name}' is not a class or callable")

    def names(self) -> list[str]:
        return sorted(self._schedulers)

    def __len__(self) -> int:
        return len(self._schedulers)

    def __bool__(self) -> bool:
        return bool(self._schedulers)


# Process-wide default registry with built-in schedulers.
scheduler_registry = SchedulerRegistry()
scheduler_registry.register("serial", SerialScheduler)
scheduler_registry.register("parallel", ParallelScheduler)
scheduler_registry.register("progress", ProgressScheduler)


# Convenience functions
def register_scheduler(name: str, scheduler: type[Scheduler] | Callable[..., Scheduler]) -> None:
    """Register a custom scheduler plugin."""
    scheduler_registry.register(name, scheduler)


def get_scheduler(name: str) -> Scheduler:
    """Get a scheduler instance by name (returns the default parameterless version)."""
    return scheduler_registry.create(name)
