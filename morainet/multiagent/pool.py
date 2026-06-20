"""Agent pool: pre-warm and reuse agents across tasks.

Maintains a pool of idle agents. When a task arrives, an idle agent
is picked from the pool; after the task completes, the agent returns
to the pool. This avoids the cost of repeated agent creation/destruction.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from morainet.core.agent import Agent
from morainet.multiagent.factory import AgentFactory, SpawnedAgent


class PoolStrategy(str, Enum):
    """How agents are selected from the pool."""
    ROUND_ROBIN = "round_robin"       # cycle through agents in order
    LEAST_BUSY = "least_busy"         # pick the agent with the fewest tasks
    RANDOM = "random"                 # pick any idle agent
    FIRST_AVAILABLE = "first_available"  # first idle agent found


@dataclass
class PoolConfig:
    """Configuration for an AgentPool.

    Parameters
    ----------
    min_size:
        Minimum number of agents to keep warm at all times.
    max_size:
        Maximum number of agents in the pool.
    idle_timeout:
        Seconds before an idle agent is retired (destroyed).
    strategy:
        Agent selection strategy.
    prewarm:
        Whether to immediately create min_size agents on pool creation.
    """
    min_size: int = 2
    max_size: int = 10
    idle_timeout: float = 600.0
    strategy: PoolStrategy = PoolStrategy.LEAST_BUSY
    prewarm: bool = True


class AgentPool:
    """Pre-warm and reuse agents to amortize creation cost.

    Usage::

        pool = AgentPool(factory, "coder", PoolConfig(min_size=2, max_size=5))
        agent = await pool.acquire()
        try:
            result = await agent.arun("write a function")
        finally:
            await pool.release(agent.agent_id)
    """

    def __init__(
        self,
        factory: AgentFactory,
        role: str,
        config: PoolConfig | None = None,
    ) -> None:
        self.factory = factory
        self.role = role
        self.config = config or PoolConfig()
        self._idle: list[SpawnedAgent] = []
        self._busy: dict[str, SpawnedAgent] = {}
        self._lock = asyncio.Lock()
        self._created = 0
        self._destroyed = 0

    # -- lifecycle --

    async def start(self) -> None:
        """Prewarm the pool to min_size."""
        if not self.config.prewarm:
            return
        async with self._lock:
            while len(self._idle) < self.config.min_size:
                agent = self._create_agent()
                self._idle.append(agent)

    async def stop(self) -> None:
        """Destroy all agents in the pool."""
        async with self._lock:
            all_agents = self._idle + list(self._busy.values())
            for agent in all_agents:
                self.factory.destroy(agent.agent_id)
            self._destroyed += len(all_agents)
            self._idle.clear()
            self._busy.clear()

    # -- acquire / release --

    async def acquire(self) -> Agent:
        """Get an agent from the pool. Blocks if none available and at max size."""
        async with self._lock:
            # Retire stale idle agents
            self._retire_stale()

            # Pick from idle pool
            agent = self._pick()
            if agent is not None:
                self._idle = [a for a in self._idle if a.agent_id != agent.agent_id]
                self._busy[agent.agent_id] = agent
                agent.lifecycle = "busy"
                return agent.agent

            # Need to create a new one
            if len(self._idle) + len(self._busy) < self.config.max_size:
                agent = self._create_agent()
                self._busy[agent.agent_id] = agent
                agent.lifecycle = "busy"
                return agent.agent

        # Pool full — wait for a release
        while True:
            async with self._lock:
                agent = self._pick()
                if agent is not None:
                    self._idle = [a for a in self._idle if a.agent_id != agent.agent_id]
                    self._busy[agent.agent_id] = agent
                    agent.lifecycle = "busy"
                    return agent.agent
            await asyncio.sleep(0.1)

    async def release(self, agent_id: str) -> None:
        """Return an agent to the idle pool."""
        async with self._lock:
            agent = self._busy.pop(agent_id, None)
            if agent is None:
                return
            agent.lifecycle = "idle"
            agent.task_count += 1
            self._idle.append(agent)

    # -- parallel execution --

    async def execute_all(
        self,
        queries: list[str],
        max_parallel: int = 0,
    ) -> list[dict[str, Any]]:
        """Execute multiple queries across the pool concurrently.

        Parameters
        ----------
        queries:
            List of tasks to execute.
        max_parallel:
            Maximum concurrent agents. 0 = use full pool size.

        Returns
        -------
        list[dict]
            Each dict contains ``{"query": ..., "result": ..., "agent_id": ..., "error": ...}``.
        """
        max_workers = max_parallel or self.config.max_size
        semaphore = asyncio.Semaphore(max_workers)

        async def _run_one(query: str) -> dict[str, Any]:
            async with semaphore:
                try:
                    agent = await self.acquire()
                    try:
                        result = await agent.arun(query)
                        return {
                            "query": query,
                            "agent_id": agent.system_prompt[:40] if agent.system_prompt else "?",
                            "result": result.final_answer,
                            "error": None,
                        }
                    finally:
                        await self.release(agent.agent_id)
                except Exception as e:
                    return {"query": query, "agent_id": "", "result": "", "error": str(e)}

        return await asyncio.gather(*[_run_one(q) for q in queries])

    # -- internals --

    def _create_agent(self) -> SpawnedAgent:
        _agent = self.factory.spawn(self.role)
        self._created += 1
        # Get the most recently spawned agent from the factory
        active = self.factory.list_active()
        if active:
            return active[-1]
        raise RuntimeError("AgentFactory.spawn returned but no active agents found")

    def _pick(self) -> SpawnedAgent | None:
        if not self._idle:
            return None

        strategy = self.config.strategy
        if strategy == PoolStrategy.RANDOM:
            import random
            return random.choice(self._idle)
        elif strategy == PoolStrategy.LEAST_BUSY:
            return min(self._idle, key=lambda a: a.task_count)
        elif strategy == PoolStrategy.FIRST_AVAILABLE:
            return self._idle[0]
        elif strategy == PoolStrategy.ROUND_ROBIN:
            agent = self._idle[0]
            self._idle = self._idle[1:] + [agent]
            return agent
        return self._idle[0]

    def _retire_stale(self) -> None:
        """Remove agents that have been idle beyond the timeout."""
        if self.config.idle_timeout <= 0:
            return
        now = time.time()
        cutoff = now - self.config.idle_timeout
        to_retire = []
        for agent in self._idle:
            if agent.created_at < cutoff and len(self._idle) > self.config.min_size:
                to_retire.append(agent.agent_id)
        for agent_id in to_retire:
            self.factory.destroy(agent_id)
            self._destroyed += 1
            self._idle = [a for a in self._idle if a.agent_id != agent_id]

    # -- stats --

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "idle": len(self._idle),
            "busy": len(self._busy),
            "created": self._created,
            "destroyed": self._destroyed,
            "total": len(self._idle) + len(self._busy),
            "role": self.role,
        }

    def __len__(self) -> int:
        return len(self._idle) + len(self._busy)
