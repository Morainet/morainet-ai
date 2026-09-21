"""Offline tests for the multi-agent pool (fake factory, no real agents).

Covers prewarm, acquire/release, selection strategies, idle retirement,
parallel execution, and stats — all with an in-memory fake factory.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from morainet.multiagent.pool import AgentPool, PoolConfig, PoolStrategy


class FakeAgent:
    def __init__(self, agent_id: str, system_prompt: str = "sys") -> None:
        self.agent_id = agent_id
        self.system_prompt = system_prompt

    async def arun(self, query: str):
        return SimpleNamespace(final_answer=f"ans:{query}")


class FakeFactory:
    def __init__(self) -> None:
        self._active: list = []
        self.destroyed: list[str] = []
        self._n = 0

    def spawn(self, role: str, *, parent_id: str = "", **kw) -> FakeAgent:
        self._n += 1
        aid = f"{role}_{self._n}"
        agent = FakeAgent(aid, system_prompt=f"{role} prompt")
        self._active.append(
            SimpleNamespace(agent_id=aid, agent=agent, created_at=time.time(), task_count=0)
        )
        return agent

    def list_active(self) -> list:
        return list(self._active)

    def destroy(self, agent_id: str) -> bool:
        self.destroyed.append(agent_id)
        self._active = [a for a in self._active if a.agent_id != agent_id]
        return True


def _pool(**kw) -> AgentPool:
    return AgentPool(FakeFactory(), "coder", PoolConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_pool_config_defaults():
    cfg = PoolConfig()
    assert cfg.min_size == 2
    assert cfg.max_size == 10
    assert cfg.strategy is PoolStrategy.LEAST_BUSY
    assert cfg.prewarm is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_start_prewarms_to_min_size():
    pool = _pool(min_size=3, max_size=5, prewarm=True)
    await pool.start()
    assert pool.stats["idle"] == 3
    assert len(pool) == 3


async def test_start_no_prewarm():
    pool = _pool(min_size=2, prewarm=False)
    await pool.start()
    assert pool.stats["idle"] == 0


async def test_stop_destroys_all():
    factory = FakeFactory()
    pool = AgentPool(factory, "coder", PoolConfig(min_size=2, max_size=5))
    await pool.start()
    await pool.acquire()
    await pool.stop()
    assert len(factory.destroyed) == 2
    assert len(pool) == 0


# ---------------------------------------------------------------------------
# Acquire / release
# ---------------------------------------------------------------------------


async def test_acquire_from_idle():
    pool = _pool(min_size=2, max_size=5)
    await pool.start()
    agent = await pool.acquire()
    assert isinstance(agent, FakeAgent)
    assert pool.stats["busy"] == 1
    assert pool.stats["idle"] == 1


async def test_acquire_creates_new_when_empty():
    pool = _pool(min_size=0, max_size=5)
    agent = await pool.acquire()
    assert isinstance(agent, FakeAgent)
    assert pool.stats["busy"] == 1
    assert pool.stats["created"] == 1


async def test_release_returns_agent_to_idle():
    pool = _pool(min_size=0, max_size=5)
    agent = await pool.acquire()
    await pool.release(agent.agent_id)
    assert pool.stats["idle"] == 1
    assert pool.stats["busy"] == 0


async def test_release_unknown_is_noop():
    pool = _pool(min_size=0)
    await pool.release("nope")
    assert pool.stats["idle"] == 0


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def test_pick_strategies():
    pool = _pool(min_size=0)
    a1 = SimpleNamespace(agent_id="1", task_count=5)
    a2 = SimpleNamespace(agent_id="2", task_count=1)

    pool._idle = [a1, a2]
    pool.config.strategy = PoolStrategy.FIRST_AVAILABLE
    assert pool._pick() is a1

    pool._idle = [a1, a2]
    pool.config.strategy = PoolStrategy.LEAST_BUSY
    assert pool._pick() is a2

    pool._idle = [a1, a2]
    pool.config.strategy = PoolStrategy.RANDOM
    assert pool._pick() in (a1, a2)

    pool._idle = [a1, a2]
    pool.config.strategy = PoolStrategy.ROUND_ROBIN
    assert pool._pick() in (a1, a2)

    pool._idle = []
    assert pool._pick() is None


# ---------------------------------------------------------------------------
# Idle retirement
# ---------------------------------------------------------------------------


def test_retire_stale():
    factory = FakeFactory()
    pool = AgentPool(factory, "coder", PoolConfig(min_size=1, max_size=5, idle_timeout=0.001))
    old = SimpleNamespace(agent_id="old", created_at=time.time() - 100, task_count=0)
    new = SimpleNamespace(agent_id="new", created_at=time.time(), task_count=0)
    pool._idle = [old, new]
    pool._retire_stale()
    assert "old" in factory.destroyed
    assert [a.agent_id for a in pool._idle] == ["new"]


def test_retire_stale_disabled():
    pool = _pool(min_size=0, idle_timeout=0)
    old = SimpleNamespace(agent_id="old", created_at=time.time() - 100, task_count=0)
    pool._idle = [old]
    pool._retire_stale()
    assert pool._idle == [old]


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


async def test_execute_all():
    pool = _pool(min_size=0, max_size=3)
    results = await pool.execute_all(["q1", "q2"])
    assert len(results) == 2
    assert results[0]["result"] == "ans:q1"
    assert results[0]["error"] is None


async def test_execute_all_records_error(monkeypatch):
    pool = _pool(min_size=0, max_size=2)

    async def _bad_acquire():
        return FakeAgent("x")

    monkeypatch.setattr(pool, "acquire", _bad_acquire)

    async def _raise(self, query: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeAgent, "arun", _raise)
    results = await pool.execute_all(["q"])
    assert results[0]["error"] == "boom"
