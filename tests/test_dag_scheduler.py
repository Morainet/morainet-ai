"""Offline tests for the DAG scheduler plugins (no external services).

Covers :class:`SerialScheduler`, :class:`ParallelScheduler` (concurrency,
timeout, retry), :class:`ProgressScheduler` (progress callbacks), and the
:class:`SchedulerRegistry` / module-level factories.
"""

from __future__ import annotations

import asyncio

import pytest

from morainet.exceptions import MorainetError
from morainet.workflow import Workflow
from morainet.workflow.dag_scheduler import (
    NodeProgress,
    ParallelScheduler,
    ProgressScheduler,
    Scheduler,
    SchedulerProgress,
    SchedulerRegistry,
    SerialScheduler,
    get_scheduler,
    register_scheduler,
)


def _boom(ctx):  # noqa: ANN001
    raise ValueError("boom")


# ---------------------------------------------------------------------------
# Progress dataclasses
# ---------------------------------------------------------------------------


def test_scheduler_progress_pct_empty():
    assert SchedulerProgress().progress_pct == 0.0


def test_scheduler_progress_pct():
    assert SchedulerProgress(total=4, completed=2).progress_pct == 50.0


def test_node_progress_defaults():
    np = NodeProgress(name="n")
    assert np.name == "n"
    assert np.status == "pending"
    assert np.retries == 0


# ---------------------------------------------------------------------------
# SerialScheduler
# ---------------------------------------------------------------------------


async def test_serial_scheduler_runs_in_order():
    wf = Workflow()
    wf.add_node("a", lambda ctx: ctx["x"] + 1)
    wf.add_node("b", lambda ctx: ctx["a"] * 2)
    wf.connect("a", "b")
    out = await SerialScheduler().run(wf, {"x": 1})
    assert out["a"] == 2
    assert out["b"] == 4


# ---------------------------------------------------------------------------
# ParallelScheduler
# ---------------------------------------------------------------------------


async def test_parallel_scheduler_runs_independent_nodes():
    wf = Workflow()
    wf.add_node("root", lambda ctx: 1)
    wf.add_node("left", lambda ctx: ctx["root"] + 1)
    wf.add_node("right", lambda ctx: ctx["root"] + 2)
    wf.connect("root", "left")
    wf.connect("root", "right")
    out = await ParallelScheduler().run(wf, {})
    assert out["left"] == 2
    assert out["right"] == 3


async def test_parallel_scheduler_max_workers_runs():
    wf = Workflow()
    wf.add_node("a", lambda ctx: "a")
    wf.add_node("b", lambda ctx: "b")
    wf.add_node("c", lambda ctx: "c")
    out = await ParallelScheduler(max_workers=1).run(wf, {})
    assert set(out) >= {"a", "b", "c"}


async def test_parallel_scheduler_timeout():
    async def slow(ctx):  # noqa: ANN001
        await asyncio.sleep(1)
        return "done"

    wf = Workflow()
    wf.add_node("s", slow)
    with pytest.raises(asyncio.TimeoutError):
        await ParallelScheduler(timeout=0.01).run(wf, {})


async def test_parallel_scheduler_retry_then_success():
    calls = {"n": 0}

    async def flaky(ctx):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    wf = Workflow()
    wf.add_node("f", flaky)
    out = await ParallelScheduler(retry_count=3, retry_delay=0.0).run(wf, {})
    assert out["f"] == "ok"
    assert calls["n"] == 3


async def test_parallel_scheduler_retry_exhausted_raises():
    wf = Workflow()
    wf.add_node("f", _boom)
    with pytest.raises(MorainetError):
        await ParallelScheduler(retry_count=2, retry_delay=0.0).run(wf, {})


async def test_parallel_scheduler_node_failure_raises():
    wf = Workflow()
    wf.add_node("f", _boom)
    with pytest.raises(MorainetError):
        await ParallelScheduler().run(wf, {})


# ---------------------------------------------------------------------------
# ProgressScheduler
# ---------------------------------------------------------------------------


async def test_progress_scheduler_tracks_progress():
    events: list[int] = []
    wf = Workflow()
    wf.add_node("a", lambda ctx: 1)
    wf.add_node("b", lambda ctx: ctx["a"] + 1)
    wf.connect("a", "b")
    sched = ProgressScheduler(on_progress=lambda p: events.append(p.completed))
    out = await sched.run(wf, {})
    assert out["b"] == 2
    assert sched.progress.total == 2
    assert sched.progress.completed == 2
    assert sched.progress.progress_pct == 100.0
    assert events  # on_progress fired


async def test_progress_scheduler_failure():
    wf = Workflow()
    wf.add_node("f", _boom)
    sched = ProgressScheduler()
    with pytest.raises(MorainetError):
        await sched.run(wf, {})
    assert sched.progress.failed == 1
    assert sched.progress.nodes["f"].status == "failed"


# ---------------------------------------------------------------------------
# SchedulerRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    reg = SchedulerRegistry()
    reg.register("ser", SerialScheduler)
    assert reg.get("ser") is SerialScheduler


def test_registry_get_unknown_raises():
    reg = SchedulerRegistry()
    with pytest.raises(MorainetError):
        reg.get("nope")


def test_registry_create_class():
    reg = SchedulerRegistry()
    reg.register("par", ParallelScheduler)
    s = reg.create("par", max_workers=2)
    assert isinstance(s, ParallelScheduler)
    assert s.max_workers == 2


def test_registry_create_factory_returns_scheduler():
    reg = SchedulerRegistry()
    reg.register("factory", lambda **kw: SerialScheduler())
    assert isinstance(reg.create("factory"), SerialScheduler)


def test_registry_create_factory_returns_non_scheduler_raises():
    reg = SchedulerRegistry()
    reg.register("bad", lambda **kw: 42)
    with pytest.raises(MorainetError):
        reg.create("bad")


def test_registry_create_unknown_raises():
    reg = SchedulerRegistry()
    with pytest.raises(MorainetError):
        reg.create("nope")


def test_registry_names_len_bool():
    reg = SchedulerRegistry()
    assert len(reg) == 0
    assert not reg
    reg.register("a", SerialScheduler)
    reg.register("b", ParallelScheduler)
    assert reg.names() == ["a", "b"]
    assert len(reg) == 2
    assert reg


# ---------------------------------------------------------------------------
# Module-level factories
# ---------------------------------------------------------------------------


def test_module_get_scheduler_builtins():
    assert isinstance(get_scheduler("serial"), SerialScheduler)
    assert isinstance(get_scheduler("parallel"), ParallelScheduler)
    assert isinstance(get_scheduler("progress"), ProgressScheduler)


def test_module_get_scheduler_unknown_raises():
    with pytest.raises(MorainetError):
        get_scheduler("nope")


def test_register_scheduler_plugin():
    class MySched(Scheduler):
        async def run(self, workflow, inputs):  # noqa: ANN001
            return {}

    register_scheduler("my-custom", MySched)
    assert isinstance(get_scheduler("my-custom"), MySched)
