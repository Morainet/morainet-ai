"""Offline tests for the team topologies (fake scripted agents).

Covers HierarchicalTeam, ReviewTeam, DebateTeam, SharedMemoryPool, and the
Router paths not exercised by test_multiagent_topology.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from morainet.multiagent.topologies import (
    DebateTeam,
    HierarchicalTeam,
    ReviewTeam,
    Route,
    Router,
    SharedMemoryPool,
    SubTask,
    TeamStatus,
)


class FakeAgent:
    """Scripted agent: returns queued strings, or raises queued exceptions."""

    def __init__(self, *contents, provider=None) -> None:
        self._contents = list(contents)
        self._i = 0
        self.provider = provider
        self.system_prompt = ""

    async def arun(self, prompt):
        if self._i < len(self._contents):
            content = self._contents[self._i]
            self._i += 1
            if isinstance(content, Exception):
                raise content
            return SimpleNamespace(final_answer=content)
        raise RuntimeError("no scripted response left")


# ---------------------------------------------------------------------------
# HierarchicalTeam
# ---------------------------------------------------------------------------


async def test_hierarchical_decompose_and_synthesize():
    orch = FakeAgent("coder | write code\nwriter | write docs", "final synth")
    team = HierarchicalTeam(
        orchestrator=orch,
        specialists={"coder": FakeAgent("code out"), "writer": FakeAgent("docs out")},
    )
    result = await team.run("build")
    assert result.status is TeamStatus.SUCCESS
    assert result.final_answer == "final synth"
    assert result.contributor_count >= 3


async def test_hierarchical_explicit_subtasks():
    orch = FakeAgent("synth")
    team = HierarchicalTeam(orchestrator=orch, specialists={"coder": FakeAgent("c")})
    result = await team.run(
        "task", sub_tasks=[SubTask(description="d", specialist_role="coder")]
    )
    assert result.final_answer == "synth"


async def test_hierarchical_no_subtasks():
    orch = FakeAgent("nothing useful")
    team = HierarchicalTeam(orchestrator=orch, specialists={})
    result = await team.run("task")
    assert "No sub-tasks executed." in result.final_answer


async def test_hierarchical_synthesis_failure_falls_back():
    orch = FakeAgent("coder | do it", RuntimeError("synth fail"))
    team = HierarchicalTeam(orchestrator=orch, specialists={"coder": FakeAgent("code out")})
    result = await team.run("task")
    assert "code out" in result.final_answer


# ---------------------------------------------------------------------------
# ReviewTeam
# ---------------------------------------------------------------------------


async def test_review_approved_first_cycle():
    team = ReviewTeam(
        producer=FakeAgent("draft1"), reviewers=[FakeAgent("APPROVED looks good")], max_cycles=3
    )
    result = await team.run("task")
    assert result.status is TeamStatus.SUCCESS
    assert result.final_answer == "draft1"


async def test_review_revise_then_approve():
    team = ReviewTeam(
        producer=FakeAgent("d1", "d2"),
        reviewers=[FakeAgent("needs work", "APPROVED")],
        max_cycles=3,
    )
    result = await team.run("task")
    assert result.final_answer == "d2"


async def test_review_auto_approve_after():
    team = ReviewTeam(
        producer=FakeAgent("d1", "d2", "d3"),
        reviewers=[FakeAgent("nope", "nope", "nope")],
        max_cycles=5,
        auto_approve_after=1,
    )
    result = await team.run("task")
    assert result.status is TeamStatus.SUCCESS


async def test_review_producer_failure_returns_failed():
    team = ReviewTeam(
        producer=FakeAgent(RuntimeError("boom")),
        reviewers=[FakeAgent("APPROVED")],
        max_cycles=1,
    )
    result = await team.run("task")
    assert result.status is TeamStatus.FAILED


async def test_review_reviewer_error_is_not_approved():
    team = ReviewTeam(
        producer=FakeAgent("d1", "d2"),
        reviewers=[FakeAgent(RuntimeError("reviewer down"))],
        max_cycles=1,
    )
    result = await team.run("task")
    assert result.status is TeamStatus.SUCCESS


# ---------------------------------------------------------------------------
# DebateTeam
# ---------------------------------------------------------------------------


async def test_debate_team_arun():
    team = DebateTeam(
        arbiter=FakeAgent("verdict"),
        debaters=[FakeAgent("a1"), FakeAgent("b1")],
        rounds=1,
    )
    result = await team.arun("topic")
    assert result.final_answer == "verdict"
    assert result.contributor_count >= 2


# ---------------------------------------------------------------------------
# SharedMemoryPool
# ---------------------------------------------------------------------------


async def test_shared_memory_pool_runs_all_agents():
    pool = SharedMemoryPool(agents={"a": FakeAgent("ra"), "b": FakeAgent("rb")})
    result = await pool.run("task")
    assert result.status is TeamStatus.SUCCESS
    assert "ra" in result.final_answer
    assert "rb" in result.final_answer


async def test_shared_memory_pool_subset():
    pool = SharedMemoryPool(agents={"a": FakeAgent("ra"), "b": FakeAgent("rb")})
    result = await pool.run("task", agent_names=["a"])
    assert "ra" in result.final_answer
    assert "rb" not in result.final_answer


async def test_shared_memory_pool_agent_error():
    pool = SharedMemoryPool(agents={"a": FakeAgent(RuntimeError("boom"))})
    result = await pool.run("task")
    assert "ERROR" in result.final_answer


async def test_shared_memory_pool_without_bus():
    pool = SharedMemoryPool(agents={"a": FakeAgent("ra")}, enable_bus=False)
    assert pool.bus is None
    result = await pool.run("task")
    assert "ra" in result.final_answer


async def test_shared_memory_pool_close():
    pool = SharedMemoryPool(agents={"a": FakeAgent("ra")})
    await pool.close()


# ---------------------------------------------------------------------------
# Router (remaining paths)
# ---------------------------------------------------------------------------


async def test_router_async_selector():
    async def selector(query: str) -> str:
        return "b"

    router = Router(
        routes=[Route(name="a", agent=FakeAgent("ra")), Route(name="b", agent=FakeAgent("rb"))],
        selector=selector,
    )
    result = await router.arun("q")
    assert result.route == "b"
    assert result.final == "rb"


async def test_router_agent_error_captured():
    router = Router(
        routes=[Route(name="a", agent=FakeAgent(RuntimeError("boom")))],
        selector=lambda q: "a",
    )
    result = await router.arun("q")
    assert "Error" in result.final


async def test_router_explicit_default():
    router = Router(
        routes=[Route(name="a", agent=FakeAgent("ra")), Route(name="b", agent=FakeAgent("rb"))],
        selector=lambda q: "zzz",
        default="b",
    )
    result = await router.arun("q")
    assert result.route == "b"


async def test_router_provider_exception_falls_back():
    class BadProvider:
        async def chat(self, messages, tools=None, response_format=None):
            raise RuntimeError("provider down")

    router = Router(routes=[Route(name="a", agent=FakeAgent("ra"))], provider=BadProvider())
    result = await router.arun("q")
    assert result.route == "a"
    assert result.final == "ra"
