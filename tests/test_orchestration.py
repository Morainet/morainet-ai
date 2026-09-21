"""Offline tests for TeamOrchestrator (fake factory + patched topologies).

The orchestrator only spawns agents and delegates to a topology, so we inject a
fake factory and replace the topology classes with fakes — no real agents,
providers, or LLM calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import morainet.multiagent.orchestration as orch_mod
from morainet.multiagent.factory import AgentBlueprint
from morainet.multiagent.orchestration import TeamOrchestrator
from morainet.multiagent.topologies import TeamResult, TeamStatus


class FakeFactory:
    def __init__(self) -> None:
        self.blueprints: dict = {}
        self._active: list = []
        self.destroyed: list[str] = []
        self._n = 0

    def register_blueprint(self, name: str, blueprint: AgentBlueprint) -> None:
        self.blueprints[name] = blueprint

    def spawn(self, role: str, *, parent_id: str = "", **kw) -> SimpleNamespace:
        self._n += 1
        aid = f"{role}_{self._n}"
        self._active.append(SimpleNamespace(agent_id=aid))
        return SimpleNamespace(agent_id=aid, role=role)

    def list_active(self) -> list:
        return list(self._active)

    def destroy(self, agent_id: str) -> bool:
        self.destroyed.append(agent_id)
        self._active = [a for a in self._active if a.agent_id != agent_id]
        return True


class _FakeDebate:
    def __init__(self, arbiter, debaters, rounds) -> None:
        self.debaters = debaters
        self.rounds = rounds

    async def arun(self, topic: str) -> TeamResult:
        return TeamResult(status=TeamStatus.SUCCESS, final_answer=f"debate:{topic}", trace_id="t")


class _FakeReview:
    def __init__(self, producer, reviewers, max_cycles) -> None:
        self.reviewers = reviewers

    async def run(self, task: str, criteria: str) -> TeamResult:
        return TeamResult(final_answer=f"review:{task}", trace_id="t")


class _FakeHier:
    def __init__(self, orchestrator, specialists, auto_decompose=True) -> None:
        self.specialists = specialists

    async def run(self, task: str) -> TeamResult:
        return TeamResult(final_answer=f"delegate:{task}", trace_id="t")


class _FakePipeline:
    def __init__(self, stages) -> None:
        self.stages = stages

    async def arun(self, task: str) -> SimpleNamespace:
        return SimpleNamespace(final=f"pipe:{task}")


class _FakeGroupChat:
    def __init__(self, members, max_rounds) -> None:
        self.members = members

    async def arun(self, topic: str) -> SimpleNamespace:
        return SimpleNamespace(final=f"chat:{topic}")


class _FakeStage:
    def __init__(self, name, agent, instruction) -> None:
        self.name = name
        self.agent = agent
        self.instruction = instruction


class _FakeMember:
    def __init__(self, agent, name) -> None:
        self.agent = agent
        self.name = name


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orch_mod, "DebateTeam", _FakeDebate)
    monkeypatch.setattr(orch_mod, "ReviewTeam", _FakeReview)
    monkeypatch.setattr(orch_mod, "HierarchicalTeam", _FakeHier)
    monkeypatch.setattr(orch_mod, "Pipeline", _FakePipeline)
    monkeypatch.setattr(orch_mod, "GroupChat", _FakeGroupChat)
    monkeypatch.setattr(orch_mod, "Stage", _FakeStage)
    monkeypatch.setattr(orch_mod, "GroupChatMember", _FakeMember)
    return monkeypatch


def _orch() -> tuple[TeamOrchestrator, FakeFactory]:
    factory = FakeFactory()
    return TeamOrchestrator(provider=SimpleNamespace(), factory=factory), factory


# ---------------------------------------------------------------------------
# Spawn helpers
# ---------------------------------------------------------------------------


def test_register_blueprint_forwards_to_factory():
    orch, factory = _orch()
    bp = AgentBlueprint(role="coder")
    orch.register_blueprint("coder", bp)
    assert factory.blueprints["coder"] is bp


def test_spawn_tracks_agent_ids():
    orch, _ = _orch()
    agent = orch._spawn("coder", parent_id="p")
    assert agent.role == "coder"
    assert len(orch._spawned_ids) == 1


def test_spawn_many():
    orch, _ = _orch()
    agents = orch._spawn_many("coder", 3)
    assert len(agents) == 3
    assert len(orch._spawned_ids) == 3


def test_cleanup_destroys_and_clears():
    orch, factory = _orch()
    orch._spawn("a")
    orch._spawn("b")
    orch._cleanup()
    assert len(factory.destroyed) == 2
    assert orch._spawned_ids == []


# ---------------------------------------------------------------------------
# Topology shortcuts
# ---------------------------------------------------------------------------


async def test_debate(patched):
    orch, factory = _orch()
    result = await orch.debate("topic", count=2)
    assert result.final_answer == "debate:topic"
    assert orch._spawned_ids == []
    assert len(factory.destroyed) >= 1


async def test_review(patched):
    orch, factory = _orch()
    result = await orch.review("task", reviewer_count=2)
    assert result.final_answer == "review:task"
    assert orch._spawned_ids == []


async def test_delegate(patched):
    orch, factory = _orch()
    result = await orch.delegate("task", roles=["planner", "coder"])
    assert result.final_answer == "delegate:task"
    assert orch._spawned_ids == []


async def test_pipeline(patched):
    orch, factory = _orch()
    result = await orch.pipeline("task", stage_roles=["a", "b"])
    assert isinstance(result, TeamResult)
    assert result.final_answer == "pipe:task"
    assert result.status is TeamStatus.SUCCESS
    assert orch._spawned_ids == []


async def test_group_chat(patched):
    orch, factory = _orch()
    result = await orch.group_chat("topic", member_roles=["a", "b"], max_turns=5)
    assert result.final_answer == "chat:topic"
    assert orch._spawned_ids == []
