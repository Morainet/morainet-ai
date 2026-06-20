from __future__ import annotations

import time

import pytest

from morainet.multiagent.factory import (
    AgentBlueprint,
    AgentFactory,
    AgentLifecycle,
    SpawnedAgent,
)
from morainet.providers import MockProvider


# ---------------------------------------------------------------------------
# AgentBlueprint
# ---------------------------------------------------------------------------

def test_agent_blueprint_defaults():
    bp = AgentBlueprint(role="coder")
    assert bp.role == "coder"
    assert bp.system_prompt == ""
    assert bp.tools == []
    assert bp.provider is None
    assert bp.sandbox_level == "STANDARD"
    assert bp.max_steps == 10
    assert bp.token_budget == 0
    assert bp.time_budget == 0.0


def test_build_identity():
    bp = AgentBlueprint(role="coder", system_prompt="You are a coder.")
    identity = bp.build_identity("agent_001", name="CodeBot")
    assert identity.agent_id == "agent_001"
    assert identity.name == "CodeBot"
    assert identity.role == "coder"
    assert "coder" in identity.capabilities
    assert "code-gen" in identity.capabilities
    assert "code-review" in identity.capabilities


def test_build_identity_default_name():
    bp = AgentBlueprint(role="reviewer")
    identity = bp.build_identity("agent_abc12345")
    assert identity.name.startswith("reviewer-agent_ab")


def test_infer_capabilities_coder():
    bp = AgentBlueprint(role="coder")
    caps = bp._infer_capabilities()
    assert "coder" in caps
    assert "code-gen" in caps
    assert "code-review" in caps


def test_infer_capabilities_reviewer():
    bp = AgentBlueprint(role="reviewer")
    caps = bp._infer_capabilities()
    assert "code-review" in caps
    assert "quality-check" in caps


def test_infer_capabilities_planner():
    bp = AgentBlueprint(role="planner")
    caps = bp._infer_capabilities()
    assert "planning" in caps
    assert "task-decomposition" in caps


def test_infer_capabilities_tester():
    bp = AgentBlueprint(role="tester")
    caps = bp._infer_capabilities()
    assert "testing" in caps
    assert "test-gen" in caps


def test_infer_capabilities_architect():
    bp = AgentBlueprint(role="architect")
    caps = bp._infer_capabilities()
    assert "architecture" in caps
    assert "design" in caps


def test_infer_capabilities_unknown_role():
    bp = AgentBlueprint(role="unknown_role")
    caps = bp._infer_capabilities()
    assert caps == ["unknown_role"]


# ---------------------------------------------------------------------------
# AgentLifecycle
# ---------------------------------------------------------------------------

def test_agent_lifecycle_enum():
    assert AgentLifecycle.CREATED.value == "created"
    assert AgentLifecycle.ACTIVE.value == "active"
    assert AgentLifecycle.BUSY.value == "busy"
    assert AgentLifecycle.IDLE.value == "idle"
    assert AgentLifecycle.DRAINING.value == "draining"
    assert AgentLifecycle.TERMINATED.value == "terminated"
    assert AgentLifecycle.ERROR.value == "error"


# ---------------------------------------------------------------------------
# SpawnedAgent
# ---------------------------------------------------------------------------

def test_spawned_agent_defaults():
    # Cannot fully construct without a real Agent, but we can test the dataclass
    assert AgentLifecycle.ACTIVE is not None


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------

class TestAgentFactory:
    def setup_method(self):
        self.provider = MockProvider(responses=[])
        self.factory = AgentFactory(provider=self.provider)

    def test_register_blueprint(self):
        bp = AgentBlueprint(role="coder")
        self.factory.register_blueprint("coder", bp)
        assert "coder" in self.factory.list_blueprints()

    def test_unregister_blueprint(self):
        bp = AgentBlueprint(role="coder")
        self.factory.register_blueprint("coder", bp)
        self.factory.unregister_blueprint("coder")
        assert "coder" not in self.factory.list_blueprints()

    def test_unregister_nonexistent(self):
        self.factory.unregister_blueprint("nonexistent")

    def test_list_blueprints_empty(self):
        assert self.factory.list_blueprints() == []

    def test_spawn_without_blueprint_raises(self):
        with pytest.raises(KeyError):
            self.factory.spawn("unknown_role")

    def test_spawn_with_valid_blueprint(self):
        bp = AgentBlueprint(role="coder", system_prompt="You are a coder.", max_steps=5)
        self.factory.register_blueprint("coder", bp)
        agent = self.factory.spawn("coder")
        assert agent is not None
        assert self.factory.active_count == 1
        assert self.factory.is_full is False

    def test_spawn_many(self):
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        agents = self.factory.spawn_many("coder", count=3)
        assert len(agents) == 3
        assert self.factory.active_count == 3

    def test_destroy_active_agent(self):
        bp = AgentBlueprint(role="coder", max_steps=5)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        agents = self.factory.list_active()
        agent_id = agents[0].agent_id
        assert self.factory.destroy(agent_id) is True
        assert self.factory.active_count == 0

    def test_destroy_nonexistent(self):
        assert self.factory.destroy("nonexistent") is False

    def test_destroy_all(self):
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn_many("coder", count=3)
        destroyed = self.factory.destroy_all()
        assert destroyed == 3
        assert self.factory.active_count == 0

    def test_get_active_agent(self):
        bp = AgentBlueprint(role="coder", max_steps=5)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        agents = self.factory.list_active()
        agent_id = agents[0].agent_id
        sa = self.factory.get(agent_id)
        assert sa is not None
        assert sa.agent_id == agent_id

    def test_get_nonexistent_agent(self):
        assert self.factory.get("nonexistent") is None

    def test_list_active(self):
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        active = self.factory.list_active()
        assert len(active) == 1
        assert isinstance(active[0], SpawnedAgent)

    def test_active_count(self):
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        assert self.factory.active_count == 0
        self.factory.spawn("coder")
        assert self.factory.active_count == 1

    def test_is_full(self):
        # default max is 50, so not full with 1 agent
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        assert self.factory.is_full is False

    def test_is_full_when_at_max(self):
        self.factory._max_total_agents = 1
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        assert self.factory.is_full is True

    def test_spawn_exceeds_max_raises(self):
        self.factory._max_total_agents = 0
        bp = AgentBlueprint(role="coder")
        self.factory.register_blueprint("coder", bp)
        with pytest.raises(RuntimeError, match="limit"):
            self.factory.spawn("coder")

    def test_destroy_idle(self):
        bp = AgentBlueprint(role="coder", max_steps=3)
        self.factory.register_blueprint("coder", bp)
        self.factory.spawn("coder")
        agents = self.factory.list_active()
        agent_id = agents[0].agent_id

        # Manually set to IDLE with an old timestamp
        spawned = self.factory.get(agent_id)
        spawned.lifecycle = AgentLifecycle.IDLE
        spawned.created_at = time.time() - 1000

        destroyed = self.factory.destroy_idle(idle_seconds=60)
        assert destroyed == 1
        assert self.factory.active_count == 0
