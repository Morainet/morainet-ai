"""Dynamic agent generation: spawn sub-agents on demand, auto-destroy when done.

AgentBlueprint  drives the factory; it defines what every spawned agent looks like.
AgentFactory   creates Agent instances from blueprints, enforces sandboxes,
               and manages their lifecycle.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from morainet.core.agent import Agent
from morainet.multiagent.protocol import AgentIdentity, A2AChannel
from morainet.multiagent.sandbox import AgentSandbox, MemoryNamespace, PermissionProfile, ResourceQuota
from morainet.providers.base import Provider


# ============================================================================
#  Agent Blueprint
# ============================================================================

@dataclass
class AgentBlueprint:
    """Template for dynamically creating agents.

    Each blueprint defines the role, tools, system prompt, memory, and
    sandbox configuration for one class of agents.
    """

    role: str                                     # e.g., "planner", "coder", "reviewer"
    system_prompt: str = ""
    tools: list[Any] = field(default_factory=list)
    provider: Provider | None = None
    memory: Any = None                            # shared memory instance
    sandbox_level: str = "STANDARD"               # LIMITED / STANDARD / ELEVATED / FULL
    max_steps: int = 10
    token_budget: int = 0
    time_budget: float = 0.0

    def build_identity(self, agent_id: str, name: str = "") -> AgentIdentity:
        return AgentIdentity(
            agent_id=agent_id,
            name=name or f"{self.role}-{agent_id[:8]}",
            role=self.role,
            capabilities=self._infer_capabilities(),
        )

    def _infer_capabilities(self) -> list[str]:
        caps: list[str] = [self.role]
        if "code" in self.role.lower() or "coder" in self.role.lower():
            caps.extend(["code-gen", "code-review"])
        if "review" in self.role.lower() or "reviewer" in self.role.lower():
            caps.extend(["code-review", "quality-check"])
        if "plan" in self.role.lower() or "planner" in self.role.lower():
            caps.extend(["planning", "task-decomposition"])
        if "test" in self.role.lower() or "tester" in self.role.lower():
            caps.extend(["testing", "test-gen"])
        if "architect" in self.role.lower():
            caps.extend(["architecture", "design"])
        return caps


# ============================================================================
#  Lifecycle state
# ============================================================================

class AgentLifecycle(str, Enum):
    """Lifecycle states for dynamically spawned agents."""
    CREATED = "created"           # blueprint ready, not yet instantiated
    ACTIVE = "active"             # Agent is running / ready to run
    BUSY = "busy"                 # Agent is currently executing a task
    IDLE = "idle"                 # Agent is alive but waiting
    DRAINING = "draining"         # finishing current task, then will terminate
    TERMINATED = "terminated"     # Agent has been destroyed
    ERROR = "error"               # Creation or execution failed


@dataclass
class SpawnedAgent:
    """Wrapper around a dynamically created Agent with lifecycle tracking."""
    agent_id: str
    agent: Agent
    blueprint: AgentBlueprint
    identity: AgentIdentity
    sandbox: AgentSandbox
    lifecycle: AgentLifecycle = AgentLifecycle.CREATED
    created_at: float = field(default_factory=time.time)
    terminated_at: float = 0.0
    task_count: int = 0
    channel: A2AChannel | None = None


# ============================================================================
#  Agent Factory
# ============================================================================

class AgentFactory:
    """Dynamically creates and destroys agents based on blueprints.

    Usage::

        factory = AgentFactory(provider=llm)
        factory.register_blueprint("coder", AgentBlueprint(
            role="coder",
            system_prompt="You are a senior software engineer.",
            tools=[code_search_tool, file_write_tool],
        ))

        agent = factory.spawn("coder", parent_id="orchestrator")
        result = agent.arun("Implement a REST endpoint")
        factory.destroy(agent_id)
    """

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self._blueprints: dict[str, AgentBlueprint] = {}
        self._active_agents: dict[str, SpawnedAgent] = {}
        self._idle_timeout: float = 300.0     # auto-destroy after 5 min idle
        self._max_total_agents: int = 50

    # -- blueprint registry --

    def register_blueprint(self, name: str, blueprint: AgentBlueprint) -> None:
        self._blueprints[name] = blueprint

    def unregister_blueprint(self, name: str) -> None:
        self._blueprints.pop(name, None)

    def list_blueprints(self) -> list[str]:
        return list(self._blueprints.keys())

    # -- spawn / destroy --

    def spawn(
        self,
        role: str,
        *,
        parent_id: str = "",
        agent_id: str = "",
        custom_prompt: str = "",
        extra_tools: list[Any] | None = None,
    ) -> Agent:
        """Create a new agent from a registered blueprint.

        Parameters
        ----------
        role:
            Blueprint name to instantiate.
        parent_id:
            Parent orchestrator ID for tracing.
        agent_id:
            Custom agent ID; auto-generated if omitted.
        custom_prompt:
            Additional system prompt instructions appended to blueprint prompt.
        extra_tools:
            Extra tools on top of the blueprint's tool set.

        Returns
        -------
        Agent
            The newly created Agent instance.
        """
        if len(self._active_agents) >= self._max_total_agents:
            raise RuntimeError(f"Agent limit ({self._max_total_agents}) reached")

        blueprint = self._blueprints.get(role)
        if blueprint is None:
            raise KeyError(f"No blueprint registered for role '{role}'")

        _id = agent_id or f"{role}_{uuid.uuid4().hex[:8]}"
        identity = blueprint.build_identity(_id)

        # Build sandbox
        sandbox = AgentSandbox(
            agent_id=_id,
            quota=ResourceQuota(
                max_steps=blueprint.max_steps,
                token_budget=blueprint.token_budget,
                time_budget=blueprint.time_budget,
            ),
            profile=PermissionProfile.standard(_id),
            memory=MemoryNamespace(namespace_id=_id, store=blueprint.memory),
        )

        # Prepare tools
        tools = list(blueprint.tools)
        if extra_tools:
            tools.extend(extra_tools)

        # Prepare prompt
        prompt = blueprint.system_prompt
        if custom_prompt:
            prompt = f"{prompt}\n\n{custom_prompt}"
        if parent_id:
            prompt = f"{prompt}\n\n[parent agent: {parent_id}]"

        agent = Agent(
            provider=blueprint.provider or self.provider,
            tools=tools,
            memory=sandbox.memory,
            system_prompt=prompt.strip(),
            max_steps=blueprint.max_steps,
            token_budget=blueprint.token_budget or None,
        )

        sandbox.activate()

        spawned = SpawnedAgent(
            agent_id=_id,
            agent=agent,
            blueprint=blueprint,
            identity=identity,
            sandbox=sandbox,
            lifecycle=AgentLifecycle.ACTIVE,
        )
        self._active_agents[_id] = spawned

        return agent

    def spawn_many(
        self,
        role: str,
        count: int,
        *,
        parent_id: str = "",
    ) -> list[Agent]:
        """Spawn multiple agents of the same role in parallel."""
        return [self.spawn(role, parent_id=parent_id) for _ in range(count)]

    def destroy(self, agent_id: str) -> bool:
        """Terminate an agent and release its resources."""
        spawned = self._active_agents.pop(agent_id, None)
        if spawned is None:
            return False
        spawned.lifecycle = AgentLifecycle.TERMINATED
        spawned.terminated_at = time.time()
        spawned.sandbox.deactivate()
        return True

    def destroy_idle(self, idle_seconds: float | None = None) -> int:
        """Destroy agents that have been idle for too long. Returns count destroyed."""
        threshold = idle_seconds or self._idle_timeout
        now = time.time()
        destroyed = 0
        for agent_id in list(self._active_agents):
            spawned = self._active_agents[agent_id]
            if spawned.lifecycle == AgentLifecycle.IDLE:
                if now - spawned.created_at > threshold:
                    self.destroy(agent_id)
                    destroyed += 1
        return destroyed

    def destroy_all(self) -> int:
        """Destroy all active agents."""
        count = len(self._active_agents)
        for agent_id in list(self._active_agents):
            self.destroy(agent_id)
        return count

    # -- query --

    def get(self, agent_id: str) -> SpawnedAgent | None:
        return self._active_agents.get(agent_id)

    def list_active(self) -> list[SpawnedAgent]:
        return list(self._active_agents.values())

    @property
    def active_count(self) -> int:
        return len(self._active_agents)

    @property
    def is_full(self) -> bool:
        return len(self._active_agents) >= self._max_total_agents
