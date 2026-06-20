"""Team orchestrator: unified coordinator for multi-agent runs.

Manages the full lifecycle of a multi-agent run:
1. Spawn specialist agents from blueprints
2. Apply sandbox constraints
3. Execute a chosen topology
4. Collect results
5. Tear down spawned agents
"""

from __future__ import annotations

import uuid

from morainet.core.agent import Agent
from morainet.memory.base import Memory
from morainet.multiagent.factory import AgentBlueprint, AgentFactory
from morainet.multiagent.topologies import (
    DebateTeam,
    GroupChat,
    GroupChatMember,
    HierarchicalTeam,
    Pipeline,
    ReviewTeam,
    Stage,
    TeamResult,
    TeamStatus,
)
from morainet.providers.base import Provider


class TeamOrchestrator:
    """Unified coordinator that can compose multiple topology strategies.

    The orchestrator manages the full lifecycle of a multi-agent run:
    1. Spawn specialist agents from blueprints
    2. Apply sandbox constraints
    3. Execute a chosen topology
    4. Collect results
    5. Tear down spawned agents

    Usage::

        orch = TeamOrchestrator(provider=llm, factory=AgentFactory(llm))
        orch.register_blueprint("coder", AgentBlueprint(
            role="coder", system_prompt="You are a coder.",
        ))

        result = await orch.debate("Which framework should we use?", count=3)
        result = await orch.review("Write a REST API", reviewer_count=1)
        result = await orch.delegate("Build a login system", roles=["planner", "coder", "tester"])
    """

    def __init__(
        self,
        provider: Provider,
        factory: AgentFactory | None = None,
        shared_memory: Memory | None = None,
    ) -> None:
        self.provider = provider
        self.factory = factory or AgentFactory(provider)
        self.shared_memory = shared_memory
        self._spawned_ids: list[str] = []

    def register_blueprint(self, name: str, blueprint: AgentBlueprint) -> None:
        """Register a specialist blueprint."""
        self.factory.register_blueprint(name, blueprint)

    def _spawn(self, role: str, parent_id: str = "") -> Agent:
        agent = self.factory.spawn(role, parent_id=parent_id)
        spawned_list = self.factory.list_active()
        if spawned_list:
            self._spawned_ids.append(spawned_list[-1].agent_id)
        return agent

    def _spawn_many(self, role: str, count: int, parent_id: str = "") -> list[Agent]:
        agents = [self._spawn(role, parent_id) for _ in range(count)]
        return agents

    def _cleanup(self) -> None:
        for agent_id in list(self._spawned_ids):
            try:
                self.factory.destroy(agent_id)
            except Exception:
                pass
        self._spawned_ids.clear()

    # -- topology shortcuts --

    async def debate(
        self,
        topic: str,
        count: int = 3,
        rounds: int = 1,
        arbiter_role: str = "arbiter",
        debater_role: str = "debater",
    ) -> TeamResult:
        """Run a structured debate on a topic."""
        arbiter = self._spawn(arbiter_role, parent_id="debate_arbiter")
        debaters = self._spawn_many(debater_role, count, parent_id="debate")

        team = DebateTeam(arbiter=arbiter, debaters=debaters, rounds=rounds)
        try:
            return await team.arun(topic)
        finally:
            self._cleanup()

    async def review(
        self,
        task: str,
        acceptance_criteria: str = "",
        max_cycles: int = 3,
        reviewer_count: int = 1,
        producer_role: str = "producer",
        reviewer_role: str = "reviewer",
    ) -> TeamResult:
        """Run a review cycle: produce -> review -> revise."""
        producer = self._spawn(producer_role, parent_id="review_producer")
        reviewers = self._spawn_many(reviewer_role, reviewer_count, parent_id="review")

        team = ReviewTeam(
            producer=producer,
            reviewers=reviewers,
            max_cycles=max_cycles,
        )
        try:
            return await team.run(task, acceptance_criteria)
        finally:
            self._cleanup()

    async def delegate(
        self,
        task: str,
        roles: list[str],
        orchestrator_role: str = "orchestrator",
    ) -> TeamResult:
        """Hierarchical delegation: orchestrator + specialists."""
        orch = self._spawn(orchestrator_role, parent_id="orchestrator")
        specialists = {
            role: self._spawn(role, parent_id=f"specialist_{role}")
            for role in roles
        }

        team = HierarchicalTeam(
            orchestrator=orch,
            specialists=specialists,
            auto_decompose=True,
        )
        try:
            return await team.run(task)
        finally:
            self._cleanup()

    async def pipeline(
        self,
        task: str,
        stage_roles: list[str],
    ) -> TeamResult:
        """Run a sequential pipeline through multiple specialist roles."""
        stages: list[Stage] = []
        for role in stage_roles:
            agent = self._spawn(role, parent_id=f"pipeline_{role}")
            stages.append(Stage(name=role, agent=agent, instruction=f"Stage: {role}: {{query}}"))

        pipeline = Pipeline(stages)
        try:
            result = await pipeline.arun(task)
            return TeamResult(
                status=TeamStatus.SUCCESS,
                final_answer=result.final,
                trace_id=uuid.uuid4().hex[:16],
            )
        finally:
            self._cleanup()

    async def group_chat(
        self,
        topic: str,
        member_roles: list[str],
        max_turns: int = 10,
    ) -> TeamResult:
        """Run a group chat with multiple agents."""
        members: list[GroupChatMember] = []
        for role in member_roles:
            agent = self._spawn(role, parent_id=f"chat_{role}")
            members.append(GroupChatMember(agent=agent, name=role))

        chat = GroupChat(members=members, max_rounds=max_turns)
        try:
            result = await chat.arun(topic)
            return TeamResult(
                status=TeamStatus.SUCCESS,
                final_answer=result.final,
                trace_id=uuid.uuid4().hex[:16],
            )
        finally:
            self._cleanup()
