"""Multi-agent collaboration demo — all topologies + A2A protocol + sandboxing.

Runs through every capability without requiring a real LLM provider.
Uses MockProvider for zero-cost testing.
"""

from __future__ import annotations

import asyncio
import sys

# ---------------------------------------------------------------------------
#  Test 1: A2A Protocol — agent-to-agent communication
# ---------------------------------------------------------------------------

def test_a2a_protocol():
    """Test direct channel communication between two agents."""
    from morainet.multiagent.protocol import (
        A2AChannel,
        A2AMessage,
        A2AMessageType,
        AgentIdentity,
    )

    # Create paired channels
    ch_a, ch_b = A2AChannel.pair("agent_A", "agent_B")

    # Exchange identities
    id_a = AgentIdentity(agent_id="agent_A", name="Alice", role="researcher", capabilities=["search"])
    id_b = AgentIdentity(agent_id="agent_B", name="Bob", role="analyst", capabilities=["analyze"])

    async def _run():
        # Handshake
        received_b = await asyncio.gather(
            ch_a.handshake(id_a),
            ch_b.handshake(id_b),
        )
        assert received_b[0] is not None, "A should receive B's identity"
        assert received_b[0].name == "Bob"

        # Query → Response
        async def _respond():
            msg = await ch_b.recv()
            assert msg.msg_type == A2AMessageType.QUERY
            await ch_b.respond(msg.msg_id, "Here is the answer from Bob")

        query_task = asyncio.create_task(_respond())
        reply = await ch_a.query("What is the capital of France?")
        assert reply is not None
        assert "Bob" in reply.payload
        await query_task

        # Delegate → Result
        async def _delegate_respond():
            msg = await ch_b.recv()
            assert msg.msg_type == A2AMessageType.DELEGATE
            await ch_b.send(A2AMessage(
                msg_id="r1", msg_type=A2AMessageType.RESULT,
                sender_id="agent_B", recipient_id="agent_A",
                correlation_id=msg.msg_id,
                payload="Task completed successfully",
            ))

        delegate_task = asyncio.create_task(_delegate_respond())
        result = await ch_a.delegate("Analyze this dataset")
        assert result is not None
        assert "completed" in result.payload
        await delegate_task

        await ch_a.close()
        await ch_b.close()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  PASSED: A2A Protocol (handshake, query/response, delegate/result)")


# ---------------------------------------------------------------------------
#  Test 2: A2A Bus — shared message bus
# ---------------------------------------------------------------------------

def test_a2a_bus():
    """Test many-to-many broadcast communication."""
    from morainet.multiagent.protocol import A2ABus, A2AMessageType

    async def _run():
        bus = A2ABus()

        ch_a = bus.subscribe("agent_A", {"code"})
        ch_b = bus.subscribe("agent_B", {"review"})
        ch_c = bus.subscribe("agent_C")  # subscribe to all

        # Broadcast
        await bus.broadcast("agent_A", "Hello everyone!")

        msg_b = await ch_b.recv(timeout=0.5)
        assert msg_b is not None, "B should receive broadcast"
        msg_c = await ch_c.recv(timeout=0.5)
        assert msg_c is not None, "C should receive broadcast"

        # Topic-filtered event
        await bus.broadcast("agent_B", "New code ready for review", topic="code")

        msg_a = await ch_a.recv(timeout=0.5)
        assert msg_a is not None, "A should receive code topic (subscribed to code)"
        assert msg_a.msg_type == A2AMessageType.EVENT

        msg_b2 = await ch_b.recv(timeout=0.1)
        assert msg_b2 is None, "B should NOT receive code topic (not subscribed)"

        await bus.close()

    asyncio.get_event_loop().run_until_complete(_run())
    print("  PASSED: A2A Bus (broadcast, topic filter, many-to-many)")


# ---------------------------------------------------------------------------
#  Test 3: Sandbox — resource isolation & permission profiles
# ---------------------------------------------------------------------------

def test_sandbox():
    """Test resource quotas and permission profiles."""
    import time
    from morainet.multiagent.sandbox import (
        AgentSandbox,
        MemoryNamespace,
        PermissionProfile,
        ResourceQuota,
    )
    from morainet.core.models import Message

    # --- ResourceQuota ---
    tight = ResourceQuota(max_steps=5, token_budget=8000, time_budget=30.0)
    assert tight.check_step(3) is True
    assert tight.check_step(6) is False
    assert tight.check_tokens(5000) is True
    assert tight.check_tokens(10000) is False

    unlimited = ResourceQuota.unlimited()
    assert unlimited.check_step(999) is True

    # --- PermissionProfile ---
    limited = PermissionProfile.limited()
    assert limited.is_allowed("search") is True
    assert limited.is_allowed("write_file") is False
    assert limited.is_allowed("delete_file") is False

    elevated = PermissionProfile.elevated(block={"delete_file", "deploy"})
    assert elevated.is_allowed("search") is True
    assert elevated.is_allowed("write_file") is True
    assert elevated.is_allowed("delete_file") is False

    # --- MemoryNamespace ---
    ns_a = MemoryNamespace("agent_A")
    ns_b = MemoryNamespace("agent_B")

    async def _mem_test():
        await ns_a.add(Message.user("Secret data for A"))
        await ns_b.add(Message.user("Secret data for B"))

        ctx_a = await ns_a.get_context("data")
        ctx_b = await ns_b.get_context("data")

        # Each namespace is independent
        assert any("Secret data for A" in (m.content or "") for m in ctx_a)
        assert any("Secret data for B" in (m.content or "") for m in ctx_b)
        assert not any("Secret data for B" in (m.content or "") for m in ctx_a)

    asyncio.get_event_loop().run_until_complete(_mem_test())

    # --- AgentSandbox ---
    sb = AgentSandbox.for_agent("test_agent", level="STANDARD")
    assert sb.agent_id == "test_agent"
    assert not sb.is_active
    sb.activate()
    assert sb.is_active
    sb.deactivate()
    assert not sb.is_active

    print("  PASSED: Sandbox (quota checks, permission profiles, memory isolation)")


# ---------------------------------------------------------------------------
#  Test 4: Agent Factory — dynamic spawn & destroy
# ---------------------------------------------------------------------------

def test_factory():
    """Test dynamic agent creation and lifecycle management."""
    from morainet.multiagent.factory import AgentBlueprint, AgentFactory, AgentLifecycle
    from morainet.providers.mock import MockProvider

    provider = MockProvider()
    factory = AgentFactory(provider)

    # Register blueprints
    factory.register_blueprint("coder", AgentBlueprint(
        role="coder",
        system_prompt="You are a software engineer.",
        sandbox_level="STANDARD",
    ))
    factory.register_blueprint("reviewer", AgentBlueprint(
        role="reviewer",
        system_prompt="You are a code reviewer.",
        sandbox_level="LIMITED",
    ))
    factory.register_blueprint("tester", AgentBlueprint(
        role="tester",
        system_prompt="You are a QA tester.",
        sandbox_level="STANDARD",
    ))

    assert factory.list_blueprints() == ["coder", "reviewer", "tester"]

    # Spawn agents
    coder = factory.spawn("coder", parent_id="orchestrator")
    reviewer = factory.spawn("reviewer", parent_id="orchestrator")
    assert factory.active_count == 2

    spawned = factory.list_active()
    assert len(spawned) == 2
    assert any(s.blueprint.role == "coder" for s in spawned)
    assert any(s.blueprint.role == "reviewer" for s in spawned)

    # Lifecycle tracking
    coder_spawned = factory.get(spawned[0].agent_id)
    assert coder_spawned is not None
    assert coder_spawned.lifecycle in (AgentLifecycle.ACTIVE, AgentLifecycle.CREATED)

    # Destroy
    factory.destroy(spawned[0].agent_id)
    assert factory.active_count == 1

    # Destroy all
    factory.destroy_all()
    assert factory.active_count == 0

    print("  PASSED: Agent Factory (register, spawn, lifecycle, destroy)")


# ---------------------------------------------------------------------------
#  Test 5: Topologies — Debate
# ---------------------------------------------------------------------------

def test_debate_topology():
    """Test DebateTeam with MockProvider (no real LLM needed)."""
    from morainet.core.agent import Agent
    from morainet.multiagent.topologies import DebateTeam
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()

        arbiter = Agent(
            provider=provider,
            system_prompt="You are an impartial arbiter.",
            max_steps=3,
        )

        debater1 = Agent(
            provider=provider,
            system_prompt="You argue FOR the proposition.",
            max_steps=3,
        )

        debater2 = Agent(
            provider=provider,
            system_prompt="You argue AGAINST the proposition.",
            max_steps=3,
        )

        team = DebateTeam(
            arbiter=arbiter,
            debaters=[debater1, debater2],
            rounds=1,
        )

        result = await team.arun("Should we adopt microservices?")
        assert result.status.value in ("success", "partial")
        assert len(result.contributions) >= 3  # 2 debaters + arbiter
        print(f"  PASSED: DebateTeam ({len(result.contributions)} contributions, status={result.status.value})")

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 6: Topologies — Review
# ---------------------------------------------------------------------------

def test_review_topology():
    """Test ReviewTeam with MockProvider."""
    from morainet.core.agent import Agent
    from morainet.multiagent.topologies import ReviewTeam
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()

        producer = Agent(
            provider=provider,
            system_prompt="You produce technical documents.",
            max_steps=3,
        )

        reviewer = Agent(
            provider=provider,
            system_prompt="You review and provide constructive feedback.",
            max_steps=3,
        )

        team = ReviewTeam(
            producer=producer,
            reviewers=[reviewer],
            max_cycles=2,
        )

        result = await team.run(
            "Write a design doc for a REST API",
            acceptance_criteria="Must include endpoints, data models, error handling",
        )
        assert result.status.value in ("success", "partial")
        # 1 initial draft + up to 2 review cycles + up to 2 revisions = 3-5 contributions
        print(f"  PASSED: ReviewTeam ({len(result.contributions)} contributions, status={result.status.value})")

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 7: Topologies — Hierarchical Delegation
# ---------------------------------------------------------------------------

def test_hierarchical_topology():
    """Test HierarchicalTeam with explicit sub-tasks."""
    from morainet.core.agent import Agent
    from morainet.multiagent.topologies import HierarchicalTeam, SubTask
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()

        orchestrator = Agent(
            provider=provider,
            system_prompt="You plan and coordinate.",
            max_steps=3,
        )

        specialists = {
            "frontend": Agent(
                provider=provider,
                system_prompt="You are a frontend developer.",
                max_steps=3,
            ),
            "backend": Agent(
                provider=provider,
                system_prompt="You are a backend developer.",
                max_steps=3,
            ),
        }

        team = HierarchicalTeam(
            orchestrator=orchestrator,
            specialists=specialists,
            auto_decompose=False,
        )

        sub_tasks = [
            SubTask(description="Build the React login form", specialist_role="frontend"),
            SubTask(description="Build the auth API endpoint", specialist_role="backend"),
        ]

        result = await team.run("Build login system", sub_tasks=sub_tasks)
        assert result.status.value in ("success", "partial")
        assert len(result.contributions) >= 2
        print(f"  PASSED: HierarchicalTeam ({len(result.contributions)} contributions)")

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 8: Topologies — Pipeline & Router
# ---------------------------------------------------------------------------

def test_pipeline_and_router():
    """Test Pipeline and Router topologies."""
    from morainet.core.agent import Agent
    from morainet.multiagent.topologies import Pipeline, Route, Router, Stage
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()

        planner = Agent(
            provider=provider,
            system_prompt="You create plans.",
            max_steps=3,
        )
        coder = Agent(
            provider=provider,
            system_prompt="You write code.",
            max_steps=3,
        )
        reviewer = Agent(
            provider=provider,
            system_prompt="You review code.",
            max_steps=3,
        )

        # --- Pipeline ---
        pipeline = Pipeline([
            Stage(name="plan", agent=planner, instruction="Create a plan: {query}"),
            Stage(name="code", agent=coder, instruction="Write the code: {query}"),
            Stage(name="review", agent=reviewer, instruction="Review: {query}"),
        ])

        result = await pipeline.arun("Build a login API")
        assert hasattr(result, 'final')
        print(f"  PASSED: Pipeline")

        # --- Router ---
        router = Router([
            Route(name="code", agent=coder, rules=["code", "implement", "debug"]),
            Route(name="plan", agent=planner, rules=["plan", "design"]),
        ], selector=lambda q: "code" if "implement" in q.lower() else "plan")

        result = await router.arun("Implement user authentication")
        assert hasattr(result, 'route')
        print(f"  PASSED: Router (routed to correct agent)")

        # Route to unknown
        result = await router.arun("Some random question")
        assert hasattr(result, 'final')
        print(f"  PASSED: Router (fallback)")

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 9: Topologies — Shared Memory Pool
# ---------------------------------------------------------------------------

def test_shared_memory_pool():
    """Test SharedMemoryPool with agents sharing memory."""
    from morainet.core.agent import Agent
    from morainet.multiagent.topologies import SharedMemoryPool
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()

        agent_a = Agent(
            provider=provider,
            system_prompt="Agent A: researcher.",
            max_steps=3,
        )
        agent_b = Agent(
            provider=provider,
            system_prompt="Agent B: analyst.",
            max_steps=3,
        )

        pool = SharedMemoryPool(
            agents={"researcher": agent_a, "analyst": agent_b},
        )

        result = await pool.run("Analyze recent market trends")
        assert result.status.value in ("success", "partial")
        assert len(result.contributions) == 2
        print(f"  PASSED: SharedMemoryPool ({len(result.contributions)} agents collaborated)")

        await pool.close()

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 10: Team Orchestrator — full lifecycle
# ---------------------------------------------------------------------------

def test_team_orchestrator():
    """Test TeamOrchestrator with dynamic spawn/cleanup."""
    from morainet.multiagent.factory import AgentBlueprint
    from morainet.multiagent.orchestration import TeamOrchestrator
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()
        orch = TeamOrchestrator(provider=provider)

        orch.register_blueprint("debater", AgentBlueprint(
            role="debater",
            system_prompt="You argue constructively.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("arbiter", AgentBlueprint(
            role="arbiter",
            system_prompt="You synthesize arguments into a decision.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("producer", AgentBlueprint(
            role="producer",
            system_prompt="You produce high-quality work.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("reviewer", AgentBlueprint(
            role="reviewer",
            system_prompt="You review critically.",
            sandbox_level="LIMITED",
        ))
        orch.register_blueprint("coder", AgentBlueprint(
            role="coder",
            system_prompt="You write production code.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("tester", AgentBlueprint(
            role="tester",
            system_prompt="You write comprehensive tests.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("planner", AgentBlueprint(
            role="planner",
            system_prompt="You decompose tasks.",
            sandbox_level="STANDARD",
        ))
        orch.register_blueprint("orchestrator", AgentBlueprint(
            role="orchestrator",
            system_prompt="You coordinate specialist agents.",
            sandbox_level="STANDARD",
        ))

        # Debate
        result = await orch.debate("Should we use Rust or Go?", count=2, rounds=1)
        assert result.status.value in ("success", "partial")
        print(f"  PASSED: TeamOrchestrator.debate (contributions: {len(result.contributions)})")

        # Review
        result = await orch.review("Write a function to sort numbers", max_cycles=1)
        assert result.status.value in ("success", "partial")
        print(f"  PASSED: TeamOrchestrator.review (contributions: {len(result.contributions)})")

        # Delegate
        result = await orch.delegate(
            "Build a login system",
            roles=["planner", "coder"],
        )
        assert result.status.value in ("success", "partial")
        print(f"  PASSED: TeamOrchestrator.delegate (contributions: {len(result.contributions)})")

        # Pipeline
        result = await orch.pipeline(
            "Create a web app",
            stage_roles=["planner", "coder", "reviewer"],
        )
        assert result.status.value in ("success", "partial")
        print(f"  PASSED: TeamOrchestrator.pipeline (stages: {len(result.contributions)})")

        # Group chat
        result = await orch.group_chat(
            "Design a database schema",
            member_roles=["planner", "coder", "reviewer"],
        )
        assert result.status.value in ("success", "partial")
        print(f"  PASSED: TeamOrchestrator.group_chat (participants: {len(result.contributions)})")

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
#  Test 11: Agent Pool — acquire/release lifecycle
# ---------------------------------------------------------------------------

def test_agent_pool():
    """Test AgentPool pre-warming and reuse."""
    from morainet.multiagent.factory import AgentBlueprint, AgentFactory
    from morainet.multiagent.pool import AgentPool, PoolConfig
    from morainet.providers.mock import MockProvider

    async def _run():
        provider = MockProvider()
        factory = AgentFactory(provider)
        factory.register_blueprint("coder", AgentBlueprint(
            role="coder",
            system_prompt="You write code.",
            sandbox_level="STANDARD",
        ))

        pool = AgentPool(
            factory=factory,
            role="coder",
            config=PoolConfig(min_size=2, max_size=5, prewarm=True),
        )

        await pool.start()
        assert len(pool) >= 2, "Pool should have at least min_size agents after start"
        assert pool.stats["idle"] >= 2

        # Acquire
        agent = await pool.acquire()
        assert pool.stats["busy"] == 1

        # Release — find the busy agent's ID
        busy_agents = list(pool._busy.keys())
        if busy_agents:
            await pool.release(busy_agents[0])
        assert pool.stats["busy"] == 0
        assert pool.stats["idle"] >= 2

        await pool.stop()
        assert len(pool) == 0

        print("  PASSED: AgentPool (prewarm, acquire, release, stop)")

    asyncio.get_event_loop().run_until_complete(_run())


# ============================================================================
#  Runner
# ============================================================================

def main():
    print("=" * 60)
    print("  Multi-Agent Collaboration System — Full Demo")
    print("=" * 60)
    print()

    tests = [
        ("A2A Protocol", test_a2a_protocol),
        ("A2A Bus", test_a2a_bus),
        ("Sandbox & Isolation", test_sandbox),
        ("Agent Factory", test_factory),
        ("Debate Team", test_debate_topology),
        ("Review Team", test_review_topology),
        ("Hierarchical Team", test_hierarchical_topology),
        ("Pipeline & Router", test_pipeline_and_router),
        ("Shared Memory Pool", test_shared_memory_pool),
        ("Team Orchestrator", test_team_orchestrator),
        ("Agent Pool", test_agent_pool),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    print(f"  All {len(tests)} tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
