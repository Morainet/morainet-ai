"""Multi-agent collaboration: topologies, A2A protocol, sandboxing, dynamic spawn.

This package provides:
  - A2A native protocol       : agents talk directly, no tool intermediary
  - Debate / Review / Hierarchical / SharedMemory topologies
  - Pipeline / Router / GroupChat orchestration
  - Dynamic agent spawning    : create sub-agents on demand, auto-destroy on completion
  - Resource & permission isolation: sandbox, quota, permission profiles
  - Agent pool                : pre-warm and reuse agents
"""

from morainet.multiagent.factory import AgentBlueprint, AgentFactory, AgentLifecycle, SpawnedAgent
from morainet.multiagent.orchestration import TeamOrchestrator
from morainet.multiagent.pool import AgentPool, PoolConfig, PoolStrategy
from morainet.multiagent.protocol import (
    A2ABus,
    A2AChannel,
    A2AMessage,
    A2AMessageType,
    AgentIdentity,
)
from morainet.multiagent.sandbox import (
    AgentSandbox,
    MemoryNamespace,
    PermissionProfile,
    ResourceQuota,
)
from morainet.multiagent.topologies import (
    AgentContribution,
    Debate,
    DebateTeam,
    GroupChat,
    GroupChatMember,
    HierarchicalTeam,
    Pipeline,
    ReviewTeam,
    Route,
    Router,
    SharedMemoryPool,
    Stage,
    SubTask,
    TeamResult,
    TeamStatus,
)

__all__ = [
    # A2A Protocol
    "AgentIdentity",
    "A2AMessage",
    "A2AMessageType",
    "A2AChannel",
    "A2ABus",
    # Topologies
    "TeamResult",
    "TeamStatus",
    "AgentContribution",
    "Debate",
    "DebateTeam",
    "GroupChat",
    "GroupChatMember",
    "ReviewTeam",
    "HierarchicalTeam",
    "SharedMemoryPool",
    "Pipeline",
    "Stage",
    "Router",
    "Route",
    "SubTask",
    # Orchestration
    "TeamOrchestrator",
    # Factory & lifecycle
    "AgentFactory",
    "AgentBlueprint",
    "AgentLifecycle",
    "SpawnedAgent",
    # Pool
    "AgentPool",
    "PoolConfig",
    "PoolStrategy",
    # Sandbox & isolation
    "AgentSandbox",
    "ResourceQuota",
    "PermissionProfile",
    "MemoryNamespace",
]
