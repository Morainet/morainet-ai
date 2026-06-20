"""Resource isolation, permission profiles, and independent memory spaces.

Each agent in a multi-agent team can be sandboxed with:
  - ResourceQuota     : token / step / time budgets
  - PermissionProfile : allowlist / denylist of tools
  - MemoryNamespace   : scoped memory that other agents cannot read
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from morainet.core.models import Message
from morainet.memory.base import Memory
from morainet.memory.short_memory import ShortMemory


# ============================================================================
#  Resource Quota
# ============================================================================

@dataclass
class ResourceQuota:
    """Limits per-agent resource consumption.

    Parameters
    ----------
    max_steps:
        Maximum reasoning steps. 0 = unlimited.
    token_budget:
        Maximum tokens consumed. 0 = unlimited.
    time_budget:
        Maximum wall-clock seconds for a single run. 0 = unlimited.
    max_concurrent_tasks:
        Maximum parallel sub-tasks the agent can spawn. 0 = unlimited.
    """
    max_steps: int = 0
    token_budget: int = 0
    time_budget: float = 0.0
    max_concurrent_tasks: int = 0

    def check_step(self, step_count: int) -> bool:
        if self.max_steps > 0 and step_count >= self.max_steps:
            return False
        return True

    def check_tokens(self, total_tokens: int) -> bool:
        if self.token_budget > 0 and total_tokens >= self.token_budget:
            return False
        return True

    def check_time(self, started_at: float) -> bool:
        if self.time_budget > 0 and (time.time() - started_at) >= self.time_budget:
            return False
        return True

    @classmethod
    def unlimited(cls) -> "ResourceQuota":
        return cls()

    @classmethod
    def tight(cls) -> "ResourceQuota":
        """Conservative limits for simple sub-tasks."""
        return cls(max_steps=5, token_budget=8000, time_budget=30.0)


# ============================================================================
#  Permission Profile
# ============================================================================

@dataclass
class PermissionProfile:
    """Controls which tools a sandboxed agent may call.

    Modes:
      - ALLOW_ALL (default) : any tool is permitted
      - ALLOWLIST           : only explicitly listed tools
      - DENYLIST            : all tools except blocked ones

    Levels:
      - LIMITED : read-only + safe tools only
      - STANDARD: read/write, no destructive operations
      - ELEVATED: all tools except explicitly blocked
      - FULL    : unrestricted
    """
    agent_id: str = ""
    allow_all: bool = True
    allowlist: set[str] = field(default_factory=set)
    denylist: set[str] = field(default_factory=set)
    level: str = "STANDARD"           # LIMITED / STANDARD / ELEVATED / FULL

    # Pre-defined tool categories for easy allowlist/denylist
    READ_ONLY_TOOLS = {"search", "read_file", "list_dir", "grep", "fetch", "get"}
    SAFE_TOOLS = {"search", "read_file", "list_dir", "grep", "fetch", "get",
                   "write_file", "replace_in_file", "run_command"}
    DESTRUCTIVE_TOOLS = {"delete_file", "execute_command", "deploy", "drop",
                          "rm", "force_push", "hard_reset"}

    def is_allowed(self, tool_name: str) -> bool:
        """Check whether a tool is permitted for this profile."""
        if self.denylist and tool_name in self.denylist:
            return False
        if not self.allow_all:
            if self.allowlist and tool_name not in self.allowlist:
                return False
        return True

    @classmethod
    def limited(cls, agent_id: str = "") -> "PermissionProfile":
        """Read-only + safe tools only."""
        return cls(
            agent_id=agent_id,
            allow_all=False,
            allowlist=cls.READ_ONLY_TOOLS,
            level="LIMITED",
        )

    @classmethod
    def standard(cls, agent_id: str = "") -> "PermissionProfile":
        """Read/write but no destructive operations."""
        return cls(
            agent_id=agent_id,
            allow_all=False,
            allowlist=cls.SAFE_TOOLS,
            level="STANDARD",
        )

    @classmethod
    def elevated(cls, agent_id: str = "", block: set[str] | None = None) -> "PermissionProfile":
        """All tools except explicitly blocked ones."""
        return cls(
            agent_id=agent_id,
            allow_all=True,
            denylist=block or set(),
            level="ELEVATED",
        )

    @classmethod
    def full(cls, agent_id: str = "") -> "PermissionProfile":
        """Unrestricted access."""
        return cls(agent_id=agent_id, allow_all=True, level="FULL")


# ============================================================================
#  Memory Namespace: per-agent isolated memory
# ============================================================================

class MemoryNamespace(Memory):
    """Scoped, isolated memory for a single agent within a multi-agent team.

    Each sandboxed agent gets its own MemoryNamespace. The orchestrator
    may also provision a shared 'team' namespace that multiple agents can read.

    Parameters
    ----------
    namespace_id:
        Unique namespace key (usually = agent_id).
    store:
        Underlying Memory implementation. Default ShortMemory keeps
        everything in-process with no persistence.
    """

    def __init__(self, namespace_id: str, store: Memory | None = None) -> None:
        self.namespace_id = namespace_id
        self._store: Memory = store or ShortMemory()
        self._message_count = 0
        self._created_at = time.time()

    async def add(self, message: Message) -> None:
        self._message_count += 1
        await self._store.add(message)

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        return await self._store.get_context(query, limit)

    def __len__(self) -> int:
        return self._message_count

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "messages": self._message_count,
            "age_seconds": time.time() - self._created_at,
        }


# ============================================================================
#  Agent Sandbox
# ============================================================================

class AgentSandbox:
    """Complete isolation boundary for a single agent.

    Bundles resource quota, permission profile, and memory namespace
    into one unit that can be attached to any Agent.

    Usage::

        sandbox = AgentSandbox.for_agent("coder_1")
        agent = Agent(
            provider=provider,
            tools=tools,
            memory=sandbox.memory,       # isolated memory
            max_steps=sandbox.quota.max_steps,
            token_budget=sandbox.quota.token_budget,
        )
        # Enforce permissions at tool-execution time
        sandbox.profile.is_allowed(tool_name)
    """

    def __init__(
        self,
        agent_id: str,
        quota: ResourceQuota | None = None,
        profile: PermissionProfile | None = None,
        memory: MemoryNamespace | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.quota = quota or ResourceQuota.unlimited()
        self.profile = profile or PermissionProfile.standard(agent_id)
        self.memory = memory or MemoryNamespace(namespace_id=agent_id)
        self._active = False
        self._started_at: float = 0.0

    def activate(self) -> None:
        self._active = True
        self._started_at = time.time()

    def deactivate(self) -> None:
        self._active = False

    @property
    def elapsed(self) -> float:
        if self._started_at == 0:
            return 0.0
        return time.time() - self._started_at

    @property
    def is_active(self) -> bool:
        return self._active

    @classmethod
    def for_agent(
        cls,
        agent_id: str,
        level: str = "STANDARD",
    ) -> "AgentSandbox":
        """Create a sandbox with sensible defaults for a given permission level."""
        profile_map = {
            "LIMITED": PermissionProfile.limited,
            "STANDARD": PermissionProfile.standard,
            "ELEVATED": PermissionProfile.elevated,
            "FULL": PermissionProfile.full,
        }
        factory = profile_map.get(level.upper(), PermissionProfile.standard)
        return cls(
            agent_id=agent_id,
            quota=ResourceQuota.unlimited() if level.upper() == "FULL" else ResourceQuota.tight(),
            profile=factory(agent_id),
        )
