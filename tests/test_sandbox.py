from __future__ import annotations

import time

from morainet.core.models import Message
from morainet.multiagent.sandbox import (
    AgentSandbox,
    MemoryNamespace,
    PermissionProfile,
    ResourceQuota,
)


# ============================================================================
#  ResourceQuota
# ============================================================================

class TestResourceQuota:
    def test_check_step_without_limit(self):
        quota = ResourceQuota()
        assert quota.check_step(0) is True
        assert quota.check_step(1000) is True

    def test_check_step_with_limit(self):
        quota = ResourceQuota(max_steps=3)
        assert quota.check_step(0) is True
        assert quota.check_step(2) is True
        assert quota.check_step(3) is False
        assert quota.check_step(4) is False

    def test_check_tokens_without_limit(self):
        quota = ResourceQuota()
        assert quota.check_tokens(0) is True
        assert quota.check_tokens(99999) is True

    def test_check_tokens_with_limit(self):
        quota = ResourceQuota(token_budget=100)
        assert quota.check_tokens(0) is True
        assert quota.check_tokens(99) is True
        assert quota.check_tokens(100) is False
        assert quota.check_tokens(200) is False

    def test_check_time(self):
        quota = ResourceQuota(time_budget=0.5)
        now = time.time()
        assert quota.check_time(now) is True
        past = now - 1.0
        assert quota.check_time(past) is False

    def test_check_time_no_limit(self):
        quota = ResourceQuota()
        assert quota.check_time(time.time()) is True

    def test_unlimited_factory(self):
        quota = ResourceQuota.unlimited()
        assert quota.max_steps == 0
        assert quota.token_budget == 0
        assert quota.time_budget == 0.0

    def test_tight_factory(self):
        quota = ResourceQuota.tight()
        assert quota.max_steps == 5
        assert quota.token_budget == 8000
        assert quota.time_budget == 30.0


# ============================================================================
#  PermissionProfile
# ============================================================================

class TestPermissionProfile:
    def test_allow_all_permission(self):
        profile = PermissionProfile(allow_all=True)
        assert profile.is_allowed("any_tool") is True
        assert profile.is_allowed("delete_file") is True

    def test_allowlist_blocks_unknown(self):
        profile = PermissionProfile(allow_all=False, allowlist={"search", "read"})
        assert profile.is_allowed("search") is True
        assert profile.is_allowed("write") is False

    def test_denylist_blocks_even_allow_all(self):
        profile = PermissionProfile(allow_all=True, denylist={"delete_file"})
        assert profile.is_allowed("search") is True
        assert profile.is_allowed("delete_file") is False

    def test_allowlist_and_denylist_together(self):
        profile = PermissionProfile(
            allow_all=False,
            allowlist={"search", "delete_file"},
            denylist={"delete_file"},
        )
        assert profile.is_allowed("search") is True
        assert profile.is_allowed("delete_file") is False

    def test_limited_factory(self):
        profile = PermissionProfile.limited("agent_1")
        assert profile.agent_id == "agent_1"
        assert profile.level == "LIMITED"
        assert profile.allow_all is False
        assert profile.is_allowed("search") is True
        assert profile.is_allowed("delete_file") is False

    def test_standard_factory(self):
        profile = PermissionProfile.standard("agent_1")
        assert profile.level == "STANDARD"
        assert profile.is_allowed("write_file") is True
        assert profile.is_allowed("delete_file") is False
        assert profile.is_allowed("deploy") is False

    def test_elevated_factory(self):
        profile = PermissionProfile.elevated("agent_1", block={"deploy"})
        assert profile.level == "ELEVATED"
        assert profile.allow_all is True
        assert profile.is_allowed("search") is True
        assert profile.is_allowed("deploy") is False

    def test_full_factory(self):
        profile = PermissionProfile.full("agent_1")
        assert profile.level == "FULL"
        assert profile.allow_all is True
        assert profile.is_allowed("any_tool") is True
        assert profile.is_allowed("delete_file") is True


# ============================================================================
#  MemoryNamespace
# ============================================================================

class TestMemoryNamespace:
    async def test_add_increments_count(self):
        ns = MemoryNamespace("ns_1")
        msg = Message.user("hello")
        await ns.add(msg)
        assert len(ns) == 1

    async def test_multiple_adds(self):
        ns = MemoryNamespace("ns_1")
        await ns.add(Message.user("msg1"))
        await ns.add(Message.user("msg2"))
        await ns.add(Message.user("msg3"))
        assert len(ns) == 3

    async def test_get_context_delegates(self):
        ns = MemoryNamespace("ns_1")
        result = await ns.get_context("query", limit=5)
        assert isinstance(result, list)

    def test_stats_property(self):
        ns = MemoryNamespace("ns_1")
        stats = ns.stats
        assert stats["namespace_id"] == "ns_1"
        assert stats["messages"] == 0
        assert stats["age_seconds"] >= 0


# ============================================================================
#  AgentSandbox
# ============================================================================

class TestAgentSandbox:
    def test_initialization(self):
        sandbox = AgentSandbox("agent_1")
        assert sandbox.agent_id == "agent_1"
        assert sandbox.quota is not None
        assert sandbox.profile is not None
        assert sandbox.memory is not None
        assert sandbox.is_active is False

    def test_activate_deactivate(self):
        sandbox = AgentSandbox("agent_1")
        assert sandbox.is_active is False
        sandbox.activate()
        assert sandbox.is_active is True
        sandbox.deactivate()
        assert sandbox.is_active is False

    def test_elapsed_time(self):
        sandbox = AgentSandbox("agent_1")
        assert sandbox.elapsed == 0.0
        sandbox.activate()
        assert sandbox.elapsed >= 0

    def test_for_agent_standard(self):
        sandbox = AgentSandbox.for_agent("agent_1", "STANDARD")
        assert sandbox.profile.level == "STANDARD"
        assert sandbox.quota.max_steps == 5

    def test_for_agent_full(self):
        sandbox = AgentSandbox.for_agent("agent_1", "FULL")
        assert sandbox.profile.level == "FULL"
        assert sandbox.profile.allow_all is True

    def test_for_agent_limited(self):
        sandbox = AgentSandbox.for_agent("agent_1", "LIMITED")
        assert sandbox.profile.level == "LIMITED"

    def test_for_agent_elevated(self):
        sandbox = AgentSandbox.for_agent("agent_1", "ELEVATED")
        assert sandbox.profile.level == "ELEVATED"

    def test_for_agent_unknown_level_defaults_to_standard(self):
        sandbox = AgentSandbox.for_agent("agent_1", "UNKNOWN")
        assert sandbox.profile.level == "STANDARD"
