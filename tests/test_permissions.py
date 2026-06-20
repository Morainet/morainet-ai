from __future__ import annotations

import pytest

from morainet.tools.permissions import (
    PermissionEnforcer,
    PermissionRegistry,
    PermissionRule,
    ToolPermissionError,
    create_default_registry,
)


# ---------------------------------------------------------------------------
# PermissionRule
# ---------------------------------------------------------------------------

def test_permission_rule_fields():
    rule = PermissionRule(role="admin", action="execute", tool_name="delete_file")
    assert rule.role == "admin"
    assert rule.action == "execute"
    assert rule.tool_name == "delete_file"


# ---------------------------------------------------------------------------
# PermissionRegistry.grant / check
# ---------------------------------------------------------------------------

def test_grant_and_check_exact_match():
    reg = PermissionRegistry()
    reg.grant("user", "read", "weather")
    assert reg.check("user", "read", "weather") is True
    assert reg.check("user", "write", "weather") is False
    assert reg.check("admin", "read", "weather") is False


def test_check_wildcard_role():
    reg = PermissionRegistry()
    reg.grant("*", "read", "weather")
    assert reg.check("alice", "read", "weather") is True
    assert reg.check("bob", "read", "weather") is True
    assert reg.check("alice", "write", "weather") is False


def test_check_wildcard_action():
    reg = PermissionRegistry()
    reg.grant("user", "*", "weather")
    assert reg.check("user", "read", "weather") is True
    assert reg.check("user", "write", "weather") is True
    assert reg.check("user", "execute", "weather") is True


def test_check_wildcard_tool():
    reg = PermissionRegistry()
    reg.grant("user", "read", "*")
    assert reg.check("user", "read", "weather") is True
    assert reg.check("user", "read", "news") is True
    assert reg.check("user", "write", "weather") is False


def test_check_wildcard_role_action_grants_everything():
    reg = PermissionRegistry()
    reg.grant("*", "*", "*")
    assert reg.check("anyone", "anything", "any-tool") is True


def test_check_denied_by_default():
    reg = PermissionRegistry()
    assert reg.check("user", "read", "weather") is False


# ---------------------------------------------------------------------------
# PermissionRegistry.revoke
# ---------------------------------------------------------------------------

def test_revoke_removes_specific_rule():
    reg = PermissionRegistry()
    reg.grant("user", "read", "weather")
    reg.grant("user", "read", "news")
    reg.revoke("user", "read", "weather")
    assert reg.check("user", "read", "weather") is False
    assert reg.check("user", "read", "news") is True


# ---------------------------------------------------------------------------
# PermissionRegistry.get_role_permissions
# ---------------------------------------------------------------------------

def test_get_role_permissions():
    reg = PermissionRegistry()
    reg.grant("user", "read", "weather")
    reg.grant("user", "execute", "calculator")
    perms = reg.get_role_permissions("user")
    assert len(perms) == 2
    actions = {p["action"] for p in perms}
    assert actions == {"read", "execute"}


def test_get_role_permissions_wildcard_role_includes_all():
    reg = PermissionRegistry()
    reg.grant("*", "read", "public-tool")
    reg.grant("user", "execute", "private-tool")
    perms = reg.get_role_permissions("user")
    assert len(perms) == 2


# ---------------------------------------------------------------------------
# PermissionRegistry.clear
# ---------------------------------------------------------------------------

def test_clear_removes_all_rules():
    reg = PermissionRegistry()
    reg.grant("user", "read", "weather")
    reg.clear()
    assert reg.check("user", "read", "weather") is False
    assert reg.get_role_permissions("user") == []


# ---------------------------------------------------------------------------
# PermissionEnforcer
# ---------------------------------------------------------------------------

def test_enforcer_check_passing():
    reg = PermissionRegistry()
    reg.grant("user", "execute", "tool_a")
    enforcer = PermissionEnforcer(reg)
    enforcer.check("user", "tool_a")


def test_enforcer_check_failing():
    reg = PermissionRegistry()
    enforcer = PermissionEnforcer(reg)
    with pytest.raises(ToolPermissionError):
        enforcer.check("user", "tool_a")


def test_enforcer_default_action():
    reg = PermissionRegistry()
    reg.grant("user", "execute", "tool_a")
    enforcer = PermissionEnforcer(reg, default_action="execute")
    enforcer.check("user", "tool_a")


def test_enforcer_custom_action():
    reg = PermissionRegistry()
    reg.grant("user", "read", "tool_a")
    enforcer = PermissionEnforcer(reg)
    enforcer.check("user", "tool_a", action="read")


def test_enforcer_check_arguments_delegates():
    reg = PermissionRegistry()
    reg.grant("user", "execute", "tool_a")
    enforcer = PermissionEnforcer(reg)
    enforcer.check_arguments("user", "tool_a", {"key": "val"})


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------

def test_create_default_registry():
    reg = create_default_registry(["weather", "calculator", "search"])
    assert reg.check("default", "execute", "weather") is True
    assert reg.check("default", "execute", "calculator") is True
    assert reg.check("default", "execute", "search") is True
    assert reg.check("other", "execute", "weather") is False
