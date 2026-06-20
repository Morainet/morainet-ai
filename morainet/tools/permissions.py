"""Fine-grained tool permission system (RBAC-style).

Usage::

    perms = PermissionRegistry()
    perms.grant("user", "read", "weather")
    perms.grant("admin", "*", "*")  # wildcard
    enforcer = PermissionEnforcer(perms)
    enforcer.check("user", "read", "weather")  # True
    enforcer.check("user", "write", "weather") # False → raises ToolPermissionError
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ToolPermissionError(Exception):
    """Raised when a role lacks permission for a tool action."""


@dataclass
class PermissionRule:
    role: str
    action: str  # "read" | "write" | "execute" | "*"
    tool_name: str  # tool name or "*"


class PermissionRegistry:
    """Registry of role → (action, tool) permission grants.

    Supports wildcards: ``"*"`` matches any action or tool name.
    """

    def __init__(self) -> None:
        self._rules: list[PermissionRule] = []

    def grant(self, role: str, action: str, tool_name: str) -> None:
        """Grant ``action`` on ``tool_name`` to ``role``."""
        self._rules.append(PermissionRule(role=role, action=action, tool_name=tool_name))

    def revoke(self, role: str, action: str, tool_name: str) -> None:
        """Revoke a previously granted rule."""
        self._rules = [
            r
            for r in self._rules
            if not (r.role == role and r.action == action and r.tool_name == tool_name)
        ]

    def check(self, role: str, action: str, tool_name: str) -> bool:
        """Return ``True`` if ``role`` is permitted ``action`` on ``tool_name``."""
        for rule in self._rules:
            if rule.role != role and rule.role != "*":
                continue
            if rule.action != action and rule.action != "*":
                continue
            if rule.tool_name != tool_name and rule.tool_name != "*":
                continue
            return True
        return False

    def get_role_permissions(self, role: str) -> list[dict[str, str]]:
        """List all (action, tool_name) pairs granted to ``role``."""
        return [
            {"action": r.action, "tool_name": r.tool_name}
            for r in self._rules
            if r.role == role or r.role == "*"
        ]

    def clear(self) -> None:
        self._rules.clear()


class PermissionEnforcer:
    """Gatekeeper that wraps a PermissionRegistry for enforcement.

    ``default_action`` is the action to check when not explicitly specified
    (typically ``"execute"`` for tool calls).
    """

    def __init__(
        self,
        registry: PermissionRegistry,
        default_action: str = "execute",
    ) -> None:
        self.registry = registry
        self.default_action = default_action

    def check(self, role: str, tool_name: str, action: str | None = None) -> None:
        """Raise ``ToolPermissionError`` if permission is denied."""
        action = action or self.default_action
        if not self.registry.check(role, action, tool_name):
            raise ToolPermissionError(
                f"role '{role}' is not permitted to '{action}' tool '{tool_name}'"
            )

    def check_arguments(
        self,
        role: str,
        tool_name: str,
        arguments: dict[str, Any],
        action: str | None = None,
    ) -> None:
        """Check tool permission with optional argument-based restrictions.

        Override this in subclasses to implement field-level access control.
        """
        self.check(role, tool_name, action)


def create_default_registry(tool_names: list[str]) -> PermissionRegistry:
    """Create a registry with all tools granted to the 'default' role.

    Useful as a starting point: grant all, then revoke dangerous ones.
    """
    reg = PermissionRegistry()
    for name in tool_names:
        reg.grant("default", "execute", name)
    return reg
