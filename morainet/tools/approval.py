"""Human-in-the-loop approval workflow for dangerous tool calls.

Supports interactive CLI approval and async callback-based approval.

Usage::

    # CLI-based interactive approval
    approver = InteractiveApprover()
    result = await approver.request_approval("delete_file", {"path": "/tmp/x"})
    if result.approved:
        await tool.invoke(...)

    # Programmatic callback approval
    def my_approver(name, args) -> bool:
        return args.get("amount", 0) < 1000

    # Use with Agent:
    agent = Agent(
        provider=...,
        tools=[tool(dangerous=True)(delete_file)],
        approve_tool=my_approver,
    )
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from morainet.observability.tracing import logger


@dataclass
class ApprovalRequest:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    requested_at: float = field(default_factory=lambda: __import__("time").time())

    def describe(self) -> str:
        args_repr = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        reason = f" — {self.reason}" if self.reason else ""
        return f"[{self.id}] {self.tool_name}({args_repr}){reason}"


@dataclass
class ApprovalResponse:
    request_id: str
    approved: bool
    reason: str = ""
    approved_by: str = ""
    approved_at: float = field(default_factory=lambda: __import__("time").time())


class Approver(ABC):
    """Abstract approver for dangerous tool calls."""

    @abstractmethod
    async def approve(self, request: ApprovalRequest) -> ApprovalResponse: ...


class InteractiveApprover(Approver):
    """CLI-based interactive approval — prompts user via stdin.

    ``timeout_seconds`` — if > 0, auto-deny after this many seconds.
    """

    def __init__(
        self,
        timeout_seconds: float = 0.0,
        auto_deny_on_timeout: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.auto_deny_on_timeout = auto_deny_on_timeout

    async def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        prompt = (
            f"\n⚠️  Dangerous tool call requested:\n"
            f"   {request.describe()}\n"
            f"   Approve? [y/N]: "
        )

        try:
            if self.timeout_seconds > 0:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(input, prompt),
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Approval timeout ({self.timeout_seconds}s) for {request.tool_name}"
                    )
                    return ApprovalResponse(
                        request_id=request.id,
                        approved=not self.auto_deny_on_timeout,
                        reason="timeout" if self.auto_deny_on_timeout else "auto-approved",
                        approved_by="system",
                    )
            else:
                result = await asyncio.to_thread(input, prompt)
        except EOFError:
            return ApprovalResponse(
                request_id=request.id, approved=False, reason="EOF on stdin", approved_by="system"
            )

        approved = result.strip().lower() in ("y", "yes")
        return ApprovalResponse(
            request_id=request.id,
            approved=approved,
            reason="user input" if approved else "user denied",
            approved_by="human" if approved else "",
        )


class CallbackApprover(Approver):
    """Programmatic approver using an async callback.

    ``callback`` receives (tool_name, arguments) and returns bool.
    """

    def __init__(
        self,
        callback: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self.callback = callback

    async def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        import inspect

        result = self.callback(request.tool_name, request.arguments)
        if inspect.isawaitable(result):
            approved = await result
        else:
            approved = result

        return ApprovalResponse(
            request_id=request.id,
            approved=bool(approved),
            reason="callback",
            approved_by="programmatic",
        )


class ApprovalStore(ABC):
    """Abstract store for pending/reviewed approval requests."""

    @abstractmethod
    async def save(self, request: ApprovalRequest) -> None: ...

    @abstractmethod
    async def get(self, request_id: str) -> ApprovalRequest | None: ...

    @abstractmethod
    async def list_pending(self) -> list[ApprovalRequest]: ...


class InMemoryApprovalStore(ApprovalStore):
    """In-memory approval request store."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    async def save(self, request: ApprovalRequest) -> None:
        self._requests[request.id] = request

    async def get(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    async def list_pending(self) -> list[ApprovalRequest]:
        return list(self._requests.values())


class ApprovalFlow:
    """Orchestrates the full approval lifecycle: request → approve → audit.

    ``approver``   — the Approver implementation.
    ``store``      — optional persistence for approval requests.
    ``audit_store``— optional audit store for recording decisions.
    ``risk_levels``— dict mapping tool_name → "low" | "medium" | "high" | "critical".
                     Tools not listed default to "low".
    ``require_approval_for``— set of risk levels that require human approval.
    """

    def __init__(
        self,
        approver: Approver,
        store: ApprovalStore | None = None,
        audit_store: Any = None,  # AuditStore
        risk_levels: dict[str, str] | None = None,
        require_approval_for: set[str] | None = None,
    ) -> None:
        self.approver = approver
        self.store = store or InMemoryApprovalStore()
        self._audit_store = audit_store
        self.risk_levels = risk_levels or {}
        self.require_approval_for = require_approval_for or {"high", "critical"}

    def _risk(self, tool_name: str) -> str:
        return self.risk_levels.get(tool_name, "low")

    async def check_and_approve(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str = "",
    ) -> ApprovalResponse:
        """Check if approval is needed, and if so, request it.

        Returns ``ApprovalResponse(approved=True)`` immediately for tools
        whose risk level does not require approval.
        """
        risk = self._risk(tool_name)
        if risk not in self.require_approval_for:
            return ApprovalResponse(
                request_id="auto",
                approved=True,
                reason=f"risk level '{risk}' does not require approval",
                approved_by="system",
            )

        request = ApprovalRequest(
            tool_name=tool_name,
            arguments=arguments,
            reason=reason,
        )
        await self.store.save(request)

        response = await self.approver.approve(request)
        logger.info(
            f"[{trace_id}] approval: {tool_name} → "
            f"{'APPROVED' if response.approved else 'DENIED'} by {response.approved_by}"
        )

        # Record to audit store if available
        if self._audit_store is not None:
            try:
                from morainet.tools.audit import AuditEntry

                entry = AuditEntry(
                    trace_id=trace_id,
                    role="approval-flow",
                    tool_name=tool_name,
                    action="approve" if response.approved else "deny",
                    arguments=arguments,
                    result=response.reason,
                )
                await self._audit_store.write(entry)
            except Exception:
                pass

        return response
