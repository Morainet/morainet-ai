from __future__ import annotations

from morainet.tools.approval import (
    ApprovalFlow,
    ApprovalRequest,
    ApprovalResponse,
    CallbackApprover,
    InMemoryApprovalStore,
)


# ---------------------------------------------------------------------------
# ApprovalRequest
# ---------------------------------------------------------------------------

def test_approval_request_construction():
    req = ApprovalRequest(
        tool_name="delete_file",
        arguments={"path": "/tmp/x"},
        reason="cleanup",
    )
    assert req.tool_name == "delete_file"
    assert req.arguments == {"path": "/tmp/x"}
    assert req.reason == "cleanup"
    assert len(req.id) == 12


def test_approval_request_describe():
    req = ApprovalRequest(
        tool_name="delete_file",
        arguments={"path": "/tmp/x"},
        reason="cleanup",
    )
    desc = req.describe()
    assert "delete_file" in desc
    assert "path='/tmp/x'" in desc
    assert "cleanup" in desc


def test_approval_request_describe_no_reason():
    req = ApprovalRequest(tool_name="search", arguments={"query": "weather"})
    desc = req.describe()
    assert "search" in desc
    assert "query='weather'" in desc


# ---------------------------------------------------------------------------
# ApprovalResponse
# ---------------------------------------------------------------------------

def test_approval_response_construction():
    resp = ApprovalResponse(
        request_id="abc123",
        approved=True,
        reason="callback",
        approved_by="programmatic",
    )
    assert resp.request_id == "abc123"
    assert resp.approved is True
    assert resp.reason == "callback"
    assert resp.approved_by == "programmatic"


# ---------------------------------------------------------------------------
# CallbackApprover
# ---------------------------------------------------------------------------

async def test_callback_approver_approved():
    approver = CallbackApprover(lambda name, args: True)
    req = ApprovalRequest(tool_name="search", arguments={"query": "weather"})
    resp = await approver.approve(req)
    assert resp.approved is True
    assert resp.approved_by == "programmatic"


async def test_callback_approver_denied():
    approver = CallbackApprover(lambda name, args: False)
    req = ApprovalRequest(tool_name="delete_file", arguments={"path": "/"})
    resp = await approver.approve(req)
    assert resp.approved is False


async def test_callback_approver_receives_correct_args():
    seen = {}

    def callback(name, args):
        seen["name"] = name
        seen["args"] = args
        return True

    approver = CallbackApprover(callback)
    req = ApprovalRequest(tool_name="search", arguments={"q": "test"})
    await approver.approve(req)
    assert seen["name"] == "search"
    assert seen["args"] == {"q": "test"}


# ---------------------------------------------------------------------------
# InMemoryApprovalStore
# ---------------------------------------------------------------------------

async def test_inmemory_approval_store():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(tool_name="search", arguments={"q": "test"})
    await store.save(req)
    retrieved = await store.get(req.id)
    assert retrieved is not None
    assert retrieved.tool_name == "search"


async def test_inmemory_approval_store_get_nonexistent():
    store = InMemoryApprovalStore()
    assert await store.get("nonexistent") is None


async def test_inmemory_approval_store_list_pending():
    store = InMemoryApprovalStore()
    req1 = ApprovalRequest(tool_name="search", arguments={"q": "1"})
    req2 = ApprovalRequest(tool_name="calc", arguments={"expr": "2+2"})
    await store.save(req1)
    await store.save(req2)
    pending = await store.list_pending()
    assert len(pending) == 2


# ---------------------------------------------------------------------------
# ApprovalFlow
# ---------------------------------------------------------------------------

async def test_approval_flow_low_risk_auto_approved():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: False),
        risk_levels={"search": "low"},
    )
    resp = await flow.check_and_approve(
        trace_id="t1",
        tool_name="search",
        arguments={"q": "test"},
    )
    assert resp.approved is True
    assert resp.approved_by == "system"
    assert "does not require approval" in resp.reason


async def test_approval_flow_high_risk_requires_approval():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: True),
        risk_levels={"delete_file": "high"},
    )
    resp = await flow.check_and_approve(
        trace_id="t1",
        tool_name="delete_file",
        arguments={"path": "/"},
    )
    assert resp.approved is True
    assert resp.approved_by == "programmatic"


async def test_approval_flow_high_risk_denied():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: False),
        risk_levels={"delete_file": "high"},
    )
    resp = await flow.check_and_approve(
        trace_id="t1",
        tool_name="delete_file",
        arguments={"path": "/"},
    )
    assert resp.approved is False


async def test_approval_flow_risk_defaults_to_low():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: False),
    )
    resp = await flow.check_and_approve(
        trace_id="t1",
        tool_name="unknown_tool",
        arguments={},
    )
    assert resp.approved is True


async def test_approval_flow_critical_requires_approval():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: False),
        risk_levels={"drop_db": "critical"},
        require_approval_for={"critical"},
    )
    resp = await flow.check_and_approve(
        trace_id="t1",
        tool_name="drop_db",
        arguments={},
    )
    assert resp.approved is False


def test_approval_flow_risk_method():
    flow = ApprovalFlow(
        approver=CallbackApprover(lambda n, a: True),
        risk_levels={"tool_a": "high", "tool_b": "critical"},
    )
    assert flow._risk("tool_a") == "high"
    assert flow._risk("tool_b") == "critical"
    assert flow._risk("unknown") == "low"
