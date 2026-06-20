from __future__ import annotations

from morainet.tools.audit import AuditEntry, AuditLogger, InMemoryAuditStore


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------

def test_audit_entry_construction():
    entry = AuditEntry(
        trace_id="trace_1",
        role="agent",
        tool_name="search",
        action="execute",
        arguments={"query": "weather"},
        result="sunny",
        duration_ms=150.0,
    )
    assert entry.trace_id == "trace_1"
    assert entry.role == "agent"
    assert entry.tool_name == "search"
    assert entry.action == "execute"
    assert entry.arguments == {"query": "weather"}
    assert entry.result == "sunny"
    assert entry.error is None
    assert entry.duration_ms == 150.0
    assert isinstance(entry.timestamp, float)


def test_audit_entry_defaults():
    entry = AuditEntry(trace_id="t1", role="admin", tool_name="delete", action="deny")
    assert entry.arguments == {}
    assert entry.result is None
    assert entry.error is None
    assert entry.duration_ms == 0.0


def test_audit_entry_to_dict():
    entry = AuditEntry(
        trace_id="trace_1",
        role="agent",
        tool_name="search",
        action="execute",
    )
    d = entry.to_dict()
    assert d["trace_id"] == "trace_1"
    assert d["role"] == "agent"
    assert d["tool_name"] == "search"
    assert d["action"] == "execute"


# ---------------------------------------------------------------------------
# InMemoryAuditStore
# ---------------------------------------------------------------------------

async def test_write_and_query():
    store = InMemoryAuditStore()
    entry = AuditEntry(trace_id="t1", role="agent", tool_name="search", action="execute")
    await store.write(entry)
    results = await store.query()
    assert len(results) == 1
    assert results[0].tool_name == "search"


async def test_query_filter_by_trace_id():
    store = InMemoryAuditStore()
    await store.write(AuditEntry(trace_id="t1", role="agent", tool_name="search", action="execute"))
    await store.write(AuditEntry(trace_id="t2", role="agent", tool_name="calc", action="execute"))
    results = await store.query(trace_id="t1")
    assert len(results) == 1
    assert results[0].tool_name == "search"


async def test_query_filter_by_tool_name():
    store = InMemoryAuditStore()
    await store.write(AuditEntry(trace_id="t1", role="agent", tool_name="search", action="execute"))
    await store.write(AuditEntry(trace_id="t2", role="agent", tool_name="calc", action="execute"))
    results = await store.query(tool_name="calc")
    assert len(results) == 1


async def test_query_filter_by_role():
    store = InMemoryAuditStore()
    await store.write(AuditEntry(trace_id="t1", role="agent", tool_name="search", action="execute"))
    await store.write(AuditEntry(trace_id="t2", role="admin", tool_name="search", action="execute"))
    results = await store.query(role="admin")
    assert len(results) == 1


async def test_query_filter_by_action():
    store = InMemoryAuditStore()
    await store.write(AuditEntry(trace_id="t1", role="agent", tool_name="search", action="execute"))
    await store.write(AuditEntry(trace_id="t2", role="agent", tool_name="search", action="deny"))
    results = await store.query(action="deny")
    assert len(results) == 1


async def test_query_limit_and_offset():
    store = InMemoryAuditStore()
    for i in range(5):
        await store.write(AuditEntry(trace_id=f"t{i}", role="agent", tool_name="t", action="execute"))
    results = await store.query(limit=2, offset=1)
    assert len(results) == 2


async def test_query_empty_store():
    store = InMemoryAuditStore()
    results = await store.query()
    assert results == []


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

async def test_log_execution():
    store = InMemoryAuditStore()
    logger = AuditLogger(store)
    await logger.log_execution(
        trace_id="trace_1",
        tool_name="search",
        arguments={"query": "weather"},
        result="sunny",
        duration_ms=100.0,
    )
    results = await store.query()
    assert len(results) == 1
    assert results[0].action == "execute"
    assert results[0].tool_name == "search"
    assert results[0].result == "sunny"


async def test_log_execution_with_error():
    store = InMemoryAuditStore()
    logger = AuditLogger(store)
    await logger.log_execution(
        trace_id="trace_1",
        tool_name="search",
        error="connection refused",
    )
    results = await store.query()
    assert results[0].action == "error"
    assert results[0].error == "connection refused"


async def test_log_execution_with_custom_role():
    store = InMemoryAuditStore()
    logger = AuditLogger(store, default_role="admin")
    await logger.log_execution(
        trace_id="trace_1",
        tool_name="search",
        role="supervisor",
    )
    results = await store.query()
    assert results[0].role == "supervisor"


async def test_log_execution_uses_default_role():
    store = InMemoryAuditStore()
    logger = AuditLogger(store, default_role="agent")
    await logger.log_execution(trace_id="trace_1", tool_name="search")
    results = await store.query()
    assert results[0].role == "agent"


async def test_log_approve():
    store = InMemoryAuditStore()
    logger = AuditLogger(store)
    await logger.log_approve(
        trace_id="trace_1",
        tool_name="delete_file",
        arguments={"path": "/tmp/x"},
    )
    results = await store.query()
    assert results[0].action == "approve"


async def test_log_deny():
    store = InMemoryAuditStore()
    logger = AuditLogger(store)
    await logger.log_approve(
        trace_id="trace_1",
        tool_name="delete_file",
        approved=False,
    )
    results = await store.query()
    assert results[0].action == "deny"
