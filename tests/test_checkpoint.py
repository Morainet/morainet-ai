"""Tests for morainet.persistence.checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from morainet.core.models import Message, Step, Usage
from morainet.persistence.checkpoint import (
    Checkpoint,
    CheckpointHook,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)


# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------

def test_checkpoint_creation():
    cp = Checkpoint(
        trace_id="trace-1",
        query="hello",
        messages=[Message.user("hello")],
        steps=[],
        cursor=5,
    )
    assert cp.trace_id == "trace-1"
    assert cp.query == "hello"
    assert len(cp.messages) == 1
    assert cp.cursor == 5
    assert cp.usage.total_tokens == 0
    assert cp.created_at is not None


def test_checkpoint_defaults():
    cp = Checkpoint(trace_id="t1", query="q")
    assert cp.messages == []
    assert cp.steps == []
    assert cp.cursor == 0


def test_checkpoint_with_usage():
    cp = Checkpoint(trace_id="t1", query="q", usage=Usage(total_tokens=100))
    assert cp.usage.total_tokens == 100


def test_checkpoint_from_context():
    """from_context builds a checkpoint from a context object."""
    from morainet.core.context import Context

    ctx = Context(trace_id="trace-ctx", query="test query")
    ctx.messages.append(Message.user("hi"))
    ctx.usage = Usage(total_tokens=42)

    cp = Checkpoint.from_context(ctx, cursor=3)
    assert cp.trace_id == "trace-ctx"
    assert cp.query == "test query"
    assert len(cp.messages) == 1
    assert cp.messages[0].content == "hi"
    assert cp.cursor == 3
    assert cp.usage.total_tokens == 42


# ---------------------------------------------------------------------------
# InMemoryCheckpointStore
# ---------------------------------------------------------------------------

async def test_inmemory_save_and_load():
    store = InMemoryCheckpointStore()
    cp = Checkpoint(trace_id="t1", query="q", cursor=1)

    await store.save(cp)
    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"
    assert loaded.query == "q"
    assert loaded.cursor == 1


async def test_inmemory_load_missing():
    store = InMemoryCheckpointStore()
    assert await store.load("nonexistent") is None


async def test_inmemory_overwrite():
    store = InMemoryCheckpointStore()
    cp1 = Checkpoint(trace_id="t1", query="v1")
    cp2 = Checkpoint(trace_id="t1", query="v2")

    await store.save(cp1)
    await store.save(cp2)
    loaded = await store.load("t1")
    assert loaded.query == "v2"


# ---------------------------------------------------------------------------
# FileCheckpointStore
# ---------------------------------------------------------------------------

async def test_file_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileCheckpointStore(directory=tmpdir)
        cp = Checkpoint(trace_id="trace-f1", query="file test", cursor=3)
        await store.save(cp)

        loaded = await store.load("trace-f1")
        assert loaded is not None
        assert loaded.trace_id == "trace-f1"
        assert loaded.query == "file test"
        assert loaded.cursor == 3


async def test_file_load_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileCheckpointStore(directory=tmpdir)
        assert await store.load("nonexistent") is None


async def test_file_creates_directory():
    with tempfile.TemporaryDirectory() as parent:
        subdir = str(Path(parent) / "subdir")
        store = FileCheckpointStore(directory=subdir)
        assert Path(subdir).exists()

        cp = Checkpoint(trace_id="t1", query="q")
        await store.save(cp)
        assert (Path(subdir) / "t1.json").exists()


# ---------------------------------------------------------------------------
# SQLiteCheckpointStore
# ---------------------------------------------------------------------------

async def test_sqlite_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteCheckpointStore(path=db_path)
        cp = Checkpoint(trace_id="trace-sql1", query="sql test", cursor=7)
        await store.save(cp)

        loaded = await store.load("trace-sql1")
        assert loaded is not None
        assert loaded.trace_id == "trace-sql1"
        assert loaded.query == "sql test"
        assert loaded.cursor == 7

        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


async def test_sqlite_load_missing():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteCheckpointStore(path=db_path)
        assert await store.load("nonexistent") is None
        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


async def test_sqlite_overwrite():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteCheckpointStore(path=db_path)
        await store.save(Checkpoint(trace_id="t1", query="v1"))
        await store.save(Checkpoint(trace_id="t1", query="v2"))

        loaded = await store.load("t1")
        assert loaded.query == "v2"
        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


async def test_sqlite_close():
    store = SQLiteCheckpointStore(path=":memory:")
    store.close()
    # Should not raise


# ---------------------------------------------------------------------------
# CheckpointHook
# ---------------------------------------------------------------------------

async def test_checkpoint_hook_on_run_start():
    from morainet.core.context import Context

    store = InMemoryCheckpointStore()
    hook = CheckpointHook(store=store)
    ctx = Context(trace_id="hook-test", query="hook")

    await hook.on_run_start(ctx)
    assert hook._cursor == 0


async def test_checkpoint_hook_on_llm_end():
    from morainet.core.context import Context
    from morainet.core.models import ChatResponse, Message

    store = InMemoryCheckpointStore()
    hook = CheckpointHook(store=store)
    ctx = Context(trace_id="hook-test-2", query="llm hook")
    ctx.messages.append(Message.user("test"))

    await hook.on_llm_end(ctx, ChatResponse(message=Message.assistant(content="ok")))
    assert hook._cursor == 1

    loaded = await store.load("hook-test-2")
    assert loaded is not None
    assert loaded.cursor == 1
    assert loaded.query == "llm hook"


async def test_checkpoint_hook_on_tool_end():
    from morainet.core.context import Context
    from morainet.core.models import Step, StepStatus, ToolCall

    store = InMemoryCheckpointStore()
    hook = CheckpointHook(store=store)
    ctx = Context(trace_id="hook-tool", query="tool test")

    step = Step(
        index=1,
        description="echo",
        output="hi",
        status=StepStatus.SUCCESS,
    )
    await hook.on_tool_end(ctx, step)
    assert hook._cursor == 1

    loaded = await store.load("hook-tool")
    assert loaded is not None


async def test_checkpoint_hook_on_run_end():
    from morainet.core.context import Context
    from morainet.core.models import AgentResult, Message, Usage

    store = InMemoryCheckpointStore()
    hook = CheckpointHook(store=store)
    ctx = Context(trace_id="hook-run-end", query="end test")

    result = AgentResult(
        trace_id="hook-run-end",
        steps=[],
        final_answer="answer",
        usage=Usage(total_tokens=10),
    )
    await hook.on_run_end(ctx, result)
    loaded = await store.load("hook-run-end")
    assert loaded is not None
    assert loaded.query == "end test"
