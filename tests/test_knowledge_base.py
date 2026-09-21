"""Offline tests for the knowledge base (in-memory store + hash embedder).

Covers ingestion, TTL expiry, incremental removal/update, snapshots (create /
restore / rotation), catalogue persistence, search, and stats — no external
services required.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from morainet.memory.embeddings import HashEmbedder
from morainet.memory.knowledge_base import (
    DocumentRecord,
    KnowledgeBase,
    SnapshotMeta,
    run_cleanup,
)
from morainet.memory.stores import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


def test_document_record_expired():
    assert not DocumentRecord(id="x", source="s").expired
    assert DocumentRecord(id="x", source="s", expires_at=time.time() - 1).expired
    assert not DocumentRecord(id="x", source="s", expires_at=time.time() + 100).expired


def test_snapshot_meta_defaults():
    meta = SnapshotMeta(id="s1", name="v1")
    assert meta.document_count == 0
    assert meta.total_chunks == 0
    assert isinstance(meta.created_at, float)


# ---------------------------------------------------------------------------
# Construction / basic stats
# ---------------------------------------------------------------------------


def test_init_defaults():
    kb = KnowledgeBase()
    assert isinstance(kb.store, InMemoryVectorStore)
    assert isinstance(kb.embedder, HashEmbedder)
    assert kb.chunk_count == 0
    assert kb.document_count == 0
    assert kb.version == 1


async def test_ingest_text():
    kb = KnowledgeBase()
    n = await kb.ingest_text("hello world", source="doc1", title="Doc")
    assert n == 1
    assert kb.chunk_count == 1
    assert kb.document_count == 1
    assert kb.version == 1


async def test_ingest_text_with_tags_and_metadata():
    kb = KnowledgeBase()
    await kb.ingest_text("alpha", source="a", tags=["t1"], metadata={"k": "v"})
    assert kb.chunk_count == 1
    assert await kb.remove_by_tag("t1") == 1
    assert kb.chunk_count == 0


async def test_search_returns_results():
    kb = KnowledgeBase()
    await kb.ingest_text("the quick brown fox", source="a")
    results = await kb.search("quick brown", top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1


async def test_stats():
    kb = KnowledgeBase()
    await kb.ingest_text("x", source="a")
    stats = kb.stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1
    assert stats["store_backend"] == "InMemoryVectorStore"
    assert stats["embedder"] == "HashEmbedder"
    assert stats["snapshot_count"] == 0


# ---------------------------------------------------------------------------
# Ingestion from files / directories
# ---------------------------------------------------------------------------


async def test_ingest_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    kb = KnowledgeBase()
    n = await kb.ingest_file(str(p))
    assert n >= 1
    assert kb.chunk_count >= 1
    assert kb.document_count == 1


async def test_ingest_directory(tmp_path):
    (tmp_path / "x.txt").write_text("alpha beta")
    (tmp_path / "y.txt").write_text("gamma delta")
    kb = KnowledgeBase()
    n = await kb.ingest_directory(str(tmp_path), glob="*.txt")
    assert n >= 2
    assert kb.document_count == 2


async def test_update_document(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("version one")
    kb = KnowledgeBase()
    await kb.ingest_file(str(p))
    p.write_text("version two changed")
    n = await kb.update_document(str(p))
    assert n >= 1
    assert kb.document_count == 1


# ---------------------------------------------------------------------------
# Removal / cleanup
# ---------------------------------------------------------------------------


async def test_remove_document():
    kb = KnowledgeBase()
    await kb.ingest_text("hello", source="doc1")
    removed = await kb.remove_document("doc1")
    assert removed == 1
    assert kb.chunk_count == 0


async def test_remove_document_unknown_source():
    kb = KnowledgeBase()
    assert await kb.remove_document("nope") == 0


async def test_cleanup_expired():
    kb = KnowledgeBase()
    await kb.ingest_text("expire me", source="a", ttl=0.001)
    await asyncio.sleep(0.01)
    removed = await kb.cleanup_expired()
    assert removed == 1
    assert kb.chunk_count == 0


# ---------------------------------------------------------------------------
# Catalogue persistence
# ---------------------------------------------------------------------------


async def test_catalogue_persistence(tmp_path):
    path = tmp_path / "cat.json"
    kb = KnowledgeBase(catalogue_path=path)
    await kb.ingest_text("hello", source="a")
    assert path.exists()

    kb2 = KnowledgeBase(catalogue_path=path)
    assert kb2.chunk_count == 1
    assert kb2.document_count == 1


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


async def test_create_and_list_snapshots():
    kb = KnowledgeBase()
    await kb.ingest_text("one", source="a")
    await kb.ingest_text("two", source="b")
    snap = kb.create_snapshot("v1", description="first")
    assert snap.document_count == 2
    assert snap.total_chunks == 2
    listed = kb.list_snapshots()
    assert len(listed) == 1
    assert listed[0].name == "v1"


async def test_snapshot_rotation():
    kb = KnowledgeBase(max_versions=2)
    for i in range(4):
        kb.create_snapshot(f"s{i}")
        time.sleep(0.001)
    assert len(kb.list_snapshots()) == 2


def test_snapshot_rotation_removes_files(tmp_path):
    path = tmp_path / "cat.json"
    kb = KnowledgeBase(catalogue_path=path, max_versions=1)
    first = kb.create_snapshot("a")
    time.sleep(0.001)
    kb.create_snapshot("b")
    assert not (tmp_path / "snapshots" / f"{first.id}.json").exists()


async def test_restore_snapshot_with_path(tmp_path):
    path = tmp_path / "cat.json"
    kb = KnowledgeBase(catalogue_path=path)
    await kb.ingest_text("one", source="a")
    snap = kb.create_snapshot("v1")
    assert (tmp_path / "snapshots" / f"{snap.id}.json").exists()

    await kb.ingest_text("two", source="b")
    assert kb.document_count == 2

    assert kb.restore_snapshot(snap.id) is True
    assert kb.document_count == 1


async def test_restore_snapshot_without_path_returns_false():
    kb = KnowledgeBase()
    await kb.ingest_text("x", source="a")
    snap = kb.create_snapshot("v1")
    assert kb.restore_snapshot(snap.id) is False


def test_restore_unknown_snapshot_returns_false():
    kb = KnowledgeBase()
    assert kb.restore_snapshot("nope") is False


# ---------------------------------------------------------------------------
# Close / background cleanup
# ---------------------------------------------------------------------------


async def test_close():
    kb = KnowledgeBase()
    await kb.ingest_text("x", source="a")
    await kb.close()


async def test_run_cleanup_task_cancellable():
    kb = KnowledgeBase()
    task = asyncio.create_task(run_cleanup(kb, interval_seconds=0.01))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
