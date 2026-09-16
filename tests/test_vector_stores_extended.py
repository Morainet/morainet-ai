"""Offline tests for the extended vector store backends.

The external backends (pgvector / qdrant / milvus / faiss) require optional
libraries that are not installed in CI, so those tests exercise the
lazy-import :class:`MemoryStoreError` guard. The factory and registry helpers
are fully testable offline.
"""

from __future__ import annotations

import pytest

from morainet.exceptions import MemoryStoreError
from morainet.memory.base import VectorStore
from morainet.memory.vector_stores_extended import (
    FaissStore,
    MilvusStore,
    PgVectorStore,
    QdrantStore,
    create_vector_store,
    list_vector_store_backends,
)


EXPECTED_BACKENDS = {"inmemory", "chroma", "pgvector", "qdrant", "faiss", "milvus"}


def test_list_backends_contains_all():
    assert EXPECTED_BACKENDS.issubset(set(list_vector_store_backends()))


def test_create_unknown_backend_raises():
    with pytest.raises(ValueError):
        create_vector_store("does-not-exist")


def test_create_inmemory():
    store = create_vector_store("inmemory")
    assert isinstance(store, VectorStore)


async def test_create_inmemory_roundtrip():
    store = create_vector_store("inmemory")
    iid = await store.upsert("hello", [0.1, 0.2, 0.3], {"k": "v"})
    assert isinstance(iid, str)
    assert await store.count() == 1
    results = await store.search([0.1, 0.2, 0.3], top_k=1)
    assert results[0]["text"] == "hello"
    assert await store.delete(iid) is True
    assert await store.count() == 0


def test_pgvector_missing_library_raises():
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        with pytest.raises(MemoryStoreError):
            PgVectorStore()
        return
    pytest.skip("psycopg2 installed")


def test_qdrant_missing_library_raises():
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        with pytest.raises(MemoryStoreError):
            QdrantStore()
        return
    pytest.skip("qdrant-client installed")


def test_milvus_missing_library_raises():
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        with pytest.raises(MemoryStoreError):
            MilvusStore()
        return
    pytest.skip("pymilvus installed")


def test_faiss_missing_library_raises():
    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(MemoryStoreError):
            FaissStore(dimension=4)
        return
    pytest.skip("faiss installed")


async def test_faiss_store_operations():
    try:
        import faiss  # noqa: F401
    except ImportError:
        pytest.skip("faiss not installed")

    store = FaissStore(dimension=4)
    iid = await store.upsert("t", [0.1, 0.2, 0.3, 0.4], {"a": 1})
    assert isinstance(iid, str)
    results = await store.search([0.1, 0.2, 0.3, 0.4], top_k=1)
    assert results[0]["text"] == "t"
    assert await store.count() == 1
    assert await store.delete(iid) is True
    assert await store.count() == 0
    store.rebuild()  # no-op best effort
