"""Offline tests for the Redis and PostgreSQL checkpoint stores.

Both stores are exercised with fake backends (no redis server, no Postgres).
The lazy-import error paths are also covered when the optional drivers are
absent.
"""

from __future__ import annotations

import fnmatch

import pytest

from morainet.persistence.checkpoint import Checkpoint
from morainet.persistence.postgres_checkpoint import PostgresCheckpointStore
from morainet.persistence.redis_checkpoint import RedisCheckpointStore


def _checkpoint(trace_id: str = "t1") -> Checkpoint:
    return Checkpoint(trace_id=trace_id, query="q")


# ===========================================================================
# Redis
# ===========================================================================


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict = {}
        self.setex_calls: list = []
        self.closed = False

    async def set(self, key, value):
        self.data[key] = value

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl))
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        return 1 if self.data.pop(key, None) is not None else 0

    async def keys(self, pattern):
        return [k for k in self.data if fnmatch.fnmatch(k, pattern)]

    async def aclose(self):
        self.closed = True


def _redis(**kw) -> tuple[RedisCheckpointStore, FakeRedis]:
    store = RedisCheckpointStore(**kw)
    client = FakeRedis()
    store._client = client
    return store, client


def test_redis_key_prefix():
    store, _ = _redis(key_prefix="pfx:")
    assert store._key("abc") == "pfx:abc"


async def test_redis_save_and_load():
    store, client = _redis()
    await store.save(_checkpoint("t1"))
    assert "morainet:checkpoint:t1" in client.data
    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"


async def test_redis_save_with_ttl_uses_setex():
    store, client = _redis(ttl_seconds=60)
    await store.save(_checkpoint("t2"))
    assert client.setex_calls == [("morainet:checkpoint:t2", 60)]


async def test_redis_load_missing_returns_none():
    store, _ = _redis()
    assert await store.load("nope") is None


async def test_redis_delete():
    store, _ = _redis()
    await store.save(_checkpoint("t1"))
    assert await store.delete("t1") is True
    assert await store.delete("t1") is False


async def test_redis_list_trace_ids():
    store, _ = _redis()
    await store.save(_checkpoint("a"))
    await store.save(_checkpoint("b"))
    assert sorted(await store.list_trace_ids()) == ["a", "b"]


async def test_redis_list_trace_ids_pattern():
    store, _ = _redis()
    await store.save(_checkpoint("job_1"))
    await store.save(_checkpoint("other"))
    assert await store.list_trace_ids("job_*") == ["job_1"]


async def test_redis_close():
    store, client = _redis()
    await store.close()
    assert client.closed is True
    assert store._client is None


def test_redis_client_requires_package():
    try:
        import redis  # noqa: F401
    except ImportError:
        store = RedisCheckpointStore()
        with pytest.raises(ImportError):
            _ = store.client
        return
    pytest.skip("redis installed")


# ===========================================================================
# PostgreSQL
# ===========================================================================


class FakeCursor:
    def __init__(self, rows, rowcount) -> None:
        self._rows = rows
        self.rowcount = rowcount

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def execute(self, sql, params=None):
        self._pool.executed.append((sql, params))
        return FakeCursor(self._pool.result_rows, self._pool.rowcount)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self) -> None:
        self.executed: list = []
        self.result_rows: list = []
        self.rowcount = 0
        self.closed = False

    def connection(self) -> FakeConn:
        return FakeConn(self)

    async def close(self):
        self.closed = True


def _pg(**kw) -> tuple[PostgresCheckpointStore, FakePool]:
    store = PostgresCheckpointStore(**kw)
    pool = FakePool()
    store._pool = pool
    store._table_created = True
    return store, pool


def test_postgres_table_name():
    store = PostgresCheckpointStore(schema="myschema")
    assert store._table_name == "myschema.morainet_checkpoints"


async def test_postgres_save():
    store, pool = _pg()
    await store.save(_checkpoint("t1"))
    assert any("INSERT INTO" in sql for sql, _ in pool.executed)


async def test_postgres_load_found():
    store, pool = _pg()
    cp = _checkpoint("t1")
    pool.result_rows = [(cp.model_dump(),)]
    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.trace_id == "t1"


async def test_postgres_load_missing():
    store, pool = _pg()
    pool.result_rows = []
    assert await store.load("nope") is None


async def test_postgres_delete():
    store, pool = _pg()
    pool.rowcount = 1
    assert await store.delete("t1") is True
    pool.rowcount = 0
    assert await store.delete("t1") is False


async def test_postgres_list_trace_ids():
    store, pool = _pg()
    pool.result_rows = [("a",), ("b",)]
    assert await store.list_trace_ids() == ["a", "b"]


async def test_postgres_cleanup_older_than():
    store, pool = _pg()
    pool.rowcount = 3
    assert await store.cleanup_older_than(7) == 3


async def test_postgres_close():
    store, pool = _pg()
    await store.close()
    assert pool.closed is True
    assert store._pool is None


async def test_postgres_ensure_pool_requires_psycopg():
    store = PostgresCheckpointStore()
    try:
        import psycopg  # noqa: F401
        import psycopg_pool  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            await store._ensure_pool()
        return
    pytest.skip("psycopg installed")
