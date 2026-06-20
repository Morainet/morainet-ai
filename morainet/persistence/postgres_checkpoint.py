"""PostgreSQL-backed checkpoint store for distributed breakpoint resume.

Requires ``psycopg`` (``pip install "psycopg[binary]"``).

Usage::

    store = PostgresCheckpointStore(
        dsn="postgresql://user:pass@localhost:5432/morainet"
    )
    agent = Agent(provider=..., checkpoint_store=store)
"""

from __future__ import annotations

import json
from typing import Any

from morainet.persistence.checkpoint import Checkpoint, CheckpointStore


class PostgresCheckpointStore(CheckpointStore):
    """PostgreSQL-backed checkpoint persistence using ``psycopg`` (3.x).

    ``dsn``     — libpq connection string.
    ``schema``  — optional schema qualifier for the checkpoints table.
    ``min_conn``/``max_conn`` — connection pool size.

    Table is auto-created on first use.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/morainet",
        schema: str = "public",
        min_conn: int = 1,
        max_conn: int = 5,
    ) -> None:
        self._dsn = dsn
        self._schema = schema
        self._min_conn = min_conn
        self._max_conn = max_conn
        self._pool: Any = None
        self._table_created = False

    @property
    def _table_name(self) -> str:
        return f"{self._schema}.morainet_checkpoints"

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            try:
                from importlib.util import find_spec

                if find_spec("psycopg") is None or find_spec("psycopg_pool") is None:
                    raise ImportError
                from psycopg_pool import AsyncConnectionPool  # type: ignore[import-untyped,unused-ignore]
            except ImportError:
                raise ImportError(
                    "psycopg and psycopg_pool are required for PostgresCheckpointStore. "
                    "Install with: pip install morainet-ai[postgres]"
                ) from None
            self._pool = AsyncConnectionPool(
                conninfo=self._dsn,
                min_size=self._min_conn,
                max_size=self._max_conn,
                open=True,
            )
        if not self._table_created:
            await self._create_table()
        return self._pool

    async def _create_table(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    trace_id    TEXT PRIMARY KEY,
                    data        JSONB NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Index on created_at for cleanup queries
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_morainet_checkpoints_created_at
                ON {self._table_name} (created_at)
            """)
        self._table_created = True

    async def save(self, checkpoint: Checkpoint) -> None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"""INSERT INTO {self._table_name} (trace_id, data, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (trace_id) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = NOW()""",
                (checkpoint.trace_id, json.dumps(checkpoint.model_dump())),
            )

    async def load(self, trace_id: str) -> Checkpoint | None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    f"SELECT data FROM {self._table_name} WHERE trace_id = %s",
                    (trace_id,),
                )
            ).fetchone()
            if row is None:
                return None
            return Checkpoint.model_validate(row[0])

    async def delete(self, trace_id: str) -> bool:
        """Delete a checkpoint. Returns ``True`` if it existed."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table_name} WHERE trace_id = %s",
                (trace_id,),
            )
            return result.rowcount is not None and result.rowcount > 0

    async def list_trace_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List trace_ids ordered by most recent first."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    f"SELECT trace_id FROM {self._table_name} "
                    "ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            ).fetchall()
            return [r[0] for r in rows]

    async def cleanup_older_than(self, days: int = 30) -> int:
        """Delete checkpoints older than ``days``. Returns count deleted."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table_name} WHERE updated_at < NOW() - INTERVAL '%s days'",
                (str(days),),
            )
            return result.rowcount or 0

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
