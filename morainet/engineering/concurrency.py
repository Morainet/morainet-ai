"""Global concurrency limiter for LLM calls, tool executions, and agent runs.

Usage::

    limiter = ConcurrencyLimiter(max_concurrent=5)
    async with limiter:
        result = await agent.arun("...")
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any


class ConcurrencyLimiter:
    """Semaphore-based global concurrency guard.

    Wrap any async block to limit simultaneous executions across the process.
    """

    def __init__(self, max_concurrent: int = 10, *, fair: bool = False) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._fair = fair
        self._waiters: int = 0  # approximate

    @property
    def max_concurrent(self) -> int:
        return getattr(self._semaphore, "_value", 0)

    @property
    def running(self) -> int:
        return getattr(self._semaphore, "_value", 0)

    @property
    def waiting(self) -> int:
        return self._waiters

    async def acquire(self) -> None:
        """Block until a slot is available."""
        self._waiters += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiters -= 1

    def release(self) -> None:
        """Release a previously acquired slot."""
        self._semaphore.release()

    async def __aenter__(self) -> "ConcurrencyLimiter":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def wrap(self, coro: Any) -> Any:
        """Wrap a coroutine so it respects the concurrency limit.

        Returns an awaitable that auto-acquires and releases the semaphore.
        """

        async def _guarded() -> Any:
            async with self:
                return await coro

        return _guarded()
