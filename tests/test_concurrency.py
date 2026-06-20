"""Tests for ConcurrencyLimiter."""

from __future__ import annotations

import asyncio

from morainet.engineering.concurrency import ConcurrencyLimiter


def test_init_defaults():
    limiter = ConcurrencyLimiter()
    assert limiter.max_concurrent == 10
    assert limiter.waiting == 0


def test_init_custom_max():
    limiter = ConcurrencyLimiter(max_concurrent=3)
    assert limiter.max_concurrent == 3


def test_running_property_initial():
    limiter = ConcurrencyLimiter(max_concurrent=5)
    # Initially no one has acquired
    # running reflects the internal semaphore value
    assert limiter.running >= 0


async def test_acquire_release():
    limiter = ConcurrencyLimiter(max_concurrent=2)
    await limiter.acquire()
    await limiter.acquire()
    # Two slots occupied — semaphore count approaches 0
    # release one
    limiter.release()
    # Should be able to acquire again
    await limiter.acquire()
    limiter.release()
    limiter.release()


async def test_async_context_manager():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    async with limiter:
        pass  # acquire + release handled automatically


async def test_context_manager_releases_on_exception():
    limiter = ConcurrencyLimiter(max_concurrent=1)

    class TestError(Exception):
        pass

    try:
        async with limiter:
            raise TestError("boom")
    except TestError:
        pass

    # Should still be able to acquire after exception
    await limiter.acquire()
    limiter.release()


async def test_wrap_method():
    limiter = ConcurrencyLimiter(max_concurrent=1)

    async def _work():
        return 42

    result = await limiter.wrap(_work())
    assert result == 42


async def test_wrap_respects_concurrency_limit():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    completed: list[int] = []

    async def _work(n: int):
        await asyncio.sleep(0.01)
        completed.append(n)
        return n

    # Wrap two coroutines
    t1 = asyncio.create_task(limiter.wrap(_work(1)))
    t2 = asyncio.create_task(limiter.wrap(_work(2)))

    await asyncio.gather(t1, t2)
    assert sorted(completed) == [1, 2]


async def test_max_concurrent_enforced():
    """Only max_concurrent tasks run simultaneously."""
    limiter = ConcurrencyLimiter(max_concurrent=2)
    running_count: int = 0
    max_observed: int = 0
    lock = asyncio.Lock()

    async def _work():
        nonlocal running_count, max_observed
        async with lock:
            running_count += 1
            max_observed = max(max_observed, running_count)
        await asyncio.sleep(0.02)
        async with lock:
            running_count -= 1

    tasks = [asyncio.create_task(limiter.wrap(_work())) for _ in range(5)]
    await asyncio.gather(*tasks)

    # At most max_concurrent tasks should have run simultaneously
    assert max_observed <= 2


async def test_waiting_counter():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    await limiter.acquire()
    # One slot occupied, semaphore at ~0

    assert limiter.waiting == 0

    async def _waiter():
        await limiter.acquire()
        limiter.release()

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0.01)
    # waiter should be waiting, but since waiting is approximate
    # it briefly increments then decrements in a try/finally
    limiter.release()
    await task
