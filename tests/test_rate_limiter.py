"""Tests for morainet.engineering.rate_limiter."""

from __future__ import annotations

import asyncio
import time


from morainet.engineering.rate_limiter import (
    RateLimitConfig,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    _AsyncLock,
)


# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------

def test_rate_limit_config_defaults():
    cfg = RateLimitConfig()
    assert cfg.rate == 10.0
    assert cfg.burst == 20
    assert cfg.max_requests == 100
    assert cfg.window_seconds == 60.0


def test_rate_limit_config_custom():
    cfg = RateLimitConfig(rate=5.0, burst=10, max_requests=50, window_seconds=30.0)
    assert cfg.rate == 5.0
    assert cfg.burst == 10
    assert cfg.max_requests == 50
    assert cfg.window_seconds == 30.0


# ---------------------------------------------------------------------------
# _AsyncLock
# ---------------------------------------------------------------------------

async def test_async_lock_acquire_release():
    lock = _AsyncLock()
    async with lock:
        pass  # should not deadlock


async def test_async_lock_exclusion():
    lock = _AsyncLock()
    results = []

    async def guarded(val: int) -> None:
        async with lock:
            await asyncio.sleep(0.01)
            results.append(val)

    await asyncio.gather(guarded(1), guarded(2), guarded(3))
    assert results == [1, 2, 3]  # sequential execution preserved order


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------

def test_token_bucket_defaults():
    tb = TokenBucketRateLimiter()
    assert tb.rate == 10.0
    assert tb.burst == 20.0
    assert tb._tokens == 20.0


def test_token_bucket_custom():
    tb = TokenBucketRateLimiter(rate=5.0, burst=10)
    assert tb.rate == 5.0
    assert tb.burst == 10.0
    assert tb._tokens == 10.0


async def test_token_bucket_consume_when_tokens_available():
    tb = TokenBucketRateLimiter(rate=100, burst=10)  # lots of tokens
    assert await tb.consume() is True


async def test_token_bucket_consume_when_empty():
    tb = TokenBucketRateLimiter(rate=100, burst=0)  # no initial tokens
    # Tokens are 0, rate is 100/s, but initially 0
    tb._tokens = 0.0
    assert await tb.consume() is False


async def test_token_bucket_consume_refills_over_time(monkeypatch):
    tb = TokenBucketRateLimiter(rate=100, burst=10)
    tb._tokens = 0.0

    # Advance time by 1 second → 100 tokens refilled, capped at burst=10
    monkeypatch.setattr(time, "monotonic", lambda: tb._last_fill + 1.0)
    assert await tb.consume() is True
    assert tb._tokens == 9.0


async def test_token_bucket_acquire_blocks_then_gets_token(monkeypatch):
    tb = TokenBucketRateLimiter(rate=100, burst=0)
    tb._tokens = 0.0

    # _refill needs to add at least 1 token; advance by 0.02s → 2 tokens
    start = tb._last_fill
    monkeypatch.setattr(time, "monotonic", lambda: start + 0.02)
    await tb.acquire()
    # Should have consumed 1 token


async def test_token_bucket_async_context_manager():
    tb = TokenBucketRateLimiter(rate=100, burst=10)
    async with tb:
        pass  # acquire called, token consumed


async def test_token_bucket_aexit_noop():
    tb = TokenBucketRateLimiter(rate=100, burst=10)
    await tb.__aexit__(None, None, None)  # should not raise


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter
# ---------------------------------------------------------------------------

def test_sliding_window_defaults():
    sw = SlidingWindowRateLimiter()
    assert sw.max_requests == 100
    assert sw.window_seconds == 60.0


def test_sliding_window_custom():
    sw = SlidingWindowRateLimiter(max_requests=5, window_seconds=10.0)
    assert sw.max_requests == 5
    assert sw.window_seconds == 10.0


async def test_sliding_window_acquire_under_limit():
    sw = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert await sw.acquire() is True
    assert sw.current_count == 5


async def test_sliding_window_acquire_over_limit():
    sw = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert await sw.acquire() is True
    assert await sw.acquire() is False
    assert sw.current_count == 3


async def test_sliding_window_rejects_when_full():
    sw = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    await sw.acquire()
    await sw.acquire()
    await sw.acquire()
    assert await sw.acquire() is False


async def test_sliding_window_prune_old_entries(monkeypatch):
    sw = SlidingWindowRateLimiter(max_requests=3, window_seconds=1.0)
    await sw.acquire()
    await sw.acquire()

    # Advance time beyond window
    monkeypatch.setattr(time, "monotonic", lambda: time.monotonic() + 2.0)
    await sw.acquire()
    # Old entries should be pruned
    assert sw.current_count <= 1


async def test_sliding_window_remaining():
    sw = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    assert sw.remaining == 5
    await sw.acquire()
    assert sw.remaining == 4
    await sw.acquire()
    await sw.acquire()
    assert sw.remaining == 2
