from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from morainet.engineering.rate_limiter import (
    RateLimitConfig,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)


# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------

def test_rate_limit_config_defaults():
    config = RateLimitConfig()
    assert config.rate == 10.0
    assert config.burst == 20
    assert config.max_requests == 100
    assert config.window_seconds == 60.0


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------

def test_token_bucket_defaults():
    limiter = TokenBucketRateLimiter()
    assert limiter.rate == 10.0
    assert limiter.burst == 20.0
    assert limiter._tokens == 20.0


async def test_consume_returns_true_when_tokens_available():
    limiter = TokenBucketRateLimiter(rate=100, burst=10)
    for _ in range(10):
        assert await limiter.consume() is True
    assert await limiter.consume() is False


async def test_consume_returns_false_when_depleted():
    limiter = TokenBucketRateLimiter(rate=0, burst=0)
    assert await limiter.consume() is False


def test_refill_replenishes_tokens():
    limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
    limiter._tokens = 0
    now = limiter._last_fill + 2.0
    with patch.object(time, "monotonic", return_value=now):
        limiter._refill()
    assert limiter._tokens == pytest.approx(20.0)


def test_refill_caps_at_burst():
    limiter = TokenBucketRateLimiter(rate=100.0, burst=20)
    limiter._tokens = 19
    now = limiter._last_fill + 10.0
    with patch.object(time, "monotonic", return_value=now):
        limiter._refill()
    assert limiter._tokens == 20.0


async def test_acquire_blocks_until_token_available():
    limiter = TokenBucketRateLimiter(rate=100, burst=1)
    await limiter.acquire()
    assert limiter._tokens < 1


async def test_aenter_aexit():
    limiter = TokenBucketRateLimiter(rate=100, burst=10)
    async with limiter:
        pass


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter
# ---------------------------------------------------------------------------

def test_sliding_window_defaults():
    limiter = SlidingWindowRateLimiter()
    assert limiter.max_requests == 100
    assert limiter.window_seconds == 60.0


async def test_acquire_within_limit():
    limiter = SlidingWindowRateLimiter(max_requests=5)
    for _ in range(5):
        assert await limiter.acquire() is True


async def test_acquire_exceeds_limit():
    limiter = SlidingWindowRateLimiter(max_requests=3)
    for _ in range(3):
        assert await limiter.acquire() is True
    assert await limiter.acquire() is False


def test_prune_removes_old_entries():
    limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=10.0)
    now = 1000.0
    limiter._timestamps.extend([989.0, 995.0, 1000.0])
    limiter._prune(now)
    assert len(limiter._timestamps) == 2
    assert 989.0 not in limiter._timestamps
    assert 995.0 in limiter._timestamps
    assert 1000.0 in limiter._timestamps


def test_current_count_property():
    limiter = SlidingWindowRateLimiter(max_requests=100)
    limiter._timestamps.extend([1.0, 2.0, 3.0])
    assert limiter.current_count == 3


def test_remaining_property():
    limiter = SlidingWindowRateLimiter(max_requests=10)
    limiter._timestamps.extend([1.0, 2.0, 3.0])
    assert limiter.remaining == 7


def test_remaining_never_negative():
    limiter = SlidingWindowRateLimiter(max_requests=2)
    limiter._timestamps.extend([1.0, 2.0, 3.0, 4.0])
    assert limiter.remaining == 0
