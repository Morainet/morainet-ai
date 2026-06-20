"""Tests for morainet.engineering.circuit_breaker."""

from __future__ import annotations

import time

import pytest

from morainet.engineering.circuit_breaker import CircuitBreaker, CircuitState
from morainet.exceptions import CircuitBreakerOpenError


# ---------------------------------------------------------------------------
# Construction & initial state
# ---------------------------------------------------------------------------

def test_default_construction():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.is_open is False
    assert cb.failure_threshold == 5
    assert cb.cooldown_seconds == 30.0
    assert cb.half_open_max_calls == 1
    assert cb.success_threshold == 2
    assert cb.name == "default"


def test_custom_construction():
    cb = CircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=10.0,
        half_open_max_calls=2,
        success_threshold=1,
        name="custom",
    )
    assert cb.failure_threshold == 3
    assert cb.cooldown_seconds == 10.0
    assert cb.half_open_max_calls == 2
    assert cb.success_threshold == 1
    assert cb.name == "custom"


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

def test_circuit_state_values():
    assert CircuitState.CLOSED.value == "closed"
    assert CircuitState.OPEN.value == "open"
    assert CircuitState.HALF_OPEN.value == "half_open"


# ---------------------------------------------------------------------------
# CLOSED → OPEN (failure threshold)
# ---------------------------------------------------------------------------

def test_closed_to_open_on_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.on_failure()
    cb.on_failure()
    assert cb.state == CircuitState.CLOSED
    cb.on_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open is True


def test_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3)
    cb.on_failure()
    cb.on_failure()
    cb.on_success()
    assert cb.state == CircuitState.CLOSED  # failure count reset
    cb.on_failure()
    cb.on_failure()
    assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN (cooldown)
# ---------------------------------------------------------------------------

def test_open_to_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
    cb.on_failure()
    # Cooldown is 0, so immediately transitions to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN


def test_open_stays_open_during_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=3600.0)
    cb.on_failure()
    assert cb.state == CircuitState.OPEN
    # Check again - still OPEN
    assert cb.state == CircuitState.OPEN
    assert cb.is_open is True


# ---------------------------------------------------------------------------
# HALF_OPEN behavior
# ---------------------------------------------------------------------------

def test_half_open_success_then_close():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0, success_threshold=2)
    cb.on_failure()
    # force transition now
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_calls = 0

    cb.on_success()
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1)
    cb.on_failure()
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_calls = 0

    cb.on_failure()
    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Async context manager: __aenter__ / __aexit__
# ---------------------------------------------------------------------------

async def test_aenter_when_closed():
    cb = CircuitBreaker()
    result = await cb.__aenter__()
    assert result is cb
    assert cb.state == CircuitState.CLOSED


async def test_aenter_raises_when_open():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=999.0)
    cb.on_failure()
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        await cb.__aenter__()


async def test_aenter_half_open_max_calls_exceeded():
    cb = CircuitBreaker(half_open_max_calls=1)
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_calls = 1  # already at max

    with pytest.raises(CircuitBreakerOpenError):
        await cb.__aenter__()


async def test_aexit_on_exception_calls_on_failure():
    cb = CircuitBreaker(failure_threshold=3)
    await cb.__aexit__(ValueError, ValueError("boom"), None)
    assert cb._failure_count == 1


async def test_aexit_no_exception_no_failure():
    cb = CircuitBreaker(failure_threshold=3)
    await cb.__aexit__(None, None, None)
    assert cb._failure_count == 0


# ---------------------------------------------------------------------------
# Async context manager: full usage pattern
# ---------------------------------------------------------------------------

async def test_async_with_closed():
    cb = CircuitBreaker()
    async with cb:
        cb.on_success()
    assert cb.state == CircuitState.CLOSED


async def test_async_with_exception_triggers_failure():
    cb = CircuitBreaker(failure_threshold=2)
    try:
        async with cb:
            raise RuntimeError("test error")
    except RuntimeError:
        pass
    assert cb._failure_count == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset():
    cb = CircuitBreaker(failure_threshold=1)
    cb.on_failure()
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb._failure_count == 0
    assert cb._success_count == 0
    assert cb._half_open_calls == 0


# ---------------------------------------------------------------------------
# aexit with BaseException subclasses (keyboard interrupt, etc.)
# ---------------------------------------------------------------------------

async def test_aexit_on_base_exception():
    cb = CircuitBreaker(failure_threshold=3)
    await cb.__aexit__(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert cb._failure_count == 1


async def test_aexit_on_asyncio_cancelled():
    import asyncio
    cb = CircuitBreaker(failure_threshold=3)
    await cb.__aexit__(asyncio.CancelledError, asyncio.CancelledError(), None)
    assert cb._failure_count == 1
