from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from morainet.engineering.circuit_breaker import CircuitBreaker, CircuitState
from morainet.exceptions import CircuitBreakerOpenError


# ---------------------------------------------------------------------------
# CircuitState
# ---------------------------------------------------------------------------

def test_circuit_state_enum_values():
    assert CircuitState.CLOSED.value == "closed"
    assert CircuitState.OPEN.value == "open"
    assert CircuitState.HALF_OPEN.value == "half_open"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_initialization():
    breaker = CircuitBreaker()
    assert breaker.failure_threshold == 5
    assert breaker.cooldown_seconds == 30.0
    assert breaker.half_open_max_calls == 1
    assert breaker.success_threshold == 2
    assert breaker.name == "default"
    assert breaker._state == CircuitState.CLOSED


def test_custom_initialization():
    breaker = CircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=10.0,
        half_open_max_calls=2,
        success_threshold=3,
        name="my-breaker",
    )
    assert breaker.failure_threshold == 3
    assert breaker.cooldown_seconds == 10.0
    assert breaker.name == "my-breaker"


# ---------------------------------------------------------------------------
# State property
# ---------------------------------------------------------------------------

def test_state_initial():
    breaker = CircuitBreaker()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.is_open is False


# ---------------------------------------------------------------------------
# on_failure
# ---------------------------------------------------------------------------

def test_on_failure_increments_count():
    breaker = CircuitBreaker()
    breaker.on_failure()
    assert breaker._failure_count == 1


def test_on_failure_transitions_to_open_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.on_failure()
    breaker.on_failure()
    assert breaker._state == CircuitState.CLOSED
    breaker.on_failure()
    assert breaker._state == CircuitState.OPEN
    assert breaker.is_open is True


def test_on_failure_in_half_open_transitions_back_to_open():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=0)
    # Force to OPEN
    for _ in range(3):
        breaker.on_failure()
    assert breaker._state == CircuitState.OPEN
    # After cooldown=0, should go to HALF_OPEN
    breaker._transition()
    assert breaker._state == CircuitState.HALF_OPEN
    # Failure in HALF_OPEN goes back to OPEN
    breaker.on_failure()
    assert breaker._state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# _transition (OPEN -> HALF_OPEN)
# ---------------------------------------------------------------------------

def test_transition_open_to_half_open_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    breaker.on_failure()
    assert breaker._state == CircuitState.OPEN

    # Simulate time passing
    with patch.object(time, "monotonic", return_value=breaker._opened_at + 31.0):
        breaker._transition()
        assert breaker._state == CircuitState.HALF_OPEN


def test_transition_stays_open_before_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    breaker.on_failure()
    assert breaker._state == CircuitState.OPEN

    with patch.object(time, "monotonic", return_value=breaker._opened_at + 10.0):
        breaker._transition()
        assert breaker._state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# __aenter__ / __aexit__
# ---------------------------------------------------------------------------

async def test_aenter_when_closed_allows_entry():
    breaker = CircuitBreaker()
    async with breaker:
        assert breaker._state == CircuitState.CLOSED


async def test_aenter_when_open_raises():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.on_failure()
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            pass


async def test_aenter_when_half_open_allows_entry():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
    breaker.on_failure()
    breaker._transition()
    assert breaker._state == CircuitState.HALF_OPEN
    async with breaker:
        pass


async def test_aenter_when_half_open_max_calls_reached():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0, half_open_max_calls=0)
    breaker.on_failure()
    breaker._transition()
    with pytest.raises(CircuitBreakerOpenError):
        async with breaker:
            pass


async def test_aexit_on_exception_calls_on_failure():
    breaker = CircuitBreaker(failure_threshold=3)
    try:
        async with breaker:
            raise ValueError("boom")
    except ValueError:
        pass
    assert breaker._failure_count == 1


async def test_aexit_no_exception_does_not_call_failure():
    breaker = CircuitBreaker(failure_threshold=3)
    async with breaker:
        pass
    assert breaker._failure_count == 0


# ---------------------------------------------------------------------------
# on_success
# ---------------------------------------------------------------------------

def test_on_success_in_closed_resets_failure_count():
    breaker = CircuitBreaker(failure_threshold=5)
    breaker.on_failure()
    breaker.on_failure()
    assert breaker._failure_count == 2
    breaker.on_success()
    assert breaker._failure_count == 0


def test_on_success_in_half_open_transitions_to_closed_after_threshold():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0, success_threshold=2)
    breaker.on_failure()
    breaker._transition()
    assert breaker._state == CircuitState.HALF_OPEN

    breaker.on_success()
    assert breaker._state == CircuitState.HALF_OPEN
    assert breaker._success_count == 1

    breaker.on_success()
    assert breaker._state == CircuitState.CLOSED
    assert breaker._failure_count == 0
    assert breaker._success_count == 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_to_closed():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.on_failure()
    assert breaker._state == CircuitState.OPEN
    breaker.reset()
    assert breaker._state == CircuitState.CLOSED
    assert breaker._failure_count == 0
    assert breaker._success_count == 0
    assert breaker._half_open_calls == 0
