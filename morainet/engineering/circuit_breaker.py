"""Circuit breaker with CLOSED → OPEN → HALF_OPEN state machine.

Prevents cascading failures by stopping all calls to a failing provider for a
cooldown period, then allowing a trial request in HALF_OPEN.

Usage::

    breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
    async with breaker:
        try:
            result = await provider.chat(...)
            breaker.on_success()
        except Exception:
            breaker.on_failure()
            raise
"""

from __future__ import annotations

import enum
import time
from types import TracebackType

from morainet.exceptions import CircuitBreakerOpenError
from morainet.observability.tracing import logger


class CircuitState(enum.Enum):
    CLOSED = "closed"           # normal operation
    OPEN = "open"               # rejecting requests
    HALF_OPEN = "half_open"     # testing recovery


class CircuitBreaker:
    """State machine that opens when failures exceed threshold.

    ``failure_threshold``  — consecutive failures before opening.
    ``cooldown_seconds``    — time spent OPEN before transitioning to HALF_OPEN.
    ``half_open_max_calls`` — max trial calls in HALF_OPEN before deciding to CLOSE or re-OPEN.
    ``success_threshold``   — consecutive successes in HALF_OPEN needed to CLOSE.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 2,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        self._transition()
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _transition(self) -> None:
        """Check if state should change based on elapsed time."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"[circuit-breaker:{self.name}] OPEN → HALF_OPEN")

    async def __aenter__(self) -> "CircuitBreaker":
        self._transition()
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN; "
                f"retry in {self.cooldown_seconds - (time.monotonic() - self._opened_at):.1f}s"
            )
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' HALF_OPEN max calls reached"
                )
            self._half_open_calls += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.on_failure()

    def on_success(self) -> None:
        """Report a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"[circuit-breaker:{self.name}] HALF_OPEN → CLOSED")
        else:
            # Reset failure count on any success in CLOSED state
            self._failure_count = 0

    def on_failure(self) -> None:
        """Report a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._success_count = 0
            logger.warning(f"[circuit-breaker:{self.name}] HALF_OPEN → OPEN (trial failure)")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                f"[circuit-breaker:{self.name}] CLOSED → OPEN "
                f"({self._failure_count} failures)"
            )

    def reset(self) -> None:
        """Force the breaker back to CLOSED (manual override)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
