"""Token-bucket and sliding-window rate limiters for provider call throttling.

Usage::

    limiter = TokenBucketRateLimiter(rate=10, burst=20)
    async with limiter:
        await provider.chat(...)

    window = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    if await window.acquire():
        await provider.chat(...)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType


@dataclass
class RateLimitConfig:
    """Shared configuration constants for rate limiting."""

    # Token bucket defaults
    rate: float = 10.0  # tokens per second
    burst: int = 20     # max burst capacity

    # Sliding window defaults
    max_requests: int = 100
    window_seconds: float = 60.0


class _AsyncLock:
    """Minimal async Lock (stdlib asyncio.Lock is fine; this avoids typing noise)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._lock.release()


class TokenBucketRateLimiter:
    """Token bucket algorithm: smooth bursts, configurable steady rate.

    ``rate`` tokens are added per second.  ``burst`` is the max bucket size.
    ``consume()`` returns ``True`` if a token was available, ``False`` otherwise.
    ``__aenter__`` / ``__aexit__`` conveniently call ``acquire()`` / ``release()``.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        self.rate = float(rate)
        self.burst = float(burst)
        self._tokens = float(burst)
        self._last_fill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_fill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_fill = now

    async def consume(self) -> bool:
        """Try to consume 1 token.  Returns ``True`` on success."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while not await self.consume():
            # Calculate how long to wait for next token
            async with self._lock:
                self._refill()
                wait = (1 - self._tokens) / self.rate if self._tokens < 1 else 0
            if wait > 0:
                await asyncio.sleep(wait)

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # token already consumed


class SlidingWindowRateLimiter:
    """Sliding window algorithm: allow at most ``max_requests`` in ``window_seconds``.

    Uses a deque of timestamps; old entries are pruned on each ``acquire()``.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> bool:
        """Return ``True`` if within the limit, ``False`` if rate-limited."""
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            if len(self._timestamps) >= self.max_requests:
                return False
            self._timestamps.append(now)
            return True

    @property
    def current_count(self) -> int:
        return len(self._timestamps)

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - len(self._timestamps))
