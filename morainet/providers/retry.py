"""Retry wrapper for providers: exponential backoff on transient failures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from morainet.config import settings
from morainet.core.models import ChatResponse, Message
from morainet.exceptions import ProviderTimeoutError, RateLimitError
from morainet.observability.tracing import logger
from morainet.providers.base import Provider


@dataclass
class RetryPolicy:
    max_retries: int = field(default_factory=lambda: settings.max_retries)
    base_delay: float = 1.0
    backoff: float = 2.0
    # Only transient, idempotent failures are retried.
    retry_on: tuple[type[Exception], ...] = (RateLimitError, ProviderTimeoutError)


class RetryingProvider(Provider):
    """Wrap any provider, retrying ``chat`` on transient errors.

    Streaming is delegated without retry (a partial stream can't be replayed).
    """

    def __init__(
        self,
        inner: Provider,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.inner = inner
        self.policy = policy or RetryPolicy()
        self._sleep = sleep

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        attempt = 0
        while True:
            try:
                return await self.inner.chat(messages, tools)
            except self.policy.retry_on as exc:
                if attempt >= self.policy.max_retries:
                    raise
                delay = self.policy.base_delay * (self.policy.backoff**attempt)
                logger.warning(
                    f"provider error ({type(exc).__name__}); retry "
                    f"{attempt + 1}/{self.policy.max_retries} in {delay:.1f}s"
                )
                await self._sleep(delay)
                attempt += 1

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        return self.inner.stream(messages, tools)
