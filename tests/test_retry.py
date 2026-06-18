from __future__ import annotations

import pytest

from morainet.core.models import ChatResponse, Message
from morainet.exceptions import AuthError, RateLimitError
from morainet.providers import MockProvider, RetryingProvider, RetryPolicy
from morainet.providers.base import Provider


class FlakyProvider(Provider):
    """Fails with the given error N times, then succeeds."""

    def __init__(self, fails: int, error: Exception):
        self.fails = fails
        self.error = error
        self.attempts = 0

    async def chat(self, messages, tools=None, response_format=None):
        self.attempts += 1
        if self.attempts <= self.fails:
            raise self.error
        return ChatResponse(message=Message.assistant(content="ok"))


async def _noop_sleep(_):
    return None


async def test_retries_then_succeeds():
    inner = FlakyProvider(fails=2, error=RateLimitError("429"))
    provider = RetryingProvider(inner, RetryPolicy(max_retries=3), sleep=_noop_sleep)
    resp = await provider.chat([Message.user("hi")])
    assert resp.message.content == "ok"
    assert inner.attempts == 3  # 2 failures + 1 success


async def test_exhausts_retries_and_raises():
    inner = FlakyProvider(fails=5, error=RateLimitError("429"))
    provider = RetryingProvider(inner, RetryPolicy(max_retries=2), sleep=_noop_sleep)
    with pytest.raises(RateLimitError):
        await provider.chat([Message.user("hi")])
    assert inner.attempts == 3  # initial + 2 retries


async def test_non_retryable_error_propagates_immediately():
    inner = FlakyProvider(fails=5, error=AuthError("401"))
    provider = RetryingProvider(inner, RetryPolicy(max_retries=3), sleep=_noop_sleep)
    with pytest.raises(AuthError):
        await provider.chat([Message.user("hi")])
    assert inner.attempts == 1  # not retried


async def test_stream_delegates_without_retry():
    inner = MockProvider(handler=lambda m, t: ChatResponse(message=Message.assistant(content="hello")))
    provider = RetryingProvider(inner, sleep=_noop_sleep)
    chunks = [c async for c in provider.stream([Message.user("hi")])]
    assert "".join(chunks) == "hello"


async def test_agent_level_retry_wraps_provider():
    from morainet import Agent

    inner = FlakyProvider(fails=2, error=RateLimitError("429"))
    # base_delay=0 keeps retries instant in the test.
    agent = Agent(provider=inner, retry=RetryPolicy(max_retries=3, base_delay=0.0))
    result = await agent.arun("hi")
    assert result.final_answer == "ok"
    assert inner.attempts == 3
