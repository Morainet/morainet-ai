"""Offline tests for the full-chain error classifier and categorized retry.

Covers :func:`classify_error` (status-code + exception-type routing),
:class:`CategoryStrategy`/`CategorizedRetryPolicy` (per-category backoff), and
:class:`CategorizedRetryingProvider` (async retry with a fake provider). All
pure logic / hermetic — no network.
"""

from __future__ import annotations

import types

import pytest

from morainet.exceptions import (
    AuthError,
    ContextLengthError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from morainet.providers.error_classifier import (
    CategorizedRetryPolicy,
    CategorizedRetryingProvider,
    CategoryStrategy,
    ErrorCategory,
    _compute_delay,
    classify_error,
)
from morainet.providers.retry import RetryPolicy


async def _noop_sleep(_d: float) -> None:
    return None


class _FakeProvider:
    def __init__(self, fail_times: int = 0, exc: BaseException | None = None) -> None:
        self.fail_times = fail_times
        self.exc = exc or RateLimitError("rate")
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "OK"

    def stream(self, *args, **kwargs):
        return "STREAM"


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


def test_classify_status_429_rate_limit():
    assert classify_error(ValueError("x"), 429) == ErrorCategory.RATE_LIMIT


def test_classify_status_400_422_parameter():
    assert classify_error(ValueError("x"), 400) == ErrorCategory.PARAMETER
    assert classify_error(ValueError("x"), 422) == ErrorCategory.PARAMETER


def test_classify_status_401_403_auth():
    assert classify_error(ValueError("x"), 401) == ErrorCategory.AUTH
    assert classify_error(ValueError("x"), 403) == ErrorCategory.AUTH


def test_classify_status_413_context():
    assert classify_error(ValueError("x"), 413) == ErrorCategory.CONTEXT


def test_classify_status_5xx_server():
    assert classify_error(ValueError("x"), 500) == ErrorCategory.SERVER
    assert classify_error(ValueError("x"), 599) == ErrorCategory.SERVER


def test_classify_exception_rate_limit():
    assert classify_error(RateLimitError("x")) == ErrorCategory.RATE_LIMIT


def test_classify_exception_timeout_network():
    assert classify_error(ProviderTimeoutError("x")) == ErrorCategory.NETWORK


def test_classify_exception_auth():
    assert classify_error(AuthError("x")) == ErrorCategory.AUTH


def test_classify_exception_context():
    assert classify_error(ContextLengthError("x")) == ErrorCategory.CONTEXT


def test_classify_exception_connection_by_name():
    assert classify_error(ConnectionError("x")) == ErrorCategory.NETWORK


def test_classify_http_status_error_recurses():
    class _HTTPStatusError(Exception):
        pass

    _HTTPStatusError.__name__ = "HTTPStatusError"
    exc = _HTTPStatusError("boom")
    exc.response = types.SimpleNamespace(status_code=503)
    assert classify_error(exc) == ErrorCategory.SERVER


def test_classify_provider_error_unknown():
    assert classify_error(ProviderError("x")) == ErrorCategory.UNKNOWN


def test_classify_generic_exception_unknown():
    assert classify_error(ValueError("x")) == ErrorCategory.UNKNOWN
    assert classify_error(KeyError("x")) == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# _compute_delay
# ---------------------------------------------------------------------------


def test_compute_delay_zero_base():
    s = CategoryStrategy(max_retries=0, base_delay=0, backoff=0, jitter=False)
    assert _compute_delay(s, 0) == 0


def test_compute_delay_formula_no_jitter():
    s = CategoryStrategy(max_retries=3, base_delay=0.5, backoff=1.5, jitter=False)
    assert _compute_delay(s, 0) == 0.5
    assert _compute_delay(s, 1) == 0.75
    assert _compute_delay(s, 2) == 1.125


def test_compute_delay_jitter_within_bounds():
    s = CategoryStrategy(max_retries=3, base_delay=1.0, backoff=2.0, jitter=True)
    for attempt in range(3):
        d = _compute_delay(s, attempt)
        base = 1.0 * (2.0 ** attempt)
        assert base * 0.9 <= d <= base * 1.1


# ---------------------------------------------------------------------------
# CategorizedRetryPolicy
# ---------------------------------------------------------------------------


def test_policy_covers_all_categories():
    p = CategorizedRetryPolicy()
    assert set(p.strategies.keys()) == set(ErrorCategory)


def test_policy_merge_custom_overrides_default():
    custom = CategoryStrategy(max_retries=10, base_delay=0.1, backoff=1.1, jitter=False)
    p = CategorizedRetryPolicy(strategies={ErrorCategory.NETWORK: custom})
    assert p.get_strategy(ErrorCategory.NETWORK).max_retries == 10
    # other categories keep defaults
    assert p.get_strategy(ErrorCategory.RATE_LIMIT).max_retries == 5


def test_policy_should_retry_matrix():
    p = CategorizedRetryPolicy()
    assert p.should_retry(RateLimitError("x")) is True
    assert p.should_retry(AuthError("x")) is False
    assert p.should_retry(ContextLengthError("x")) is True   # CONTEXT max_retries=1
    assert p.should_retry(ProviderError("x")) is True        # UNKNOWN max_retries=2


def test_policy_max_retries_for():
    p = CategorizedRetryPolicy()
    assert p.max_retries_for(ContextLengthError("x")) == 1
    assert p.max_retries_for(AuthError("x")) == 0


def test_policy_compute_delay_positive():
    p = CategorizedRetryPolicy()
    assert p.compute_delay(RateLimitError("x"), 0) > 0


def test_policy_to_retry_policy():
    p = CategorizedRetryPolicy()
    rp = p.to_retry_policy()
    assert isinstance(rp, RetryPolicy)
    assert rp.max_retries == 5  # RATE_LIMIT (5) is the max across categories


# ---------------------------------------------------------------------------
# CategorizedRetryingProvider (async)
# ---------------------------------------------------------------------------


async def test_retry_provider_success_first_try():
    prov = _FakeProvider(fail_times=0)
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    assert await rp.chat([]) == "OK"
    assert prov.calls == 1


async def test_retry_provider_retries_then_succeeds():
    prov = _FakeProvider(fail_times=1, exc=RateLimitError("rl"))
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    assert await rp.chat([]) == "OK"
    assert prov.calls == 2


async def test_retry_provider_exhausts_retries():
    prov = _FakeProvider(fail_times=99, exc=RateLimitError("rl"))
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    with pytest.raises(RateLimitError):
        await rp.chat([])
    assert prov.calls == 6  # 1 initial + 5 retries


async def test_retry_provider_no_retry_on_auth():
    prov = _FakeProvider(fail_times=99, exc=AuthError("auth"))
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    with pytest.raises(AuthError):
        await rp.chat([])
    assert prov.calls == 1


async def test_retry_provider_status_from_response():
    class _HTTPStatusError(Exception):
        pass

    _HTTPStatusError.__name__ = "HTTPStatusError"
    err = _HTTPStatusError("e")
    err.response = types.SimpleNamespace(status_code=400)  # PARAMETER → no retry
    prov = _FakeProvider(fail_times=99, exc=err)
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    with pytest.raises(_HTTPStatusError):
        await rp.chat([])
    assert prov.calls == 1


async def test_retry_provider_stream_delegates():
    prov = _FakeProvider()
    rp = CategorizedRetryingProvider(prov, sleep=_noop_sleep)
    assert rp.stream([]) == "STREAM"
