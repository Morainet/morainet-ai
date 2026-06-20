"""Full-chain error classification: categorize failures for differentiated retry.

Each error is assigned a category, and each category has its own backoff strategy:

- NETWORK     → fast retry (linear, short delay)
- RATE_LIMIT  → exponential backoff + jitter (standard)
- PARAMETER   → no retry (fatal)
- AUTH        → no retry (fatal)
- SERVER      → slow exponential backoff
- CONTEXT     → truncate and retry once
- UNKNOWN     → cautious exponential backoff

Usage::

    from morainet.providers.error_classifier import (
        classify_error, ErrorCategory, CategorizedRetryPolicy,
    )
    policy = CategorizedRetryPolicy()
    retrying = RetryingProvider(provider, policy=policy.to_retry_policy())
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from morainet.exceptions import (
    AuthError,
    ContextLengthError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from morainet.observability.tracing import logger
from morainet.providers.retry import RetryPolicy


class ErrorCategory(enum.Enum):
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    PARAMETER = "parameter"
    AUTH = "auth"
    SERVER = "server"
    CONTEXT = "context"
    UNKNOWN = "unknown"


# --- Classification ---------------------------------------------------------


def classify_error(exc: BaseException, status_code: int | None = None) -> ErrorCategory:
    """Classify an exception into an ErrorCategory based on type and status code."""

    # HTTP status-based classification (takes priority)
    if status_code is not None:
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if status_code in (400, 422):
            return ErrorCategory.PARAMETER
        if status_code in (401, 403):
            return ErrorCategory.AUTH
        if status_code == 413:
            return ErrorCategory.CONTEXT
        if 500 <= status_code < 600:
            return ErrorCategory.SERVER

    # Exception type-based classification
    if isinstance(exc, RateLimitError):
        return ErrorCategory.RATE_LIMIT
    if isinstance(exc, ProviderTimeoutError):
        return ErrorCategory.NETWORK
    if isinstance(exc, AuthError):
        return ErrorCategory.AUTH
    if isinstance(exc, ContextLengthError):
        return ErrorCategory.CONTEXT

    # Check for network-level exceptions
    exc_name = type(exc).__name__
    if exc_name in (
        "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
        "RemoteProtocolError", "NetworkError", "ConnectionError", "TimeoutError",
    ):
        return ErrorCategory.NETWORK

    # HTTP status errors from httpx
    if exc_name == "HTTPStatusError" and hasattr(exc, "response"):
        response = getattr(exc, "response", None)
        if response is not None:
            code = getattr(response, "status_code", 0)
            return classify_error(exc, status_code=code)

    if isinstance(exc, ProviderError):
        return ErrorCategory.UNKNOWN

    return ErrorCategory.UNKNOWN


# --- Per-category retry configuration ---------------------------------------


@dataclass
class CategoryStrategy:
    """Retry strategy for a single error category."""

    max_retries: int
    base_delay: float
    backoff: float  # multiplier per attempt
    jitter: bool = True
    jitter_factor: float = 0.1  # ±10%


# Default strategies per category
DEFAULT_STRATEGIES: dict[ErrorCategory, CategoryStrategy] = {
    ErrorCategory.NETWORK: CategoryStrategy(
        max_retries=3, base_delay=0.5, backoff=1.5, jitter=True,
    ),
    ErrorCategory.RATE_LIMIT: CategoryStrategy(
        max_retries=5, base_delay=1.0, backoff=2.0, jitter=True,
    ),
    ErrorCategory.PARAMETER: CategoryStrategy(
        max_retries=0, base_delay=0, backoff=0, jitter=False,
    ),
    ErrorCategory.AUTH: CategoryStrategy(
        max_retries=0, base_delay=0, backoff=0, jitter=False,
    ),
    ErrorCategory.SERVER: CategoryStrategy(
        max_retries=3, base_delay=2.0, backoff=2.0, jitter=True,
    ),
    ErrorCategory.CONTEXT: CategoryStrategy(
        max_retries=1, base_delay=0, backoff=0, jitter=False,
    ),
    ErrorCategory.UNKNOWN: CategoryStrategy(
        max_retries=2, base_delay=1.0, backoff=2.0, jitter=True,
    ),
}


def _compute_delay(strategy: CategoryStrategy, attempt: int) -> float:
    """Compute delay for given attempt, with optional jitter."""
    delay = strategy.base_delay * (strategy.backoff ** attempt)
    if strategy.jitter and delay > 0:
        delay *= 1.0 + random.uniform(-strategy.jitter_factor, strategy.jitter_factor)
    return max(0, delay)


# --- Categorized retry policy -----------------------------------------------


@dataclass
class CategorizedRetryPolicy:
    """Retry policy with per-category backoff strategies.

    Usage::

        policy = CategorizedRetryPolicy()
        retrying = RetryingProvider(provider, policy=policy.to_retry_policy())
    """

    strategies: dict[ErrorCategory, CategoryStrategy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Merge user-provided strategies with defaults
        merged = {**DEFAULT_STRATEGIES, **self.strategies}
        self.strategies = merged

    def get_strategy(self, category: ErrorCategory) -> CategoryStrategy:
        return self.strategies[category]

    def should_retry(self, exc: BaseException, status_code: int | None = None) -> bool:
        """Return ``True`` if this exception category allows retries."""
        category = classify_error(exc, status_code)
        strategy = self.get_strategy(category)
        return strategy.max_retries > 0

    def compute_delay(
        self, exc: BaseException, attempt: int, status_code: int | None = None
    ) -> float:
        """Compute retry delay for a given exception and attempt number."""
        category = classify_error(exc, status_code)
        strategy = self.get_strategy(category)
        return _compute_delay(strategy, attempt)

    def max_retries_for(self, exc: BaseException, status_code: int | None = None) -> int:
        """Maximum retry count for this exception type."""
        category = classify_error(exc, status_code)
        return self.get_strategy(category).max_retries

    def to_retry_policy(self) -> RetryPolicy:
        """Convert to a legacy ``RetryPolicy`` usable with ``RetryingProvider``.

        Note: the legacy RetryPolicy only supports a single backoff strategy.
        For full per-category behavior, use ``CategorizedRetryingProvider``.
        """
        return RetryPolicy(
            max_retries=max(s.max_retries for s in self.strategies.values()),
            base_delay=1.0,
            backoff=2.0,
            retry_on=(RateLimitError, ProviderTimeoutError),
        )


# --- Enhanced RetryingProvider with category-aware retry --------------------


class CategorizedRetryingProvider:
    """Provider wrapper that retries with per-category backoff strategies.

    Differs from ``RetryingProvider``:
    - Different retry count per error type
    - Different delay formula per error type
    - Network errors retry fast; rate-limit errors retry with long backoff
    - Parameter/auth errors are never retried

    Usage::

        provider = CategorizedRetryingProvider(OpenAIProvider(...))
        response = await provider.chat(messages)
    """

    def __init__(
        self,
        inner: Any,  # Provider
        policy: CategorizedRetryPolicy | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        import asyncio as _asyncio

        self.inner = inner
        self.policy = policy or CategorizedRetryPolicy()
        self._sleep = sleep or _asyncio.sleep

    async def chat(
        self,
        messages: list[Any],  # list[Message]
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:  # ChatResponse
        category: ErrorCategory | None = None
        max_retries = 0
        attempt = 0

        while True:
            try:
                result = await self.inner.chat(messages, tools, response_format=response_format)
                return result
            except Exception as exc:
                # Extract status code if available (httpx)
                status_code: int | None = None
                if hasattr(exc, "response"):
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        status_code = getattr(resp, "status_code", None)

                current_category = classify_error(exc, status_code)
                strategy = self.policy.get_strategy(current_category)

                # On first failure, lock in the category for this call chain
                if category is None:
                    category = current_category
                    max_retries = strategy.max_retries

                # Log the classification
                logger.debug(
                    f"[categorized-retry] error={type(exc).__name__} "
                    f"category={current_category.value} "
                    f"attempt={attempt + 1}/{max_retries + 1}"
                )

                if attempt >= max_retries:
                    logger.error(
                        f"[categorized-retry] exhausted retries "
                        f"({attempt}/{max_retries}) for category={current_category.value}"
                    )
                    raise

                delay = _compute_delay(strategy, attempt)
                if delay > 0:
                    logger.info(
                        f"[categorized-retry] retrying in {delay:.2f}s "
                        f"(category={current_category.value}, attempt={attempt + 1})"
                    )
                    await self._sleep(delay)

                attempt += 1

    def stream(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Stream is delegated without retry."""
        return self.inner.stream(messages, tools, response_format=response_format)
