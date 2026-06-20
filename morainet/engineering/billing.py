"""Per-model billing tracker with token counting and cost estimation.

Tracks input/output tokens per model call, maps them to configured pricing tiers,
and provides cumulative cost summaries for budget monitoring.

Usage::

    tracker = BillingTracker(
        pricing={"gpt-4o": {"input": 2.50, "output": 10.00}},  # per 1M tokens
        budget_usd=5.00,
    )
    tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
    print(tracker.stats_summary())  # {"total_calls": 1, "estimated_cost_usd": "$0.003250", ...}
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from morainet.exceptions import BudgetExceededError


@dataclass
class ModelUsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class BillingStats:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    budget_usd: float | None = None
    budget_remaining_usd: float | None = None
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": f"${self.estimated_cost_usd:.6f}",
            "budget_usd": f"${self.budget_usd:.6f}" if self.budget_usd is not None else None,
            "budget_remaining_usd": (
                f"${self.budget_remaining_usd:.6f}"
                if self.budget_remaining_usd is not None
                else None
            ),
            "per_model": self.per_model,
        }


class BillingTracker:
    """Tracks cumulative token usage and cost across all model calls.

    ``pricing`` is a dict mapping model name → {"input": price_per_1M, "output": price_per_1M}.
    Prices are in USD per **million** tokens (standard OpenAI-style pricing).
    """

    # Default pricing (USD per 1M tokens), approximate as of mid-2025
    DEFAULT_PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "qwen-plus": {"input": 0.55, "output": 1.10},
        "qwen-turbo": {"input": 0.27, "output": 0.55},
        "glm-4": {"input": 1.40, "output": 1.40},
        "moonshot-v1": {"input": 1.60, "output": 1.60},
        "ernie-4.0": {"input": 0.80, "output": 1.60},
    }

    def __init__(
        self,
        pricing: dict[str, dict[str, float]] | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self._pricing = {**self.DEFAULT_PRICING, **(pricing or {})}
        self._budget_usd = budget_usd
        self._records: list[ModelUsageRecord] = []
        self._model_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )

        # Cumulative counters
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def budget_usd(self) -> float | None:
        return self._budget_usd

    @property
    def budget_remaining_usd(self) -> float | None:
        if self._budget_usd is None:
            return None
        return max(0.0, self._budget_usd - self.total_cost_usd)

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute cost in USD from token counts and configured pricing."""
        prices = self._pricing.get(model)
        if prices is None:
            # Try prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
            for known in sorted(self._pricing, key=len, reverse=True):
                if model.startswith(known):
                    prices = self._pricing[known]
                    break
        if prices is None:
            return 0.0

        return (input_tokens / 1_000_000) * prices["input"] + (
            output_tokens / 1_000_000
        ) * prices["output"]

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        """Record a model call, updating cumulative stats.

        Raises ``BudgetExceededError`` if ``budget_usd`` is set and the new cost
        would push total cost over budget.
        """
        cost = self._cost(model, input_tokens, output_tokens)

        if self._budget_usd is not None and self.total_cost_usd + cost > self._budget_usd:
            raise BudgetExceededError(
                f"billing budget exceeded: "
                f"${self.total_cost_usd + cost:.6f} > ${self._budget_usd:.6f}"
            )

        self._records.append(
            ModelUsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        )
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        ms = self._model_stats[model]
        ms["calls"] += 1
        ms["input_tokens"] += input_tokens
        ms["output_tokens"] += output_tokens
        ms["cost_usd"] += cost

    def stats(self) -> BillingStats:
        """Return a snapshot of cumulative billing statistics."""
        return BillingStats(
            total_calls=self.total_calls,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_tokens=self.total_input_tokens + self.total_output_tokens,
            estimated_cost_usd=self.total_cost_usd,
            budget_usd=self._budget_usd,
            budget_remaining_usd=self.budget_remaining_usd,
            per_model=dict(self._model_stats),
        )

    def stats_summary(self) -> dict[str, Any]:
        """Return a human-readable dict summary."""
        return self.stats().to_dict()

    def reset(self) -> None:
        """Clear all records and counters."""
        self._records.clear()
        self._model_stats.clear()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
