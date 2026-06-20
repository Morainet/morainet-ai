"""Tests for morainet.engineering.billing."""

from __future__ import annotations

import pytest

from morainet.engineering.billing import (
    BillingStats,
    BillingTracker,
    ModelUsageRecord,
)
from morainet.exceptions import BudgetExceededError


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------

def test_model_usage_record():
    r = ModelUsageRecord(model="gpt-4o", input_tokens=100, output_tokens=50)
    assert r.model == "gpt-4o"
    assert r.input_tokens == 100
    assert r.output_tokens == 50
    assert r.timestamp > 0


# ---------------------------------------------------------------------------
# BillingStats
# ---------------------------------------------------------------------------

def test_billing_stats_defaults():
    s = BillingStats()
    assert s.total_calls == 0
    assert s.total_input_tokens == 0
    assert s.total_output_tokens == 0
    assert s.total_tokens == 0
    assert s.estimated_cost_usd == 0.0
    assert s.budget_usd is None
    assert s.budget_remaining_usd is None
    assert s.per_model == {}


def test_billing_stats_to_dict():
    s = BillingStats(
        total_calls=5,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.005,
        budget_usd=1.0,
        budget_remaining_usd=0.995,
        per_model={"gpt-4o": {"calls": 5, "input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.005}},
    )
    d = s.to_dict()
    assert d["total_calls"] == 5
    assert d["estimated_cost_usd"] == "$0.005000"
    assert d["budget_usd"] == "$1.000000"
    assert d["budget_remaining_usd"] == "$0.995000"
    assert d["per_model"]["gpt-4o"]["calls"] == 5


def test_billing_stats_to_dict_no_budget():
    s = BillingStats(total_calls=1, estimated_cost_usd=0.001)
    d = s.to_dict()
    assert d["budget_usd"] is None
    assert d["budget_remaining_usd"] is None


# ---------------------------------------------------------------------------
# BillingTracker: construction
# ---------------------------------------------------------------------------

def test_tracker_default_construction():
    bt = BillingTracker()
    assert bt.total_calls == 0
    assert bt.total_input_tokens == 0
    assert bt.total_output_tokens == 0
    assert bt.total_cost_usd == 0.0
    assert bt.budget_usd is None
    assert bt.budget_remaining_usd is None


def test_tracker_with_custom_pricing():
    bt = BillingTracker(pricing={"my-model": {"input": 1.0, "output": 2.0}})
    assert "my-model" in bt._pricing
    # Default pricing still present
    assert "gpt-4o" in bt._pricing


def test_tracker_with_budget():
    bt = BillingTracker(budget_usd=5.0)
    assert bt.budget_usd == 5.0
    assert bt.budget_remaining_usd == 5.0


def test_tracker_default_pricing():
    bt = BillingTracker()
    assert "gpt-4o" in bt._pricing
    assert "deepseek-chat" in bt._pricing
    assert "glm-4" in bt._pricing
    assert "moonshot-v1" in bt._pricing


# ---------------------------------------------------------------------------
# BillingTracker: record
# ---------------------------------------------------------------------------

def test_record_with_known_model():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=1000, output_tokens=500)
    assert bt.total_calls == 1
    assert bt.total_input_tokens == 1000
    assert bt.total_output_tokens == 500
    expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
    assert bt.total_cost_usd == pytest.approx(expected_cost)


def test_record_multiple_calls():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=100, output_tokens=50)
    bt.record("gpt-4o", input_tokens=200, output_tokens=100)
    assert bt.total_calls == 2
    assert bt.total_input_tokens == 300
    assert bt.total_output_tokens == 150


def test_record_unknown_model():
    bt = BillingTracker()
    bt.record("unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert bt.total_calls == 1
    assert bt.total_cost_usd == 0.0  # no pricing → free


def test_record_prefix_match():
    """Model name prefix match: unknown variant → match closest known pricing."""
    bt = BillingTracker()
    bt.record("gpt-4o-2024-08-06", input_tokens=1_000_000, output_tokens=0)
    # Should match gpt-4o pricing (longest prefix match)
    assert bt.total_cost_usd > 0
    assert bt.total_cost_usd < 100  # sanity check: not zero, not astronomical


def test_record_model_stats_accumulation():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=100, output_tokens=50)
    bt.record("deepseek-chat", input_tokens=200, output_tokens=50)
    assert bt._model_stats["gpt-4o"]["calls"] == 1
    assert bt._model_stats["deepseek-chat"]["calls"] == 1


# ---------------------------------------------------------------------------
# BillingTracker: budget exceeded
# ---------------------------------------------------------------------------

def test_record_budget_exceeded():
    bt = BillingTracker(budget_usd=0.0001)
    # 1M input tokens @ $2.50/1M = $2.50 which exceeds budget
    with pytest.raises(BudgetExceededError) as exc:
        bt.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert "budget exceeded" in str(exc.value).lower()


def test_record_budget_not_exceeded_on_zero_cost():
    bt = BillingTracker(budget_usd=10.0)
    bt.record("unknown-model", input_tokens=1_000_000)
    assert bt.total_calls == 1


# ---------------------------------------------------------------------------
# BillingTracker: stats
# ---------------------------------------------------------------------------

def test_stats_empty():
    bt = BillingTracker()
    s = bt.stats()
    assert s.total_calls == 0
    assert s.estimated_cost_usd == 0.0


def test_stats_after_records():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=1000, output_tokens=500)
    s = bt.stats()
    assert s.total_calls == 1
    assert s.total_input_tokens == 1000
    assert s.total_tokens == 1500
    assert s.estimated_cost_usd > 0


def test_stats_with_budget():
    bt = BillingTracker(budget_usd=10.0)
    bt.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    s = bt.stats()
    assert s.budget_usd == 10.0
    assert s.budget_remaining_usd == pytest.approx(10.0 - 2.50, rel=0.01)


# ---------------------------------------------------------------------------
# BillingTracker: stats_summary
# ---------------------------------------------------------------------------

def test_stats_summary():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=100, output_tokens=50)
    summary = bt.stats_summary()
    assert isinstance(summary, dict)
    assert "total_calls" in summary
    assert "$" in str(summary["estimated_cost_usd"])


# ---------------------------------------------------------------------------
# BillingTracker: reset
# ---------------------------------------------------------------------------

def test_reset():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=100, output_tokens=50)
    assert bt.total_calls == 1

    bt.reset()
    assert bt.total_calls == 0
    assert bt.total_input_tokens == 0
    assert bt.total_output_tokens == 0
    assert bt.total_cost_usd == 0.0
    assert len(bt._records) == 0
    assert bt.budget_remaining_usd is None


def test_reset_with_budget():
    bt = BillingTracker(budget_usd=10.0)
    bt.record("gpt-4o", input_tokens=100, output_tokens=50)
    bt.reset()
    assert bt.total_cost_usd == 0.0
    assert bt.budget_remaining_usd == 10.0


# ---------------------------------------------------------------------------
# BillingTracker: cost edge cases
# ---------------------------------------------------------------------------

def test_cost_longest_prefix_match():
    """When multiple known models share a prefix, check cost is non-zero."""
    bt = BillingTracker(pricing={
        "gpt": {"input": 1.0, "output": 1.0},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    })
    bt.record("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    # Should match either gpt-4o or gpt-4o-mini (from defaults); cost > 0
    assert bt.total_cost_usd > 0
    assert bt.total_cost_usd < 10


def test_record_zero_output_tokens():
    bt = BillingTracker()
    bt.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert bt.total_cost_usd == pytest.approx(2.50)
    assert bt.total_output_tokens == 0
