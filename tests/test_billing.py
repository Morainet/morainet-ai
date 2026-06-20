from __future__ import annotations

import pytest

from morainet.engineering.billing import BillingStats, BillingTracker, ModelUsageRecord
from morainet.exceptions import BudgetExceededError


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------

def test_model_usage_record_fields():
    record = ModelUsageRecord(model="gpt-4o", input_tokens=500, output_tokens=200)
    assert record.model == "gpt-4o"
    assert record.input_tokens == 500
    assert record.output_tokens == 200
    assert isinstance(record.timestamp, float)


def test_model_usage_record_default_timestamp():
    record = ModelUsageRecord(model="gpt-4o", input_tokens=100, output_tokens=50)
    assert record.timestamp > 0


# ---------------------------------------------------------------------------
# BillingStats
# ---------------------------------------------------------------------------

def test_billing_stats_to_dict_defaults():
    stats = BillingStats()
    d = stats.to_dict()
    assert d["total_calls"] == 0
    assert d["total_input_tokens"] == 0
    assert d["total_output_tokens"] == 0
    assert d["total_tokens"] == 0
    assert d["estimated_cost_usd"] == "$0.000000"
    assert d["budget_usd"] is None
    assert d["budget_remaining_usd"] is None
    assert d["per_model"] == {}


def test_billing_stats_to_dict_with_budget():
    stats = BillingStats(
        total_calls=5,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.0075,
        budget_usd=5.00,
        budget_remaining_usd=4.9925,
        per_model={"gpt-4o": {"calls": 5, "input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.0075}},
    )
    d = stats.to_dict()
    assert d["total_calls"] == 5
    assert d["estimated_cost_usd"] == "$0.007500"
    assert d["budget_usd"] == "$5.000000"
    assert d["budget_remaining_usd"] == "$4.992500"


# ---------------------------------------------------------------------------
# BillingTracker._cost()
# ---------------------------------------------------------------------------

def test_cost_calculation_gpt4o():
    tracker = BillingTracker()
    cost = tracker._cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(12.50)


def test_cost_calculation_gpt4o_only_input():
    tracker = BillingTracker()
    cost = tracker._cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(2.50)


def test_cost_calculation_prefix_match():
    tracker = BillingTracker()
    cost = tracker._cost("gpt-4o-2024-08-06", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(2.50)


def test_cost_calculation_unknown_model_returns_zero():
    tracker = BillingTracker()
    cost = tracker._cost("unknown-model-v99", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.0


def test_cost_calculation_custom_pricing():
    tracker = BillingTracker(pricing={"my-model": {"input": 1.0, "output": 2.0}})
    cost = tracker._cost("my-model", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(2.0)


def test_cost_calculation_custom_pricing_overrides_default():
    tracker = BillingTracker(pricing={"gpt-4o": {"input": 1.0, "output": 2.0}})
    cost = tracker._cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# BillingTracker.record()
# ---------------------------------------------------------------------------

def test_record_single_call():
    tracker = BillingTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
    assert tracker.total_calls == 1
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500
    assert tracker.total_cost_usd > 0


def test_record_multiple_calls_accumulate():
    tracker = BillingTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=0)
    tracker.record("gpt-4o", input_tokens=500, output_tokens=0)
    assert tracker.total_calls == 2
    assert tracker.total_input_tokens == 1500


def test_record_within_budget_succeeds():
    tracker = BillingTracker(budget_usd=0.01)
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=0)


def test_record_exceeds_budget_raises():
    tracker = BillingTracker(budget_usd=0.001)
    tracker.record("gpt-4o", input_tokens=100, output_tokens=0)
    with pytest.raises(BudgetExceededError):
        tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)


def test_record_no_budget_no_error():
    tracker = BillingTracker()
    tracker.record("gpt-4o", input_tokens=10_000_000, output_tokens=10_000_000)
    assert tracker.total_cost_usd > 0


# ---------------------------------------------------------------------------
# BillingTracker.stats() / stats_summary()
# ---------------------------------------------------------------------------

def test_stats_snapshot():
    tracker = BillingTracker(budget_usd=5.0)
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
    stats = tracker.stats()
    assert isinstance(stats, BillingStats)
    assert stats.total_calls == 1
    assert stats.budget_usd == 5.0
    assert stats.budget_remaining_usd is not None
    assert "gpt-4o" in stats.per_model


def test_stats_summary():
    tracker = BillingTracker(budget_usd=10.0)
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=0)
    d = tracker.stats_summary()
    assert isinstance(d, dict)
    assert d["total_calls"] == 1
    assert d["budget_usd"] == "$10.000000"


def test_stats_per_model_aggregation():
    tracker = BillingTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=100)
    tracker.record("gpt-4o", input_tokens=500, output_tokens=50)
    tracker.record("claude-3-5-sonnet", input_tokens=200, output_tokens=100)
    stats = tracker.stats()
    assert stats.per_model["gpt-4o"]["calls"] == 2
    assert stats.per_model["claude-3-5-sonnet"]["calls"] == 1


# ---------------------------------------------------------------------------
# BillingTracker.budget_remaining_usd
# ---------------------------------------------------------------------------

def test_budget_remaining_no_budget():
    tracker = BillingTracker()
    assert tracker.budget_remaining_usd is None


def test_budget_remaining_with_budget():
    tracker = BillingTracker(budget_usd=5.0)
    tracker.record("gpt-4o", input_tokens=100_000, output_tokens=0)
    remaining = tracker.budget_remaining_usd
    assert remaining is not None
    assert remaining > 0
    assert remaining < 5.0


def test_budget_remaining_never_below_zero():
    tracker = BillingTracker(budget_usd=5.0)
    tracker.record("gpt-4o", input_tokens=2_000_000, output_tokens=0)
    assert tracker.budget_remaining_usd == 0.0


# ---------------------------------------------------------------------------
# BillingTracker.reset()
# ---------------------------------------------------------------------------

def test_reset_clears_all():
    tracker = BillingTracker(budget_usd=5.0)
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
    tracker.record("gpt-4o", input_tokens=500, output_tokens=250)
    tracker.reset()
    assert tracker.total_calls == 0
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0
    assert tracker.total_cost_usd == 0.0
    stats = tracker.stats()
    assert stats.per_model == {}
