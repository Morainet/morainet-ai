from __future__ import annotations

from morainet.tokens import estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_roughly_quarter_of_length():
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("a") == 1
