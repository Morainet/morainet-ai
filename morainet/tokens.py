"""Rough, dependency-free token estimation.

A cheap heuristic (~4 chars per token) good enough for context-window
budgeting. Swap in a real tokenizer (e.g. tiktoken) for precise accounting.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))
