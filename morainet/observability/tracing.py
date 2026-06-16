"""Lightweight tracing helpers."""

from __future__ import annotations

import uuid

from loguru import logger


def new_trace_id() -> str:
    return uuid.uuid4().hex


__all__ = ["new_trace_id", "logger"]
