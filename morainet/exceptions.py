"""Exception hierarchy for Morainet AI.

All framework exceptions derive from :class:`MorainetError`, so callers can
catch the whole family with a single ``except``.
"""

from __future__ import annotations


class MorainetError(Exception):
    """Root exception for all framework errors."""


class ConfigError(MorainetError):
    """Invalid or missing configuration."""


# --- Provider errors -------------------------------------------------------


class ProviderError(MorainetError):
    """Base class for LLM provider failures."""


class RateLimitError(ProviderError):
    """Provider rate limit hit (retryable)."""


class ProviderTimeoutError(ProviderError):
    """Provider request timed out (retryable)."""


class AuthError(ProviderError):
    """Authentication / authorization failure (not retryable)."""


class ContextLengthError(ProviderError):
    """Request exceeded the model context window."""


# --- Tool errors -----------------------------------------------------------


class ToolError(MorainetError):
    """Base class for tool-related failures."""


class ToolNotFoundError(ToolError):
    """Referenced tool is not registered."""


class ToolValidationError(ToolError):
    """Tool arguments failed validation (fed back to the model for self-repair)."""


class ToolExecutionError(ToolError):
    """Tool raised an exception during execution."""


# --- Reasoning errors ------------------------------------------------------


class ReasoningError(MorainetError):
    """Base class for reasoning-loop failures."""


class MaxStepsExceededError(ReasoningError):
    """Reasoning loop hit ``max_steps`` without converging."""


class BudgetExceededError(ReasoningError):
    """Run exceeded its configured token budget before converging."""


class MaxConsecutiveErrorsError(ReasoningError):
    """Too many consecutive tool failures; the run was aborted."""


class PlanError(ReasoningError):
    """Planning phase failed — could not decompose task into actionable steps."""


class ReflectionError(ReasoningError):
    """Reflection phase failed — could not evaluate progress."""


class VerificationError(ReasoningError):
    """Self-verification detected an issue with the answer."""


# --- Memory errors ---------------------------------------------------------


class MemoryStoreError(MorainetError):
    """Memory backend (vector store) failure."""


class DocumentParseError(MorainetError):
    """Document parsing failure (unsupported format, corrupt file, etc.)."""


# --- Workflow errors -------------------------------------------------------


class WorkflowError(MorainetError):
    """Base class for workflow failures."""


class CycleError(WorkflowError):
    """The workflow graph contains a cycle."""


class NodeNotFoundError(WorkflowError):
    """Referenced node is not in the graph."""
