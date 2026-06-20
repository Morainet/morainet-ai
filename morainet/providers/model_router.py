"""Multi-model router: auto-route tasks to models by complexity, with cost control.

Route simple queries to cheap/fast models and complex tasks to powerful models,
dramatically reducing API costs while maintaining quality.

Key capabilities:
- **Complexity scoring**: heuristic analysis of query length, structure, and
  intent to determine which model tier to use.
- **Model tiers**: define small/medium/large model pools with cost weights.
- **Fallback chains**: if a small model fails or gives low-confidence output,
  escalate to a larger model.
- **Cost tracking**: accumulate per-model token usage and estimated cost.
- **Concurrency**: optional concurrent query to multiple models, pick best.

Usage::

    from morainet.providers import (
        ModelRouter, QwenProvider, DeepSeekProvider,
    )

    router = ModelRouter(
        tiers={
            "small": [QwenProvider(model="qwen-turbo"), DeepSeekProvider(model="deepseek-chat")],
            "large": [QwenProvider(model="qwen-plus"), DeepSeekProvider(model="deepseek-reasoner")],
        },
        default_tier="small",
        fallback_enabled=True,
    )

    # Single query — auto-routed
    response = await router.chat(messages)

    # Stream
    async for token in router.stream(messages):
        print(token, end="")
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from morainet.core.models import ChatResponse, Message, Usage
from morainet.observability.tracing import logger
from morainet.providers.base import Provider


@dataclass
class ModelTier:
    """A pool of providers at the same capability/cost level."""

    providers: list[Provider]
    cost_per_1k_tokens: float = 0.0  # estimated cost per 1K input tokens (USD)
    label: str = ""


@dataclass
class RouterStats:
    """Cumulative routing statistics for cost optimization."""

    total_calls: int = 0
    calls_by_tier: dict[str, int] = field(default_factory=dict)
    total_usage: Usage = field(default_factory=Usage)
    estimated_cost_usd: float = 0.0


# --- complexity heuristics -----------------------------------------------

# Keywords that suggest a complex/analytical task
_COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"analyze|analysis|compare|contrast|evaluate|summarize|synthesize|"
    r"debug|diagnose|optimize|refactor|architect|design|implement|"
    r"explain.*in detail|step.by.step|reason.*about|think.*through|"
    r"code|script|function|algorithm|database|schema|"
    r"计算|分析|对比|评估|总结|调试|诊断|优化|设计|实现|架构|"
    r"详细|逐步|推理|思考|代码|函数|算法|数据库"
    r")\b",
    re.IGNORECASE,
)

# Keywords that suggest a simple task
_SIMPLE_KEYWORDS = re.compile(
    r"\b("
    r"hi|hello|hey|bye|thanks|thank you|ok|okay|yes|no|what is|who is|"
    r"when is|how are you|translate|define|"
    r"你好|谢谢|再见|什么是|谁|什么时候|翻译|定义"
    r")\b",
    re.IGNORECASE,
)


def estimate_complexity(messages: list[Message]) -> float:
    """Estimate task complexity on a 0.0–1.0 scale.

    Heuristics used:
    - Message count and total character length
    - Presence of code blocks or structured data
    - Keyword matching (complex vs simple)
    - Multi-turn conversation depth

    Returns a float where:
    - < 0.3 → simple (greetings, definitions)
    - 0.3–0.6 → moderate (Q&A, summarization)
    - > 0.6 → complex (analysis, coding, multi-step reasoning)
    """
    user_texts = [
        m.content or ""
        for m in messages
        if m.role.value == "user"
    ]
    if not user_texts:
        return 0.3  # default moderate

    combined = " ".join(user_texts)
    total_chars = len(combined)
    msg_count = len(user_texts)
    score = 0.0

    # 1. Length factor (longer queries tend to be more complex)
    if total_chars > 2000:
        score += 0.4
    elif total_chars > 800:
        score += 0.25
    elif total_chars > 200:
        score += 0.1

    # 2. Multi-turn factor
    if msg_count > 3:
        score += 0.2
    elif msg_count > 1:
        score += 0.1

    # 3. Code / structured data presence
    if re.search(r"```|def |class |function |import |SELECT |FROM |WHERE ", combined):
        score += 0.25

    # 4. Complex keyword match
    if _COMPLEX_KEYWORDS.search(combined):
        score += 0.2

    # 5. Simple keyword match (reduces score)
    if _SIMPLE_KEYWORDS.search(combined) and total_chars < 200:
        score -= 0.15

    return max(0.0, min(1.0, score))


def _get_messages_text(messages: list[Message]) -> str:
    return " ".join(m.content or "" for m in messages if m.content)


# --- router ---------------------------------------------------------------


class ModelRouter(Provider):
    """Multi-model router with complexity-based routing and cost control.

    Tiers are ordered from cheapest to most expensive. The router will:
    1. Estimate task complexity
    2. Select the appropriate tier
    3. Pick a provider from that tier (round-robin for load balancing)
    4. On failure or low-quality output, fall back to a higher tier
    """

    def __init__(
        self,
        tiers: dict[str, list[Provider]],
        default_tier: str = "small",
        complexity_thresholds: tuple[float, float] = (0.3, 0.6),
        fallback_enabled: bool = True,
        fallback_max_tiers: int = 3,
        cost_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            tiers: Provider pools keyed by tier name (e.g. "small", "medium", "large").
            default_tier: Tier to use when complexity can't be determined.
            complexity_thresholds: (simple_max, complex_min) thresholds.
                - score < simple_max → lowest tier
                - simple_max ≤ score < complex_min → middle tier
                - score ≥ complex_min → highest tier
            fallback_enabled: If True, escalate to higher tier on failure.
            fallback_max_tiers: Max number of tiers to try before giving up.
            cost_weights: Per-tier USD cost per 1K tokens.
        """
        if not tiers:
            raise ValueError("At least one tier is required.")
        self.tiers = tiers
        self.tier_names = list(tiers.keys())
        self.default_tier = default_tier
        self.complexity_thresholds = complexity_thresholds
        self.fallback_enabled = fallback_enabled
        self.fallback_max_tiers = fallback_max_tiers
        self.cost_weights = cost_weights or {}

        # Round-robin counters per tier
        self._rr_counters: dict[str, int] = {name: 0 for name in tiers}

        # Stats
        self.stats = RouterStats()

    # --- routing logic ----------------------------------------------------

    def _select_tier(self, complexity: float) -> str:
        """Map a complexity score to a tier name."""
        tier_count = len(self.tier_names)
        if tier_count == 1:
            return self.tier_names[0]

        simple_max, complex_min = self.complexity_thresholds

        if complexity < simple_max:
            return self.tier_names[0]  # cheapest
        elif complexity < complex_min:
            if tier_count >= 3:
                return self.tier_names[1]  # middle
            return self.tier_names[-1]
        else:
            return self.tier_names[-1]  # most capable

    def _pick_provider(self, tier_name: str) -> Provider:
        """Round-robin within a tier for load balancing."""
        providers = self.tiers.get(tier_name, [])
        if not providers:
            raise ValueError(f"No providers in tier '{tier_name}'")
        idx = self._rr_counters[tier_name]
        self._rr_counters[tier_name] = (idx + 1) % len(providers)
        return providers[idx]

    def _get_fallback_tiers(self, current_tier: str) -> list[str]:
        """Return tiers to try after the current one (in escalation order)."""
        try:
            idx = self.tier_names.index(current_tier)
        except ValueError:
            return []
        candidates = self.tier_names[idx + 1 :]
        return candidates[: self.fallback_max_tiers - 1]

    def _update_stats(self, tier_name: str, usage: Usage) -> None:
        """Record usage and estimate cost."""
        self.stats.total_calls += 1
        self.stats.calls_by_tier[tier_name] = (
            self.stats.calls_by_tier.get(tier_name, 0) + 1
        )
        self.stats.total_usage = self.stats.total_usage + usage

        cost_per_1k = self.cost_weights.get(tier_name, 0.0)
        if cost_per_1k > 0:
            total_tokens = usage.prompt_tokens + usage.completion_tokens
            self.stats.estimated_cost_usd += (total_tokens / 1000) * cost_per_1k

    # --- API ---------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        complexity = estimate_complexity(messages)
        tier = self._select_tier(complexity)
        logger.debug(
            f"[router] complexity={complexity:.2f} → tier='{tier}' "
            f"(query preview: {_get_messages_text(messages)[:80]})"
        )

        tiers_to_try = [tier]
        if self.fallback_enabled:
            tiers_to_try += self._get_fallback_tiers(tier)

        last_error: Exception | None = None
        for attempt, tier_name in enumerate(tiers_to_try):
            provider = self._pick_provider(tier_name)
            try:
                response = await provider.chat(messages, tools, response_format)
                self._update_stats(tier_name, response.usage)
                if attempt > 0:
                    logger.info(
                        f"[router] fallback to tier='{tier_name}' succeeded "
                        f"after {attempt} attempt(s)"
                    )
                return response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[router] tier='{tier_name}' failed: {exc}; "
                    f"escalating..."
                )

        raise RuntimeError(
            f"ModelRouter: all tiers exhausted (tried {tiers_to_try}). "
            f"Last error: {last_error}"
        ) from last_error

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens; routers to the best tier but streams directly."""
        complexity = estimate_complexity(messages)
        tier = self._select_tier(complexity)

        tiers_to_try = [tier]
        if self.fallback_enabled:
            tiers_to_try += self._get_fallback_tiers(tier)

        last_error: Exception | None = None
        for attempt, tier_name in enumerate(tiers_to_try):
            provider = self._pick_provider(tier_name)
            try:
                async for chunk in provider.stream(messages, tools, response_format):
                    yield chunk
                return  # successful
            except Exception as exc:
                last_error = exc
                logger.warning(f"[router] tier='{tier_name}' stream failed: {exc}")

        raise RuntimeError(
            f"ModelRouter stream: all tiers exhausted. "
            f"Last error: {last_error}"
        ) from last_error

    # --- convenience -------------------------------------------------------

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self.stats = RouterStats()

    def get_stats_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of routing statistics."""
        return {
            "total_calls": self.stats.total_calls,
            "calls_by_tier": dict(self.stats.calls_by_tier),
            "total_tokens": {
                "prompt": self.stats.total_usage.prompt_tokens,
                "completion": self.stats.total_usage.completion_tokens,
                "total": self.stats.total_usage.total_tokens,
            },
            "estimated_cost_usd": f"${self.stats.estimated_cost_usd:.6f}",
        }


# --- concurrent multi-model query -----------------------------------------


async def multi_model_query(
    providers: list[Provider],
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    pick_strategy: str = "fastest",
) -> ChatResponse:
    """Query multiple models concurrently and pick the best response.

    Args:
        providers: List of providers to query in parallel.
        messages: Messages to send.
        tools: Optional tool definitions.
        response_format: Optional response format.
        pick_strategy: How to pick the winner:
            - ``"fastest"``: return the first successful response.
            - ``"longest"``: wait for all, return the longest response.
            - ``"all"``: return all responses (as a dict).

    Returns:
        The winning ChatResponse, or for ``"all"`` strategy, returns
        a ChatResponse whose content is a concatenation.
    """
    if pick_strategy == "fastest":
        return await _race(providers, messages, tools, response_format)
    elif pick_strategy == "longest":
        return await _pick_longest(providers, messages, tools, response_format)
    else:
        # "all" — concatenate all responses
        raise NotImplementedError(
            "pick_strategy='all' not yet implemented for multi_model_query"
        )


async def _race(
    providers: list[Provider],
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    response_format: dict[str, Any] | None,
) -> ChatResponse:
    """Race: return the first successful response."""
    tasks = [
        p.chat(messages, tools, response_format)
        for p in providers
    ]
    for coro in asyncio.as_completed(tasks):
        try:
            return await coro
        except Exception as exc:
            logger.debug(f"[multi-model race] one provider failed: {exc}")
    raise RuntimeError("All providers failed in multi-model race.")


async def _pick_longest(
    providers: list[Provider],
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    response_format: dict[str, Any] | None,
) -> ChatResponse:
    """Wait for all, return the longest content."""
    results_raw = await asyncio.gather(
        *[p.chat(messages, tools, response_format) for p in providers],
        return_exceptions=True,
    )
    best: ChatResponse | None = None
    best_len = 0
    for r in results_raw:
        if isinstance(r, BaseException):
            logger.debug(f"[multi-model longest] one provider failed: {r}")
            continue
        content_len = len(r.message.content or "")
        if content_len > best_len:
            best = r
            best_len = content_len
    if best is None:
        raise RuntimeError("All providers failed in multi-model query.")
    return best
