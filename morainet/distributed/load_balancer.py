"""Model inference load balancing — route requests across provider endpoints.

Provides:

- :class:`LoadBalancer`: abstract load-balancing interface.
- :class:`RoundRobinBalancer`: simple round-robin.
- :class:`WeightedRoundRobinBalancer`: weighted round-robin for heterogeneous nodes.
- :class:`ModelRouter`: tiered routing (small/medium/large models).
- :class:`ProviderShard`: maps model families to endpoint lists.
- :class:`HybridRouter`: edge + cloud hybrid routing with complexity-based dispatch.

Usage::

    from morainet.distributed import (
        Endpoint, ProviderShard, WeightedRoundRobinBalancer, HybridRouter
    )

    shard = ProviderShard(
        name="openai",
        endpoints=[
            Endpoint(url="https://api.openai.com/v1", weight=3),
            Endpoint(url="https://azure.openai.com/v1", weight=2),
        ],
    )
    balancer = WeightedRoundRobinBalancer(shard.endpoints)
    ep = balancer.next()  # -> Endpoint
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    SMALL = "small"     # qwen2.5:3b, llama3.2:3b
    MEDIUM = "medium"   # llama3.1:8b, qwen2.5:7b
    LARGE = "large"     # gpt-4o, claude-3-opus
    EDGE = "edge"       # local embedded models


@dataclass
class Endpoint:
    """A provider API endpoint with optional weight."""

    url: str
    weight: float = 1.0
    tier: Tier = Tier.MEDIUM
    labels: dict[str, str] = field(default_factory=dict)
    healthy: bool = True
    latency_ms: float = 0.0  # last measured latency
    error_count: int = 0
    max_errors: int = 5

    @property
    def available(self) -> bool:
        return self.healthy and self.error_count < self.max_errors

    def record_success(self, elapsed_ms: float = 0.0) -> None:
        self.latency_ms = elapsed_ms
        self.error_count = max(0, self.error_count - 1)

    def record_error(self) -> None:
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.healthy = False

    def reset(self) -> None:
        self.healthy = True
        self.error_count = 0


@dataclass
class ProviderShard:
    """A group of endpoints for a model family."""

    name: str                  # e.g. "openai", "ollama", "anthropic"
    endpoints: list[Endpoint] = field(default_factory=list)
    tier: Tier = Tier.MEDIUM
    default_model: str = ""

    @property
    def healthy_endpoints(self) -> list[Endpoint]:
        return [ep for ep in self.endpoints if ep.available]

    def add_endpoint(self, ep: Endpoint) -> None:
        self.endpoints.append(ep)

    def remove_endpoint(self, url: str) -> None:
        self.endpoints = [ep for ep in self.endpoints if ep.url != url]


# ---------------------------------------------------------------------------
# Abstract load balancer
# ---------------------------------------------------------------------------


class LoadBalancer(ABC):
    """Abstract load balancer for selecting an endpoint."""

    @abstractmethod
    def next(self) -> Endpoint | None:
        """Return the next available endpoint, or None."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        ...


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------


class RoundRobinBalancer(LoadBalancer):
    """Simple round-robin across available endpoints."""

    def __init__(self, endpoints: list[Endpoint]) -> None:
        self._endpoints = endpoints
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> Endpoint | None:
        healthy = [ep for ep in self._endpoints if ep.available]
        if not healthy:
            return None
        with self._lock:
            ep = healthy[self._idx % len(healthy)]
            self._idx += 1
        return ep

    def reset(self) -> None:
        with self._lock:
            self._idx = 0


# ---------------------------------------------------------------------------
# Weighted round-robin (smooth algorithm)
# ---------------------------------------------------------------------------


class WeightedRoundRobinBalancer(LoadBalancer):
    """Smooth weighted round-robin — fair distribution respecting weights.

    Uses Nginx-style smooth weighting: each endpoint has an effective
    weight that accumulates, and the highest one wins, then subtracts total.
    """

    def __init__(self, endpoints: list[Endpoint]) -> None:
        self._endpoints = endpoints
        self._current_weights: dict[str, float] = {}
        self._lock = threading.Lock()
        self._total_weight = 0.0
        self._recalc()

    def _recalc(self) -> None:
        self._total_weight = sum(ep.weight for ep in self._endpoints if ep.available)
        for ep in self._endpoints:
            if ep.url not in self._current_weights:
                self._current_weights[ep.url] = 0.0

    def next(self) -> Endpoint | None:
        healthy = [ep for ep in self._endpoints if ep.available]
        if not healthy:
            return None

        with self._lock:
            self._recalc()
            if self._total_weight == 0:
                return healthy[0] if healthy else None

            # Accumulate effective weights
            for ep in healthy:
                self._current_weights[ep.url] += ep.weight

            # Pick the endpoint with highest effective weight
            best = max(healthy, key=lambda ep: self._current_weights[ep.url])
            best_url = best.url
            self._current_weights[best_url] -= self._total_weight
            return best

    def reset(self) -> None:
        with self._lock:
            self._current_weights.clear()
            self._recalc()


# ---------------------------------------------------------------------------
# Model router — tiered + provider-aware
# ---------------------------------------------------------------------------


class ModelRouter:
    """Route inference requests based on model tier and provider availability.

    Usage::

        router = ModelRouter()
        router.register("openai", openai_shard)
        router.register("ollama", ollama_shard)

        endpoint = router.route(tier="large", provider="openai")
        # -> Endpoint for GPT-4o
    """

    def __init__(self) -> None:
        self._shards: dict[str, ProviderShard] = {}
        self._balancers: dict[str, LoadBalancer] = {}
        self._tier_map: dict[Tier, list[str]] = {}

    def register(self, name: str, shard: ProviderShard, balancer: LoadBalancer | None = None) -> None:
        self._shards[name] = shard
        self._balancers[name] = balancer or WeightedRoundRobinBalancer(shard.endpoints)
        self._tier_map.setdefault(shard.tier, []).append(name)

    def route(self, tier: str | Tier = Tier.LARGE, provider: str | None = None,
              preferred_labels: dict[str, str] | None = None) -> Endpoint | None:
        """Route to the best available endpoint.

        Args:
            tier: Model size tier.
            provider: Specific provider name (or ``None`` for any).
            preferred_labels: Filter by labels (e.g. ``{"region": "us-east"}``).

        Returns:
            The selected Endpoint, or ``None`` if nothing is available.
        """
        tier = Tier(tier) if isinstance(tier, str) else tier
        candidates = self._tier_map.get(tier, [])
        if provider and provider in candidates:
            candidates = [provider]

        for name in candidates:
            balancer = self._balancers.get(name)
            if balancer is None:
                continue
            shard = self._shards[name]
            # Try to respect label preferences
            if preferred_labels:
                ep = self._route_with_labels(shard, preferred_labels)
                if ep:
                    return ep
            ep = balancer.next()
            if ep:
                return ep

        # Fallback: try next tier (e.g. large → medium → small)
        fallback_order = [Tier.LARGE, Tier.MEDIUM, Tier.SMALL, Tier.EDGE]
        start = fallback_order.index(tier) if tier in fallback_order else 0
        for ft in fallback_order[start + 1:]:
            for name in self._tier_map.get(ft, []):
                balancer = self._balancers.get(name)
                if balancer:
                    ep = balancer.next()
                    if ep:
                        return ep
        return None

    def _route_with_labels(self, shard: ProviderShard, labels: dict[str, str]) -> Endpoint | None:
        """Try to find an endpoint matching all labels."""
        for ep in shard.healthy_endpoints:
            if all(ep.labels.get(k) == v for k, v in labels.items()):
                return ep
        return None

    @property
    def shard_names(self) -> list[str]:
        return list(self._shards)

    def get_shard(self, name: str) -> ProviderShard | None:
        return self._shards.get(name)


# ---------------------------------------------------------------------------
# Hybrid router (edge + cloud)
# ---------------------------------------------------------------------------


class HybridRouter:
    """Edge + Cloud hybrid scheduler.

    Simple queries (complexity < threshold) → edge (local model).
    Complex queries (complexity >= threshold) → cloud (distributed cluster).

    Assumes a lightweight complexity estimator (e.g. string length, keyword
    matching) — can be swapped for an ML-based estimator.

    Usage::

        hybrid = HybridRouter(cloud_endpoint="http://cloud-api:8080", threshold=0.5)
        route, endpoint = hybrid.decide(query="What is 2+2?")
        # -> ("edge", local_endpoint)
        route, endpoint = hybrid.decide(query="Explain transformer architecture in detail...")
        # -> ("cloud", cloud_endpoint)
    """

    def __init__(
        self,
        cloud_endpoint: str = "http://localhost:8080",
        cloud_shard: ProviderShard | None = None,
        edge_shard: ProviderShard | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.cloud_endpoint = cloud_endpoint
        self.cloud_balancer = WeightedRoundRobinBalancer(
            cloud_shard.endpoints
        ) if cloud_shard else WeightedRoundRobinBalancer([
            Endpoint(url=cloud_endpoint, tier=Tier.LARGE)
        ])
        self.edge_balancer = WeightedRoundRobinBalancer(
            edge_shard.endpoints
        ) if edge_shard else WeightedRoundRobinBalancer([
            Endpoint(url="http://localhost:11434", tier=Tier.EDGE)
        ])
        self.threshold = threshold
        self._keywords = [
            "explain", "analyze", "optimize", "debug", "refactor",
            "summarize", "translate", "generate code for",
            "design", "architecture", "deploy", "implement",
            "comparison", "trade-off", "best practice",
        ]

    def estimate_complexity(self, query: str) -> float:
        """Simple heuristic: word count + keyword match → 0-1 score."""
        words = query.split()
        word_count = len(words)
        # Base score from word count (logarithmic, saturates at ~100 words)
        word_score = min(1.0, word_count / 50.0)

        # Keyword bonus
        kw_count = sum(1 for kw in self._keywords if kw.lower() in query.lower())
        kw_score = min(0.5, kw_count * 0.1)

        return min(1.0, word_score + kw_score)

    def decide(self, query: str) -> tuple[str, Endpoint | None]:
        """Decide routing and return (route_name, endpoint)."""
        complexity = self.estimate_complexity(query)
        if complexity > self.threshold:
            return ("cloud", self.cloud_balancer.next())
        return ("edge", self.edge_balancer.next())

    def route(self, query: str) -> Endpoint | None:
        """Return only the endpoint (auto-decide edge vs cloud)."""
        _, ep = self.decide(query)
        return ep
