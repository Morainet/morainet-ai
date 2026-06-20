from __future__ import annotations

from morainet.multiagent.factory import AgentBlueprint, AgentFactory
from morainet.multiagent.pool import AgentPool, PoolConfig, PoolStrategy
from morainet.providers import MockProvider


# ---------------------------------------------------------------------------
# PoolConfig
# ---------------------------------------------------------------------------

def test_pool_config_defaults():
    config = PoolConfig()
    assert config.min_size == 2
    assert config.max_size == 10
    assert config.idle_timeout == 600.0
    assert config.strategy == PoolStrategy.LEAST_BUSY
    assert config.prewarm is True


def test_pool_config_custom():
    config = PoolConfig(
        min_size=3,
        max_size=5,
        idle_timeout=300.0,
        strategy=PoolStrategy.ROUND_ROBIN,
        prewarm=False,
    )
    assert config.min_size == 3
    assert config.max_size == 5
    assert config.strategy == PoolStrategy.ROUND_ROBIN
    assert config.prewarm is False


# ---------------------------------------------------------------------------
# PoolStrategy
# ---------------------------------------------------------------------------

def test_pool_strategy_enum_values():
    assert PoolStrategy.ROUND_ROBIN.value == "round_robin"
    assert PoolStrategy.LEAST_BUSY.value == "least_busy"
    assert PoolStrategy.RANDOM.value == "random"
    assert PoolStrategy.FIRST_AVAILABLE.value == "first_available"


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------

class TestAgentPool:
    def setup_method(self):
        self.provider = MockProvider(responses=[])
        self.factory = AgentFactory(provider=self.provider)
        self.factory.register_blueprint(
            "coder",
            AgentBlueprint(role="coder", max_steps=3),
        )

    def test_agent_pool_initialization(self):
        pool = AgentPool(
            factory=self.factory,
            role="coder",
            config=PoolConfig(min_size=1, max_size=3, prewarm=False),
        )
        assert pool.role == "coder"
        assert pool.config.min_size == 1
        assert len(pool) == 0

    def test_agent_pool_default_config(self):
        pool = AgentPool(factory=self.factory, role="coder")
        assert pool.config.min_size == 2
        assert pool.config.max_size == 10
        assert pool.config.strategy == PoolStrategy.LEAST_BUSY

    def test_agent_pool_stats_initial(self):
        pool = AgentPool(
            factory=self.factory,
            role="coder",
            config=PoolConfig(min_size=1, max_size=3, prewarm=False),
        )
        stats = pool.stats
        assert stats["idle"] == 0
        assert stats["busy"] == 0
        assert stats["created"] == 0
        assert stats["destroyed"] == 0
        assert stats["role"] == "coder"

    def test_agent_pool_len_initially_zero(self):
        pool = AgentPool(
            factory=self.factory,
            role="coder",
            config=PoolConfig(min_size=1, max_size=3, prewarm=False),
        )
        assert len(pool) == 0

    def test_agent_pool_different_strategies(self):
        for strategy in PoolStrategy:
            config = PoolConfig(
                min_size=1,
                max_size=3,
                strategy=strategy,
                prewarm=False,
            )
            pool = AgentPool(factory=self.factory, role="coder", config=config)
            assert pool.config.strategy == strategy

    def test_pool_prewarm_disabled(self):
        pool = AgentPool(
            factory=self.factory,
            role="coder",
            config=PoolConfig(min_size=2, max_size=5, prewarm=False),
        )
        assert len(pool) == 0
