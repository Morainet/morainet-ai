"""Runtime configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MORAINET_", env_file=".env", extra="ignore")

    # -- international providers ------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    ollama_base_url: str = "http://localhost:11434"

    # -- Chinese LLM providers --------------------------------------------
    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    wenxin_api_key: str | None = None
    wenxin_secret_key: str | None = None
    wenxin_base_url: str = "https://qianfan.baidubce.com/v2"

    zhipu_api_key: str | None = None
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    moonshot_api_key: str | None = None
    moonshot_base_url: str = "https://api.moonshot.cn/v1"

    minimax_api_key: str | None = None
    minimax_base_url: str = "https://api.minimax.chat/v1"

    siliconflow_api_key: str | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # -- routing ----------------------------------------------------------
    router_default_tier: str = "small"
    router_fallback_enabled: bool = True

    # -- general ----------------------------------------------------------
    default_model: str = "gpt-4o"
    max_steps: int = 10
    request_timeout: float = 60.0
    max_retries: int = 3
    log_level: str = "INFO"

    # -- vector store ----------------------------------------------------
    vector_store_backend: str = "inmemory"  # inmemory | chroma | pgvector | qdrant | faiss | milvus
    vector_store_path: str = ""             # disk path for chroma/faiss
    vector_store_connection: str = ""       # connection string for pgvector/qdrant/milvus
    vector_dimension: int = 1536            # embedding dimension (384 for all-MiniLM, 1536 for OpenAI)

    # -- document parsing ------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_mode: str = "recursive"           # recursive | fixed
    default_document_ttl: float = 0.0       # seconds, 0 = never expire
    knowledge_base_path: str = ""           # catalogue directory for KnowledgeBase

    # -- reasoning -------------------------------------------------------
    compress_after_messages: int = 30       # trigger context compression beyond this many msgs
    max_decomposition_depth: int = 3        # max depth for recursive task decomposition
    self_verify: bool = True                # verify answer before returning it
    tool_cache_ttl: float = 300.0           # seconds, 0 = never expire
    tool_cache_max_size: int = 1000         # max cached entries
    tool_cache_path: str = ""               # disk path for cache persistence
    max_reflect_rounds: int = 3             # Plan-Solve-Reflect max replan cycles

    # -- engineering / production -----------------------------------------
    # Rate limiting
    rate_limit_tokens_per_sec: float = 10.0    # token bucket refill rate
    rate_limit_burst: int = 20                 # max burst capacity
    rate_limit_window_max: int = 100           # sliding window max requests
    rate_limit_window_seconds: float = 60.0    # sliding window duration

    # Concurrency
    max_concurrent_llm_calls: int = 10         # global async semaphore for LLM calls
    max_concurrent_tool_calls: int = 20        # global async semaphore for tool executions

    # Circuit breaker
    circuit_breaker_failures: int = 5          # consecutive failures to OPEN
    circuit_breaker_cooldown: float = 30.0     # seconds in OPEN before HALF_OPEN
    circuit_breaker_half_open_max: int = 1     # max trial calls in HALF_OPEN

    # Billing
    billing_budget_usd: float = 0.0            # 0 = no budget; >0 = enforce cost cap

    # Persistence
    checkpoint_redis_url: str = ""             # e.g. redis://localhost:6379/0
    checkpoint_redis_ttl: int = 0              # seconds, 0 = no expiry
    checkpoint_postgres_dsn: str = ""          # e.g. postgresql://user:pass@localhost/morainet

    # Audit
    audit_log_path: str = ""                   # file path for FileAuditStore
    audit_db_path: str = "morainet_audit.db"   # SQLite path for SQLiteAuditStore

    # Error classification retry
    retry_network_max: int = 3                 # max retries for network errors
    retry_network_base_delay: float = 0.5
    retry_rate_limit_max: int = 5              # max retries for rate limit errors
    retry_rate_limit_base_delay: float = 1.0
    retry_server_max: int = 3                  # max retries for server errors
    retry_server_base_delay: float = 2.0

    # -- MCP connection pool -----------------------------------------------
    mcp_pool_reconnect: bool = True            # auto-reconnect failed MCP servers
    mcp_pool_reconnect_delay: float = 5.0      # seconds between reconnects
    mcp_pool_reconnect_attempts: int = 3       # max reconnect attempts
    mcp_pool_health_interval: float = 30.0     # health check interval in seconds

    # -- MCP resource cache ------------------------------------------------
    mcp_cache_ttl: float = 300.0               # seconds, 0 = no expiry
    mcp_cache_max_size: int = 1000             # max cached entries per category
    mcp_cache_path: str = ""                   # disk path for persistent cache

    # -- Plugin marketplace ------------------------------------------------
    plugin_marketplace_path: str = ""           # local plugins directory path
    plugin_marketplace_index_url: str = ""      # remote registry index URL


settings = Settings()
