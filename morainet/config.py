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


settings = Settings()
