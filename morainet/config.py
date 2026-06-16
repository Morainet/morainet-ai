"""Runtime configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MORAINET_", env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    ollama_base_url: str = "http://localhost:11434"

    default_model: str = "gpt-4o"
    max_steps: int = 10
    request_timeout: float = 60.0
    max_retries: int = 3
    log_level: str = "INFO"


settings = Settings()
