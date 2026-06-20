"""Generic OpenAI-compatible provider — one-click access to any private LLM gateway.

Use this provider for:
- Self-hosted vLLM / TGI / LocalAI / text-generation-webui / llama.cpp server
- Any private LLM gateway that speaks the OpenAI Chat Completions API
- Third-party proxies (e.g. OneAPI, FastGPT, LobeChat gateway)

No vendor-specific assumptions — just provide ``base_url``, ``api_key``, and
``model``, and the framework treats it like any other provider.
"""

from __future__ import annotations

from morainet.providers.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Generic OpenAI-compatible provider for private LLM gateways.

    Usage::

        from morainet.providers import OpenAICompatibleProvider

        # Local vLLM server
        provider = OpenAICompatibleProvider(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",  # optional
        )

        # OneAPI / FastGPT proxy
        provider = OpenAICompatibleProvider(
            model="gpt-4",
            base_url="https://your-proxy.com/v1",
            api_key="sk-xxx",
        )

    Unlike specific providers (OpenAIProvider, DeepSeekProvider, etc.), this
    provider does **not** fall back to ``settings.*_api_key`` — you must pass
    all parameters explicitly.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float | None = None,
    ) -> None:
        from morainet.config import settings

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout or settings.request_timeout,
        )
