"""Ollama provider (native /api/chat) over httpx — for local models.

No API key required. Tool calling is supported by recent Ollama versions.
Ollama returns tool-call arguments as an object and assigns no call id, so we
synthesize ids.

Enhanced with:
- Batch inference: send multiple prompts concurrently, aggregate results.
- Quantization config: set num_ctx, num_gpu, num_thread, etc. via ``options``.
- Multi-model scheduling: semaphore-based concurrency control across models.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from morainet.config import settings
from morainet.core.models import ChatResponse, Message, Role, ToolCall, Usage
from morainet.exceptions import ProviderError, ProviderTimeoutError
from morainet.observability.tracing import logger
from morainet.providers._streaming import parse_ollama_ndjson_line
from morainet.providers.base import Provider


# --- model options (quantization / inference tuning) ----------------------


@dataclass
class OllamaOptions:
    """Ollama model inference options (passed as ``options`` in the request).

    These control quantization, context length, GPU usage, and more.
    See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md

    Usage::

        provider = OllamaProvider(
            model="qwen2.5:7b",
            options=OllamaOptions(
                num_ctx=4096,
                num_gpu=1,
                num_thread=8,
                temperature=0.7,
            ),
        )
    """

    # Context / memory
    num_ctx: int | None = None           # context window size (default: 2048)
    num_predict: int | None = None       # max tokens to generate (-1 = infinite)

    # GPU / quantization
    num_gpu: int | None = None           # number of GPU layers (-1 = all)
    num_thread: int | None = None        # number of CPU threads
    f16_kv: bool | None = None           # use fp16 for KV cache (saves VRAM)
    low_vram: bool | None = None         # low VRAM mode
    use_mmap: bool | None = None         # memory-mapped model loading
    use_mlock: bool | None = None        # lock model in RAM

    # Sampling
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None

    # Misc
    numa: bool | None = None             # NUMA optimization
    vocab_only: bool | None = None       # load only vocabulary

    def to_dict(self) -> dict[str, Any]:
        """Serialize non-None fields to a dict for the Ollama API."""
        result: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is not None:
                result[k] = v
        return result


# --- Ollama scheduler (concurrent multi-model access) ---------------------


class OllamaScheduler:
    """Concurrency limiter for a single Ollama instance running multiple models.

    Ollama has limited GPU VRAM — running too many models simultaneously can
    cause OOM. This semaphore-based scheduler prevents that by limiting how many
    concurrent ``chat``/``stream`` calls can be in flight at once.

    Usage::

        scheduler = OllamaScheduler(max_concurrency=2)

        provider_a = OllamaProvider("qwen2.5:7b", scheduler=scheduler)
        provider_b = OllamaProvider("llama3.1:8b", scheduler=scheduler)

        # Both share the same semaphore; at most 2 calls run at once.
        async with asyncio.TaskGroup() as tg:
            tg.create_task(provider_a.chat(messages1))
            tg.create_task(provider_b.chat(messages2))
            tg.create_task(provider_a.chat(messages3))  # waits

    Without a scheduler, each ``OllamaProvider`` works independently (no limit).
    """

    def __init__(self, max_concurrency: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    def __enter__(self) -> "OllamaScheduler":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# --- serialization --------------------------------------------------------


def to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role.value, "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in m.tool_calls
            ]
        converted.append(msg)
    return converted


def parse_response(data: dict[str, Any], model: str) -> ChatResponse:
    raw = data.get("message", {})
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(raw.get("tool_calls") or []):
        fn = tc.get("function", {})
        tool_calls.append(
            ToolCall(
                id=f"call_{i}", name=fn.get("name", ""), arguments=fn.get("arguments", {})
            )
        )

    prompt = data.get("prompt_eval_count", 0)
    completion = data.get("eval_count", 0)
    return ChatResponse(
        message=Message(
            role=Role.ASSISTANT,
            content=raw.get("content") or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
        model=data.get("model", model),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


# --- provider -------------------------------------------------------------


class OllamaProvider(Provider):
    """Ollama local LLM provider with quantization control and scheduling.

    Args:
        model: Ollama model name (e.g. ``"qwen2.5:7b"``, ``"llama3.1:8b"``).
        base_url: Ollama server URL (default: ``http://localhost:11434``).
        timeout: HTTP request timeout in seconds.
        options: Inference/quantization options (context length, GPU layers, etc.).
        scheduler: Optional concurrency limiter for multi-model scenarios.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str | None = None,
        timeout: float | None = None,
        options: OllamaOptions | None = None,
        scheduler: OllamaScheduler | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        self.options = options or OllamaOptions()
        self.scheduler = scheduler

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama(messages),
            "stream": stream,
            "options": self.options.to_dict(),
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": s} for s in tools]
        if response_format:
            payload["format"] = response_format
        return payload

    async def _acquire_slot(self) -> None:
        if self.scheduler is not None:
            await self.scheduler.acquire()

    def _release_slot(self) -> None:
        if self.scheduler is not None:
            self.scheduler.release()

    # --- single chat / stream ---------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        payload = self._build_payload(messages, tools, response_format, stream=False)
        try:
            await self._acquire_slot()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            finally:
                self._release_slot()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        return parse_response(resp.json(), self.model)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, tools, response_format, stream=True)
        try:
            await self._acquire_slot()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}/api/chat", json=payload
                    ) as resp:
                        if resp.status_code >= 400:
                            body = (await resp.aread()).decode("utf-8", "replace")
                            raise ProviderError(f"{resp.status_code}: {body}")
                        async for line in resp.aiter_lines():
                            delta = parse_ollama_ndjson_line(line)
                            if delta:
                                yield delta
            finally:
                self._release_slot()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

    # --- batch inference --------------------------------------------------

    async def batch_chat(
        self,
        messages_list: list[list[Message]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_concurrency: int = 4,
    ) -> list[ChatResponse]:
        """Run multiple chat requests concurrently and return all results.

        Order is preserved: ``results[i]`` corresponds to ``messages_list[i]``.

        Args:
            messages_list: One message list per request.
            tools: Shared tool definitions for all requests.
            response_format: Shared response format for all requests.
            max_concurrency: Max simultaneous requests (semaphore-controlled).

        Returns:
            List of ChatResponse, one per input.
        """
        if not messages_list:
            return []

        sem = asyncio.Semaphore(max_concurrency)
        results: dict[int, ChatResponse | Exception] = {}

        async def _worker(idx: int, msgs: list[Message]) -> None:
            async with sem:
                try:
                    results[idx] = await self.chat(msgs, tools, response_format)
                except Exception as exc:
                    logger.warning(f"[ollama batch] request {idx} failed: {exc}")
                    results[idx] = exc

        tasks = [_worker(i, msgs) for i, msgs in enumerate(messages_list)]
        await asyncio.gather(*tasks)

        ordered: list[ChatResponse] = []
        for i in range(len(messages_list)):
            r = results.get(i)
            if isinstance(r, ChatResponse):
                ordered.append(r)
            elif isinstance(r, Exception):
                raise RuntimeError(
                    f"Batch request {i} failed: {r}"
                ) from r
            else:
                raise RuntimeError(f"Batch request {i} returned no result.")
        return ordered

    async def batch_generate(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_concurrency: int = 4,
    ) -> list[ChatResponse]:
        """Convenience: batch chat with plain-text prompts.

        Args:
            prompts: Text prompts (one per request).
            system_prompt: Optional shared system prompt.
            tools: Shared tool definitions.
            response_format: Shared response format.
            max_concurrency: Max simultaneous requests.

        Returns:
            List of ChatResponse, one per prompt.
        """
        messages_list: list[list[Message]] = []
        for p in prompts:
            msgs: list[Message] = []
            if system_prompt:
                msgs.append(Message.system(system_prompt))
            msgs.append(Message.user(p))
            messages_list.append(msgs)
        return await self.batch_chat(
            messages_list, tools, response_format, max_concurrency
        )


# --- multi-model concurrent scheduling helper -----------------------------


@dataclass
class MultiOllamaResult:
    """Result from a multi-model Ollama query."""
    model: str
    response: ChatResponse | None = None
    error: str | None = None


async def multi_ollama_query(
    providers: dict[str, OllamaProvider],
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    return_all: bool = False,
) -> MultiOllamaResult | list[MultiOllamaResult]:
    """Query multiple Ollama models concurrently.

    Useful for comparing model outputs or running ensemble voting.

    Args:
        providers: Dict of label → OllamaProvider to query.
        messages: Shared messages for all models.
        tools: Shared tools.
        response_format: Shared response format.
        return_all: If True, return all results; if False, return the first
            successful one.

    Returns:
        Single MultiOllamaResult if ``return_all=False``, list otherwise.
    """
    async def _query(name: str, p: OllamaProvider) -> MultiOllamaResult:
        try:
            resp = await p.chat(messages, tools, response_format)
            return MultiOllamaResult(model=name, response=resp)
        except Exception as exc:
            return MultiOllamaResult(model=name, error=str(exc))

    coros = [_query(name, p) for name, p in providers.items()]

    if return_all:
        results = await asyncio.gather(*coros)
        return list(results)

    # Race: return first successful result
    tasks = [asyncio.ensure_future(c) for c in coros]
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            if result.response is not None:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return result
        except Exception as exc:
            logger.debug(f"[multi-ollama] one model failed: {exc}")

    raise RuntimeError("All Ollama models failed in multi-model query.")
