"""Morainet Agent 异步服务集成 Demo。

展示 Morainet 与常见异步框架的集成方式：

1. aiohttp 服务
2. 后台任务队列（asyncio.Queue）
3. 并发 Agent 调用（信号量控制）
4. WebSocket 实时流式对话
5. Celery 任务队列集成模式

运行：
    pip install aiohttp websockets morainet-ai[openai]
    MORAINET_OPENAI_API_KEY=sk-xxx python deployment/async_service_demo.py

测试 WebSocket：
    wscat -c ws://localhost:8001/ws/agent
    > {"query": "今天天气怎么样？"}
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import datetime, timezone

from loguru import logger

from morainet import Agent, ShortMemory, tool
from morainet.core.models import Message
from morainet.providers import OllamaProvider, OpenAIProvider
from morainet.providers.base import Provider


# ===================== 工具 =====================


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，如 "上海"
        unit: 温度单位，celsius 或 fahrenheit
    """
    return f"{city}：晴，26°C"


def build_provider() -> Provider:
    """自动选择 Provider。"""
    if os.getenv("MORAINET_OPENAI_API_KEY"):
        return OpenAIProvider(model=os.getenv("MORAINET_DEFAULT_MODEL", "gpt-4o"))
    if os.getenv("MORAINET_OLLAMA_BASE_URL"):
        return OllamaProvider(model=os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b"))
    from morainet.providers import MockProvider

    return MockProvider(
        handler=lambda m, t: __import__("morainet.core.models", fromlist=["ChatResponse", "Message"])
        .ChatResponse(message=Message.assistant(content="（Mock 模式）"))
    )


def build_agent() -> Agent:
    """构建 Agent 实例。"""
    return Agent(
        provider=build_provider(),
        tools=[get_weather],
        memory=ShortMemory(max_messages=20),
        max_steps=10,
        system_prompt="你是智能助手，回答简洁专业。",
    )


# ===================== 1. aiohttp 服务 =====================


async def aiohttp_demo():
    """aiohttp 集成示例。"""
    try:
        from aiohttp import web
    except ImportError:
        logger.warning("aiohttp 未安装，跳过")
        return

    agent = build_agent()

    async def handle_agent(request: web.Request) -> web.Response:
        data = await request.json()
        query = data.get("query", "")
        if not query:
            return web.json_response({"error": "query required"}, status=400)

        result = await agent.arun(query)
        return web.json_response(
            {
                "answer": result.final_answer,
                "steps": len(result.steps),
                "tokens": result.usage.total_tokens,
                "trace_id": result.trace_id,
            }
        )

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_post("/agent/run", handle_agent)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8001)
    await site.start()
    logger.info("aiohttp service: http://0.0.0.0:8001")
    logger.info("  POST /agent/run  —  Agent call")
    logger.info("  GET  /health     —  Health check")


# ===================== 2. 任务队列模式 =====================


class AgentJob:
    """异步任务描述。"""

    def __init__(self, query: str):
        self.id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self.query = query
        self.status = "pending"
        self.result: str | None = None
        self.error: str | None = None
        self.future: asyncio.Future = asyncio.Future()


class AgentWorkerPool:
    """Agent 任务队列工作池。"""

    def __init__(self, num_workers: int = 3, max_concurrent: int = 5):
        self.queue: asyncio.Queue[AgentJob] = asyncio.Queue()
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.agent = build_agent()
        self.jobs: dict[str, AgentJob] = {}

    async def submit(self, query: str) -> AgentJob:
        """提交一个 Agent 任务。"""
        job = AgentJob(query)
        self.jobs[job.id] = job
        await self.queue.put(job)
        return job

    async def get_result(self, job_id: str) -> str | None:
        """获取任务结果。"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return await job.future

    async def worker(self, worker_id: int):
        """工作协程。"""
        while True:
            job = await self.queue.get()
            try:
                job.status = "running"
                async with self.semaphore:
                    result = await self.agent.arun(job.query)
                job.result = result.final_answer
                job.status = "done"
                job.future.set_result(result.final_answer)
            except Exception as e:
                job.error = str(e)
                job.status = "failed"
                job.future.set_exception(e)
            finally:
                self.queue.task_done()

    async def start(self):
        """启动工作池。"""
        for i in range(self.num_workers):
            asyncio.create_task(self.worker(i), name=f"worker-{i}")
        logger.info(f"Worker pool started ({self.num_workers} workers, {self.semaphore._value} concurrent)")


async def task_queue_demo():
    """任务队列模式演示。"""
    pool = AgentWorkerPool(num_workers=2, max_concurrent=3)
    await pool.start()

    # 批量提交任务
    queries = ["今天天气怎么样？", "上海适合穿什么？", "北京的温度是多少？"]
    jobs = [await pool.submit(q) for q in queries]
    logger.info(f"Submitted {len(jobs)} jobs")

    # 等待所有完成
    results = await asyncio.gather(*[job.future for job in jobs], return_exceptions=True)
    for i, r in enumerate(results):
        status = "✓" if not isinstance(r, Exception) else "✗"
        logger.info(f"  [{status}] {jobs[i].query[:20]}... → {str(r)[:40]}...")


# ===================== 3. WebSocket 流式对话 =====================


async def handle_websocket(websocket, agent: Agent):
    """WebSocket 连接处理。"""
    async for raw in websocket:
        try:
            data = json.loads(raw)
            query = data.get("query", "")
            if not query:
                await websocket.send(json.dumps({"error": "query required"}))
                continue

            # 流式输出
            async for token in agent.astream(query):
                await websocket.send(json.dumps({"token": token}))
            await websocket.send(json.dumps({"done": True}))

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))


async def websocket_demo():
    """WebSocket 实时流式对话。"""
    agent = build_agent()

    async def handler(websocket):
        await handle_websocket(websocket, agent)

    try:
        import websockets
    except ImportError:
        logger.warning("websockets 未安装，跳过 WebSocket demo")
        return

    server = await websockets.serve(handler, "0.0.0.0", 8002)
    logger.info("WebSocket service: ws://0.0.0.0:8002/ws/agent")


# ===================== 4. 并发压力测试 =====================


async def concurrency_demo():
    """并发 Agent 调用演示。"""
    agent = build_agent()
    sem = asyncio.Semaphore(5)

    async def call(query: str) -> tuple[str, float]:
        start = datetime.now(timezone.utc)
        async with sem:
            result = await agent.arun(query)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        return result.final_answer[:50], elapsed

    queries = [
        "今天天气怎么样？",
        "上海的温度是多少？",
        "北京冷不冷？",
        "深圳适合穿什么衣服？",
        "明天会下雨吗？",
    ]

    logger.info(f"并发测试：{len(queries)} 个请求，最大并发 5")
    start = datetime.now(timezone.utc)
    results = await asyncio.gather(*[call(q) for q in queries])
    total = (datetime.now(timezone.utc) - start).total_seconds()

    for i, (answer, elapsed) in enumerate(results):
        logger.info(f"  [{i+1}] {elapsed:.2f}s → {answer}...")
    logger.info(f"  总耗时: {total:.2f}s（串行理论耗时: {sum(r[1] for r in results):.2f}s）")


# ===================== 主流程 =====================


async def main():
    logger.info("=" * 60)
    logger.info("Morainet 异步服务集成 Demo")
    logger.info("=" * 60)

    # 1. 启动 aiohttp 服务（后台）
    aiohttp_task = asyncio.create_task(aiohttp_demo())
    await asyncio.sleep(1)  # 等待服务启动

    # 2. 启动 WebSocket 服务（后台）
    ws_task = asyncio.create_task(websocket_demo())
    await asyncio.sleep(0.5)

    # 3. 任务队列模式演示
    logger.info("\n--- 任务队列模式 ---")
    await task_queue_demo()

    # 4. 并发调用演示
    logger.info("\n--- 并发调用演示 ---")
    await concurrency_demo()

    # 保持运行，等待 Ctrl+C
    logger.info("\n服务已启动。按 Ctrl+C 停止。")
    logger.info("  aiohttp:     http://localhost:8001/agent/run")
    logger.info("  WebSocket:   ws://localhost:8002/ws/agent")
    logger.info("  Celery:      见代码注释")

    stop = asyncio.Event()

    def shutdown():
        logger.info("Shutting down...")
        stop.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    await stop.wait()

    # 清理
    aiohttp_task.cancel()
    ws_task.cancel()
    logger.info("Done.")


# ===================== Celery 集成模式（代码示例） =====================

"""
# deployment/celery_worker.py
# 安装：pip install celery redis
# 启动：celery -A celery_worker worker --loglevel=info

from celery import Celery
from morainet import Agent
from morainet.providers import OpenAIProvider

celery = Celery("morainet", broker="redis://localhost:6379/0")

agent = Agent(
    provider=OpenAIProvider(model="gpt-4o"),
    tools=[...],
    memory=ShortMemory(),
)

@celery.task(bind=True, max_retries=3)
def run_agent_task(self, query: str) -> dict:
    try:
        result = agent.run(query)
        return {
            "answer": result.final_answer,
            "steps": len(result.steps),
            "tokens": result.usage.total_tokens,
        }
    except Exception as e:
        self.retry(exc=e, countdown=60)

# 调用：
# from celery_worker import run_agent_task
# task = run_agent_task.delay("今天天气怎么样？")
# result = task.get(timeout=60)
"""


if __name__ == "__main__":
    asyncio.run(main())
