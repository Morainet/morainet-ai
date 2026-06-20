"""Morainet Agent + FastAPI 嵌入示例。

完整可运行的生产级 API 服务，展示：
- Agent 生命周期管理（启动/关闭）
- 同步/流式两种接口
- 健康检查
- 多 Agent 实例路由
- 会话管理（ShortMemory）
- 请求日志与错误处理

运行：
    pip install fastapi uvicorn morainet-ai[openai]
    MORAINET_OPENAI_API_KEY=sk-xxx python deployment/fastapi_demo.py

测试：
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/agent/run -H "Content-Type: application/json" -d '{"query":"今天天气怎么样？"}'
    curl -N -X POST http://localhost:8000/agent/stream -H "Content-Type: application/json" -d '{"query":"写一首诗"}'
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from morainet import Agent, ShortMemory, tool
from morainet.core.models import Message
from morainet.memory import InMemoryVectorStore, LongMemory
from morainet.providers import OllamaProvider, OpenAIProvider
from morainet.providers.base import Provider


# ===================== 配置 =====================

class ServiceConfig:
    """服务配置，优先读环境变量。"""

    model: str = os.getenv("MORAINET_DEFAULT_MODEL", "gpt-4o")
    max_steps: int = int(os.getenv("MORAINET_MAX_STEPS", "10"))
    request_timeout: float = float(os.getenv("MORAINET_REQUEST_TIMEOUT", "60.0"))
    max_concurrent_llm: int = int(os.getenv("MORAINET_MAX_CONCURRENT_LLM_CALLS", "10"))
    max_concurrent_tools: int = int(os.getenv("MORAINET_MAX_CONCURRENT_TOOL_CALLS", "20"))
    log_level: str = os.getenv("MORAINET_LOG_LEVEL", "INFO")
    ollama_base_url: str = os.getenv("MORAINET_OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def build_provider() -> Provider:
        """根据环境变量自动选择 Provider。"""
        if os.getenv("MORAINET_OPENAI_API_KEY"):
            return OpenAIProvider(model=ServiceConfig.model)
        if os.getenv("MORAINET_OLLAMA_BASE_URL"):
            ollama_model = os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b")
            return OllamaProvider(model=ollama_model)
        # 离线模式：MockProvider
        from morainet.providers import MockProvider
        from morainet.core.models import ChatResponse

        return MockProvider(
            handler=lambda msgs, tools: ChatResponse(
                message=Message.assistant(content="（离线模式）请设置 MORAINET_OPENAI_API_KEY 或 MORAINET_OLLAMA_BASE_URL")
            )
        )


# ===================== 工具 =====================


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，如 "上海"
        unit: 温度单位，celsius 或 fahrenheit
    """
    fake_db = {
        "上海": "晴，26°C",
        "北京": "多云，22°C",
        "深圳": "阵雨，28°C",
        "杭州": "阴，24°C",
        "成都": "小雨，20°C",
    }
    result = fake_db.get(city, f"{city}：暂无数据")
    return result if unit == "celsius" else result


@tool
def current_time(timezone_offset: str = "+8") -> str:
    """获取当前时间。

    Args:
        timezone_offset: 时区偏移，如 +8（北京时间）
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ===================== 应用状态 =====================


class AppState:
    """全局应用状态管理。"""

    def __init__(self):
        self.default_agent: Agent | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self.start_time: datetime | None = None

    async def initialize(self):
        """初始化 Agent 和资源。"""
        self.start_time = datetime.now(timezone.utc)
        self.semaphore = asyncio.Semaphore(ServiceConfig.max_concurrent_llm)

        provider = ServiceConfig.build_provider()
        memory = ShortMemory(max_messages=20)

        self.default_agent = Agent(
            provider=provider,
            tools=[get_weather, current_time],
            memory=memory,
            max_steps=ServiceConfig.max_steps,
            system_prompt="你是智能助手，提供准确简洁的解答。回答用中文。",
        )
        logger.info(f"Agent initialized (model={ServiceConfig.model})")

    async def shutdown(self):
        """清理资源。"""
        logger.info("Shutting down agent service")


state = AppState()


# ===================== FastAPI 应用 =====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # 配置日志
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=ServiceConfig.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    await state.initialize()
    yield
    await state.shutdown()


app = FastAPI(
    title="Morainet Agent Service",
    description="Lightweight AI Agent Runtime API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 请求/响应模型 =====================

class QueryRequest(BaseModel):
    """Agent 调用请求。"""

    query: str = Field(..., description="用户输入", min_length=1, max_length=10000)
    stream: bool = Field(False, description="是否流式输出")
    session_id: str | None = Field(None, description="会话 ID（用于多轮对话）")


class QueryResponse(BaseModel):
    """Agent 调用响应。"""

    answer: str
    steps: int
    tokens: int
    trace_id: str
    model: str = ""


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    version: str
    model: str
    uptime_seconds: float


# ===================== 中间件 =====================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件。"""
    start = datetime.now(timezone.utc)
    response = await call_next(request)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.2f}s)"
    )
    return response


# ===================== API 端点 =====================


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    if not state.default_agent or not state.start_time:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=ServiceConfig.model,
        uptime_seconds=round(uptime, 1),
    )


@app.post("/agent/run", response_model=QueryResponse)
async def run_agent(req: QueryRequest):
    """同步 Agent 调用：返回完整结果。"""
    if not state.default_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async with state.semaphore:
        try:
            async with asyncio.timeout(ServiceConfig.request_timeout):
                result = await state.default_agent.arun(req.query)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Agent call timed out")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result.final_answer,
        steps=len(result.steps),
        tokens=result.usage.total_tokens,
        trace_id=result.trace_id,
        model=ServiceConfig.model,
    )


@app.post("/agent/stream")
async def stream_agent(req: QueryRequest):
    """流式 Agent 调用：SSE 逐 token 输出。"""
    if not state.default_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def generate():
        try:
            async with asyncio.timeout(ServiceConfig.request_timeout):
                async for token in state.default_agent.astream(req.query):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
        except asyncio.TimeoutError:
            yield "data: [ERROR] Request timed out\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/chat")
async def chat_agent(req: QueryRequest):
    """多轮对话接口（带会话记忆）。"""
    if not state.default_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    result = await state.default_agent.arun(req.query)
    return {
        "answer": result.final_answer,
        "trace_id": result.trace_id,
        "tokens": result.usage.total_tokens,
    }


@app.get("/stats")
async def stats():
    """服务统计信息。"""
    if not state.start_time:
        raise HTTPException(503, "Not initialized")

    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    return {
        "uptime_seconds": round(uptime, 1),
        "model": ServiceConfig.model,
        "max_concurrent_llm": ServiceConfig.max_concurrent_llm,
        "max_concurrent_tools": ServiceConfig.max_concurrent_tools,
    }


# ===================== 启动 =====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deployment.fastapi_demo:app" if __file__.endswith("fastapi_demo.py") else app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=ServiceConfig.log_level.lower(),
    )
