# Morainet AI 私有化部署指南

> 从零开始，将 Morainet Agent 部署到你的生产环境。

---

## 目录

1. [部署架构概览](#1-部署架构概览)
2. [Docker 一键部署](#2-docker-一键部署)
3. [Docker Compose 完整栈](#3-docker-compose-完整栈)
4. [Kubernetes 部署](#4-kubernetes-部署)
5. [FastAPI 嵌入示例](#5-fastapi-嵌入示例)
6. [异步服务集成](#6-异步服务集成)
7. [本地 Ollama 模式](#7-本地-ollama-模式)
8. [配置管理](#8-配置管理)
9. [日志与监控](#9-日志与监控)
10. [安全最佳实践](#10-安全最佳实践)
11. [运维手册](#11-运维手册)

---

## 1. 部署架构概览

### 最小部署

```
┌─────────────┐
│    Client   │
└──────┬──────┘
       │ HTTP
┌──────▼──────────────┐
│  FastAPI + Morainet │  ← 单进程
│  ┌────────────────┐ │
│  │  Agent Runtime  │ │
│  │  (Python 3.11+) │ │
│  └────────────────┘ │
└──────┬──────────────┘
       │
┌──────▼──────┐  ┌────────────┐
│  LLM API    │  │  Redis/DB  │  ← 可选
│  (云端/本地)  │  │  (持久化)   │
└─────────────┘  └────────────┘
```

### 生产部署

```
                  ┌──────────────┐
                  │  Nginx/Caddy │  ← 反向代理 + TLS
                  └──────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼─────┐  ┌──────▼─────┐  ┌─────▼────────┐
│ FastAPI #1   │  │ FastAPI #2 │  │ FastAPI #3    │
│ Morainet     │  │ Morainet   │  │ Morainet      │
└──────┬───────┘  └──────┬─────┘  └──────┬────────┘
       │                 │               │
       └─────────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼──────┐
   │   Redis   │  │  PostgreSQL │  │  Qdrant    │
   │ (缓存/CKP) │  │  (业务数据)   │  │  (向量检索)  │
   └───────────┘  └─────────────┘  └────────────┘
```

---

## 2. Docker 一键部署

### 2.1 最小的 Dockerfile

```dockerfile
# deployment/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 安装 Morainet
COPY . .
RUN pip install --no-cache-dir -e ".[openai,redis]"

# 复制应用代码
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 构建与运行

```bash
# 构建镜像
docker build -t morainet-app -f deployment/Dockerfile .

# 运行
docker run -d \
  --name morainet-agent \
  -p 8000:8000 \
  -e MORAINET_OPENAI_API_KEY=sk-xxx \
  -e MORAINET_DEFAULT_MODEL=gpt-4o \
  -e MORAINET_LOG_LEVEL=INFO \
  morainet-app

# 健康检查
curl http://localhost:8000/health
# {"status": "ok"}

# 测试调用
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "今天天气怎么样？"}'
```

---

## 3. Docker Compose 完整栈

```yaml
# deployment/docker-compose.yml
version: "3.8"

services:
  app:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MORAINET_OPENAI_API_KEY=${MORAINET_OPENAI_API_KEY}
      - MORAINET_DEFAULT_MODEL=${MORAINET_DEFAULT_MODEL:-gpt-4o}
      - MORAINET_LOG_LEVEL=${MORAINET_LOG_LEVEL:-INFO}
      - MORAINET_MAX_STEPS=10
      - MORAINET_REQUEST_TIMEOUT=60.0
      - MORAINET_CHECKPOINT_REDIS_URL=redis://redis:6379/0
      - MORAINET_VECTOR_STORE_BACKEND=qdrant
      - MORAINET_VECTOR_STORE_CONNECTION=http://qdrant:6333
      - MORAINET_BILLING_BUDGET_USD=50.0
    depends_on:
      redis:
        condition: service_healthy
      qdrant:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  redis_data:
  qdrant_data:
```

```bash
# 启动完整栈
docker compose -f deployment/docker-compose.yml up -d

# 查看日志
docker compose -f deployment/docker-compose.yml logs -f app

# 扩容 Agent 服务
docker compose -f deployment/docker-compose.yml up -d --scale app=3

# 停止
docker compose -f deployment/docker-compose.yml down
```

---

## 4. Kubernetes 部署

```yaml
# deployment/k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morainet-agent
  labels:
    app: morainet
spec:
  replicas: 3
  selector:
    matchLabels:
      app: morainet
  template:
    metadata:
      labels:
        app: morainet
    spec:
      containers:
      - name: app
        image: your-registry/morainet-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: MORAINET_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: morainet-secrets
              key: openai-api-key
        - name: MORAINET_DEFAULT_MODEL
          value: "gpt-4o"
        - name: MORAINET_LOG_LEVEL
          value: "INFO"
        - name: MORAINET_CHECKPOINT_REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: morainet-service
spec:
  selector:
    app: morainet
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Secret
metadata:
  name: morainet-secrets
type: Opaque
stringData:
  openai-api-key: "sk-your-key-here"
```

```bash
# 部署
kubectl apply -f deployment/k8s-deployment.yaml

# 查看状态
kubectl get pods -l app=morainet
kubectl get svc morainet-service

# 扩容
kubectl scale deployment morainet-agent --replicas=5

# 日志
kubectl logs -f deployment/morainet-agent
```

---

## 5. FastAPI 嵌入示例

完整可运行的 FastAPI 服务示例（见 `deployment/fastapi_demo.py`）：

```python
# deployment/fastapi_demo.py
"""Morainet Agent as a FastAPI service."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from morainet import Agent, tool
from morainet.memory import ShortMemory
from morainet.providers import OpenAIProvider


# ---------- Application State ----------
class AppState:
    def __init__(self):
        self.agent: Agent | None = None

state = AppState()


# ---------- Tools ----------
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，如 "上海"
        unit: 温度单位，celsius 或 fahrenheit
    """
    fake = {"上海": "晴，26°C", "北京": "多云，22°C", "深圳": "阵雨，28°C"}
    return fake.get(city, f"{city}：暂无数据")


# ---------- Lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    provider = OpenAIProvider(model=os.getenv("MORAINET_DEFAULT_MODEL", "gpt-4o"))
    memory = ShortMemory(max_messages=20)
    state.agent = Agent(
        provider=provider,
        tools=[get_weather],
        memory=memory,
        max_steps=10,
        system_prompt="你是智能助手，提供准确简洁的解答。",
    )
    yield
    # Shutdown
    ...


# ---------- API ----------
app = FastAPI(title="Morainet Agent Service", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    steps: int
    tokens: int
    trace_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/run", response_model=QueryResponse)
async def run_agent(req: QueryRequest):
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    result = await state.agent.arun(req.query)
    return QueryResponse(
        answer=result.final_answer,
        steps=len(result.steps),
        tokens=result.usage.total_tokens,
        trace_id=result.trace_id,
    )


@app.post("/agent/stream")
async def stream_agent(req: QueryRequest):
    from fastapi.responses import StreamingResponse

    if not state.agent:
        raise HTTPException(503, "Agent not initialized")

    async def generate():
        async for token in state.agent.astream(req.query):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**运行：**

```bash
# 安装依赖
pip install fastapi uvicorn morainet-ai[openai]

# 启动服务
MORAINET_OPENAI_API_KEY=sk-xxx python deployment/fastapi_demo.py

# 测试
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "上海今天天气怎么样？"}'
```

---

## 6. 异步服务集成

### 6.1 Celery 任务队列

```python
# deployment/celery_worker.py
from celery import Celery

from morainet import Agent, tool
from morainet.providers import OpenAIProvider

celery = Celery("morainet", broker="redis://localhost:6379/0")

agent = Agent(
    provider=OpenAIProvider(model="gpt-4o"),
    tools=[...],
    memory=ShortMemory(),
)

@celery.task
def run_agent_task(query: str) -> dict:
    """Celery task wrapper (uses asyncio.run internally)."""
    result = agent.run(query)
    return {
        "answer": result.final_answer,
        "steps": len(result.steps),
        "tokens": result.usage.total_tokens,
    }

# Usage:
# run_agent_task.delay("What is the weather?")
```

### 6.2 aiohttp 服务

```python
# deployment/aiohttp_demo.py
from aiohttp import web

from morainet import Agent
from morainet.providers import OpenAIProvider

agent = Agent(provider=OpenAIProvider(model="gpt-4o"), tools=[...])

async def handle_agent(request: web.Request) -> web.Response:
    data = await request.json()
    result = await agent.arun(data["query"])
    return web.json_response({
        "answer": result.final_answer,
        "trace_id": result.trace_id,
    })

app = web.Application()
app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
app.router.add_post("/agent/run", handle_agent)

if __name__ == "__main__":
    web.run_app(app, port=8000)
```

### 6.3 gRPC 集成

```protobuf
// deployment/protos/agent.proto
service AgentService {
    rpc Run (QueryRequest) returns (QueryResponse);
}

message QueryRequest { string query = 1; }
message QueryResponse {
    string answer = 1;
    int32 steps = 2;
    int32 tokens = 3;
    string trace_id = 4;
}
```

---

## 7. 本地 Ollama 模式

完全离线、零成本的部署方式（适合内网 / 合规场景）：

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# 2. 拉取模型
ollama pull qwen2.5:14b          # 工具调用用
ollama pull nomic-embed-text     # embedding 用

# 3. 部署 Morainet（无需任何 API key）
MORAINET_OLLAMA_BASE_URL=http://localhost:11434 python deployment/fastapi_demo.py
```

```python
# 代码中切换
from morainet import Agent
from morainet.providers import OllamaProvider

agent = Agent(
    provider=OllamaProvider(model="qwen2.5:14b"),
    tools=[...],
)
result = agent.run("今天天气怎么样？")
```

---

## 8. 配置管理

### 8.1 环境变量（推荐生产）

```bash
# .env.production
MORAINET_OPENAI_API_KEY=sk-prod-xxx
MORAINET_ANTHROPIC_API_KEY=sk-ant-xxx
MORAINET_DEFAULT_MODEL=gpt-4o
MORAINET_MAX_STEPS=10
MORAINET_REQUEST_TIMEOUT=30.0
MORAINET_LOG_LEVEL=INFO

# 持久化
MORAINET_CHECKPOINT_REDIS_URL=redis://redis.internal:6379/0

# 向量库
MORAINET_VECTOR_STORE_BACKEND=qdrant
MORAINET_VECTOR_STORE_CONNECTION=http://qdrant.internal:6333

# 工程化
MORAINET_MAX_CONCURRENT_LLM_CALLS=10
MORAINET_BILLING_BUDGET_USD=100.0
MORAINET_CIRCUIT_BREAKER_FAILURES=3
```

### 8.2 .env 文件（开发）

```bash
# 开发环境
MORAINET_LOG_LEVEL=DEBUG
MORAINET_DEFAULT_MODEL=gpt-4o-mini
MORAINET_MAX_STEPS=20

# 本地 Ollama
MORAINET_OLLAMA_BASE_URL=http://localhost:11434
```

### 8.3 代码中覆盖

```python
from morainet.config import Settings

settings = Settings(
    openai_api_key=os.getenv("OPENAI_KEY"),
    max_steps=15,
    log_level="DEBUG",
)
```

---

## 9. 日志与监控

### 9.1 结构化日志

```python
# Morainet 使用 loguru，开箱即用
import os
os.environ["MORAINET_LOG_LEVEL"] = "INFO"

# 或在代码中
from loguru import logger

logger.add(
    "logs/morainet_{time}.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
)
```

### 9.2 OpenTelemetry 集成

```python
from morainet.observability import OTelHook

agent = Agent(
    provider=...,
    hooks=[OTelHook(service_name="morainet-agent")],
)
# 自动导出 trace 到 OTLP collector
```

### 9.3 自定义 Hook 监控

```python
from morainet.observability import Hook
from datetime import datetime, timezone

class MetricsHook(Hook):
    def __init__(self):
        self.total_calls = 0
        self.total_tokens = 0

    def on_run_end(self, context, result):
        self.total_calls += 1
        self.total_tokens += result.usage.total_tokens
        latency = (datetime.now(timezone.utc) - context.started_at).total_seconds()
        print(f"[METRICS] calls={self.total_calls} tokens={self.total_tokens} latency={latency:.2f}s")
```

---

## 10. 安全最佳实践

### 10.1 API Key 管理

```bash
# ❌ 不要硬编码
agent = Agent(provider=OpenAIProvider(api_key="sk-xxx"))

# ✅ 环境变量
agent = Agent(provider=OpenAIProvider())  # 自动读取 MORAINET_OPENAI_API_KEY

# ✅ Secret Manager (AWS/GCP/Vault)
key = get_secret("openai-api-key")
agent = Agent(provider=OpenAIProvider(api_key=key))
```

### 10.2 工具安全

```python
# 标记危险工具
@tool(dangerous=True)
def delete_database(name: str) -> str: ...

# 人工审批
def approve(name: str, args: dict) -> bool:
    # 发送审批请求到企业 IM / 工单系统
    return send_approval_request(name, args)

agent = Agent(provider=..., tools=[delete_database], approve_tool=approve)
```

### 10.3 输入校验

```python
from fastapi import Request
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    stream: bool = False

@app.post("/agent/run")
async def run_agent(request: Request, req: QueryRequest):
    # 限流
    await rate_limiter.acquire(request.client.host)
    ...
```

---

## 11. 运维手册

### 11.1 健康检查

```python
@app.get("/health")
async def health():
    checks = {
        "agent": state.agent is not None,
        "provider": state.agent.provider is not None if state.agent else False,
    }
    status = all(checks.values())
    return {"status": "ok" if status else "degraded", "checks": checks}
```

### 11.2 优雅关闭

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.agent = Agent(...)
    yield
    # Shutdown — 清理资源
    if state.agent and state.agent.checkpoint_store:
        await state.agent.checkpoint_store.close()
```

### 11.3 故障恢复

```python
# 基于 Checkpoint 的自动恢复
from morainet import FileCheckpointStore

store = FileCheckpointStore("./.checkpoints")
agent = Agent(provider=..., checkpoint_store=store)

# 崩溃后恢复
try:
    result = await agent.arun("长时间任务...")
except Exception:
    # 从最近 checkpoint 恢复
    cp = await store.load_latest()
    if cp:
        result = await agent.resume(cp)
```

### 11.4 资源限制

```yaml
# Docker Compose 中限制资源
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M

# K8s 中配置
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

---

## 快速开始检查清单

- [ ] 选择部署方式（Docker / Compose / K8s）
- [ ] 配置 LLM Provider API Key（或本地 Ollama）
- [ ] 配置 Checkpoint 持久化后端
- [ ] 配置向量库（如需 RAG）
- [ ] 设置日志级别和轮转
- [ ] 配置健康检查端点
- [ ] 设置资源限制（CPU/内存）
- [ ] 配置反向代理 + TLS（生产）
- [ ] 设置监控告警
- [ ] 编写运维 Runbook

---

> 更多参考：[API Reference](api-reference.md) · [性能调优指南](performance-tuning.md) · [架构设计](architecture.md)
