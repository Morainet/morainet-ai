# Morainet AI 性能调优指南

> 从开发到生产，提升 Agent 系统的吞吐量、响应速度与资源利用率。

---

## 目录

1. [性能模型概览](#1-性能模型概览)
2. [Provider 优化](#2-provider-优化)
3. [推理策略选择](#3-推理策略选择)
4. [Memory 优化](#4-memory-优化)
5. [Tool 调用优化](#5-tool-调用优化)
6. [Workflow 并行化](#6-workflow-并行化)
7. [MCP 连接池调优](#7-mcp-连接池调优)
8. [并发与限流](#8-并发与限流)
9. [Checkpoint 持久化](#9-checkpoint-持久化)
10. [向量检索优化](#10-向量检索优化)
11. [Streaming 优化](#11-streaming-优化)
12. [生产环境 Checklist](#12-生产环境-checklist)

---

## 1. 性能模型概览

一次 Agent 调用的时间消耗模型：

```
总延迟 = Σ(LLM 调用延迟) + Σ(工具执行延迟) + Σ(Memory 检索延迟) + 框架开销
```

其中：
- **LLM 调用**占主导（通常 60-90%），取决于模型大小、网络 RTT
- **工具执行**取决于工具本身（API 调用、数据库查询、文件 I/O）
- **Memory 检索**在向量检索场景下约 5-50ms（取决于后端）
- **框架开销**通常 < 5ms（纯 Python 协调逻辑）

**关键优化策略：**
1. 减少 LLM 调用次数（选择高效推理策略）
2. 并行化独立操作（工具调用、Workflow 节点）
3. 缓存重复查询（工具结果、MCP 资源、Embedding）
4. 控制上下文大小（滑动窗口、摘要压缩）

---

## 2. Provider 优化

### 2.1 模型选择

| 场景 | 推荐模型 | 延迟参考 |
|------|---------|---------|
| 简单对话 / 路由 | DeepSeek-V3 / Qwen2.5-7B | 200-500ms |
| 工具调用 / 推理 | GPT-4o / Claude Sonnet | 500-1500ms |
| 复杂规划 / 代码审查 | GPT-4o / Claude Opus | 2000-5000ms |
| 离线 / 隐私 / 低成本 | Ollama qwen2.5:3b | 50-200ms（本地） |

### 2.2 连接复用

```python
import httpx

# Morainet 默认使用 httpx 连接池，自动复用 TCP 连接
# 无需额外配置。可通过环境变量调整：

# MORAINET_REQUEST_TIMEOUT=30.0    # 超时秒数（默认 60）
```

### 2.3 重试策略

```python
from morainet.providers import RetryingProvider, RetryPolicy

provider = RetryingProvider(
    wrapped=OpenAIProvider(model="gpt-4o"),
    policy=RetryPolicy(
        max_retries=3,
        base_delay=0.5,     # 指数退避起点
        backoff=2.0,        # 每次翻倍
    ),
)
```

### 2.4 模型路由 — 用小模型省钱

```python
from morainet.providers import ModelRouter

router = ModelRouter(
    tiers={
        "small": DeepSeekProvider(model="deepseek-v3"),      # 便宜、快
        "large": OpenAIProvider(model="gpt-4o"),             # 强、贵
    },
    default="small",
    enable_fallback=True,   # small 失败自动切到 large
)
```

**路由策略建议：**
- 简单问答 → `small`
- 工具调用 → `small`（尝试）→ 失败回退 `large`
- 复杂推理 → 直接 `large`

### 2.5 流式输出

流式输出不会减少总延迟，但能**显著提升用户体验**（首 token 延迟）：

```python
async for token in agent.astream("写一段技术文章"):
    print(token, end="", flush=True)
```

首 token 延迟对比：
- 非流式：等待全部生成 → 2-5s
- 流式：立即开始输出 → 0.2-0.5s

---

## 3. 推理策略选择

### 3.1 策略延迟对比

| 策略 | LLM 调用次数 | 适用场景 | 延迟 |
|------|------------|---------|------|
| `ToolCallingStrategy` | 1 + 工具调用次数 | 明确的任务 | 低 |
| `ReActStrategy` | 每步 1 次 | 探索性任务 | 中-高 |
| `PlanSolveReflect` | 3+ (规划+执行+反思) | 复杂多步任务 | 高 |

### 3.2 场景匹配

```python
# ✅ 场景明确 → 直接用 Workflow DAG（零 LLM 决策开销）
wf = Workflow()
wf.add_node("fetch", fetch_data)
wf.add_node("process", process_data)
wf.connect("fetch", "process")

# ✅ 明确工具调用 → ToolCallingStrategy（1-2 次 LLM 调用）
agent = Agent(provider=..., tools=[...])  # 默认

# ✅ 不确定步骤 → ReActStrategy
agent = Agent(provider=..., strategy=ReActStrategy())

# ✅ 复杂规划 → PlanSolveReflect
agent = Agent(provider=..., strategy=PlanSolveReflect())
```

### 3.3 上下文压缩

长对话场景自动压缩，避免上下文溢出：

```python
from morainet import SummarizingMemory

memory = SummarizingMemory(
    provider=OllamaProvider(model="qwen2.5:3b"),
    summarize_after=20,   # 超过 20 条消息触发压缩
    keep_last=5,          # 保留最近 5 条原文
)

agent = Agent(provider=..., memory=memory)
```

**性能收益：**
- 上下文 token 减少 60-80%
- LLM 调用延迟减少 30-50%
- Token 成本降低 40-70%

---

## 4. Memory 优化

### 4.1 ShortMemory 滑动窗口

```python
# Token 预算裁剪 — 控制上下文大小
memory = ShortMemory(max_tokens=4000)  # 超出后丢弃最旧消息

# 消息数裁剪
memory = ShortMemory(max_messages=20)
```

### 4.2 LongMemory 检索调优

```python
memory = LongMemory(
    store=InMemoryVectorStore(),
    embedder=OllamaEmbedder("nomic-embed-text"),
    score_threshold=0.3,   # 调高 → 更精准但可能漏召
    top_k=3,               # 控制注入上下文的文档数量
)
```

**调参建议：**
| 参数 | 调高效果 | 调低效果 |
|------|---------|---------|
| `score_threshold` | 更精准，可能漏召 | 更全，可能引入噪声 |
| `top_k` | 更多上下文，更慢 | 更快，可能信息不足 |

### 4.3 向量库选择

| 场景 | 推荐后端 | 原因 |
|------|---------|------|
| 开发 / 小数据 (< 1万条) | `InMemoryVectorStore` | 零依赖，毫秒级 |
| 中型数据 (1万-100万) | `ChromaVectorStore` | 本地持久化，简单 |
| 大型数据 (> 100万) | `QdrantVectorStore` / `Milvus` | 高性能索引 |
| 已有 PostgreSQL | `PgVectorStore` | 复用现有基础设施 |

### 4.4 Embedding 缓存

```python
# 对同一批文档只做一次 embedding，后续直接查向量库
# LongMemory 已内置此逻辑（add 时 embedding，search 时只检索）
```

---

## 5. Tool 调用优化

### 5.1 结果缓存

```python
from morainet.reasoning import ToolCache

cache = ToolCache(ttl=300.0, max_size=1000, persist_path="./tool_cache.json")

# 对幂等查询缓存结果，避免重复 API 调用
@tool
def stock_price(symbol: str) -> dict:
    """Get stock price (cached for 5 min)."""
    # cache 自动拦截相同参数调用
    return fetch_from_api(symbol)
```

### 5.2 并发工具调用

Morainet 自动支持模型发起的一次多工具调用（parallel tool calls）并发执行：

```python
# 模型响应含多个 tool_calls 时，框架并发执行
# 无需额外配置
```

### 5.3 工具超时

```python
import asyncio

@tool
async def slow_api(query: str) -> str:
    """Potentially slow API call."""
    async with asyncio.timeout(10.0):  # 10s 超时
        return await external_api(query)
```

---

## 6. Workflow 并行化

### 6.1 识别并行机会

```
        init
       /    \
   check_a  check_b     ← 无依赖，可并行
       \    /
       report
```

```python
wf = Workflow()
wf.add_node("check_a", check_a)
wf.add_node("check_b", check_b)
wf.connect("init", "check_a")
wf.connect("init", "check_b")
wf.connect("check_a", "report")
wf.connect("check_b", "report")
```

### 6.2 并行调度器

```python
from morainet.workflow import ParallelScheduler, ProgressScheduler

# ParallelScheduler — 并发执行同层级节点
s = ParallelScheduler(
    max_workers=4,        # 最大并行数
    timeout=30.0,         # 单节点超时
    max_retries=2,        # 失败重试
)
result = await s.run(wf, {"query": "..."})

# ProgressScheduler — 带进度追踪
s = ProgressScheduler(max_workers=4)
result = await s.run(wf, {"query": "..."})
print(s.progress)  # 每个节点的实时状态
```

**性能收益：**
- 2 个独立节点：延迟减少 ~50%（近 2x 加速）
- N 个独立节点：理论上 N 倍加速（受 max_workers 限制）

---

## 7. MCP 连接池调优

### 7.1 批量连接

```python
from morainet import MCPConnectionPool

pool = MCPConnectionPool()
pool.add_server("search", command="python", args=["-m", "search_mcp"])
pool.add_server("code", command="node", args=["code-server.js"])

await pool.connect_all()  # 并行连接所有服务器
```

### 7.2 资源缓存

```python
from morainet import MCPResourceCache

cache = MCPResourceCache(
    ttl=300,          # 5 分钟缓存
    max_size=1000,
    persist_path=".morainet/mcp_cache.json",  # 磁盘持久化，重启不丢失
)

# 首次查询 → 调用 MCP 服务器
tools = await cache.get_tools("search", lambda: client.list_tools())

# 5 分钟内的后续查询 → 从缓存返回（毫秒级）
tools = await cache.get_tools("search", lambda: client.list_tools())
```

### 7.3 健康检查

```python
pool.reconnect_attempts = 5
pool.reconnect_delay = 3.0

# 后台自动健康检查
await pool.start_health_loop(interval=30)  # 每 30 秒检查一次
```

---

## 8. 并发与限流

### 8.1 并发控制

```python
# 全局配置（环境变量）
MORAINET_MAX_CONCURRENT_LLM_CALLS=10     # LLM 调用并发数
MORAINET_MAX_CONCURRENT_TOOL_CALLS=20    # 工具执行并发数

# 或代码中配置
from morainet.engineering import ConcurrencyLimiter

limiter = ConcurrencyLimiter(
    max_concurrent_llm=10,
    max_concurrent_tools=20,
)

async with limiter.llm():
    response = await provider.chat(...)
```

### 8.2 限流

```python
from morainet.engineering import TokenBucketLimiter

# Token Bucket：均匀速率
limiter = TokenBucketLimiter(tokens_per_sec=10.0, burst=20)

# Sliding Window：时间窗口计数
from morainet.engineering import SlidingWindowLimiter
limiter = SlidingWindowLimiter(max_requests=100, window_seconds=60)
```

### 8.3 熔断器

```python
from morainet.engineering import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,      # 连续 5 次失败 → OPEN
    cooldown_seconds=30.0,    # 30s 冷却
    half_open_max=1,          # 1 次探测调用
)

@cb
async def call_unreliable_service():
    ...
```

---

## 9. Checkpoint 持久化

### 9.1 存储后端选择

| 后端 | 延迟 | 持久化 | 适用场景 |
|------|------|--------|---------|
| `InMemoryCheckpointStore` | ~0ms | ❌ | 开发调试 |
| `FileCheckpointStore` | ~1-5ms | ✅ | 单机生产 |
| `SQLiteCheckpointStore` | ~1-5ms | ✅ | 单机生产 |
| `RedisCheckpointStore` | ~0.5-2ms | ✅ | 分布式 / 高并发 |
| `PostgresCheckpointStore` | ~2-10ms | ✅ | 复用现有 PG |

### 9.2 异步写入

Checkpoint 保存默认是异步的（不阻塞 Agent 执行）。如果 `checkpoint_store` 配置了，每个 step 完成后自动异步写：

```python
store = RedisCheckpointStore(url="redis://localhost:6379/0")
agent = Agent(provider=..., checkpoint_store=store)
# 写入不阻塞，对 Agent 调用延迟几乎无影响
```

---

## 10. 向量检索优化

### 10.1 Embedding 模型选择

| 模型 | 维度 | 速度 | 质量 | 部署 |
|------|------|------|------|------|
| `nomic-embed-text` (Ollama) | 768 | 快 | 好 | 本地 |
| `mxbai-embed-large` (Ollama) | 1024 | 中 | 很好 | 本地 |
| `text-embedding-3-small` (OpenAI) | 1536 | 快 | 很好 | 云端 |
| `text-embedding-3-large` (OpenAI) | 3072 | 慢 | 最好 | 云端 |

**建议：**
- 开发 / 低延迟需求 → `nomic-embed-text`（本地，毫秒级）
- 质量要求高 → `text-embedding-3-small`（云端，~50ms）
- 最高质量 → `text-embedding-3-large`（云端，~100ms）

### 10.2 索引优化

```python
# Faiss：选择索引类型
from morainet.memory import FaissVectorStore

store = FaissVectorStore(
    dimension=768,
    index_type="IVF100,Flat",  # 聚类索引，百万级数据也很快
)

# Qdrant：调整 HNSW 参数
# 默认参数适合大多数场景，大数据量时调整 ef_construct
```

---

## 11. Streaming 优化

### 11.1 首 Token 优化

```python
# 流式输出 + 缩短 system prompt
agent = Agent(
    provider=...,
    system_prompt="你是助手，回答简洁专业。",  # 简短 prompt → 更快首 token
)

async for token in agent.astream("什么是微服务？"):
    print(token, end="", flush=True)  # 逐 token 输出
```

### 11.2 流式工具调用

Morainet 在流式场景下也支持工具调用（增量解析）：

```python
# 流式场景中模型返回 tool_calls 时自动处理
async for token in agent.astream("上海天气怎样？"):
    if isinstance(token, str):
        print(token, end="", flush=True)
```

---

## 12. 生产环境 Checklist

### 部署前检查

- [ ] 选择合适的模型（参考 §2.1）
- [ ] 配置连接池复用（httpx 默认开启）
- [ ] 配置重试策略（`RetryPolicy`）
- [ ] 启用流式输出（`astream`）改善用户体验
- [ ] 为确定性流程使用 Workflow DAG（而非 LLM 推理）
- [ ] 缓存工具结果（幂等查询用 `ToolCache`）
- [ ] 控制上下文大小（`ShortMemory(max_tokens=...)`）
- [ ] 长对话启用 `SummarizingMemory`
- [ ] MCP 工具缓存（`MCPResourceCache`）
- [ ] 向量库按数据规模选择合适的后端
- [ ] 配置并发限制避免打爆下游
- [ ] 配置熔断器保护依赖服务
- [ ] 配置 Checkpoint 持久化（生产必备）
- [ ] 设置 token 预算（`token_budget`）防止成本失控

### 监控指标

| 指标 | 工具 | 告警阈值 |
|------|------|---------|
| Agent 调用延迟 (P99) | Debugger / OTelHook | > 10s |
| LLM 调用延迟 (P99) | Provider log | > 5s |
| Token 消耗速率 | Usage 统计 | > 100k/小时 |
| 工具调用失败率 | Hook | > 5% |
| Checkpoint 写入失败 | 日志 | any |
| MCP 连接断开 | 健康检查 | any |

### 成本优化

```python
# 设置硬性预算
agent = Agent(
    provider=...,
    token_budget=100_000,  # 单次调用最多 100K tokens
)

# 全局成本追踪
from morainet.engineering import BillingTracker

tracker = BillingTracker(budget_usd=50.0)  # 每月 $50 预算
```

---

## 调优速查表

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Agent 响应慢 | LLM 调用次数太多 | 换策略/WF DAG/减少 max_steps |
| 首 Token 慢 | 上下文太大 | 缩短 prompt / SummarizingMemory |
| 工具执行慢 | 外部 API 超时 | 设置 `asyncio.timeout` / 缓存 |
| 内存溢出 | 消息无限累积 | ShortMemory(max_tokens=...) |
| MCP 连接频繁断开 | 进程不稳定 | 连接池 + 自动重连 |
| Token 成本高 | 每次都调大模型 | 模型路由 → 小模型处理简单问题 |
| 向量检索不准 | embedding 质量差 | 换 `nomic-embed-text` 或 `text-embedding-3` |

---

> 更多参考：[API Reference](api-reference.md) · [架构设计](architecture.md) · [部署指南](deployment.md)
