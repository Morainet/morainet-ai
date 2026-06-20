# Morainet AI API Reference

> Complete API documentation for Morainet AI v1.0

---

## Table of Contents

1. [Agent Core](#1-agent-core)
2. [Providers](#2-providers)
3. [Tools](#3-tools)
4. [Memory](#4-memory)
5. [Reasoning Strategies](#5-reasoning-strategies)
6. [Workflow Engine](#6-workflow-engine)
7. [Multi-Agent Orchestration](#7-multi-agent-orchestration)
8. [Persistence & Checkpoint](#8-persistence--checkpoint)
9. [Observability & Hooks](#9-observability--hooks)
10. [MCP Integration](#10-mcp-integration)
11. [Plugin System](#11-plugin-system)
12. [Engineering Features](#12-engineering-features)
13. [Configuration](#13-configuration)
14. [Exceptions](#14-exceptions)

---

## 1. Agent Core

### `Agent`

The central runtime that orchestrates reasoning, memory, tool calling, and provider interaction.

```python
from morainet import Agent

agent = Agent(
    provider: Provider,                      # Required: LLM provider
    tools: list[Tool] | None = None,         # Tools the agent can use
    memory: Memory | None = None,            # Short/Long term memory
    max_steps: int = 10,                     # Max reasoning iterations
    system_prompt: str | None = None,        # System-level instruction
    strategy: ReasoningStrategy | None = None,  # Reasoning strategy
    hooks: list[Hook] | None = None,         # Observability hooks
    checkpoint_store: CheckpointStore | None = None,  # Persistence
    token_budget: int | None = None,         # Cumulative token limit
    max_consecutive_errors: int = 0,         # Auto-abort threshold
    retry: RetryPolicy | None = None,        # Retry configuration
    approve_tool: Callable | None = None,    # Dangerous tool approval
    debug: bool = False,                     # Verbose logging
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(query: str) -> AgentResult` | Synchronous execution |
| `arun` | `(query: str) -> AgentResult` | Async execution |
| `astream` | `(query: str) -> AsyncIterator[str]` | Streaming token output |
| `resume` | `(checkpoint: Checkpoint) -> AgentResult` | Resume from checkpoint |

**`AgentResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | Final output text |
| `steps` | `list[Step]` | Execution trace |
| `usage` | `Usage` | Token consumption |
| `trace_id` | `str` | Unique trace identifier |

---

## 2. Providers

### Abstract Base

```python
from morainet.providers.base import Provider

class Provider(ABC):
    async def chat(self, messages: list[Message], tools: list[dict] | None) -> ChatResponse: ...
    async def stream(self, messages: list[Message], tools: list[dict] | None) -> AsyncIterator[str]: ...
```

### Built-in Providers

| Provider | Import | Model Example | Install Extra |
|----------|--------|---------------|---------------|
| OpenAI | `from morainet.providers import OpenAIProvider` | `"gpt-4o"` | `[openai]` |
| Anthropic Claude | `from morainet.providers import ClaudeProvider` | `"claude-sonnet-4-6"` | `[anthropic]` |
| Google Gemini | `from morainet.providers import GeminiProvider` | `"gemini-2.5-flash"` | `[gemini]` |
| Ollama (local) | `from morainet.providers import OllamaProvider` | `"qwen2.5:3b"` | built-in |
| DeepSeek | `from morainet.providers import DeepSeekProvider` | `"deepseek-v3"` | built-in |
| Qwen (Tongyi) | `from morainet.providers import QwenProvider` | `"qwen-max"` | built-in |
| Wenxin (Baidu) | `from morainet.providers import WenxinProvider` | `"ernie-4.0"` | built-in |
| Zhipu | `from morainet.providers import ZhipuProvider` | `"glm-4"` | built-in |
| Moonshot | `from morainet.providers import MoonshotProvider` | `"moonshot-v1-8k"` | built-in |
| MiniMax | `from morainet.providers import MiniMaxProvider` | `"abab6.5s"` | built-in |
| SiliconFlow | `from morainet.providers import SiliconFlowProvider` | `"Qwen/Qwen2.5-7B"` | built-in |
| Mock | `from morainet.providers import MockProvider` | N/A (offline) | built-in |

### Provider Wrappers

**`RetryingProvider`** — wraps any provider with retry logic:
```python
from morainet.providers import RetryingProvider, RetryPolicy

provider = RetryingProvider(
    wrapped=OpenAIProvider(model="gpt-4o"),
    policy=RetryPolicy(max_retries=3, base_delay=1.0, backoff=2.0),
)
```

**`ModelRouter`** — tiered routing with fallback:
```python
from morainet.providers import ModelRouter

router = ModelRouter(
    tiers={
        "small": DeepSeekProvider(),
        "large": OpenAIProvider(model="gpt-4o"),
    },
    default="small",
    enable_fallback=True,
)
```

### Usage

```python
from morainet.core.models import Usage, Message, ChatResponse
# Usage: prompt_tokens, completion_tokens, total_tokens
# Message: role (system/user/assistant/tool), content, tool_calls, tool_call_id
# ChatResponse: message, usage, model, finish_reason
```

---

## 3. Tools

### `@tool` Decorator

```python
from morainet import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Query weather for a city.

    Args:
        city: City name, e.g. "Shanghai"
        unit: Temperature unit, celsius or fahrenheit
    """
    return f"{city}: sunny, 26°C"

@tool(dangerous=True)  # Requires approval callback
def delete_file(path: str) -> str:
    """Delete a file permanently."""
    ...

@tool
async def search_api(query: str) -> dict:  # Async tool
    """Search external API."""
    ...
```

### `ToolRegistry`

```python
from morainet.tools import ToolRegistry

registry = ToolRegistry()
registry.register(web_search)
registry.register_many([get_weather, delete_file])
tool = registry.get("get_weather")
```

### Tool Schema

Auto-generated from function signature + type hints + docstring:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Query weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name, e.g. \"Shanghai\""},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
      },
      "required": ["city"]
    }
  }
}
```

---

## 4. Memory

### Abstract Interface

```python
from morainet.memory.base import Memory

class Memory(ABC):
    async def add(self, message: Message) -> None: ...
    async def get_context(self, query: str, limit: int = 10) -> list[Message]: ...
```

### Built-in Implementations

| Class | Use Case | Key Parameters |
|-------|----------|---------------|
| `ShortMemory` | Multi-turn conversation | `max_messages` (sliding window), `max_tokens` (token budget trim) |
| `LongMemory` | Knowledge base / RAG | `store` (VectorStore), `embedder`, `score_threshold`, `top_k` |
| `SummarizingMemory` | Long conversation summary | `provider`, `summarize_after`, `keep_last` |
| `CompositeMemory` | Combine short + long | `memories` (list of Memory) |

```python
from morainet import ShortMemory
from morainet.memory import LongMemory, InMemoryVectorStore, SummarizingMemory

# Short-term
memory = ShortMemory(max_messages=20)

# Long-term RAG
memory = LongMemory(
    store=InMemoryVectorStore(),
    embedder=OllamaEmbedder("nomic-embed-text"),
    score_threshold=0.3,
    top_k=5,
)
await memory.add(Message.assistant(content="Knowledge base document..."))

# Summarizing
memory = SummarizingMemory(provider=provider, summarize_after=20, keep_last=5)
```

### Vector Stores

| Store | Extra Install | Key Parameter |
|-------|--------------|---------------|
| `InMemoryVectorStore` | built-in | `dimension` |
| `ChromaVectorStore` | `[chroma]` | `path`, `collection` |
| `QdrantVectorStore` | `[qdrant]` | `url`, `api_key` |
| `FaissVectorStore` | `[faiss]` | `dimension`, `index_type` |
| `MilvusVectorStore` | `[milvus]` | `uri`, `token` |
| `PgVectorStore` | `[pgvector]` | `dsn` |

### Embedders

| Class | Install | Model |
|-------|---------|-------|
| `HashEmbedder` (default) | built-in | keyword-level (offline) |
| `OllamaEmbedder` | built-in | `nomic-embed-text` |
| `OpenAIEmbedder` | `[openai]` | `text-embedding-3-small` |

---

## 5. Reasoning Strategies

### Strategy Interface

```python
from morainet.reasoning.base import ReasoningStrategy

class ReasoningStrategy(ABC):
    async def step(self, context: Context) -> StrategyDecision: ...
```

### Built-in Strategies

| Strategy | Import | Description |
|----------|--------|-------------|
| `ToolCallingStrategy` (default) | `from morainet.reasoning import ToolCallingStrategy` | Native tool calling |
| `ReActStrategy` | `from morainet.reasoning import ReActStrategy` | Reason + Act loop |
| `EnhancedReAct` | `from morainet.reasoning import EnhancedReAct` | ReAct with context compression |
| `PlanSolveReflect` | `from morainet.reasoning import PlanSolveReflect` | Plan-Execute-Reflect cycle |

```python
from morainet import Agent
from morainet.reasoning import ReActStrategy, PlanSolveReflect

agent = Agent(provider=..., strategy=ReActStrategy())
agent = Agent(provider=..., strategy=PlanSolveReflect(max_decomposition_depth=3))
```

### Context Compressor

```python
from morainet.reasoning import ContextCompressor

compressor = ContextCompressor(provider=..., compress_after_messages=30)
```

### Tool Cache

```python
from morainet.reasoning import ToolCache

cache = ToolCache(ttl=300.0, max_size=1000, persist_path="./tool_cache.json")
```

---

## 6. Workflow Engine

### `Workflow` — DAG Engine

```python
from morainet import Workflow

wf = Workflow()

# Add nodes
wf.add_node("step1", lambda ctx: {"result": ctx.get("x", 0) + 1})
wf.add_node("step2", lambda ctx: {"result": ctx["step1"]["result"] * 2})
wf.add_node("step3", async_step_fn)  # Async node

# Connect edges
wf.connect("step1", "step2")
wf.connect("step2", "step3")

# Execute
result = wf.run({"x": 5})  # Synchronous
result = await wf.arun({"x": 5})  # Async

# Visualization
print(wf.to_mermaid())
print(wf.to_dot())
```

**`Node` type:** `Callable[[dict], dict] | Callable[[dict], Awaitable[dict]]`

**Features:**
- Cycle detection (raises `WorkflowError`)
- Topological sort with parallel level execution
- Each node receives full context dict as input

### DAG Schedulers

```python
from morainet.workflow import (
    Scheduler, SerialScheduler, ParallelScheduler, ProgressScheduler,
    SchedulerRegistry, scheduler_registry, register_scheduler,
)

# Serial execution (topological order)
s = SerialScheduler()
result = await s.run(wf, {"x": 5})

# Parallel execution within levels
s = ParallelScheduler(max_workers=4, timeout=30.0, max_retries=2)
result = await s.run(wf, {"x": 5})

# Progress-tracked parallel
s = ProgressScheduler(max_workers=4)
result = await s.run(wf, {"x": 5})
print(s.progress)  # SchedulerProgress with per-node status

# Custom scheduler registration
register_scheduler("my_scheduler", MyScheduler)
s = scheduler_registry.create("my_scheduler", **kwargs)
```

---

## 7. Multi-Agent Orchestration

### `Pipeline` — Sequential Stages

```python
from morainet import Pipeline, Stage

pipe = Pipeline([
    Stage("research", research_agent),
    Stage("write", write_agent, instruction="Based on research '{research}', write about: {query}"),
])
result = pipe.run("Shanghai weather")
# result.outputs  -> {"research": "...", "write": "..."}
# result.final    -> final stage output
```

### `Router` — Intent Routing

```python
from morainet import Router, Route

router = Router(
    [
        Route("billing", billing_agent, "Billing/payment issues"),
        Route("tech", tech_agent, "Technical problems"),
    ],
    selector=lambda q: "tech" if "error" in q.lower() else "billing",  # Rule-based
    # OR provider=OpenAIProvider(...) for LLM-based routing
)
r = router.run("I got a 502 error")
# r.route -> "tech", r.final -> "..."
```

### `GroupChat` — Multi-Agent Conversation

```python
from morainet import GroupChat, GroupChatMember

member1 = GroupChatMember(name="pm", agent=pm_agent, description="Product Manager")
member2 = GroupChatMember(name="engineer", agent=dev_agent, description="Engineer")

chat = GroupChat(
    members=[member1, member2],
    speaker_selection="round_robin",  # or "auto" (LLM-driven)
    max_rounds=10,
)
result = chat.run("What should we build next sprint?")
# result.rounds -> [{"speaker": "pm", "content": "..."}, ...]
```

### `Debate` — Structured Debate

```python
from morainet import Debate

debate = Debate(
    debaters=[member_supporting, member_opposing],
    judge=judge_agent,
    rounds=3,
)
result = debate.run("Remote vs. office work?")
# result.rounds -> [{"speaker": "...", "content": "...", "round": 1}, ...]
# result.final -> judge's verdict
```

### `Agent.as_tool()` — Hierarchical Delegation

```python
orchestrator = Agent(
    provider=...,
    tools=[
        researcher.as_tool("research", "Research factual information"),
        writer.as_tool("write", "Convert facts to friendly advice"),
    ],
)
```

---

## 8. Persistence & Checkpoint

### Checkpoint Model

```python
from morainet.persistence import Checkpoint

# Fields: trace_id, query, messages, steps, cursor, usage, created_at
```

### Checkpoint Stores

| Store | Import | Key Parameter |
|-------|--------|---------------|
| `InMemoryCheckpointStore` | `from morainet.persistence import InMemoryCheckpointStore` | — |
| `FileCheckpointStore` | `from morainet.persistence import FileCheckpointStore` | `path` |
| `SQLiteCheckpointStore` | `from morainet.persistence import SQLiteCheckpointStore` | `db_path` |
| `RedisCheckpointStore` | `from morainet.persistence import RedisCheckpointStore` | `url` |
| `PostgresCheckpointStore` | `from morainet.persistence import PostgresCheckpointStore` | `dsn` |

```python
from morainet import Agent, FileCheckpointStore

store = FileCheckpointStore("./.checkpoints")
agent = Agent(provider=..., checkpoint_store=store)

result = await agent.arun("Long task...")

# Resume after crash
cp = await store.load(result.trace_id)
resumed = await agent.resume(cp)
```

---

## 9. Observability & Hooks

### Hook Interface

```python
from morainet.observability import Hook

class MyHook(Hook):
    def on_run_start(self, context): ...
    def on_run_end(self, context, result): ...
    def on_llm_start(self, context): ...
    def on_llm_end(self, context, response): ...
    def on_tool_start(self, context, name, args): ...
    def on_tool_end(self, context, name, result): ...
```

### Built-in Hooks

| Hook | Description |
|------|-------------|
| `TraceCollector` | Records all spans with timing |
| `Debugger` | Interactive timeline inspection |
| `CheckpointHook` | Auto-saves checkpoints |
| `OTelHook` | OpenTelemetry integration (requires `[otel]`) |

```python
from morainet import Debugger
from morainet.observability import TraceCollector, OTelHook

dbg = Debugger()
agent = Agent(provider=..., hooks=[dbg])
agent.run("...")
print(dbg.timeline())
# run_start → llm (1.2s) → tool:get_weather (0.1s) → llm (0.8s) → run_end
```

---

## 10. MCP Integration

### `MCPClient` — Connect to MCP Servers

```python
from morainet import MCPClient

async with MCPClient(command="python", args=["-m", "my_mcp_server"]) as client:
    tools = await client.list_tools()       # List available tools
    resources = await client.list_resources()  # List resources
    prompts = await client.list_prompts()   # List prompt templates

    result, meta = await client.call_tool("search", {"query": "..."})
    content, mime_type = await client.read_resource("docs://getting-started")
    prompt_text = await client.get_prompt("code-review", {"language": "python"})
```

### `MCPConnectionPool` — Batch Connection Management

```python
from morainet import MCPConnectionPool, ServerConfig

pool = MCPConnectionPool()
pool.add_server("search", command="python", args=["-m", "search_mcp"])
pool.add_server("code", command="node", args=["code-server.js"])
pool.add_server("db", command="python", args=["-m", "db_mcp"])

await pool.connect_all()  # Batch connect
await pool.check_health()  # Health check all

# Aggregate tools across all servers
all_tools = await pool.get_tools()

# Route tool calls to correct server
result = await pool.call_tool("search", "web_search", {"query": "..."})

# Reconnect configuration
pool.reconnect_attempts = 5
pool.reconnect_delay = 3.0
await pool.start_health_loop(interval=30)  # Background health checks
```

### `MCPResourceCache` — Tool/Prompt/Resource Caching

```python
from morainet import MCPResourceCache

cache = MCPResourceCache(ttl=300, max_size=1000, persist_path="./mcp_cache.json")

tools = await cache.get_tools(server_name, refresh_fn)
prompts = await cache.get_prompts(server_name, refresh_fn)
resources = await cache.get_resources(server_name, refresh_fn)
content = await cache.get_resource_content(server_name, uri, refresh_fn)

cache.invalidate(server_name)  # Clear server cache
stats = cache.stats()  # Usage statistics
```

---

## 11. Plugin System

### Plugin Registry

```python
from morainet.plugins import PluginRegistry, plugins

# Register
plugins.register("providers", "azure", AzureOpenAIProvider)
plugins.register("tools", "search", web_search_tool)
plugins.register("strategies", "custom", MyStrategy)

# Retrieve
provider = plugins.get("providers", "azure")
tool = plugins.get("tools", "search")

# Auto-discover from entry points
plugins.load_entry_points()
```

### Plugin Marketplace

```python
from morainet.plugins import PluginMarketplace, PluginSpec, PluginKind, marketplace

mkt = PluginMarketplace(
    plugins_dir="./plugins",
    index_url="https://plugins.example.com/index.json",
)

# Discover installed plugins
manifests = mkt.discover()

# Install from PyPI
mkt.install("morainet-plugin-slack")

# Install from local path
mkt.install_from_path("./my-plugin")

# Manage lifecycle
mkt.enable("slack")
mkt.disable("slack")
mkt.uninstall("slack")

# Search
results = mkt.search("database")
installed = mkt.list_installed()
by_kind = mkt.list_by_kind(PluginKind.TOOL)

# Export/Import index
mkt.export_index("./index.json")
mkt.import_index("./index.json")
```

### Plugin Specification

```python
from morainet.plugins import PluginSpec, PluginKind, RiskLevel

spec = PluginSpec(
    kind=PluginKind.TOOL,         # provider | tool | memory | strategy | dag_scheduler
    name="my-plugin",
    display_name="My Plugin",
    version="1.0.0",
    description="A description",
    author="author <email>",
    license="MIT",
    risk_level=RiskLevel.LOW,     # low | medium | high
    dependencies=[],
    tags=["database", "sql"],
    entry_points={
        "morainet.tools": {"search": "my_plugin.tools:search"},
    },
)
```

### Entry Point Groups

| Group | Discovered As |
|-------|--------------|
| `morainet.providers` | LLM Providers |
| `morainet.tools` | Tools |
| `morainet.memory` | Memory backends |
| `morainet.strategies` | Reasoning strategies |
| `morainet.dag_schedulers` | DAG schedulers |
| `morainet.plugins` | Plugin metadata |

---

## 12. Engineering Features

### Rate Limiting

```python
from morainet.engineering import TokenBucketLimiter, SlidingWindowLimiter

limiter = TokenBucketLimiter(tokens_per_sec=10.0, burst=20)
await limiter.acquire()  # Block until token available

limiter = SlidingWindowLimiter(max_requests=100, window_seconds=60)
```

### Circuit Breaker

```python
from morainet.engineering import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,
    cooldown_seconds=30.0,
    half_open_max=1,
)

@cb
async def call_external_api(): ...
```

### Concurrency Control

```python
from morainet.engineering import ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_concurrent_llm=10, max_concurrent_tools=20)
async with limiter.llm():
    response = await provider.chat(...)
```

### Billing Tracking

```python
from morainet.engineering import BillingTracker

tracker = BillingTracker(budget_usd=10.0)
tracker.record(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
print(tracker.total_cost)  # USD
if tracker.over_budget():
    raise BudgetExceededError("Budget exceeded")
```

### Tool Audit & Approval

```python
from morainet.tools.audit import ToolAuditor, AuditStore

auditor = ToolAuditor(store=SQLiteAuditStore("audit.db"))

# Dangerous tool approval
def manual_approve(name: str, args: dict) -> bool:
    return input(f"Execute {name}({args})? [y/N] ") == "y"

agent = Agent(provider=..., approve_tool=manual_approve)
```

---

## 13. Configuration

All configurable via environment variables (prefix `MORAINET_`) or `.env` file.

### General

| Env Var | Default | Description |
|---------|---------|-------------|
| `MORAINET_DEFAULT_MODEL` | `gpt-4o` | Default model name |
| `MORAINET_MAX_STEPS` | `10` | Max reasoning iterations |
| `MORAINET_REQUEST_TIMEOUT` | `60.0` | HTTP timeout (seconds) |
| `MORAINET_MAX_RETRIES` | `3` | Default retry count |
| `MORAINET_LOG_LEVEL` | `INFO` | Log level |

### Provider Keys

| Env Var | Provider |
|---------|----------|
| `MORAINET_OPENAI_API_KEY` | OpenAI |
| `MORAINET_ANTHROPIC_API_KEY` | Anthropic Claude |
| `MORAINET_GEMINI_API_KEY` | Google Gemini |
| `MORAINET_DEEPSEEK_API_KEY` | DeepSeek |
| `MORAINET_QWEN_API_KEY` | Qwen (Tongyi) |
| `MORAINET_WENXIN_API_KEY` | Wenxin (Baidu) |
| `MORAINET_ZHIPU_API_KEY` | Zhipu (ChatGLM) |
| `MORAINET_MOONSHOT_API_KEY` | Moonshot |
| `MORAINET_MINIMAX_API_KEY` | MiniMax |
| `MORAINET_SILICONFLOW_API_KEY` | SiliconFlow |

### Reasoning

| Env Var | Default | Description |
|---------|---------|-------------|
| `MORAINET_COMPRESS_AFTER_MESSAGES` | `30` | Auto-compress threshold |
| `MORAINET_MAX_DECOMPOSITION_DEPTH` | `3` | Plan-Solve-Reflect depth |
| `MORAINET_SELF_VERIFY` | `true` | Verify answers before return |
| `MORAINET_TOOL_CACHE_TTL` | `300.0` | Tool result cache TTL |
| `MORAINET_MAX_REFLECT_ROUNDS` | `3` | Max replan cycles |

### Engineering

| Env Var | Default | Description |
|---------|---------|-------------|
| `MORAINET_RATE_LIMIT_TOKENS_PER_SEC` | `10.0` | Token bucket refill |
| `MORAINET_CIRCUIT_BREAKER_FAILURES` | `5` | Failures to open |
| `MORAINET_BILLING_BUDGET_USD` | `0` | Cost cap (0=unlimited) |
| `MORAINET_MAX_CONCURRENT_LLM_CALLS` | `10` | LLM concurrency |
| `MORAINET_MAX_CONCURRENT_TOOL_CALLS` | `20` | Tool concurrency |
| `MORAINET_CHECKPOINT_REDIS_URL` | `""` | Redis for checkpoints |
| `MORAINET_CHECKPOINT_POSTGRES_DSN` | `""` | Postgres for checkpoints |

### MCP

| Env Var | Default | Description |
|---------|---------|-------------|
| `MORAINET_MCP_POOL_RECONNECT` | `true` | Auto-reconnect MCP |
| `MORAINET_MCP_CACHE_TTL` | `300.0` | MCP cache TTL |
| `MORAINET_MCP_CACHE_MAX_SIZE` | `1000` | MCP cache entries |

### Plugin

| Env Var | Description |
|---------|-------------|
| `MORAINET_PLUGIN_MARKETPLACE_PATH` | Local plugins directory |
| `MORAINET_PLUGIN_MARKETPLACE_INDEX_URL` | Remote registry index |

---

## 14. Exceptions

### Exception Hierarchy

```
MorainetError (base)
├── ConfigError
├── ProviderError
│   ├── RateLimitError (retryable)
│   ├── ProviderTimeoutError (retryable)
│   ├── AuthError (non-retryable)
│   └── ContextLengthError -> triggers trim
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolValidationError -> fed back to model
│   └── ToolExecutionError
├── ReasoningError
│   ├── MaxStepsExceededError
│   └── ConsecutiveErrorThresholdError
├── WorkflowError
│   ├── CyclicGraphError
│   └── UnknownNodeError
└── MemoryStoreError
```

### Key Exception Patterns

```python
from morainet.exceptions import (
    MorainetError, ProviderError, RateLimitError,
    ToolError, ToolValidationError, ToolExecutionError,
    MaxStepsExceededError, WorkflowError, MemoryStoreError,
)

try:
    result = agent.run("...")
except RateLimitError:
    # Retryable — framework retries automatically if RetryPolicy configured
    pass
except ToolValidationError as e:
    # Model provided wrong arguments — error fed back to model for self-correction
    print(f"Tool validation failed: {e}")
except MaxStepsExceededError:
    # Agent hit max_steps without producing final answer
    pass
```

---

## Core Data Models

```python
from morainet.core.models import (
    Role,           # Enum: SYSTEM | USER | ASSISTANT | TOOL
    Message,        # role, content, tool_calls, tool_call_id
    ToolCall,       # id, name, arguments
    Usage,          # prompt_tokens, completion_tokens, total_tokens
    ChatResponse,   # message, usage, model, finish_reason
    Step,           # index, description, status, output, error
    StepStatus,     # Enum: PENDING | RUNNING | SUCCESS | FAILED
    AgentResult,    # final_answer, steps, usage, trace_id
    Context,        # query, messages, tools, ...
)
```

---

## Custom Extensions

### Custom Provider

```python
from morainet.providers.base import Provider
from morainet.core.models import ChatResponse

class MyProvider(Provider):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    async def chat(self, messages, tools=None) -> ChatResponse:
        # Convert messages to your API format, make request, convert back
        ...

    async def stream(self, messages, tools=None):
        async for token in ...:
            yield token
```

### Custom Memory Backend

```python
from morainet.memory.base import Memory

class MyMemory(Memory):
    async def add(self, message: Message) -> None: ...
    async def get_context(self, query: str, limit: int = 10) -> list[Message]: ...
```

### Custom Reasoning Strategy

```python
from morainet.reasoning.base import ReasoningStrategy

class MyStrategy(ReasoningStrategy):
    async def step(self, context: Context) -> StrategyDecision: ...
```

---

For more information, see:
- [Getting Started Tutorial](wiki/Getting-Started.md)
- [Architecture Design](architecture.md)
- [Examples](../examples/)
- [Contributing Guide](../CONTRIBUTING.md)
