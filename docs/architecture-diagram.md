# Morainet AI 架构图解

> Mermaid 架构图集合，展示各层模块关系与交互流程。

---

## 1. 整体分层架构

```mermaid
flowchart TD
    subgraph App["Application Layer"]
        CLI[CLI / Script]
        WEB[Web / FastAPI]
        GRPC[gRPC Service]
    end

    subgraph Core["Agent Core"]
        AGENT[Agent Runtime]
        LIFECYCLE[Lifecycle Manager]
        CONTEXT[Context Manager]
    end

    subgraph Reasoning["Reasoning Engine"]
        TCS[ToolCallingStrategy]
        REACT[ReActStrategy]
        PSR[PlanSolveReflect]
        CC[ContextCompressor]
        TC[ToolCache]
    end

    subgraph Memory["Memory System"]
        SHORT[ShortMemory]
        LONG[LongMemory]
        SUMM[SummarizingMemory]
        COMP[CompositeMemory]
    end

    subgraph Tools["Tool System"]
        DECORATOR[@tool Decorator]
        REGISTRY[ToolRegistry]
        SCHEMA[Schema Generator]
        APPROVAL[Approval Flow]
        AUDIT[Audit System]
    end

    subgraph Workflow["Workflow Engine"]
        DAG[DAG Builder]
        SERIAL[SerialScheduler]
        PARALLEL[ParallelScheduler]
        PROGRESS[ProgressScheduler]
    end

    subgraph Provider["Provider Layer"]
        ABSTRACT[Provider Interface]
        OPENAI[OpenAI]
        CLAUDE[Claude]
        GEMINI[Gemini]
        OLLAMA[Ollama]
        DEEPSEEK[DeepSeek]
        QWEN[Qwen]
        ROUTER[ModelRouter]
        RETRY[RetryingProvider]
    end

    subgraph Infra["Infrastructure"]
        VECTOR[(VectorStore)]
        CKPT[(CheckpointStore)]
        CACHE[(Cache)]
        LIMITER[Rate Limiter]
        CB[Circuit Breaker]
        BILLING[Billing Tracker]
    end

    subgraph Ext["Extension"]
        PLUGIN[Plugin System]
        MCP[MCP Client / Pool]
        MKT[Plugin Marketplace]
    end

    subgraph Obs["Observability"]
        HOOK[Hook System]
        DEBUG[Debugger]
        TRACE[TraceCollector]
        OTEL[OpenTelemetry]
    end

    CLI --> AGENT
    WEB --> AGENT
    GRPC --> AGENT

    AGENT --> Reasoning
    AGENT --> Memory
    AGENT --> Tools
    AGENT --> Workflow

    Reasoning --> ABSTRACT
    Tools --> ABSTRACT

    Memory --> VECTOR

    ABSTRACT --> OPENAI
    ABSTRACT --> CLAUDE
    ABSTRACT --> GEMINI
    ABSTRACT --> OLLAMA
    ABSTRACT --> DEEPSEEK
    ABSTRACT --> QWEN

    AGENT --> Infra
    AGENT --> Obs
    AGENT --> Ext

    AGENT --> CKPT
```

---

## 2. Agent Runtime 执行序列

```mermaid
sequenceDiagram
    participant User
    participant Agent as Agent Core
    participant Strategy as Reasoning Strategy
    participant Memory as Memory System
    participant Provider as LLM Provider
    participant Tool as Tool Executor
    participant Hook as Hook System

    User->>Agent: run(query)
    Agent->>Hook: on_run_start()

    Agent->>Memory: get_context(query)
    Memory-->>Agent: context_messages

    loop until done or max_steps
        Agent->>Strategy: step(context)
        Strategy->>Provider: chat(messages, tools)
        Hook->>Hook: on_llm_start / on_llm_end

        alt tool_calls
            Provider-->>Strategy: tool_calls[]
            Strategy-->>Agent: execute tools
            Agent->>Hook: on_tool_start()

            par parallel tool calls
                Agent->>Tool: tool_1(args)
                Agent->>Tool: tool_2(args)
            end

            Tool-->>Agent: results[]
            Agent->>Hook: on_tool_end()
            Agent->>Memory: add(tool_results)
        else final answer
            Provider-->>Strategy: content
            Strategy-->>Agent: done
        end
    end

    Agent->>Memory: add(assistant_response)
    Agent->>Hook: on_run_end()
    Agent-->>User: AgentResult
```

---

## 3. 推理策略决策树

```mermaid
flowchart TD
    Start([Task Received]) --> Check{Is the workflow known?}

    Check -->|Yes, deterministic| WF[Use Workflow DAG]
    Check -->|No, needs reasoning| Strategy{Task type?}

    Strategy -->|Simple tool call| TCS[ToolCallingStrategy]
    Strategy -->|Exploratory, uncertain| REACT[ReActStrategy]
    Strategy -->|Complex, multi-step| PSR[PlanSolveReflect]

    TCS --> Call[LLM with tools]
    REACT --> Loop[Thought → Action → Observe loop]
    PSR --> PEP[Plan → Execute → Reflect cycle]

    Call --> Done([Return Result])
    Loop --> Done
    PEP --> Done
    WF --> Done
```

---

## 4. Memory 系统架构

```mermaid
flowchart TD
    subgraph Input["Message Input"]
        MSG[Message]
    end

    subgraph ShortTerm["Short-Term Memory"]
        SM[ShortMemory]
        SLIDING[Sliding Window]
        TOKEN_BUDGET[Token Budget Trimming]
    end

    subgraph LongTerm["Long-Term Memory"]
        LM[LongMemory]
        EMBED[Embedder]
        VS[VectorStore]
    end

    subgraph Composite["Advanced Memory"]
        SUMM_MEM[SummarizingMemory]
        COMP_MEM[CompositeMemory]
        CP[ContextCompressor]
    end

    subgraph Retrieval["Context Retrieval"]
        MERGE[Context Merge]
        INJECT[Inject into System Prompt]
    end

    MSG --> SM
    MSG --> LM

    SM --> SLIDING
    SM --> TOKEN_BUDGET

    LM --> EMBED
    EMBED --> VS

    SM --> MERGE
    LM --> MERGE

    MERGE --> INJECT
    INJECT --> AGENT[Agent Prompt]

    SUMM_MEM --> CP
    COMP_MEM --> MERGE
```

---

## 5. Provider 路由与容错

```mermaid
flowchart TD
    Request([chat request]) --> Router{ModelRouter}

    Router -->|tier=small| SmallModel[DeepSeek / Qwen]
    Router -->|tier=large| LargeModel[GPT-4o / Claude]

    SmallModel --> Retry{Error?}

    Retry -->|RateLimitError| Backoff[Exponential Backoff]
    Retry -->|TimeoutError| Backoff
    Retry -->|Success| Return([ChatResponse])

    Backoff --> RetryCount{retries < max?}
    RetryCount -->|Yes| SmallModel
    RetryCount -->|No| Fallback{Fallback enabled?}
    Fallback -->|Yes| LargeModel
    Fallback -->|No| Error([Raise Error])

    Retry -->|AuthError| Error
    Retry -->|ContextLengthError| Trim[Trim Context]
    Trim --> SmallModel

    LargeModel --> Return
```

---

## 6. Workflow DAG 执行模型

```mermaid
flowchart LR
    subgraph DAG["DAG Topology"]
        A[init] --> B[check_a]
        A --> C[check_b]
        A --> D[check_c]
        B --> E[report]
        C --> E
        D --> E
    end

    subgraph Levels["Topological Levels"]
        L0[Level 0: init]
        L1[Level 1: check_a, check_b, check_c]
        L2[Level 2: report]
    end

    L0 -->|Serial| L1
    L1 -->|Parallel| L2
```

```mermaid
sequenceDiagram
    participant Scheduler
    participant Worker1
    participant Worker2
    participant Worker3

    Scheduler->>Scheduler: Topological sort
    Scheduler->>Worker1: Level 0: init()
    Worker1-->>Scheduler: done

    par Level 1 (parallel)
        Scheduler->>Worker1: check_a()
        Scheduler->>Worker2: check_b()
        Scheduler->>Worker3: check_c()
        Worker1-->>Scheduler: done
        Worker2-->>Scheduler: done
        Worker3-->>Scheduler: done
    end

    Scheduler->>Worker1: Level 2: report()
    Worker1-->>Scheduler: done
```

---

## 7. Plugin / MCP 扩展机制

```mermaid
flowchart TD
    subgraph Discovery["Plugin Discovery"]
        ENTRY[Python entry_points]
        WHEEL[wheel/.whl]
        LOCAL[Local Directory]
    end

    subgraph Registry["Plugin Registry"]
        PROVIDERS[morainet.providers]
        TOOLS_GRP[morainet.tools]
        MEMORY_GRP[morainet.memory]
        STRATEGIES_GRP[morainet.strategies]
        SCHEDULERS[morainet.dag_schedulers]
    end

    subgraph MCP["MCP Integration"]
        POOL[MCPConnectionPool]
        CACHE_MCP[MCPResourceCache]
        CLIENT[MCPClient]
    end

    subgraph Marketplace["Plugin Marketplace"]
        MKT[PluginMarketplace]
        PIP[pip install]
        INDEX[Registry Index]
    end

    ENTRY --> Registry
    WHEEL --> PIP --> Registry
    LOCAL --> Registry

    MKT --> PIP
    MKT --> INDEX
    MKT --> LOCAL

    CLIENT --> POOL
    CLIENT --> CACHE_MCP

    Registry --> AGENT[Agent Runtime]
    POOL --> AGENT
    CACHE_MCP --> AGENT
```

---

## 8. Checkpoint 持久化流程

```mermaid
flowchart TD
    Run[agent.arun] --> Loop{Step Loop}

    Loop --> Step[Execute Step]
    Step --> Save[Async Save Checkpoint]

    Save --> Store{Backend}

    Store --> Mem[InMemory]
    Store --> File[File/JSON]
    Store --> SQLite[SQLite]
    Store --> Redis[Redis]
    Store --> PG[PostgreSQL]

    Mem --> Next{More Steps?}
    File --> Next
    SQLite --> Next
    Redis --> Next
    PG --> Next

    Next -->|Yes| Loop
    Next -->|No| Done([AgentResult])

    Crash[Crash / Restart] --> Load[load Checkpoint]
    Load --> Resume[agent.resume]
    Resume --> Loop
```

---

## 9. 异常处理与重试机制

```mermaid
flowchart TD
    Error([Exception Raised]) --> Classify{Error Type}

    Classify -->|RateLimitError| Retry[Retry with backoff]
    Classify -->|ProviderTimeoutError| Retry
    Classify -->|NetworkError| Retry

    Classify -->|AuthError| Fatal[Immediate Raise]
    Classify -->|ConfigError| Fatal

    Classify -->|ToolValidationError| Feedback[Feedback to Model]
    Feedback --> RetryModel[Model self-corrects]

    Classify -->|ContextLengthError| Trim[Auto Trim Context]
    Trim --> Continuation[Continue]

    Classify -->|MaxStepsExceededError| Partial[Return Partial Result]

    Retry --> Check{Retries Left?}
    Check -->|Yes| Wait[Exponential Backoff]
    Wait --> Call[Retry Call]
    Check -->|No| Fatal

    RetryModel --> Call
```

---

## 10. 企业级部署拓扑

```mermaid
flowchart TD
    subgraph Edge["Edge / Client"]
        BROWSER[Browser]
        MOBILE[Mobile App]
        BOT[Slack/WeCom Bot]
    end

    subgraph Gateway["API Gateway"]
        NGINX[Nginx / Caddy]
        WAF[WAF / Rate Limit]
        AUTH[Auth / OAuth2]
    end

    subgraph App["Application Tier"]
        API1[FastAPI #1]
        API2[FastAPI #2]
        API3[FastAPI #N]
    end

    subgraph Cache["Cache Tier"]
        REDIS[(Redis)]
    end

    subgraph Storage["Storage Tier"]
        PG[(PostgreSQL)]
        QDRANT[(Qdrant)]
        S3[(S3 / MinIO)]
    end

    subgraph LLM["LLM Tier"]
        OPENAI_API[OpenAI API]
        OLLAMA_SRV[Ollama Server]
        ANTHROPIC_API[Anthropic API]
    end

    subgraph Monitor["Monitoring"]
        PROM[Prometheus]
        GRAFANA[Grafana]
        TEMPO[Tempo Tracing]
    end

    BROWSER --> NGINX
    MOBILE --> NGINX
    BOT --> NGINX

    NGINX --> WAF
    WAF --> AUTH
    AUTH --> API1
    AUTH --> API2
    AUTH --> API3

    API1 --> REDIS
    API1 --> PG
    API1 --> QDRANT
    API1 --> S3

    API1 --> OPENAI_API
    API1 --> OLLAMA_SRV
    API1 --> ANTHROPIC_API

    API1 --> PROM
    PROM --> GRAFANA
    API1 --> TEMPO
```

---

## 11. 数据流全景

```mermaid
flowchart LR
    User((User)) -->|"query: str"| Agent

    Agent -->|"[memory]"| System[System Prompt + Context]
    Agent -->|"tools[]"| Schema[Tool Schemas]

    System --> LLM[LLM Provider]
    Schema --> LLM

    LLM -->|ChatResponse| Parse{Parse Response}

    Parse -->|tool_calls| Execute[Execute Tools]
    Parse -->|content| Final[Final Answer]

    Execute --> Tool1[Tool 1]
    Execute --> Tool2[Tool N]

    Tool1 -->|result| Feedback[Tool Result Messages]
    Tool2 -->|result| Feedback

    Feedback --> LLM
    Final --> User
```

---

> 配合阅读：[API Reference](api-reference.md) · [性能调优指南](performance-tuning.md) · [部署指南](deployment.md)
