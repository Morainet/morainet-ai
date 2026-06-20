"""企业级实战：自动化运维 Agent。

演示自动化运维场景：
- 监控告警处理：解析告警 → 诊断 → 执行修复 → 生成事件报告
- 预案执行（Runbook Automation）：预定义故障处理流程
- 日志分析：从日志中提取关键信息、异常模式
- 资源巡检：定期检查 CPU/内存/磁盘，生成健康报告
- 变更管理：发布上线 → 冒烟测试 → 回滚（灰度策略）

离线可跑，MockProvider 脚本化；也可切到本地 Ollama。

Run:
    python examples/enterprise_ops_agent.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/enterprise_ops_agent.py
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from morainet import Agent, Workflow, tool
from morainet.core.models import ChatResponse, Message, Role, ToolCall, Usage
from morainet.memory import ShortMemory
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider


# ---------------------------------------------------------------------------
# 运维数据模拟
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Alert:
    id: str
    service: str
    severity: AlertSeverity
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def format(self) -> str:
        return f"[{self.severity.upper()}] {self.service}: {self.message}"


@dataclass
class ServerState:
    """模拟服务器状态。"""
    cpu_pct: float = 35.0
    mem_pct: float = 60.0
    disk_pct: float = 45.0
    processes: list[str] = field(default_factory=lambda: ["nginx", "redis", "app:8080", "app:8081"])
    is_healthy: bool = True
    db_connections: int = 42


SERVER = ServerState()
INCIDENTS: list[dict] = []
LOGS_DB: list[str] = [
    "2024-01-20 10:00:01 INFO nginx 200 GET /api/users 0.01s",
    "2024-01-20 10:00:02 WARN nginx 502 GET /api/orders 30.0s upstream timeout",
    "2024-01-20 10:00:03 ERROR app:8080 ConnectionPool timeout redis:6379",
    "2024-01-20 10:00:04 ERROR app:8080 retry exhausted redis:6379",
    "2024-01-20 10:00:05 WARN app:8081 circuit_breaker OPEN redis",
    "2024-01-20 10:00:06 INFO redis recovered connections=64",
    "2024-01-20 10:00:07 INFO nginx 200 GET /api/health 0.00s",
]


# ---------------------------------------------------------------------------
# 运维工具
# ---------------------------------------------------------------------------


@tool
def check_health(service: str | None = None) -> str:
    """检查服务健康状态。

    Args:
        service: 可选，指定服务名。不传则检查全部。
    """
    if service and service == "redis":
        # 模拟 redis 故障
        if random.random() < 0.3:
            return "redis: UNHEALTHY — 连接超时"
        return "redis: HEALTHY — 64 connections"
    return (
        f"nginx: HEALTHY | app:8080: HEALTHY | app:8081: HEALTHY | "
        f"CPU: {SERVER.cpu_pct}% | MEM: {SERVER.mem_pct}% | "
        f"DISK: {SERVER.disk_pct}% | DB connections: {SERVER.db_connections}"
    )


@tool
def get_metrics(metric: str = "all") -> str:
    """获取系统监控指标。

    Args:
        metric: 指标名，可选 cpu/memory/disk/all
    """
    metrics = {
        "cpu": f"CPU: {SERVER.cpu_pct}% (avg 1min)",
        "memory": f"Memory: {SERVER.mem_pct}% (used 3.8GB / total 8GB)",
        "disk": f"Disk: {SERVER.disk_pct}% (used 45GB / total 100GB)",
        "all": (
            f"CPU: {SERVER.cpu_pct}% | "
            f"Memory: {SERVER.mem_pct}% | "
            f"Disk: {SERVER.disk_pct}% | "
            f"DB connections: {SERVER.db_connections}"
        ),
    }
    return metrics.get(metric, "未知指标")


@tool
def restart_service(service: str) -> str:
    """重启指定服务。

    Args:
        service: 服务名，如 app:8080 / nginx / redis
    """
    if service not in SERVER.processes and service not in ("nginx", "redis"):
        return f"服务 {service} 不存在"
    return f"服务 {service} 已重启（PID: {random.randint(10000, 99999)}），启动耗时 2.3s"


@tool
def search_logs(pattern: str, lines: int = 20) -> str:
    """搜索应用日志。

    Args:
        pattern: 搜索关键词
        lines: 返回最多多少行
    """
    matched = [log for log in LOGS_DB if pattern.lower() in log.lower()]
    return "\n".join(matched[-lines:]) if matched else f"未找到包含 '{pattern}' 的日志"


@tool
def create_incident(service: str, title: str, description: str, severity: str = "warning") -> str:
    """创建运维事件记录。

    Args:
        service: 受影响的服务
        title: 事件标题
        description: 事件描述
        severity: 严重程度 critical/warning/info
    """
    incident = {
        "id": f"INC-{len(INCIDENTS) + 1:04d}",
        "service": service,
        "title": title,
        "description": description,
        "severity": severity,
        "status": "open",
        "created": datetime.now(timezone.utc).isoformat(),
    }
    INCIDENTS.append(incident)
    return f"事件 {incident['id']} 已创建，已通知值班工程师。"


@tool(dangerous=True)
def rollback_deploy(version: str) -> str:
    """回滚部署到指定版本（危险操作，需审批）。

    Args:
        version: 目标版本号
    """
    return f"已回滚到 {version}。流量已切换，健康检查通过。"


@tool
def scale_up(replicas: int = 2) -> str:
    """扩容服务实例。

    Args:
        replicas: 目标副本数
    """
    old = len(SERVER.processes)
    for i in range(old + 1, replicas + 1):
        SERVER.processes.append(f"app:{8080 + i}")
    return f"扩容完成：{old} → {len(SERVER.processes)} 个实例"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def _build_provider(scenario: str = "alert") -> Provider:
    """按场景构建 MockProvider。"""
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    if scenario == "alert":
        return MockProvider(
            responses=[
                # Step 1: 查日志
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[
                            ToolCall(id="1", name="search_logs", arguments={"pattern": "ERROR redis"})
                        ]
                    ),
                    usage=Usage(total_tokens=20),
                    finish_reason="tool_calls",
                ),
                # Step 2: 查健康
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[ToolCall(id="2", name="check_health", arguments={"service": "redis"})]
                    ),
                    usage=Usage(total_tokens=20),
                    finish_reason="tool_calls",
                ),
                # Step 3: 重启服务
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[
                            ToolCall(id="3", name="restart_service", arguments={"service": "redis"})
                        ]
                    ),
                    usage=Usage(total_tokens=20),
                    finish_reason="tool_calls",
                ),
                # Step 4: 创建事件
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[
                            ToolCall(
                                id="4",
                                name="create_incident",
                                arguments={
                                    "service": "redis",
                                    "title": "Redis 连接超时导致 upstream 502",
                                    "description": "10:00:02 起 app:8080 报 redis 连接超时，重启后恢复",
                                    "severity": "warning",
                                },
                            )
                        ]
                    ),
                    usage=Usage(total_tokens=20),
                    finish_reason="tool_calls",
                ),
                # Final: 总结
                ChatResponse(
                    message=Message.assistant(
                        content="告警已处理：redis 连接超时 → 重启恢复。事件 INC-0001 已创建。"
                        "建议：redis 连接池从 64 扩容到 128，设置更积极的超时重试。"
                    ),
                ),
            ]
        )

    if scenario == "patrol":
        return MockProvider(
            responses=[
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[ToolCall(id="1", name="get_metrics", arguments={"metric": "all"})]
                    ),
                    usage=Usage(total_tokens=10),
                    finish_reason="tool_calls",
                ),
                ChatResponse(
                    message=Message.assistant(
                        content="巡检报告：所有指标正常。CPU 35% MEM 60% DISK 45%。无异常告警。"
                    ),
                ),
            ]
        )

    # 部署场景
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="1", name="check_health", arguments={}),
                    ]
                ),
                usage=Usage(total_tokens=10),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(
                    content="部署验证通过：健康检查正常，无异常日志。v2.1.0 上线成功。"
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 预案模板（Runbook）
# ---------------------------------------------------------------------------


def build_runbook_workflow(alert: Alert) -> Workflow:
    """根据告警类型构建自动化预案 Workflow。"""
    wf = Workflow()

    wf.add_node("parse", lambda ctx: {"alert": alert.format()})
    wf.add_node("diagnose", lambda ctx: {"findings": search_logs.func(alert.service, 10)})
    wf.add_node("health_check", lambda ctx: {"health": check_health.func(alert.service)})

    def decide_action(ctx: dict) -> dict:
        health = str(ctx.get("health_check", {}).get("health", ""))
        if "UNHEALTHY" in health:
            return {"action": "restart"}
        return {"action": "monitor"}

    wf.add_node("decide", decide_action)

    def execute(ctx: dict) -> dict:
        action = str(ctx.get("decide", {}).get("action", ""))
        if action == "restart":
            result = restart_service.func(alert.service)
            return {"result": result}
        return {"result": "继续监控中"}

    wf.add_node("execute", execute)
    wf.add_node(
        "report",
        lambda ctx: {
            "report": f"告警: {ctx.get('parse', {}).get('alert', '')}\n"
            f"诊断: {ctx.get('diagnose', {}).get('findings', '')}\n"
            f"动作: {ctx.get('execute', {}).get('result', '')}"
        },
    )

    wf.connect("parse", "diagnose")
    wf.connect("parse", "health_check")
    wf.connect("diagnose", "decide")
    wf.connect("health_check", "decide")
    wf.connect("decide", "execute")
    wf.connect("execute", "report")

    return wf


# ---------------------------------------------------------------------------
# 运维 Agent 系统
# ---------------------------------------------------------------------------


@dataclass
class OpsSystem:
    """运维自动化系统。"""

    agent: Agent
    memory: ShortMemory

    @classmethod
    def create(cls) -> "OpsSystem":
        memory = ShortMemory()
        agent = Agent(
            provider=_build_provider("alert"),
            tools=[
                check_health, get_metrics, restart_service,
                search_logs, create_incident, rollback_deploy, scale_up,
            ],
            memory=memory,
            system_prompt=(
                "你是资深 SRE 运维工程师。收到告警后请：\n"
                "1. 先查日志了解上下文\n"
                "2. 检查相关服务健康状态\n"
                "3. 执行修复（重启/扩容等）\n"
                "4. 创建事件记录，给出长期优化建议\n"
                "回滚等危险操作需审批确认。"
            ),
            approve_tool=lambda name, args: True,  # 自动化场景自动批准
            max_steps=10,
        )
        return cls(agent=agent, memory=memory)

    async def handle_alert(self, alert: Alert) -> str:
        """处理一条告警。"""
        print(f"\n[ALERT] 收到告警: {alert.format()}")
        result = await self.agent.arun(alert.format())
        return result.final_answer

    async def patrol(self) -> str:
        """资源巡检。"""
        agent = Agent(
            provider=_build_provider("patrol"),
            tools=[check_health, get_metrics, search_logs],
            system_prompt="你是运维巡检员。检查所有服务健康状态和资源指标，生成巡检报告。",
        )
        result = await agent.arun("执行例行资源巡检")
        return result.final_answer

    async def deploy(self, version: str) -> str:
        """部署上线并验证。"""
        agent = Agent(
            provider=_build_provider("deploy"),
            tools=[check_health, search_logs, rollback_deploy],
            system_prompt=(
                "你是发布工程师。部署后请验证健康状态、检查错误日志、"
                "确认无异常后给出上线结论。异常则建议回滚。"
            ),
        )
        result = await agent.arun(f"版本 {version} 已部署到生产环境，请验证。")
        return result.final_answer


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("企业级自动化运维 Agent")
    print("=" * 60)

    ops = OpsSystem.create()

    # 场景 1：告警处理
    print("\n--- 场景 1：Redis 连接超时告警 ---")
    alert = Alert(
        id="ALT-001",
        service="redis",
        severity=AlertSeverity.WARNING,
        message="connection timeout on redis:6379, 5s exceeded",
    )
    result = await ops.handle_alert(alert)
    print(f"处理结果：{result}")
    print(f"步数：{len(ops.agent._steps)}")

    # 场景 2：预案 Workflow
    print("\n--- 场景 2：预案 Runbook DAG ---")
    alert2 = Alert(id="ALT-002", service="app:8080", severity=AlertSeverity.CRITICAL, message="OOM killed")
    wf = build_runbook_workflow(alert2)
    out = wf.run({})
    print(f"预案执行报告：\n{out.get('report', {}).get('report', 'N/A')}")
    print(f"预案流程图：\n{wf.to_mermaid()}")

    # 场景 3：资源巡检
    print("\n--- 场景 3：资源巡检 ---")
    report = await ops.patrol()
    print(f"巡检报告：{report}")

    # 场景 4：部署上线
    print("\n--- 场景 4：部署验证 ---")
    result = await ops.deploy("v2.1.0")
    print(f"部署结果：{result}")

    # 事件汇总
    if INCIDENTS:
        print(f"\n本次产生 {len(INCIDENTS)} 个事件:")
        for inc in INCIDENTS:
            print(f"  {inc['id']} [{inc['severity']}] {inc['title']} ({inc['status']})")

    print("\n" + "=" * 60)
    print("运维 Agent 运行完毕")


if __name__ == "__main__":
    asyncio.run(main())
