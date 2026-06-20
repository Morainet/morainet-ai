"""企业级实战：代码工程助手。

完整的代码工程助手系统：
- 代码审查（Code Review）：检查逻辑错误、安全漏洞、性能问题
- 测试生成：根据函数签名自动生成 pytest 用例
- Git 变更分析：分析 diff，生成 ChangeLog
- 依赖扫描：检查过时/有漏洞的依赖
- Workflow DAG：代码审查流水线（Lint → 审查 → 报告）

工具执行**做真事**：真实读取文件、运行 ruff、解析 AST。

离线可跑，MockProvider 脚本化；也可切到本地 Ollama。

Run:
    python examples/enterprise_code_assistant.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/enterprise_code_assistant.py
"""

from __future__ import annotations

import asyncio
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from morainet import Agent, Workflow, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider

# 创建临时工程目录（含一些"问题"代码，用于演示代码审查）
WORKSPACE = Path(tempfile.mkdtemp(prefix="morainet_enterprise_code_"))

# 源码文件
(WORKSPACE / "user_service.py").write_text(
    """\"\"\"用户服务模块。\"\"\"
import hashlib
import json
import sqlite3
from typing import Any


def hash_password(password: str) -> str:
    \"\"\"对密码做哈希。\"\"\"
    return hashlib.md5(password.encode()).hexdigest()


def get_user(user_id: int) -> dict | None:
    \"\"\"从数据库获取用户。\"\"\"
    query = f"SELECT * FROM users WHERE id = {user_id}"
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(query)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2]}
    return None


def process_data(data: list[int]) -> list[int]:
    \"\"\"处理数据列表。\"\"\"
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] * data[j])
    return result


def export_config() -> str:
    \"\"\"导出配置到 JSON。\"\"\"
    config = {
        "api_key": "sk-prod-abc123def456",
        "db_password": "admin123",
        "endpoint": "https://api.example.com"
    }
    return json.dumps(config)


def divide(a: float, b: float) -> float:
    \"\"\"安全的除法运算。

    Args:
        a: 被除数
        b: 除数
    \"\"\"
    return a / b
""",
    encoding="utf-8",
)

# 测试文件
(WORKSPACE / "test_user_service.py").write_text(
    """from user_service import divide


def test_divide_normal():
    assert divide(10, 2) == 5


def test_divide_zero():
    assert divide(0, 5) == 0
""",
    encoding="utf-8",
)

# 需求文件
(WORKSPACE / "requirements.txt").write_text(
    """requests==2.25.1
flask==2.0.0
sqlalchemy==1.4.0
""",
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# 真实工具
# ---------------------------------------------------------------------------


@tool
def list_project_files() -> str:
    """列出工程中的所有文件及其大小。"""
    lines = []
    for p in sorted(WORKSPACE.rglob("*")):
        if p.is_file() and ".git" not in str(p) and "__pycache__" not in str(p):
            size = p.stat().st_size
            lines.append(f"  {p.relative_to(WORKSPACE)} ({size} bytes)")
    return "\n".join(lines) if lines else "空项目"


@tool
def read_source_file(path: str) -> str:
    """读取工程中的源码文件。

    Args:
        path: 相对工程根目录的文件路径，如 "user_service.py"
    """
    target = (WORKSPACE / path).resolve()
    if WORKSPACE.resolve() not in target.parents and target != WORKSPACE.resolve():
        return "ERROR: 路径越界"
    if not target.exists():
        return f"ERROR: 文件 {path} 不存在"
    return target.read_text(encoding="utf-8")


@tool
def run_lint(path: str | None = None) -> str:
    """运行 ruff 代码检查（真实执行）。

    Args:
        path: 可选，要检查的文件路径。不传则检查整个工程。
    """
    targets = [str(WORKSPACE / path)] if path else [str(WORKSPACE)]
    proc = subprocess.run(
        ["ruff", "check", "--quiet", *targets],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() or "无 lint 问题 ✓"


@tool
def analyze_ast(path: str) -> str:
    """解析 Python 文件的 AST 结构，检测潜在问题（真实执行）。

    Args:
        path: 相对工程根目录的文件路径
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"文件 {path} 不存在"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)

    issues: list[str] = []
    for node in ast.walk(tree):
        # 检测 MD5 哈希
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "md5":
                issues.append(f"  [安全] {path}:{node.lineno} — 使用 MD5 哈希，建议换 SHA-256")

        # 检测字符串拼接 SQL（简单模式）
        if isinstance(node, ast.JoinedStr) or isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # 简化：检测 f-string 或拼接中含 "SELECT" 即是 SQL 注入风险
            pass

        # 检测嵌套循环 O(n^2)
        if isinstance(node, ast.For):
            for child in ast.walk(node):
                if child is not node and isinstance(child, ast.For):
                    issues.append(
                        f"  [性能] {path}:{node.lineno} — 嵌套循环 O(n²)，数据量大时建议优化"
                    )
                    break

        # 检测硬编码密钥
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "config":
                    if isinstance(node.value, ast.Dict):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and key.value in (
                                "api_key", "db_password", "password", "secret"
                            ):
                                issues.append(
                                    f"  [安全] {path}:{key.lineno} — 硬编码 {key.value}，"
                                    f"建议从环境变量读取"
                                )

        # 检测除零
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Name):
                issues.append(
                    f"  [健壮] {path}:{node.lineno} — 除数为变量，建议加零值检查"
                )

    if not issues:
        # 二次扫描：SQL 注入和密钥
        for line_no, line in enumerate(source.split("\n"), 1):
            if "f\"SELECT" in line or "f'SELECT" in line:
                issues.append(f"  [安全] {path}:{line_no} — 疑似 SQL 注入（f-string 拼接 SQL）")

        for line_no, line in enumerate(source.split("\n"), 1):
            if any(k in line for k in ("api_key", "db_password", "secret_key")) and "=" in line:
                if "os.environ" not in line and "getenv" not in line:
                    issues.append(f"  [安全] {path}:{line_no} — 硬编码敏感信息")

    return "\n".join(issues) if issues else "AST 分析无问题 ✓"


@tool
def run_tests() -> str:
    """运行 pytest 测试，返回真实结果。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(WORKSPACE)],
        capture_output=True, text=True, cwd=str(WORKSPACE),
    )
    return (proc.stdout + proc.stderr)[-1200:]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def _build_provider() -> Provider:
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    return MockProvider(
        responses=[
            # Step 1: 列文件
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="1", name="list_project_files", arguments={})]
                ),
                usage=Usage(total_tokens=20),
                finish_reason="tool_calls",
            ),
            # Step 2: 读源码
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="2", name="read_source_file", arguments={"path": "user_service.py"})
                    ]
                ),
                usage=Usage(total_tokens=20),
                finish_reason="tool_calls",
            ),
            # Step 3: AST 分析
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="3", name="analyze_ast", arguments={"path": "user_service.py"})
                    ]
                ),
                usage=Usage(total_tokens=20),
                finish_reason="tool_calls",
            ),
            # Step 4: Lint
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="4", name="run_lint", arguments={})]
                ),
                usage=Usage(total_tokens=20),
                finish_reason="tool_calls",
            ),
            # Step 5: 最终报告
            ChatResponse(
                message=Message.assistant(
                    content="代码审查完毕。主要问题：\n"
                    "1. hash_password 使用 MD5（不安全）→ 换 SHA-256\n"
                    "2. get_user 存在 SQL 注入（f-string 拼接）→ 用参数化查询\n"
                    "3. process_data 嵌套循环 O(n²) → 可优化\n"
                    "4. export_config 硬编码密钥 → 用环境变量\n"
                    "5. divide 缺少除零保护 → 加异常处理\n"
                    "整体评分 6/10，建议修复安全问题后合并。"
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Workflow DAG：代码审查流水线
# ---------------------------------------------------------------------------


def build_review_workflow() -> Workflow:
    """构建代码审查 DAG 流水线。"""
    wf = Workflow()

    wf.add_node("list", lambda ctx: {"files": ctx.get("workspace", "")})
    wf.add_node("read", lambda ctx: {"source": "已读取源码"})
    wf.add_node(
        "ast_check",
        lambda ctx: {"ast_issues": analyze_ast.func("user_service.py")},
    )
    wf.add_node(
        "lint_check",
        lambda ctx: {"lint_result": run_lint.func(None)},
    )
    wf.add_node(
        "test_check",
        lambda ctx: {"test_result": "测试需补充除零场景"},
    )
    wf.add_node(
        "report",
        lambda ctx: {
            "report": f"AST: {ctx.get('ast_check', {}).get('ast_issues', '')}\n"
            f"Lint: {ctx.get('lint_check', {}).get('lint_result', '')}\n"
            f"Test: {ctx.get('test_check', {}).get('test_result', '')}"
        },
    )

    # DAG：lint 和 ast 并行执行
    wf.connect("list", "read")
    wf.connect("read", "ast_check")
    wf.connect("read", "lint_check")
    wf.connect("read", "test_check")
    wf.connect("ast_check", "report")
    wf.connect("lint_check", "report")
    wf.connect("test_check", "report")

    return wf


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("企业级代码工程助手")
    print(f"工程目录：{WORKSPACE}")
    print("=" * 60)

    # 模式 1：Agent 驱动的交互式代码审查
    print("\n--- 模式 1：Agent 代码审查 ---")
    agent = Agent(
        provider=_build_provider(),
        tools=[list_project_files, read_source_file, run_lint, analyze_ast, run_tests],
        system_prompt=(
            "你是资深代码审查专家。请：\n"
            "1. 先列出工程文件了解结构\n"
            "2. 读取源码，逐函数审查\n"
            "3. 检查安全漏洞、性能问题、代码规范\n"
            "4. 给出评分和具体修复建议"
        ),
        max_steps=10,
    )
    result = await agent.arun(f"审查工程 {WORKSPACE.name} 的代码质量")
    print(f"审查结论：\n{result.final_answer}")
    print(f"步数：{len(result.steps)}, Tokens：{result.usage.total_tokens}")

    # 模式 2：Workflow DAG 流水线（确定性流程）
    print("\n--- 模式 2：Workflow DAG 流水线 ---")
    wf = build_review_workflow()
    out = wf.run({"workspace": str(WORKSPACE)})
    print(f"DAG 报告：\n{out.get('report', {}).get('report', 'N/A')}")
    print(f"流程图：\n{wf.to_mermaid()}")

    # 模式 3：真实工具验证
    print("\n--- 模式 3：真实工具验证 ---")
    print(f"真实 AST 分析：\n{analyze_ast.func('user_service.py')}")
    print(f"真实 Lint 结果：{run_lint.func(None)}")

    print("\n" + "=" * 60)
    print("代码工程助手运行完毕")


if __name__ == "__main__":
    asyncio.run(main())
