from __future__ import annotations

import pytest

from morainet.exceptions import ToolNotFoundError, ToolValidationError
from morainet.tools import Tool, ToolRegistry, tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称
        unit: 温度单位
    """
    return f"{city}:26"


def test_schema_generation():
    schema = get_weather.schema
    assert schema["name"] == "get_weather"
    assert schema["description"].startswith("查询")
    props = schema["parameters"]["properties"]
    assert props["city"]["type"] == "string"
    assert props["city"]["description"] == "城市名称"
    assert props["unit"]["default"] == "celsius"
    assert schema["parameters"]["required"] == ["city"]


async def test_invoke_ok():
    assert await get_weather.invoke({"city": "上海"}) == "上海:26"


async def test_invoke_missing_required():
    with pytest.raises(ToolValidationError):
        await get_weather.invoke({})


def test_registry_lookup():
    reg = ToolRegistry([get_weather])
    assert reg.get("get_weather") is get_weather
    assert len(reg) == 1
    with pytest.raises(ToolNotFoundError):
        reg.get("nope")


async def test_tool_from_schema():
    seen = {}

    async def invoke(**kwargs):
        seen.update(kwargs)
        return "remote-result"

    t = Tool.from_schema(
        name="remote",
        description="a remote tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        invoke=invoke,
    )
    assert t.name == "remote"
    assert t.required_params == ["x"]
    assert await t.invoke({"x": "hi"}) == "remote-result"
    assert seen == {"x": "hi"}

    with pytest.raises(ToolValidationError):
        await t.invoke({})


def test_optional_and_list_types():
    @tool
    def f(a: int, b: list[str], c: float | None = None) -> str:
        """do.

        Args:
            a: x
            b: y
            c: z
        """
        return "ok"

    props = f.schema["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "array"
    assert props["b"]["items"]["type"] == "string"
    assert props["c"]["type"] == "number"
    assert f.schema["parameters"]["required"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Registry __bool__
# ---------------------------------------------------------------------------

def test_registry_bool():
    assert bool(ToolRegistry()) is False
    reg = ToolRegistry([get_weather])
    assert bool(reg) is True


# ---------------------------------------------------------------------------
# Tool __call__ passthrough
# ---------------------------------------------------------------------------

def test_tool_callable():
    t = Tool(get_weather.func)
    result = t("Beijing")
    assert "Beijing" in str(result)


# ---------------------------------------------------------------------------
# Schema edge cases
# ---------------------------------------------------------------------------

def test_schema_no_annotation():
    @tool
    def f(x) -> str:
        """No annotation."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert props["x"]["type"] == "string"


def test_schema_union_multiple_types():
    @tool
    def f(x: str | int) -> str:
        """Union with multiple non-None."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert props["x"]["type"] == "string"


def test_schema_dict_type():
    @tool
    def f(x: dict[str, str]) -> str:
        """Process dict."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert props["x"]["type"] == "object"


def test_schema_no_docstring():
    @tool
    def f(x: str) -> str:
        return "ok"
    assert f.schema["description"] == "f"


def test_schema_docstring_with_raises():
    @tool
    def f(x: str) -> str:
        """Do something.

        Args:
            x: the input

        Raises:
            ValueError: if invalid
        """
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert props["x"]["type"] == "string"
    assert props["x"]["description"] == "the input"


def test_schema_skip_self_param():
    @tool
    def f(self, x: str) -> str:
        """Method-like."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert "self" not in props
    assert "x" in props


def test_schema_skip_var_positional():
    @tool
    def f(x: str, *args: str) -> str:
        """With *args."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert "args" not in props
    assert "x" in props


def test_schema_skip_var_keyword():
    @tool
    def f(x: str, **kwargs: str) -> str:
        """With **kwargs."""
        return "ok"
    props = f.schema["parameters"]["properties"]
    assert "kwargs" not in props
    assert "x" in props


def test_schema_get_type_hints_fallback():
    @tool
    def f(x: "UndefinedClassXYZ") -> str:
        """Bad annotation."""
        return "ok"
    # get_type_hints fails → falls back to raw annotation string → defaults to "string"
    props = f.schema["parameters"]["properties"]
    assert props["x"]["type"] == "string"
