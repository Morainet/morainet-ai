"""Generate JSON Schema for a Python function from its signature + docstring."""

from __future__ import annotations

import inspect
import re
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

_PYTHON_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Optional[X] / X | None  (both typing.Union and PEP 604 X | None)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"type": "string"}

    if origin in (list, set, tuple):
        args = get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    return {"type": _PYTHON_TO_JSON.get(annotation, "string")}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {arg_name: description}) from a Google-style docstring."""
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    summary = lines[0].strip() if lines else ""

    arg_docs: dict[str, str] = {}
    in_args = False
    for line in lines[1:]:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Params|Parameters):$", stripped):
            in_args = True
            continue
        if in_args:
            if re.match(r"^(Returns|Raises|Yields|Examples?):$", stripped):
                break
            m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if m:
                arg_docs[m.group(1)] = m.group(2).strip()
    return summary, arg_docs


def generate_schema(func: Any) -> dict[str, Any]:
    """Build an OpenAI-style function schema for ``func``."""
    sig = inspect.signature(func)
    summary, arg_docs = _parse_docstring(inspect.getdoc(func))

    # Resolve string annotations (PEP 563 / ``from __future__ import annotations``).
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - fall back to raw annotations
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(name, param.annotation)
        schema = _json_type(annotation)
        if name in arg_docs:
            schema["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = param.default

        properties[name] = schema

    return {
        "name": func.__name__,
        "description": summary or func.__name__,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
