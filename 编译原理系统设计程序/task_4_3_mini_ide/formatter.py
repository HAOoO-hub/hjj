# formatter.py：根据花括号层级为类C代码生成简单自动缩进建议
from __future__ import annotations


def format_code(text: str) -> str:
    lines = text.splitlines()
    indent = 0
    result: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            result.append("")
            continue
        if stripped.startswith("}"):
            indent = max(0, indent - 1)
        result.append("    " * indent + stripped)
        if stripped.endswith("{"):
            indent += 1
    return "\n".join(result)

