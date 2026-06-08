# diagnostics.py：整理IDE中的错误输出和修改建议文本
from __future__ import annotations

from shared.errors import Diagnostic


def render_diagnostics(items: list[Diagnostic]) -> str:
    if not items:
        return "未发现错误"
    return "\n".join(item.as_text() for item in items)

