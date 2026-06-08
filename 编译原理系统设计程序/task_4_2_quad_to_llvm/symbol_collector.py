# symbol_collector.py：收集LLVM IR生成所需的变量和标签信息
from __future__ import annotations

from shared.quads import Quad, is_identifier, is_placeholder, is_temp


def collect_symbols(quads: list[Quad]) -> tuple[list[str], set[int]]:
    variables: set[str] = set()
    labels: set[int] = set()
    for quad in quads:
        for token in (quad.arg1, quad.arg2, quad.result):
            if is_identifier(token) and not is_temp(token):
                variables.add(token)
        if quad.op.startswith("J") and not is_placeholder(quad.result) and quad.result.isdigit():
            labels.add(int(quad.result))
    return sorted(variables), labels

