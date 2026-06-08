# ast.py：定义迷你IDE中语法分析使用的抽象语法树节点
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    kind: str
    value: str = ""
    children: list["Node"] = field(default_factory=list)
    line: int = 1
    column: int = 1

