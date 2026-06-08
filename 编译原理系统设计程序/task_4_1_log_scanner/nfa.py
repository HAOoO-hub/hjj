# nfa.py：定义日志扫描器中NFA的状态与边结构
from __future__ import annotations

from dataclasses import dataclass, field


EPSILON = "ε"


@dataclass
class NFAState:
    sid: int
    transitions: dict[str, set[int]] = field(default_factory=dict)

    def add(self, symbol: str, target: int) -> None:
        self.transitions.setdefault(symbol, set()).add(target)


@dataclass
class NFA:
    states: dict[int, NFAState]
    start: int
    accept: int

