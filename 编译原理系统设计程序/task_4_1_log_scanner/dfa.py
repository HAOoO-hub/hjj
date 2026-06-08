# dfa.py：定义日志扫描器中DFA和最小DFA的数据结构
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DFA:
    start: str
    accepts: set[str]
    transitions: dict[str, dict[str, str]]
    state_sets: dict[str, frozenset[int]]

