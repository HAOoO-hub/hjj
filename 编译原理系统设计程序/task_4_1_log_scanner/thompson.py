# thompson.py：把简化正则语法树转换为等价NFA
from __future__ import annotations

from dataclasses import dataclass

from task_4_1_log_scanner.nfa import EPSILON, NFA, NFAState
from task_4_1_log_scanner.regex_parser import RegexNode


@dataclass
class Fragment:
    start: int
    accept: int


class ThompsonBuilder:
    def __init__(self) -> None:
        self.states: dict[int, NFAState] = {}
        self.counter = 0
        self.logs: list[str] = []

    def new_state(self) -> int:
        sid = self.counter
        self.counter += 1
        self.states[sid] = NFAState(sid=sid)
        return sid

    def link(self, src: int, symbol: str, dst: int) -> None:
        self.states[src].add(symbol, dst)

    def build(self, node: RegexNode) -> tuple[NFA, list[str]]:
        fragment = self.visit(node)
        self.logs.append(f"构造完成：start={fragment.start}, accept={fragment.accept}")
        return NFA(self.states, fragment.start, fragment.accept), self.logs

    def visit(self, node: RegexNode) -> Fragment:
        kind = node.kind
        if kind == "literal":
            s, t = self.new_state(), self.new_state()
            self.link(s, str(node.value), t)
            self.logs.append(f"字面量 {node.value}: {s} -{node.value}-> {t}")
            return Fragment(s, t)
        if kind == "charset":
            s, t = self.new_state(), self.new_state()
            for ch in sorted(node.value):  # type: ignore[arg-type]
                self.link(s, ch, t)
            self.logs.append(f"字符类 {sorted(node.value)}: {s} -> {t}")
            return Fragment(s, t)
        if kind == "epsilon":
            s, t = self.new_state(), self.new_state()
            self.link(s, EPSILON, t)
            return Fragment(s, t)
        if kind == "concat":
            children = node.children or []
            fragment = self.visit(children[0])
            for child in children[1:]:
                other = self.visit(child)
                self.link(fragment.accept, EPSILON, other.start)
                fragment = Fragment(fragment.start, other.accept)
            return fragment
        if kind == "union":
            s, t = self.new_state(), self.new_state()
            for child in node.children or []:
                frag = self.visit(child)
                self.link(s, EPSILON, frag.start)
                self.link(frag.accept, EPSILON, t)
            return Fragment(s, t)
        if kind == "star":
            inner = self.visit((node.children or [])[0])
            s, t = self.new_state(), self.new_state()
            self.link(s, EPSILON, inner.start)
            self.link(s, EPSILON, t)
            self.link(inner.accept, EPSILON, inner.start)
            self.link(inner.accept, EPSILON, t)
            return Fragment(s, t)
        if kind == "plus":
            inner = self.visit((node.children or [])[0])
            s, t = self.new_state(), self.new_state()
            self.link(s, EPSILON, inner.start)
            self.link(inner.accept, EPSILON, inner.start)
            self.link(inner.accept, EPSILON, t)
            return Fragment(s, t)
        if kind == "optional":
            inner = self.visit((node.children or [])[0])
            s, t = self.new_state(), self.new_state()
            self.link(s, EPSILON, inner.start)
            self.link(s, EPSILON, t)
            self.link(inner.accept, EPSILON, t)
            return Fragment(s, t)
        raise ValueError(f"未知节点类型：{kind}")


def regex_to_nfa(node: RegexNode) -> tuple[NFA, list[str]]:
    return ThompsonBuilder().build(node)

