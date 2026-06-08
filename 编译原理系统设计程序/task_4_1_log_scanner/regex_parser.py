# regex_parser.py：解析日志扫描器需要的简化正则表达式
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegexNode:
    kind: str
    value: str | set[str] | None = None
    children: list["RegexNode"] | None = None


class RegexParser:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0

    def parse(self) -> RegexNode:
        node = self.parse_union()
        if self.pos != len(self.pattern):
            raise ValueError(f"正则未完全解析：{self.pattern[self.pos:]}")
        return node

    def peek(self) -> str | None:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def take(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def parse_union(self) -> RegexNode:
        parts = [self.parse_concat()]
        while self.peek() == "|":
            self.take()
            parts.append(self.parse_concat())
        if len(parts) == 1:
            return parts[0]
        return RegexNode("union", children=parts)

    def parse_concat(self) -> RegexNode:
        items: list[RegexNode] = []
        while self.peek() and self.peek() not in ")|":
            items.append(self.parse_repeat())
        if not items:
            return RegexNode("epsilon")
        if len(items) == 1:
            return items[0]
        return RegexNode("concat", children=items)

    def parse_repeat(self) -> RegexNode:
        node = self.parse_atom()
        while self.peek() in {"*", "+", "?", "{"}:
            ch = self.take()
            if ch == "*":
                node = RegexNode("star", children=[node])
            elif ch == "+":
                node = RegexNode("plus", children=[node])
            elif ch == "?":
                node = RegexNode("optional", children=[node])
            else:
                node = self.parse_brace_repeat(node)
        return node

    def parse_brace_repeat(self, node: RegexNode) -> RegexNode:
        start = ""
        while self.peek() and self.peek().isdigit():
            start += self.take()
        if not start:
            raise ValueError("重复次数缺少下界")
        low = int(start)
        high = low
        if self.peek() == ",":
            self.take()
            end = ""
            while self.peek() and self.peek().isdigit():
                end += self.take()
            high = int(end) if end else low
        if self.peek() != "}":
            raise ValueError("重复次数缺少右花括号")
        self.take()
        nodes = [node for _ in range(low)]
        extra = high - low
        for _ in range(extra):
            nodes.append(RegexNode("optional", children=[node]))
        if not nodes:
            return RegexNode("epsilon")
        if len(nodes) == 1:
            return nodes[0]
        return RegexNode("concat", children=nodes)

    def parse_atom(self) -> RegexNode:
        ch = self.peek()
        if ch is None:
            return RegexNode("epsilon")
        if ch == "(":
            self.take()
            node = self.parse_union()
            if self.peek() != ")":
                raise ValueError("缺少右括号")
            self.take()
            return node
        if ch == "[":
            return RegexNode("charset", value=self.parse_charset())
        return RegexNode("literal", value=self.take())

    def parse_charset(self) -> set[str]:
        self.take()
        chars: set[str] = set()
        raw: list[str] = []
        while self.peek() is not None and self.peek() != "]":
            raw.append(self.take())
        if self.peek() != "]":
            raise ValueError("字符类缺少右中括号")
        self.take()
        i = 0
        while i < len(raw):
            ch = raw[i]
            if i + 2 < len(raw) and raw[i + 1] == "-" and i not in {len(raw) - 1} and i + 2 != len(raw):
                left, right = ch, raw[i + 2]
                for code in range(ord(left), ord(right) + 1):
                    chars.add(chr(code))
                i += 3
            else:
                chars.add(ch)
                i += 1
        return chars


def parse_regex(pattern: str) -> RegexNode:
    return RegexParser(pattern).parse()

