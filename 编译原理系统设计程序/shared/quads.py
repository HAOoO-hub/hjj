# quads.py：解析和表示四元式中间代码
from __future__ import annotations

import re
from dataclasses import dataclass


LINE_RE = re.compile(
    r"^\s*(?:(?P<idx>\d+)\s*[:：]\s*)?\(\s*(?P<op>[^,]+)\s*,\s*(?P<arg1>[^,]*)\s*,\s*(?P<arg2>[^,]*)\s*,\s*(?P<result>[^)]*)\s*\)\s*$"
)

TEMP_RE = re.compile(r"^[tT]\d+$")


@dataclass#(slots=True)
class Quad:
    idx: int
    op: str
    arg1: str
    arg2: str
    result: str

    def format(self) -> str:
        return f"{self.idx}: ({self.op}, {self.arg1}, {self.arg2}, {self.result})"


def parse_quads(text: str) -> list[Quad]:
    quads: list[Quad] = []
    next_idx = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        match = LINE_RE.match(line)
        if not match:
            raise ValueError(f"无法解析四元式：{raw}")
        idx = int(match.group("idx")) if match.group("idx") is not None else next_idx
        quads.append(
            Quad(
                idx=idx,
                op=match.group("op").strip(),
                arg1=match.group("arg1").strip(),
                arg2=match.group("arg2").strip(),
                result=match.group("result").strip(),
            )
        )
        next_idx = idx + 1
    return quads


def is_placeholder(value: str) -> bool:
    return value in {"", "_"}


def is_number(value: str) -> bool:
    if is_placeholder(value):
        return False
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_temp(value: str) -> bool:
    return bool(TEMP_RE.fullmatch(value))


def is_identifier(value: str) -> bool:
    return bool(value) and not is_number(value) and not is_placeholder(value)
