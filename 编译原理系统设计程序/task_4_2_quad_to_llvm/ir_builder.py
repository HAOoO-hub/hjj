# ir_builder.py：封装LLVM IR文本生成和虚拟寄存器分配
from __future__ import annotations


class IRBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counter = 0
        self.cmp_counter = 0

    def emit(self, line: str = "") -> None:
        self.lines.append(line)

    def new_value(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    def new_cmp(self) -> str:
        self.cmp_counter += 1
        return f"%cmp{self.cmp_counter}"

    def build(self) -> str:
        return "\n".join(self.lines)

