# errors.py：定义全项目通用的错误和诊断数据结构
from __future__ import annotations

from dataclasses import dataclass


@dataclass#(slots=True)
class AppError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass#(slots=True)
class Diagnostic:
    category: str
    message: str
    line: int = 1
    column: int = 1
    suggestion: str = ""

    def as_text(self) -> str:
        tail = f" 建议：{self.suggestion}" if self.suggestion else ""
        return f"[{self.category}] 第{self.line}行第{self.column}列：{self.message}{tail}"

