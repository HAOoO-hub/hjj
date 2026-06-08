# runtime.py：维护中间代码解释器的运行时状态
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeState:
    pc: int = 0
    variables: dict[str, int] = field(default_factory=dict)
    temps: dict[str, int] = field(default_factory=dict)
    return_value: int | None = None
    halted: bool = False
    trace: list[str] = field(default_factory=list)

    call_stack: list[dict] = field(default_factory=list)
    pending_args: list[int] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.trace.append(message)

