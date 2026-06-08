# executor.py：驱动中间代码解释器逐条执行四元式
from __future__ import annotations

from shared.quads import Quad
from task_3_2_interpreter.operations import execute_quad
from task_3_2_interpreter.runtime import RuntimeState


class QuadExecutor:
    def __init__(self, programs: dict[str, list[Quad]], params_map: dict[str, list[str]] | None = None) -> None:
        self.programs = programs
        self.params_map = params_map or {}
        main_quads = programs.get("main", next(iter(programs.values()), []))
        self.state = RuntimeState()
        self.state.quads = main_quads
        self.state.program = programs
        self.state.index_map = {quad.idx: pos for pos, quad in enumerate(main_quads)}
        self.state.params_map = self.params_map

    def step(self) -> RuntimeState:
        if self.state.halted or self.state.pc >= len(self.state.quads):
            self.state.halted = True
            return self.state
        quad = self.state.quads[self.state.pc]
        jump_to = execute_quad(self.state, quad, self.state.index_map)
        if self.state.halted:
            return self.state
        self.state.pc = jump_to if jump_to is not None else self.state.pc + 1
        return self.state

    @property
    def quads(self) -> list[Quad]:
        return self.state.quads

    def run(self, max_steps: int = 10_000) -> RuntimeState:
        steps = 0
        while not self.state.halted and self.state.pc < len(self.state.quads):
            self.step()
            steps += 1
            if steps > max_steps:
                raise RuntimeError("执行步数超限，可能存在死循环")
        return self.state
