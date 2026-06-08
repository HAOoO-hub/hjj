# panel.py：构建3.2中间代码解释器的图形界面页签
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from shared.gui_base import create_labeled_frame, create_paned, create_text
from shared.quads import parse_quads
from task_3_2_interpreter.executor import QuadExecutor


class InterpreterPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self.executor: QuadExecutor | None = None
        self._build()
        self._load_default()

    def _build(self) -> None:
        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(tools, text="运行", command=self.run_all).pack(side="left", padx=4)
        ttk.Button(tools, text="单步", command=self.step_once).pack(side="left", padx=4)
        ttk.Button(tools, text="重置", command=self.reset_executor).pack(side="left", padx=4)

        pane = create_paned(self)
        left = create_labeled_frame(pane, "四元式输入")
        right = create_labeled_frame(pane, "运行结果")
        pane.add(left, weight=3)
        pane.add(right, weight=2)

        self.source = create_text(left, height=24)
        self.source.pack(fill="both", expand=True)

        info = ttk.Frame(right)
        info.pack(fill="x")
        self.current_var = tk.StringVar(value="当前PC：-")
        self.return_var = tk.StringVar(value="返回值：-")
        ttk.Label(info, textvariable=self.current_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.return_var).pack(anchor="w")

        self.vars_text = create_text(right, height=8)
        self.vars_text.pack(fill="x", pady=6)
        self.trace_text = create_text(right, height=14)
        self.trace_text.pack(fill="both", expand=True)

    def _load_default(self) -> None:
        path = Path("data/quads/interpreter_case_01.txt")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.source.insert("1.0", "\n".join(line for line in lines if not line.strip().startswith("#")))

    def reset_executor(self) -> None:
        try:
            quads = parse_quads(self.source.get("1.0", "end").strip())
            self.executor = QuadExecutor(quads)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("解释器错误", str(exc))

    def refresh(self) -> None:
        if not self.executor:
            return
        state = self.executor.state
        current_idx = self.executor.quads[state.pc].idx if state.pc < len(self.executor.quads) else "-"
        self.current_var.set(f"当前PC：{current_idx}")
        self.return_var.set(f"返回值：{state.return_value if state.return_value is not None else '-'}")
        self.vars_text.delete("1.0", "end")
        lines = ["变量表："]
        for key, value in sorted(state.variables.items()):
            lines.append(f"{key} = {value}")
        for key, value in sorted(state.temps.items()):
            lines.append(f"{key} = {value}")
        self.vars_text.insert("1.0", "\n".join(lines))
        self.trace_text.delete("1.0", "end")
        self.trace_text.insert("1.0", "\n".join(state.trace))

    def step_once(self) -> None:
        if not self.executor:
            self.reset_executor()
        try:
            assert self.executor is not None
            self.executor.step()
            self.refresh()
        except Exception as exc:
            messagebox.showerror("解释器错误", str(exc))

    def run_all(self) -> None:
        if not self.executor:
            self.reset_executor()
        try:
            assert self.executor is not None
            self.executor.run()
            self.refresh()
        except Exception as exc:
            messagebox.showerror("解释器错误", str(exc))
