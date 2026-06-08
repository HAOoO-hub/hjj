# panel.py：构建4.2四元式到LLVM IR转换器的图形界面页签
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from shared.gui_base import create_labeled_frame, create_paned, create_text
from shared.quads import parse_quads
from task_4_2_quad_to_llvm.translator import LLVMTranslator, describe_blocks


class LLVMPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self._build()
        self._load_default()

    def _build(self) -> None:
        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(tools, text="转换", command=self.translate).pack(side="left", padx=4)

        pane = create_paned(self)
        frames = [create_labeled_frame(pane, title) for title in ("四元式输入", "基本块", "LLVM IR")]
        for frame in frames:
            pane.add(frame, weight=1)
        self.source = create_text(frames[0], height=24)
        self.source.pack(fill="both", expand=True)
        self.blocks = create_text(frames[1], height=24)
        self.blocks.pack(fill="both", expand=True)
        self.output = create_text(frames[2], height=24)
        self.output.pack(fill="both", expand=True)

    def _load_default(self) -> None:
        lines = Path("data/quads/llvm_case_01.txt").read_text(encoding="utf-8").splitlines()
        self.source.insert("1.0", "\n".join(line for line in lines if not line.strip().startswith("#")))

    def translate(self) -> None:
        try:
            quads = parse_quads(self.source.get("1.0", "end").strip())
            translator = LLVMTranslator(quads)
            self.blocks.delete("1.0", "end")
            self.blocks.insert("1.0", describe_blocks(translator.blocks))
            self.output.delete("1.0", "end")
            self.output.insert("1.0", translator.translate())
        except Exception as exc:
            messagebox.showerror("LLVM 转换错误", str(exc))
