# panel.py：构建4.3迷你IDE的图形界面页签并实现实时分析
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from shared.gui_base import create_labeled_frame, create_paned, create_text
from task_4_3_mini_ide.diagnostics import render_diagnostics
from task_4_3_mini_ide.formatter import format_code
from task_4_3_mini_ide.highlighter import apply_function_highlight, apply_lexical_highlight, configure_tags
from task_4_3_mini_ide.lexer import lex
from task_4_3_mini_ide.parser import parse_tokens
from task_4_3_mini_ide.semantic import analyze_semantics


class IDEPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self.after_id: str | None = None
        self._build()
        self._load_default()
        self.analyze()

    def _build(self) -> None:
        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(tools, text="格式化", command=self.reformat).pack(side="left", padx=4)
        ttk.Button(tools, text="立即分析", command=self.analyze).pack(side="left", padx=4)

        pane = create_paned(self)
        editor_frame = create_labeled_frame(pane, "代码编辑区")
        side_frame = create_labeled_frame(pane, "分析结果")
        pane.add(editor_frame, weight=3)
        pane.add(side_frame, weight=2)

        self.editor = create_text(editor_frame, height=26)
        self.editor.pack(fill="both", expand=True)
        configure_tags(self.editor)
        self.editor.bind("<KeyRelease>", self.on_key_release)

        side_pane = create_paned(side_frame, "vertical")
        diag_frame = create_labeled_frame(side_pane, "错误与建议")
        info_frame = create_labeled_frame(side_pane, "Token / AST / 符号表")
        side_pane.add(diag_frame, weight=2)
        side_pane.add(info_frame, weight=2)

        self.diag_text = create_text(diag_frame, height=10)
        self.diag_text.pack(fill="both", expand=True)
        self.info_text = create_text(info_frame, height=12)
        self.info_text.pack(fill="both", expand=True)

    def _load_default(self) -> None:
        self.editor.insert("1.0", Path("data/code/ide_ok_01.cmini").read_text(encoding="utf-8"))

    def on_key_release(self, _event=None) -> None:
        tokens, lex_errors = lex(self.editor.get("1.0", "end-1c"))
        apply_lexical_highlight(self.editor, tokens)
        self.diag_text.delete("1.0", "end")
        if lex_errors:
            self.diag_text.insert("1.0", render_diagnostics(lex_errors))
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.after_id = self.after(300, self.analyze)

    def reformat(self) -> None:
        formatted = format_code(self.editor.get("1.0", "end-1c"))
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", formatted)
        self.analyze()

    def analyze(self) -> None:
        self.after_id = None
        text = self.editor.get("1.0", "end-1c")
        tokens, lex_errors = lex(text)
        apply_lexical_highlight(self.editor, tokens)
        root, parse_errors = parse_tokens(tokens)
        sem_errors, summary = analyze_semantics(root)
        if not parse_errors:
            apply_function_highlight(self.editor, root)
        all_errors = [*lex_errors, *parse_errors, *sem_errors]
        self.diag_text.delete("1.0", "end")
        self.diag_text.insert("1.0", render_diagnostics(all_errors))
        self.info_text.delete("1.0", "end")
        token_lines = [f"{tok.kind:<7} {tok.value}" for tok in tokens if tok.kind != "EOF"]
        info_lines = ["[Tokens]", *token_lines[:40], "", "[符号摘要]"]
        info_lines.extend(summary.get("functions", []))
        info_lines.extend(summary.get("variables", []))
        self.info_text.insert("1.0", "\n".join(info_lines))
