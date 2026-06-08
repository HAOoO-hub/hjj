# panel.py：构建4.1日志关键词扫描器的图形界面页签
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from shared.gui_base import create_labeled_frame, create_paned, create_text
from task_4_1_log_scanner.automaton_view import AutomatonView
from task_4_1_log_scanner.minimize import minimize_dfa
from task_4_1_log_scanner.regex_parser import parse_regex
from task_4_1_log_scanner.regex_spec import PRIORITY, REGEX_SPECS
from task_4_1_log_scanner.scanner import scan_text
from task_4_1_log_scanner.subset import nfa_to_dfa
from task_4_1_log_scanner.thompson import regex_to_nfa


class LogScannerPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self.compiled: dict[str, tuple] = {}
        self._build()
        self._load_default()
        self.compile_all()

    def _build(self) -> None:
        tools = ttk.Frame(self)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(tools, text="扫描", command=self.scan_logs).pack(side="left", padx=4)
        ttk.Button(tools, text="重新构造自动机", command=self.compile_all).pack(side="left", padx=4)
        self.category_var = tk.StringVar(value=PRIORITY[0])
        combo = ttk.Combobox(tools, textvariable=self.category_var, values=PRIORITY, width=10, state="readonly")
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.render_current())

        pane = create_paned(self, "vertical")
        upper = create_paned(pane)
        pane.add(upper, weight=2)
        lower = create_paned(pane)
        pane.add(lower, weight=3)

        log_frame = create_labeled_frame(upper, "日志输入")
        result_frame = create_labeled_frame(upper, "提取结果")
        upper.add(log_frame, weight=3)
        upper.add(result_frame, weight=2)
        self.source = create_text(log_frame, height=10)
        self.source.pack(fill="both", expand=True)
        self.result = create_text(result_frame, height=10)
        self.result.pack(fill="both", expand=True)

        regex_frame = create_labeled_frame(lower, "正则规则")
        view_frame = create_labeled_frame(lower, "自动机展示")
        lower.add(regex_frame, weight=2)
        lower.add(view_frame, weight=3)
        self.regex_text = create_text(regex_frame, height=14)
        self.regex_text.pack(fill="both", expand=True)
        self.auto_view = AutomatonView(view_frame)
        self.auto_view.pack(fill="both", expand=True)

    def _load_default(self) -> None:
        lines = Path("data/logs/sample_log_01.txt").read_text(encoding="utf-8").splitlines()
        self.source.insert("1.0", "\n".join(line for line in lines if not line.strip().startswith("#")))
        self.regex_text.insert("1.0", "\n".join(f"{k} = {v}" for k, v in REGEX_SPECS.items()))

    def compile_all(self) -> None:
        try:
            specs = self._parse_specs()
            self.compiled.clear()
            for name, pattern in specs.items():
                regex = parse_regex(pattern)
                nfa, nfa_logs = regex_to_nfa(regex)
                dfa, dfa_logs = nfa_to_dfa(nfa)
                min_dfa, min_logs = minimize_dfa(dfa)
                logs = [f"[{name}] 正则：{pattern}", *nfa_logs, *dfa_logs, *min_logs]
                self.compiled[name] = (nfa, dfa, min_dfa, logs)
            self.render_current()
        except Exception as exc:
            messagebox.showerror("自动机构造错误", str(exc))

    def _parse_specs(self) -> dict[str, str]:
        specs: dict[str, str] = {}
        for line in self.regex_text.get("1.0", "end").splitlines():
            text = line.strip()
            if not text or "=" not in text:
                continue
            key, value = text.split("=", 1)
            specs[key.strip()] = value.strip()
        return specs

    def render_current(self) -> None:
        name = self.category_var.get()
        if name in self.compiled:
            self.auto_view.render(*self.compiled[name])

    def scan_logs(self) -> None:
        try:
            if not self.compiled:
                self.compile_all()
            dfas = {name: item[2] for name, item in self.compiled.items()}
            matches = scan_text(self.source.get("1.0", "end").rstrip("\n"), dfas)
            self.result.delete("1.0", "end")
            lines = [f"第{m.line_no}行 [{m.category}] {m.text}" for m in matches]
            self.result.insert("1.0", "\n".join(lines) if lines else "未识别到关键词")
            self.render_current()
        except Exception as exc:
            messagebox.showerror("日志扫描错误", str(exc))
