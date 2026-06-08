# log_workspace.py：构建日志扫描工作台并展示自动机图与转移表
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from shared.gui_base import create_text
from task_4_1_log_scanner.minimize import minimize_dfa
from task_4_1_log_scanner.nfa import NFA
from task_4_1_log_scanner.regex_parser import parse_regex
from task_4_1_log_scanner.regex_spec import PRIORITY, REGEX_SPECS
from task_4_1_log_scanner.scanner import scan_text
from task_4_1_log_scanner.subset import nfa_to_dfa
from task_4_1_log_scanner.thompson import regex_to_nfa
from task_4_1_log_scanner.graphviz_renderer import nfa_to_dot, dfa_to_dot, render_dot_to_tk

# Global scale for zoom (since we're not using graph_drawer anymore)
_graph_scale = 0.475  # Default scale (95% of 0.5) for more compact initial view

def get_graph_scale():
    return _graph_scale

def set_graph_scale(scale):
    global _graph_scale
    _graph_scale = scale


def compress_symbols(symbols: list[str]) -> str:
    """Compress a list of symbols into a readable range format."""
    if not symbols:
        return ""
    if len(symbols) == 1:
        return symbols[0]
    
    # Separate single chars from multi-char symbols (like ε)
    single_chars = []
    others = []
    for sym in symbols:
        if len(sym) == 1:
            single_chars.append(sym)
        else:
            others.append(sym)
    
    result_parts = []
    
    # Compress consecutive single characters into ranges
    if single_chars:
        single_chars.sort()
        ranges = []
        start = single_chars[0]
        end = single_chars[0]
        
        for ch in single_chars[1:]:
            if ord(ch) == ord(end) + 1:
                end = ch
            else:
                if start == end:
                    ranges.append(start)
                else:
                    ranges.append(f"{start}-{end}")
                start = ch
                end = ch
        
        if start == end:
            ranges.append(start)
        else:
            ranges.append(f"{start}-{end}")
        
        result_parts.extend(ranges)
    
    if others:
        result_parts.extend(sorted(others))
    
    return ', '.join(result_parts)


def nfa_to_graph(nfa: NFA) -> dict[str, object]:
    """Convert NFA to graph format with compressed character ranges."""
    transitions: dict[str, dict[str, str]] = {}
    labels = {str(sid): str(sid) for sid in nfa.states}
    
    for sid, state in nfa.states.items():
        src = str(sid)
        transitions.setdefault(src, {})
        
        # Group symbols by target to enable compression
        target_to_symbols: dict[str, list[str]] = {}
        for symbol, targets in state.transitions.items():
            for target in targets:
                target_str = str(target)
                if target_str not in target_to_symbols:
                    target_to_symbols[target_str] = []
                target_to_symbols[target_str].append(symbol)
        
        # Create edges with compressed labels
        for target_str, symbols in target_to_symbols.items():
            if len(symbols) <= 3:
                label = ', '.join(sorted(symbols))
            else:
                label = compress_symbols(symbols)
            transitions[src][label] = target_str
    
    return {"start": str(nfa.start), "accepts": {str(nfa.accept)}, "transitions": transitions, "labels": labels}


class LogWorkspace(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self.scale = 1.0
        self.compiled: dict[str, tuple[NFA, object, object, list[str]]] = {}
        self.current_category = tk.StringVar(value=PRIORITY[0])
        self.current_graph = tk.StringVar(value="NFA")
        self._build()
        self._load_default()
        self.compile_all()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="扫描日志", command=self.scan_logs).pack(side="left", padx=4)
        ttk.Button(top, text="重建自动机", command=self.compile_all).pack(side="left", padx=4)
        category_box = ttk.Combobox(top, textvariable=self.current_category, values=PRIORITY, state="readonly", width=10)
        category_box.pack(side="left", padx=4)
        category_box.bind("<<ComboboxSelected>>", lambda _e: self.render_current())
        graph_box = ttk.Combobox(top, textvariable=self.current_graph, values=["NFA", "DFA", "最小DFA"], state="readonly", width=10)
        graph_box.pack(side="left", padx=4)
        graph_box.bind("<<ComboboxSelected>>", lambda _e: self.render_current())
        ttk.Button(top, text="放大", command=lambda: self.zoom(1.2)).pack(side="left", padx=4)
        ttk.Button(top, text="缩小", command=lambda: self.zoom(1 / 1.2)).pack(side="left", padx=4)
        ttk.Button(top, text="重置", command=self.reset_zoom).pack(side="left", padx=4)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        # Left pane: 日志输入 + 关键词输出
        left_pane = ttk.Panedwindow(left, orient="vertical")
        left_pane.pack(fill="both", expand=True)
        log_frame = ttk.LabelFrame(left_pane, text="日志输入", padding=6)
        result_frame = ttk.LabelFrame(left_pane, text="关键词输出", padding=6)
        left_pane.add(log_frame, weight=2)
        left_pane.add(result_frame, weight=1)

        self.log_text = create_text(log_frame, height=14)
        self.log_text.pack(fill="both", expand=True)
        self.result_text = create_text(result_frame, height=8)
        self.result_text.pack(fill="both", expand=True)

        # Right pane: 自动机图区域 + 正则规则
        right_pane = ttk.Panedwindow(right, orient="vertical")
        right_pane.pack(fill="both", expand=True)
        graph_frame = ttk.Frame(right_pane)
        rule_frame = ttk.LabelFrame(right_pane, text="正则规则", padding=6)
        right_pane.add(graph_frame, weight=3)
        right_pane.add(rule_frame, weight=1)

        # Graph tabs inside graph_frame
        notebook = ttk.Notebook(graph_frame)
        notebook.pack(fill="both", expand=True)
        graph_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        log_tab = ttk.Frame(notebook)
        notebook.add(graph_tab, text="自动机图")
        notebook.add(table_tab, text="转移表")
        notebook.add(log_tab, text="构造过程")

        # Graph area setup using Grid for robust layout
        graph_wrap = ttk.Frame(graph_tab)
        graph_wrap.pack(fill="both", expand=True)
        
        # Configure grid: Row 0 (Canvas) expands, Row 1 (Scrollbar) is minimal
        graph_wrap.grid_rowconfigure(0, weight=1)
        graph_wrap.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(graph_wrap, bg="#fafbfc")
        xbar = ttk.Scrollbar(graph_wrap, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(graph_wrap, orient="vertical", command=self.canvas.yview)
        
        # Connect scrollbars to canvas
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        
        # Grid placement:
        # Canvas: Top-Left, fills all available space
        # Ybar: Right edge, spans canvas height
        # Xbar: Bottom edge, spans FULL width (both columns)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, columnspan=2, sticky="ew")  # Span both columns!
        
        # Enable mouse wheel scrolling for the canvas
        def on_vertical_scroll(event):
            """Handle vertical scrolling with mouse wheel."""
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"
        
        def on_horizontal_scroll(event):
            """Handle horizontal scrolling with Shift+MouseWheel."""
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
            return "break"
        
        self.canvas.bind("<MouseWheel>", on_vertical_scroll)
        self.canvas.bind("<Shift-MouseWheel>", on_horizontal_scroll)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        self.table_text = create_text(table_tab, height=30)
        self.table_text.pack(fill="both", expand=True)
        self.log_output = create_text(log_tab, height=30)
        self.log_output.pack(fill="both", expand=True)

        # Regex rules text at bottom of right pane
        self.rule_text = create_text(rule_frame, height=6)
        self.rule_text.pack(fill="both", expand=True)

    def _load_default(self) -> None:
        path = Path("data/logs/sample_log_01.txt")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.log_text.insert("1.0", "\n".join(line for line in lines if not line.strip().startswith("#")))
        self.rule_text.insert("1.0", "\n".join(f"{k} = {v}" for k, v in REGEX_SPECS.items()))

    def parse_specs(self) -> dict[str, str]:
        specs: dict[str, str] = {}
        for line in self.rule_text.get("1.0", "end").splitlines():
            text = line.strip()
            if not text or "=" not in text:
                continue
            key, value = text.split("=", 1)
            specs[key.strip()] = value.strip()
        return specs

    def compile_all(self) -> None:
        specs = self.parse_specs()
        self.compiled.clear()
        for name, pattern in specs.items():
            regex = parse_regex(pattern)
            nfa, nfa_logs = regex_to_nfa(regex)
            dfa, dfa_logs = nfa_to_dfa(nfa)
            min_dfa, min_logs = minimize_dfa(dfa)
            self.compiled[name] = (nfa, dfa, min_dfa, [f"[{name}] {pattern}", *nfa_logs, *dfa_logs, *min_logs])
        self.render_current()

    def current_bundle(self):
        return self.compiled[self.current_category.get()]

    def render_current(self) -> None:
        if self.current_category.get() not in self.compiled:
            return
        
        nfa, dfa, min_dfa, logs = self.current_bundle()
        graph_kind = self.current_graph.get()
        
        # ALWAYS use scale=1.0 for high-quality PNG generation
        # _graph_scale controls DISPLAY size via PIL, not Graphviz generation
        if graph_kind == "NFA":
            nfa_dot = nfa_to_dot(nfa, scale=1.0)
            render_dot_to_tk(nfa_dot, self.canvas, display_scale=_graph_scale)
        elif graph_kind == "DFA":
            dfa_dot = dfa_to_dot(dfa, scale=1.0)
            render_dot_to_tk(dfa_dot, self.canvas, display_scale=_graph_scale)
        else:
            min_dfa_dot = dfa_to_dot(min_dfa, scale=1.0)
            render_dot_to_tk(min_dfa_dot, self.canvas, display_scale=_graph_scale)
        
        # Update scrollregion to match the actual image size after rendering
        self.canvas.update_idletasks()
        img = getattr(self.canvas, '_graph_image', None)
        if img:
            # Set scrollregion to allow scrolling the full image
            self.canvas.config(scrollregion=(0, 0, img.width(), img.height()))
        
        # Update table and logs (keep existing logic)
        if graph_kind == "NFA":
            graph = nfa_to_graph(nfa)
            self.table_text.delete("1.0", "end")
            self.table_text.insert("1.0", format_transition_text("NFA", graph["transitions"], graph["labels"]))
        elif graph_kind == "DFA":
            labels = {state: state for state in dfa.transitions}
            compressed_trans = compress_transitions(dfa.transitions)
            self.table_text.delete("1.0", "end")
            self.table_text.insert("1.0", format_transition_text("DFA", compressed_trans, labels))
        else:
            labels = {state: state for state in min_dfa.transitions}
            compressed_trans = compress_transitions(min_dfa.transitions)
            self.table_text.delete("1.0", "end")
            self.table_text.insert("1.0", format_transition_text("最小DFA", compressed_trans, labels))
        
        self.log_output.delete("1.0", "end")
        self.log_output.insert("1.0", "\n".join(logs))

    def scan_logs(self) -> None:
        if not self.compiled:
            self.compile_all()
        matches = scan_text(self.log_text.get("1.0", "end-1c"), {name: bundle[2] for name, bundle in self.compiled.items()})
        self.result_text.delete("1.0", "end")
        lines = [f"行{m.line_no} 列{m.start + 1}-{m.end}  {m.text:<20} {m.category}" for m in matches]
        self.result_text.insert("1.0", "\n".join(lines) if lines else "未识别到关键词")
        self.render_current()

    def zoom(self, factor: float) -> None:
        """Zoom in/out with smooth incremental steps."""
        current = get_graph_scale()
        # Extended range: 0.03x to 8.0x for better control
        new_scale = max(0.03, min(8.0, current * factor))
        set_graph_scale(new_scale)
        print(f"[Zoom] Scale: {current:.3f} → {new_scale:.3f} (factor: {factor:.2f})")
        self.render_current()

    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        set_graph_scale(1.0)
        print("[Zoom] Reset to 1.0")
        self.render_current()


def compress_transitions(transitions: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Compress DFA transitions by grouping symbols that go to the same target."""
    compressed: dict[str, dict[str, str]] = {}
    
    for src, trans_dict in transitions.items():
        # Group symbols by target
        target_to_symbols: dict[str, list[str]] = {}
        for symbol, dst in trans_dict.items():
            # Clean the symbol (remove # suffix if any)
            clean_symbol = symbol.split("#", 1)[0]
            if dst not in target_to_symbols:
                target_to_symbols[dst] = []
            target_to_symbols[dst].append(clean_symbol)
        
        # Create compressed transitions
        compressed[src] = {}
        for dst, symbols in target_to_symbols.items():
            if len(symbols) <= 3:
                label = ', '.join(sorted(symbols))
            else:
                label = compress_symbols(symbols)
            compressed[src][label] = dst
    
    return compressed


def format_transition_text(title: str, transitions: dict[str, dict[str, str]], labels: dict[str, str]) -> str:
    lines = [title]
    for state in sorted(transitions):
        lines.append(f"{labels.get(state, state)}")
        for symbol, target in sorted(transitions[state].items()):
            clean_symbol = symbol.split("#", 1)[0]
            lines.append(f"  {clean_symbol:<8} -> {labels.get(target, target)}")
    return "\n".join(lines)
