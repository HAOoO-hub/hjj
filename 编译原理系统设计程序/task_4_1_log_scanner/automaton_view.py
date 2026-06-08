# automaton_view.py：展示NFA和DFA图、状态表与构造过程日志
from __future__ import annotations

# ⚠️ DEBUG MARKER - 如果看到这个说明文件已正确加载
import sys
print("\n" + "="*80, flush=True)
print("⚠️ [AUTOMATON_VIEW.PY LOADED SUCCESSFULLY!]", flush=True)
print("️ Graphviz renderer is ACTIVE!", flush=True)
print("="*80 + "\n", flush=True)
sys.stdout.flush()

import tkinter as tk
from tkinter import ttk

from task_4_1_log_scanner.dfa import DFA
from task_4_1_log_scanner.graph_drawer import draw_graph, set_graph_scale, get_graph_scale
from task_4_1_log_scanner.nfa import NFA
from task_4_1_log_scanner.graphviz_renderer import nfa_to_dot, dfa_to_dot, render_dot_to_tk


def compress_symbols(symbols: list[str]) -> str:
    """Compress a list of symbols into a readable range format."""
    if not symbols:
        return ""
    if len(symbols) == 1:
        return symbols[0]
    
    single_chars = []
    others = []
    for sym in symbols:
        if len(sym) == 1:
            single_chars.append(sym)
        else:
            others.append(sym)
    
    result_parts = []
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


def compress_transitions(transitions: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Compress DFA transitions by grouping symbols that go to the same target."""
    compressed: dict[str, dict[str, str]] = {}
    for src, trans_dict in transitions.items():
        target_to_symbols: dict[str, list[str]] = {}
        for symbol, dst in trans_dict.items():
            clean_symbol = symbol.split("#", 1)[0]
            if dst not in target_to_symbols:
                target_to_symbols[dst] = []
            target_to_symbols[dst].append(clean_symbol)
        compressed[src] = {}
        for dst, symbols in target_to_symbols.items():
            if len(symbols) <= 3:
                label = ', '.join(sorted(symbols))
            else:
                label = compress_symbols(symbols)
            compressed[src][label] = dst
    return compressed


def compress_symbols(symbols: list[str]) -> str:
    """Compress a list of symbols into a readable range format.
    
    Converts ['A', 'B', 'C', ..., 'Z'] into 'A-Z'
    Handles individual chars and special symbols like epsilon.
    """
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
        
        # Add last range
        if start == end:
            ranges.append(start)
        else:
            ranges.append(f"{start}-{end}")
        
        result_parts.extend(ranges)
    
    # Add other symbols (like ε)
    if others:
        result_parts.extend(sorted(others))
    
    return ', '.join(result_parts)


def nfa_transitions(nfa: NFA) -> dict[str, dict[str, str]]:
    """Convert NFA to transition graph for drawing.
    
    Groups multiple target states for the same symbol into a single edge
    to avoid cluttered labels like ε:2, ε:12. Instead shows ε → 2,12.
    Also compresses character ranges to avoid showing all characters separately.
    """
    graph: dict[str, dict[str, str]] = {}
    
    for sid, state in nfa.states.items():
        # Group targets by symbol
        symbol_to_targets: dict[str, list[int]] = {}
        for symbol, targets in state.transitions.items():
            if targets:
                symbol_to_targets[symbol] = sorted(targets)
        
        # Further group symbols that go to the same target(s)
        target_pattern_to_symbols: dict[tuple, list[str]] = {}
        for symbol, targets in symbol_to_targets.items():
            target_key = tuple(targets)
            if target_key not in target_pattern_to_symbols:
                target_pattern_to_symbols[target_key] = []
            target_pattern_to_symbols[target_key].append(symbol)
        
        # Create edges with compressed labels
        graph[str(sid)] = {}
        for target_tuple, symbols in target_pattern_to_symbols.items():
            if len(target_tuple) == 1:
                target_str = str(target_tuple[0])
            else:
                # Multiple targets - show them all separated by comma
                target_str = ','.join(str(t) for t in target_tuple)
            
            # Compress symbol label
            if len(symbols) == 1:
                label = symbols[0]
            else:
                # Multiple symbols going to same target - compress into ranges
                label = compress_symbols(symbols)
            
            graph[str(sid)][label] = target_str
    
    return graph


class AutomatonView(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.canvases: dict[str, tk.Canvas] = {}
        self.scrollbars_y: dict[str, ttk.Scrollbar] = {}
        self.scrollbars_x: dict[str, ttk.Scrollbar] = {}
        self.current_nfa = None
        self.current_dfa = None
        self.current_min_dfa = None
        self.current_logs = None
        
        for name in ("NFA图", "DFA图", "最小DFA图"):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            
            # Add zoom controls
            toolbar = ttk.Frame(frame)
            toolbar.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(toolbar, text="放大", command=lambda n=name: self.zoom_in(n), width=8).pack(side="left", padx=2)
            ttk.Button(toolbar, text="缩小", command=lambda n=name: self.zoom_out(n), width=8).pack(side="left", padx=2)
            ttk.Button(toolbar, text="重置", command=lambda n=name: self.zoom_reset(n), width=8).pack(side="left", padx=2)
            
            # Create canvas with scrollbars
            canvas = tk.Canvas(frame, bg="white", height=350)
            scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            
            # Pack canvas and scrollbars
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar_y.pack(side="right", fill="y")
            scrollbar_x.pack(side="bottom", fill="x")
            
            self.canvases[name] = canvas
            self.scrollbars_y[name] = scrollbar_y
            self.scrollbars_x[name] = scrollbar_x
            
            # Enable mouse wheel scrolling - bind to frame to avoid focus issues
            def on_vertical_scroll(event, c=canvas):
                """Handle vertical scrolling with mouse wheel."""
                c.yview_scroll(int(-1*(event.delta/120)), "units")
                return "break"
            
            def on_horizontal_scroll(event, c=canvas):
                """Handle horizontal scrolling with Shift+MouseWheel."""
                c.xview_scroll(int(-1*(event.delta/120)), "units")
                return "break"
            
            # Bind to both canvas and frame for better capture
            canvas.bind("<MouseWheel>", on_vertical_scroll)
            canvas.bind("<Shift-MouseWheel>", on_horizontal_scroll)
            frame.bind("<MouseWheel>", on_vertical_scroll)
            frame.bind("<Shift-MouseWheel>", on_horizontal_scroll)
            
            # Also bind Enter/Focus events
            canvas.bind("<Enter>", lambda e, c=canvas: c.focus_set())
            frame.bind("<Enter>", lambda e, c=canvas: c.focus_set())
        
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="转移表")
        self.table = tk.Text(table_frame, wrap="none", font=("Consolas", 10))
        table_scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        table_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=table_scroll_y.set, xscrollcommand=table_scroll_x.set)
        self.table.pack(side="left", fill="both", expand=True)
        table_scroll_y.pack(side="right", fill="y")
        table_scroll_x.pack(side="bottom", fill="x")
        
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="构造过程")
        self.logs = tk.Text(log_frame, wrap="word", font=("Consolas", 10))
        log_scroll = ttk.Scrollbar(log_frame, command=self.logs.yview)
        self.logs.configure(yscrollcommand=log_scroll.set)
        self.logs.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
    
    def zoom_in(self, view_name: str) -> None:
        """Zoom in the graph."""
        current_scale = get_graph_scale()
        set_graph_scale(current_scale + 0.2)
        self._redraw_current()
    
    def zoom_out(self, view_name: str) -> None:
        """Zoom out the graph."""
        current_scale = get_graph_scale()
        set_graph_scale(current_scale - 0.2)
        self._redraw_current()
    
    def zoom_reset(self, view_name: str) -> None:
        """Reset zoom to 100%."""
        set_graph_scale(1.0)
        self._redraw_current()
    
    def _redraw_current(self) -> None:
        """Redraw all graphs with current scale."""
        if self.current_nfa is not None:
            nfa, dfa, min_dfa, logs = self.current_nfa
            self.render(nfa, dfa, min_dfa, logs)

    def render(self, nfa: NFA, dfa: DFA, min_dfa: DFA, logs: list[str]) -> None:
        # Store current data for redraw after zoom
        self.current_nfa = (nfa, dfa, min_dfa, logs)
        
        # Force use graphviz renderer (no fallback)
        print("=" * 60)
        print("开始使用 Graphviz 渲染自动机图...")
        print("=" * 60)
        
        # Render NFA with graphviz
        try:
            print("\n[1/3] 渲染 NFA 图...")
            nfa_dot = nfa_to_dot(nfa)
            render_dot_to_tk(nfa_dot, self.canvases["NFA图"])
            print("✓ NFA 图渲染完成")
        except Exception as e:
            print(f"✗ NFA 图渲染失败: {e}")
            import traceback
            traceback.print_exc()
        
        # Render DFA with graphviz
        try:
            print("\n[2/3] 渲染 DFA 图...")
            dfa_dot = dfa_to_dot(dfa)
            render_dot_to_tk(dfa_dot, self.canvases["DFA图"])
            print("✓ DFA 图渲染完成")
        except Exception as e:
            print(f"✗ DFA 图渲染失败: {e}")
            import traceback
            traceback.print_exc()
        
        # Render minimized DFA with graphviz
        try:
            print("\n[3/3] 渲染最小 DFA 图...")
            min_dfa_dot = dfa_to_dot(min_dfa)
            render_dot_to_tk(min_dfa_dot, self.canvases["最小DFA图"])
            print("✓ 最小 DFA 图渲染完成")
        except Exception as e:
            print(f"✗ 最小 DFA 图渲染失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)
        print("Graphviz 渲染流程结束")
        print("=" * 60)
        
        # Update table with state set information
        self.table.delete("1.0", "end")
        self.table.insert("1.0", format_dfa_table_with_sets("DFA", dfa) + "\n\n" + 
                         format_dfa_table_with_sets("最小DFA", min_dfa))
        
        # Update logs with state set details
        self.logs.delete("1.0", "end")
        
        # Add state set information at the beginning of logs
        state_set_logs = ["=" * 60, "DFA状态集合说明:", "=" * 60]
        for state_name in sorted(dfa.state_sets.keys()):
            nfa_states = sorted(dfa.state_sets[state_name])
            accept_marker = " (接受状态)" if state_name in dfa.accepts else ""
            state_set_logs.append(f"{state_name}{accept_marker} = {nfa_states}")
        
        state_set_logs.append("")
        state_set_logs.append("=" * 60)
        state_set_logs.append("最小DFA状态集合说明:")
        state_set_logs.append("=" * 60)
        for state_name in sorted(min_dfa.state_sets.keys()):
            dfa_states = sorted(min_dfa.state_sets[state_name])
            accept_marker = " (接受状态)" if state_name in min_dfa.accepts else ""
            state_set_logs.append(f"{state_name}{accept_marker} = {dfa_states}")
        
        state_set_logs.append("")
        state_set_logs.append("=" * 60)
        state_set_logs.append("")
        
        # Combine state set info with original logs
        all_logs = state_set_logs + logs
        self.logs.insert("1.0", "\n".join(all_logs))


def format_dfa_table_with_sets(title: str, dfa: DFA) -> str:
    """Format DFA table with state set information."""
    lines = [title]
    lines.append("状态转移表:")
    for state in sorted(dfa.transitions):
        accept_marker = " (接受)" if state in dfa.accepts else ""
        lines.append(f"  {state}{accept_marker}:")
        for symbol, target in sorted(dfa.transitions[state].items()):
            lines.append(f"    {symbol} -> {target}")
    
    lines.append("")
    lines.append("状态集合映射:")
    for state in sorted(dfa.state_sets.keys()):
        nfa_states = sorted(dfa.state_sets[state])
        accept_marker = " (接受)" if state in dfa.accepts else ""
        lines.append(f"  {state}{accept_marker} = {nfa_states}")
    
    return "\n".join(lines)

