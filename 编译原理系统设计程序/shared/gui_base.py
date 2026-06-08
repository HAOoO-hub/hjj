# gui_base.py：提供项目共享的Tkinter界面基础组件
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def create_paned(parent: tk.Widget, orient: str = "horizontal") -> ttk.Panedwindow:
    pane = ttk.Panedwindow(parent, orient=orient)
    pane.pack(fill="both", expand=True)
    return pane


def create_labeled_frame(parent: tk.Widget, title: str) -> ttk.LabelFrame:
    frame = ttk.LabelFrame(parent, text=title, padding=8)
    return frame


def create_text(parent: tk.Widget, **kwargs) -> tk.Text:
    text = tk.Text(parent, undo=True, wrap="none", font=("Consolas", 10), **kwargs)
    return text

