# highlighter.py：负责迷你IDE中的即时词法高亮和函数名叠加高亮
from __future__ import annotations

import tkinter as tk

from task_4_3_mini_ide.ast import Node
from task_4_3_mini_ide.lexer import Token


def configure_tags(text: tk.Text) -> None:
    text.tag_configure("kw", foreground="#7c3aed")
    text.tag_configure("num", foreground="#2563eb")
    text.tag_configure("comment", foreground="#6b7280")
    text.tag_configure("op", foreground="#b45309")
    text.tag_configure("func", foreground="#dc2626", font=("Consolas", 10, "bold"))


def clear_tags(text: tk.Text) -> None:
    for tag in ("kw", "num", "comment", "op", "func"):
        text.tag_remove(tag, "1.0", "end")


def index_of(token: Token) -> str:
    return f"{token.line}.{token.column - 1}"


def apply_lexical_highlight(text: tk.Text, tokens: list[Token]) -> None:
    clear_tags(text)
    for token in tokens:
        start = index_of(token)
        end = f"{start}+{len(token.value)}c"
        if token.kind == "KW":
            text.tag_add("kw", start, end)
        elif token.kind == "NUM":
            text.tag_add("num", start, end)
        elif token.kind == "COMMENT":
            text.tag_add("comment", start, end)
        elif token.kind == "OP":
            text.tag_add("op", start, end)


def apply_function_highlight(text: tk.Text, root: Node | None) -> None:
    if root is None:
        return
    for node in walk(root):
        if node.kind in {"Function", "Call"} and node.value:
            start = f"{node.line}.{node.column - 1}"
            end = f"{start}+{len(node.value)}c"
            text.tag_add("func", start, end)


def walk(node: Node):
    yield node
    for child in node.children:
        yield from walk(child)

