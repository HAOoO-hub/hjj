# lexer.py：把类C子集源代码切分为带位置信息的记号流
from __future__ import annotations

from dataclasses import dataclass

from shared.errors import Diagnostic


KEYWORDS = {"int", "void", "if", "else", "while", "return", "const", "break", "continue", "read", "write"}
DOUBLE_OPS = {"<=", ">=", "==", "!=", "&&", "||"}
SINGLE_OPS = set("+-*/=<> (){};,")


@dataclass
class Token:
    kind: str
    value: str
    line: int
    column: int


def lex(text: str) -> tuple[list[Token], list[Diagnostic]]:
    tokens: list[Token] = []
    errors: list[Diagnostic] = []
    i = 0
    line = 1
    col = 1
    while i < len(text):
        ch = text[i]
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if text.startswith("//", i):
            start_col = col
            j = i
            while j < len(text) and text[j] != "\n":
                j += 1
            tokens.append(Token("COMMENT", text[i:j], line, start_col))
            col += j - i
            i = j
            continue
        if i + 1 < len(text) and text[i : i + 2] in DOUBLE_OPS:
            tokens.append(Token("OP", text[i : i + 2], line, col))
            i += 2
            col += 2
            continue
        if ch.isalpha() or ch == "_":
            start = i
            start_col = col
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
            word = text[start:i]
            tokens.append(Token("KW" if word in KEYWORDS else "ID", word, line, start_col))
            col += i - start
            continue
        if ch.isdigit():
            start = i
            start_col = col
            while i < len(text) and text[i].isdigit():
                i += 1
            tokens.append(Token("NUM", text[start:i], line, start_col))
            col += i - start
            continue
        if ch in SINGLE_OPS:
            kind = {
                "(": "LPAREN",
                ")": "RPAREN",
                "{": "LBRACE",
                "}": "RBRACE",
                ";": "SEMI",
                ",": "COMMA",
            }.get(ch, "OP")
            tokens.append(Token(kind, ch, line, col))
            i += 1
            col += 1
            continue
        errors.append(Diagnostic("词法错误", f"无法识别字符 {ch!r}", line, col, "请删除非法字符或改为合法标识符/符号"))
        i += 1
        col += 1
    tokens.append(Token("EOF", "", line, col))
    return tokens, errors
