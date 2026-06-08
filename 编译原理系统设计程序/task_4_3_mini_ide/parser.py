# parser.py：对类C子集记号流进行递归下降语法分析
from __future__ import annotations

from shared.errors import Diagnostic
from task_4_3_mini_ide.ast import Node
from task_4_3_mini_ide.lexer import Token


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.errors: list[Diagnostic] = []

    def current(self) -> Token:
        return self.tokens[self.pos]

    def skip_comments(self) -> None:
        """Skip COMMENT tokens to avoid parsing errors."""
        while self.current().kind == "COMMENT":
            self.pos += 1

    def accept(self, kind: str, value: str | None = None) -> Token | None:
        token = self.current()
        if token.kind == kind and (value is None or token.value == value):
            self.pos += 1
            return token
        return None

    def expect(self, kind: str, value: str | None = None, message: str = "") -> Token:
        token = self.current()
        if token.kind == kind and (value is None or token.value == value):
            self.pos += 1
            return token
        suggestion = ""
        if kind == "SEMI":
            suggestion = "该语句末尾可能缺少 ';'"
        elif kind == "RPAREN":
            suggestion = "检查最近未闭合的 '('"
        elif kind == "RBRACE":
            suggestion = "请检查上一层语句块是否未正确结束"
        self.errors.append(Diagnostic("语法错误", message or f"期望 {kind}", token.line, token.column, suggestion))
        return token

    def parse(self) -> tuple[Node | None, list[Diagnostic]]:
        program = Node("Program")
        while self.current().kind != "EOF":
            self.skip_comments()
            if self.current().kind == "EOF":
                break
            func = self.parse_function()
            if func is None:
                break
            program.children.append(func)
        return (program if not self.errors else program), self.errors

    def parse_function(self) -> Node | None:
        ret = self.expect("KW", message="函数定义应以 int 或 void 开始")
        if ret.kind != "KW":
            return None
        if ret.value not in {"int", "void"}:
            self.errors.append(Diagnostic("语法错误", "函数返回类型仅支持 int 或 void", ret.line, ret.column, "请把函数返回类型改为 int 或 void"))
        name = self.expect("ID", message="函数定义缺少函数名")
        self.expect("LPAREN", message="函数定义缺少左括号")
        params = self.parse_params()
        self.expect("RPAREN", message="函数定义缺少右括号")
        body = self.parse_block()
        return Node("Function", value=name.value, children=[Node("Type", ret.value), params, body], line=name.line, column=name.column)

    def parse_params(self) -> Node:
        params = Node("Params")
        self.skip_comments()
        if self.current().kind == "RPAREN":
            return params
        while True:
            self.skip_comments()
            typ = self.expect("KW", message="参数缺少类型")
            name = self.expect("ID", message="参数缺少名字")
            params.children.append(Node("Param", value=name.value, children=[Node("Type", typ.value)], line=name.line, column=name.column))
            if not self.accept("COMMA"):
                break
            self.skip_comments()
        return params

    def parse_block(self) -> Node:
        start = self.expect("LBRACE", message="语句块缺少左花括号")
        block = Node("Block", line=start.line, column=start.column)
        while self.current().kind not in {"RBRACE", "EOF"}:
            self.skip_comments()
            stmt = self.parse_statement()
            if stmt is not None:
                block.children.append(stmt)
            else:
                self.pos += 1
        self.expect("RBRACE", message="语句块缺少右花括号")
        return block

    def parse_statement(self) -> Node | None:
        token = self.current()
        if token.kind == "KW" and token.value in {"int", "const"}:
            return self.parse_declaration()
        if token.kind == "KW" and token.value == "if":
            return self.parse_if()
        if token.kind == "KW" and token.value == "while":
            return self.parse_while()
        if token.kind == "KW" and token.value == "return":
            return self.parse_return()
        if token.kind == "KW" and token.value == "break":
            return self.parse_break()
        if token.kind == "KW" and token.value == "continue":
            return self.parse_continue()
        if token.kind == "KW" and token.value == "read":
            return self.parse_read()
        if token.kind == "KW" and token.value == "write":
            return self.parse_write()
        if token.kind == "LBRACE":
            return self.parse_block()
        return self.parse_expr_stmt()

    def parse_declaration(self) -> Node:
        is_const = False
        # Check for 'const' keyword
        if self.accept("KW", "const"):
            is_const = True
        
        self.expect("KW", value="int", message="变量声明应以 int 开始")
        name = self.expect("ID", message="变量声明缺少变量名")
        init_expr = None
        
        # Const MUST have initialization, regular vars can optionally initialize
        if self.accept("OP", "="):
            init_expr = self.parse_expression()
        elif is_const:
            self.errors.append(Diagnostic("语法错误", "常量必须初始化", name.line, name.column, "请在常量声明时赋予初始值"))
        
        self.expect("SEMI", message="变量声明缺少分号")
        node = Node("Decl", value=name.value, line=name.line, column=name.column)
        node.is_const = is_const  # Mark as constant
        if init_expr is not None:
            node.children.append(init_expr)
        return node

    def parse_if(self) -> Node:
        start = self.expect("KW", value="if", message="if 语句错误")
        self.expect("LPAREN", message="if 缺少左括号")
        cond = self.parse_expression()
        self.expect("RPAREN", message="if 缺少右括号")
        then_stmt = self.parse_statement() or Node("Empty")
        children = [cond, then_stmt]
        if self.accept("KW", "else"):
            children.append(self.parse_statement() or Node("Empty"))
        return Node("If", children=children, line=start.line, column=start.column)

    def parse_while(self) -> Node:
        start = self.expect("KW", value="while", message="while 语句错误")
        self.expect("LPAREN", message="while 缺少左括号")
        cond = self.parse_expression()
        self.expect("RPAREN", message="while 缺少右括号")
        body = self.parse_statement() or Node("Empty")
        return Node("While", children=[cond, body], line=start.line, column=start.column)

    def parse_return(self) -> Node:
        start = self.expect("KW", value="return", message="return 语句错误")
        expr = Node("Empty", line=start.line, column=start.column)
        if self.current().kind != "SEMI":
            expr = self.parse_expression()
        self.expect("SEMI", message="return 语句缺少分号")
        return Node("Return", children=[expr], line=start.line, column=start.column)

    def parse_break(self) -> Node:
        start = self.expect("KW", value="break", message="break 语句错误")
        self.expect("SEMI", message="break 语句缺少分号")
        return Node("Break", line=start.line, column=start.column)

    def parse_continue(self) -> Node:
        start = self.expect("KW", value="continue", message="continue 语句错误")
        self.expect("SEMI", message="continue 语句缺少分号")
        return Node("Continue", line=start.line, column=start.column)

    def parse_read(self) -> Node:
        """解析 read 语句: read identifier;"""
        start = self.expect("KW", value="read", message="read 语句错误")
        name = self.expect("ID", message="read 语句需要指定变量名")
        self.expect("SEMI", message="read 语句缺少分号")
        return Node("Read", value=name.value, line=start.line, column=start.column)

    def parse_write(self) -> Node:
        """解析 write 语句: write expression;"""
        start = self.expect("KW", value="write", message="write 语句错误")
        expr = self.parse_expression()
        self.expect("SEMI", message="write 语句缺少分号")
        return Node("Write", children=[expr], line=start.line, column=start.column)

    def parse_expr_stmt(self) -> Node:
        expr = self.parse_expression()
        self.expect("SEMI", message="表达式语句缺少分号")
        return Node("ExprStmt", children=[expr], line=expr.line, column=expr.column)

    def parse_expression(self) -> Node:
        return self.parse_assignment()

    def parse_assignment(self) -> Node:
        left = self.parse_logical()
        if self.accept("OP", "="):
            right = self.parse_assignment()
            return Node("Assign", children=[left, right], line=left.line, column=left.column)
        return left

    def parse_logical(self) -> Node:
        node = self.parse_relation()
        while self.current().kind == "OP" and self.current().value in {"&&", "||"}:
            op = self.current()
            self.pos += 1
            right = self.parse_relation()
            node = Node("Binary", value=op.value, children=[node, right], line=op.line, column=op.column)
        return node

    def parse_relation(self) -> Node:
        node = self.parse_add()
        while self.current().kind == "OP" and self.current().value in {"<", ">", "<=", ">=", "==", "!="}:
            op = self.current()
            self.pos += 1
            right = self.parse_add()
            node = Node("Binary", value=op.value, children=[node, right], line=op.line, column=op.column)
        return node

    def parse_add(self) -> Node:
        node = self.parse_term()
        while self.current().kind == "OP" and self.current().value in {"+", "-"}:
            op = self.current()
            self.pos += 1
            right = self.parse_term()
            node = Node("Binary", value=op.value, children=[node, right], line=op.line, column=op.column)
        return node

    def parse_term(self) -> Node:
        node = self.parse_factor()
        while self.current().kind == "OP" and self.current().value in {"*", "/"}:
            op = self.current()
            self.pos += 1
            right = self.parse_factor()
            node = Node("Binary", value=op.value, children=[node, right], line=op.line, column=op.column)
        return node

    def parse_factor(self) -> Node:
        self.skip_comments()
        token = self.current()
        if token.kind == "NUM":
            self.pos += 1
            return Node("Number", value=token.value, line=token.line, column=token.column)
        if token.kind == "ID":
            self.pos += 1
            if self.accept("LPAREN"):
                call = Node("Call", value=token.value, line=token.line, column=token.column)
                if self.current().kind != "RPAREN":
                    while True:
                        self.skip_comments()
                        call.children.append(self.parse_expression())
                        if not self.accept("COMMA"):
                            break
                self.expect("RPAREN", message="函数调用缺少右括号")
                return call
            return Node("Identifier", value=token.value, line=token.line, column=token.column)
        if self.accept("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN", message="表达式缺少右括号")
            return expr
        self.errors.append(Diagnostic("语法错误", "无法解析表达式", token.line, token.column, "请检查该处是否缺少标识符、常量或括号"))
        self.pos += 1
        return Node("Error", line=token.line, column=token.column)


def parse_tokens(tokens: list[Token]) -> tuple[Node | None, list[Diagnostic]]:
    return Parser(tokens).parse()
