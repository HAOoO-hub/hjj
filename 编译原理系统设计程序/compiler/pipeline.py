# pipeline.py：串联源码的词法语法语义分析、中间代码生成和后端处理
from __future__ import annotations

from dataclasses import dataclass, field

from compiler.ir_generator import generate_quads
from shared.errors import Diagnostic
from shared.quads import Quad
from task_3_2_interpreter.executor import QuadExecutor
from task_4_2_quad_to_llvm.translator import LLVMTranslator
from task_4_3_mini_ide.ast import Node
from task_4_3_mini_ide.lexer import Token, lex
from task_4_3_mini_ide.parser import parse_tokens


@dataclass
class AnalysisTables:
    symbols: list[dict[str, str]] = field(default_factory=list)
    variables: list[dict[str, str]] = field(default_factory=list)
    functions: list[dict[str, str]] = field(default_factory=list)
    constants: list[dict[str, str]] = field(default_factory=list)  # NEW: Constant table


@dataclass
class PipelineResult:
    source: str
    tokens: list[Token]
    ast: Node | None
    diagnostics: list[Diagnostic]
    tables: AnalysisTables
    quads: list[Quad]
    llvm_ir: str
    interpreter_trace: str
    interpreter_variables: list[tuple[str, int]]
    interpreter_return: str


def analyze_source(source: str) -> PipelineResult:
    tokens, lex_errors = lex(source)
    ast, parse_errors = parse_tokens(tokens)
    tables, semantic_errors = build_tables(ast)

    programs, ir_diags = generate_quads(ast if not parse_errors else None)

    params_map = {}
    if ast:
        for func in ast.children:
            if func.kind == "Function":
                p_names = [p.value for p in func.children[1].children] if len(func.children) > 1 else []
                params_map[func.value] = p_names

    llvm_ir = ""
    interpreter_trace = ""
    interpreter_variables: list[tuple[str, int]] = []
    interpreter_return = "-"
    backend_errors: list[Diagnostic] = []

    if programs:
        try:
            main_quads = programs.get("main", next(iter(programs.values()), []))
            llvm_ir = LLVMTranslator(main_quads).translate()
        except Exception as exc:
            backend_errors.append(Diagnostic("LLVM转换", str(exc), 1, 1, "请检查四元式生成结果"))
        try:
            state = QuadExecutor(programs, params_map).run()
            interpreter_trace = "\n".join(state.trace)
            interpreter_variables = sorted({**state.variables, **state.temps}.items())
            interpreter_return = str(state.return_value) if state.return_value is not None else "-"
        except Exception as exc:
            backend_errors.append(Diagnostic("中间代码解释", str(exc), 1, 1, "请检查输入程序和生成的四元式"))

    diagnostics = [*lex_errors, *parse_errors, *semantic_errors, *ir_diags, *backend_errors]
    quads_list = []
    for q_list in programs.values():
        quads_list.extend(q_list)

    return PipelineResult(
        source=source,
        tokens=tokens,
        ast=ast,
        diagnostics=diagnostics,
        tables=tables,
        quads=quads_list,
        llvm_ir=llvm_ir,
        interpreter_trace=interpreter_trace,
        interpreter_variables=interpreter_variables,
        interpreter_return=interpreter_return,
    )


def build_tables(root: Node | None) -> tuple[AnalysisTables, list[Diagnostic]]:
    tables = AnalysisTables()
    diagnostics: list[Diagnostic] = []
    if root is None:
        return tables, diagnostics
    function_names: set[str] = set()
    for func in root.children:
        if func.kind != "Function":
            continue
        ret_type = func.children[0].value if func.children else "int"
        params = func.children[1].children if len(func.children) > 1 else []
        if func.value in function_names:
            diagnostics.append(Diagnostic("语义错误", f"函数 {func.value} 重复定义", func.line, func.column, "请修改重复函数名"))
        function_names.add(func.value)
        tables.functions.append(
            {
                "name": func.value,
                "return_type": ret_type,
                "params": ", ".join(param.value for param in params),
                "param_count": str(len(params)),
                "line": str(func.line),
            }
        )
        tables.symbols.append(
            {
                "name": func.value,
                "kind": "function",
                "type": ret_type,
                "scope": "global",
                "line": str(func.line),
                "note": f"{len(params)} params",
            }
        )
        scope_names = set(function_names)
        for param in params:
            if param.value in scope_names:
                diagnostics.append(Diagnostic("语义错误", f"参数名 {param.value} 与已有名字冲突", param.line, param.column, "请修改参数名"))
            scope_names.add(param.value)
            row = {"name": param.value, "kind": "param", "type": param.children[0].value, "scope": func.value, "line": str(param.line), "note": ""}
            tables.variables.append(row)
            tables.symbols.append(row.copy())
        body = func.children[2]
        diagnostics.extend(collect_block_info(body, func.value, scope_names, function_names, tables))
    return tables, diagnostics


def collect_block_info(block: Node, scope: str, names: set[str], function_names: set[str], tables: AnalysisTables) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    local_names = set(names)
    for stmt in block.children:
        if stmt.kind == "Decl":
            is_const = getattr(stmt, 'is_const', False)
            if stmt.value in local_names:
                diagnostics.append(Diagnostic("语义错误", f"变量 {stmt.value} 重复定义", stmt.line, stmt.column, "请修改变量名或删除重复声明"))
            if stmt.value in function_names:
                diagnostics.append(Diagnostic("语义错误", f"变量名 {stmt.value} 与函数名冲突", stmt.line, stmt.column, "请避免变量名与函数名同名"))
            local_names.add(stmt.value)
            
            if is_const:
                # Add to constant table
                const_value = stmt.children[0].value if stmt.children else "?"
                row = {"name": stmt.value, "kind": "constant", "type": "int", "scope": scope, "line": str(stmt.line), "value": const_value}
                tables.constants.append(row)
                tables.symbols.append(row.copy())
            else:
                # Add to variable table
                row = {"name": stmt.value, "kind": "variable", "type": "int", "scope": scope, "line": str(stmt.line), "note": "local"}
                tables.variables.append(row)
                tables.symbols.append(row.copy())
        else:
            diagnostics.extend(check_node(stmt, local_names, function_names))
            if stmt.kind == "Block":
                diagnostics.extend(collect_block_info(stmt, scope, set(local_names), function_names, tables))
            elif stmt.kind == "If":
                diagnostics.extend(collect_nested(stmt, 1, scope, set(local_names), function_names, tables))
                if len(stmt.children) > 2:
                    diagnostics.extend(collect_nested(stmt, 2, scope, set(local_names), function_names, tables))
            elif stmt.kind == "While":
                diagnostics.extend(collect_nested(stmt, 1, scope, set(local_names), function_names, tables))
    return diagnostics


def collect_nested(parent: Node, index: int, scope: str, names: set[str], function_names: set[str], tables: AnalysisTables) -> list[Diagnostic]:
    node = parent.children[index]
    if node.kind == "Block":
        return collect_block_info(node, scope, names, function_names, tables)
    return check_node(node, names, function_names)


def check_node(node: Node, names: set[str], function_names: set[str]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if node.kind == "ExprStmt":
        return check_expr(node.children[0], names, function_names)
    if node.kind == "Return":
        if node.children:
            return check_expr(node.children[0], names, function_names)
        return diagnostics
    if node.kind == "If":
        diagnostics.extend(check_expr(node.children[0], names, function_names))
    if node.kind == "While":
        diagnostics.extend(check_expr(node.children[0], names, function_names))
    return diagnostics


def check_expr(node: Node, names: set[str], function_names: set[str]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if node.kind == "Identifier" and node.value not in names:
        diagnostics.append(Diagnostic("语义错误", f"变量 {node.value} 未定义", node.line, node.column, "请先声明该变量再使用"))
    elif node.kind == "Call":
        if node.value not in function_names:
            diagnostics.append(Diagnostic("语义错误", f"函数 {node.value} 未定义", node.line, node.column, "请确认函数定义是否存在"))
        for child in node.children:
            diagnostics.extend(check_expr(child, names, function_names))
    else:
        for child in node.children:
            diagnostics.extend(check_expr(child, names, function_names))
    return diagnostics


def render_ast(root: Node | None, depth: int = 0) -> str:
    if root is None:
        return "无语法树"
    lines = ["  " * depth + f"{root.kind}{(': ' + root.value) if root.value else ''}"]
    for child in root.children:
        lines.append(render_ast(child, depth + 1))
    return "\n".join(lines)

