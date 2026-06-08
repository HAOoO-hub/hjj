# semantic.py：对语法树执行符号检查并生成语义错误与建议
from __future__ import annotations

from shared.errors import Diagnostic
from task_4_3_mini_ide.ast import Node


def analyze_semantics(root: Node | None) -> tuple[list[Diagnostic], dict[str, list[str]]]:
    if root is None:
        return [], {}
    diagnostics: list[Diagnostic] = []
    function_table: dict[str, tuple[str, int]] = {}
    summary: dict[str, list[str]] = {"functions": [], "variables": [], "constants": []}
    for func in root.children:
        name = func.value
        ret_type = func.children[0].value if func.children else "int"
        params = func.children[1].children if len(func.children) > 1 else []
        if name in function_table:
            diagnostics.append(Diagnostic("语义错误", f"函数 {name} 重复定义", func.line, func.column, "请修改函数名或删除重复定义"))
        function_table[name] = (ret_type, len(params))
        summary["functions"].append(f"{ret_type} {name}({len(params)} params)")
    for func in root.children:
        diagnostics.extend(check_function(func, function_table, summary))
    return diagnostics, summary


def check_function(func: Node, function_table: dict[str, tuple[str, int]], summary: dict[str, list[str]]) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    func_name = func.value
    params_node, body = func.children[1], func.children[2]
    scope = {param.value for param in params_node.children}
    for name in scope:
        if name in function_table:
            errors.append(Diagnostic("语义错误", f"参数名 {name} 与函数名冲突", func.line, func.column, "请避免变量名与函数名同名"))
    for param in params_node.children:
        summary["variables"].append(f"{func_name}::param {param.value}")
    errors.extend(check_block(body, scope, function_table, summary, func.children[0].value))
    return errors


def check_block(block: Node, scope: set[str], function_table: dict[str, tuple[str, int]], summary: dict[str, list[str]], ret_type: str) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    for child in block.children:
        if child.kind == "Decl":
            is_const = getattr(child, 'is_const', False)
            if child.value in scope:
                errors.append(Diagnostic("语义错误", f"变量 {child.value} 重复定义", child.line, child.column, "请修改变量名或删除重复声明"))
            elif child.value in function_table:
                errors.append(Diagnostic("语义错误", f"变量名 {child.value} 与函数名冲突", child.line, child.column, "请避免变量名与函数名同名"))
            scope.add(child.value)
            if is_const:
                # Check const must have initialization
                if not child.children:
                    errors.append(Diagnostic("语义错误", f"常量 {child.value} 未初始化", child.line, child.column, "常量必须在声明时赋予初始值"))
                else:
                    summary["constants"].append(f"const {child.value} = {child.children[0]}")
            else:
                summary["variables"].append(f"local {child.value}")
        elif child.kind == "If":
            errors.extend(check_expr(child.children[0], scope, function_table))
            errors.extend(check_statement(child.children[1], set(scope), function_table, summary, ret_type))
            if len(child.children) > 2:
                errors.extend(check_statement(child.children[2], set(scope), function_table, summary, ret_type))
        elif child.kind == "While":
            errors.extend(check_expr(child.children[0], scope, function_table))
            errors.extend(check_statement(child.children[1], set(scope), function_table, summary, ret_type))
        elif child.kind == "Return":
            if child.children[0].kind != "Empty":
                errors.extend(check_expr(child.children[0], scope, function_table))
            if ret_type == "void" and child.children[0].kind not in {"Empty", "Error"}:
                errors.append(Diagnostic("语义错误", "void 函数不应返回值", child.line, child.column, "请删除 return 后的表达式或修改函数返回类型"))
            if ret_type == "int" and child.children[0].kind == "Empty":
                errors.append(Diagnostic("语义错误", "int 函数应返回一个值", child.line, child.column, "请在 return 后补上返回表达式"))
        else:
            errors.extend(check_statement(child, scope, function_table, summary, ret_type))
    return errors


def check_statement(node: Node, scope: set[str], function_table: dict[str, tuple[str, int]], summary: dict[str, list[str]], ret_type: str) -> list[Diagnostic]:
    if node.kind == "Block":
        return check_block(node, set(scope), function_table, summary, ret_type)
    if node.kind == "ExprStmt":
        return check_expr(node.children[0], scope, function_table)
    return []


def check_expr(node: Node, scope: set[str], function_table: dict[str, tuple[str, int]]) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    if node.kind == "Identifier" and node.value not in scope:
        errors.append(Diagnostic("语义错误", f"变量 {node.value} 未定义", node.line, node.column, "请先声明该变量再使用"))
    elif node.kind == "Call":
        if node.value not in function_table:
            errors.append(Diagnostic("语义错误", f"函数 {node.value} 未定义", node.line, node.column, "请确认函数是否已定义"))
        else:
            _, count = function_table[node.value]
            if count != len(node.children):
                errors.append(Diagnostic("语义错误", f"函数 {node.value} 参数个数不匹配", node.line, node.column, "请检查实参与形参数量是否一致"))
        for child in node.children:
            errors.extend(check_expr(child, scope, function_table))
    else:
        for child in node.children:
            errors.extend(check_expr(child, scope, function_table))
    return errors
