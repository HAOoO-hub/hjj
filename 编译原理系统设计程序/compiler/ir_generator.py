# ir_generator.py：把类C子集语法树转换为四元式中间代码
from __future__ import annotations

from dataclasses import dataclass

from shared.errors import Diagnostic
from shared.quads import Quad
from task_4_3_mini_ide.ast import Node


@dataclass
class RawQuad:
    op: str
    arg1: str
    arg2: str
    result: str


class QuadGenerator:
    def __init__(self) -> None:
        self.temp_counter = 0
        self.label_counter = 0
        self.raw_quads: list[RawQuad] = []
        self.diagnostics: list[Diagnostic] = []
        # Stack to track loop labels for break/continue
        self.loop_stack: list[dict[str, str]] = []  # Each entry: {"start": ..., "end": ...}

    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def new_label(self) -> str:
        self.label_counter += 1
        return f"@L{self.label_counter}"

    def emit(self, op: str, arg1: str = "_", arg2: str = "_", result: str = "_") -> None:
        self.raw_quads.append(RawQuad(op, arg1, arg2, result))

    def generate_program(self, root: Node | None) -> tuple[dict[str, list[Quad]], list[Diagnostic]]:
        if root is None:
            return {}, self.diagnostics
        all_programs: dict[str, list[Quad]] = {}
        for func in root.children:
            if func.kind == "Function":
                self.raw_quads = []
                self.temp_counter = 0
                self.label_counter = 0
                self.generate_block(func.children[2])
                all_programs[func.value] = self.resolve_labels()
        return all_programs, self.diagnostics

    def generate_block(self, block: Node) -> None:
        for stmt in block.children:
            self.generate_stmt(stmt)

    def generate_stmt(self, node: Node) -> None:
        if node.kind == "Decl":
            # Support declaration with initialization: int x = expr; or const int x = expr;
            is_const = getattr(node, 'is_const', False)
            if node.children:
                value = self.generate_expr(node.children[0])
                if is_const:
                    # Mark constant assignments with a special comment in result
                    self.emit("=", value, "_", f"const_{node.value}")
                else:
                    self.emit("=", value, "_", node.value)
            return
        if node.kind == "Read":
            # Generate read quad: (read, _, _, variable_name)
            self.emit("read", "_", "_", node.value)
            return
        if node.kind == "Write":
            # Generate write quad: (write, expression_result, _, _)
            value = self.generate_expr(node.children[0])
            self.emit("write", value, "_", "_")
            return
        if node.kind == "ExprStmt":
            self.generate_expr(node.children[0])
            return
        if node.kind == "Return":
            value = "0" if not node.children or node.children[0].kind == "Empty" else self.generate_expr(node.children[0])
            self.emit("return", value, "_", "_")
            return
        if node.kind == "Block":
            self.generate_block(node)
            return
        if node.kind == "If":
            then_label = self.new_label()
            end_label = self.new_label()
            else_label = self.new_label() if len(node.children) > 2 else end_label
            self.generate_condition(node.children[0], then_label, else_label)
            self.emit("label", "_", "_", then_label)
            self.generate_stmt(node.children[1])
            if len(node.children) > 2:
                self.emit("J", "_", "_", end_label)
                self.emit("label", "_", "_", else_label)
                self.generate_stmt(node.children[2])
            self.emit("label", "_", "_", end_label)
            return
        if node.kind == "While":
            start_label = self.new_label()
            body_label = self.new_label()
            end_label = self.new_label()
            # Push loop context for break/continue
            self.loop_stack.append({"start": start_label, "end": end_label})
            self.emit("label", "_", "_", start_label)
            self.generate_condition(node.children[0], body_label, end_label)
            self.emit("label", "_", "_", body_label)
            self.generate_stmt(node.children[1])
            self.emit("J", "_", "_", start_label)
            self.emit("label", "_", "_", end_label)
            # Pop loop context
            self.loop_stack.pop()
            return
        if node.kind == "Break":
            if not self.loop_stack:
                self.diagnostics.append(Diagnostic("语义错误", "break 语句必须在循环内使用", node.line, node.column,
                                                   "请将 break 放在 while 循环中"))
            else:
                # Jump to loop end
                self.emit("J", "_", "_", self.loop_stack[-1]["end"])
            return
        if node.kind == "Continue":
            if not self.loop_stack:
                self.diagnostics.append(Diagnostic("语义错误", "continue 语句必须在循环内使用", node.line, node.column,
                                                   "请将 continue 放在 while 循环中"))
            else:
                # Jump to loop start (condition check)
                self.emit("J", "_", "_", self.loop_stack[-1]["start"])
            return

    def generate_condition(self, node: Node, true_label: str, false_label: str) -> None:
        if node.kind == "Binary" and node.value in {"<", ">", "<=", ">=", "==", "!="}:
            left = self.generate_expr(node.children[0])
            right = self.generate_expr(node.children[1])
            self.emit(f"J{node.value}", left, right, true_label)
            self.emit("J", "_", "_", false_label)
            return
        value = self.generate_expr(node)
        self.emit("J!=", value, "0", true_label)
        self.emit("J", "_", "_", false_label)

    def generate_expr(self, node: Node) -> str:
        if node.kind == "Number":
            return node.value
        if node.kind == "Identifier":
            return node.value
        if node.kind == "Assign":
            if node.children[0].kind != "Identifier":
                self.diagnostics.append(Diagnostic("中间代码生成", "赋值左值必须是标识符", node.line, node.column,
                                                   "请将赋值左侧改为变量名"))
                return "0"
            value = self.generate_expr(node.children[1])
            self.emit("=", value, "_", node.children[0].value)
            return node.children[0].value
        if node.kind == "Binary":
            if node.value in {"+", "-", "*", "/"}:
                left = self.generate_expr(node.children[0])
                right = self.generate_expr(node.children[1])
                temp = self.new_temp()
                self.emit(node.value, left, right, temp)
                return temp
            if node.value in {"&&", "||"}:
                return self.generate_logical(node)
            if node.value in {"<", ">", "<=", ">=", "==", "!="}:
                left = self.generate_expr(node.children[0])
                right = self.generate_expr(node.children[1])
                temp = self.new_temp()
                self.emit(f"CMP{node.value}", left, right, temp)
                return temp
        if node.kind == "Call":
            for arg in node.children:
                val = self.generate_expr(arg)
                self.emit("param", val, "_", "_")
            temp = self.new_temp()
            self.emit("call", node.value, str(len(node.children)), temp)
            return temp
        if node.kind == "Empty":
            return "0"
        return "0"

    def generate_logical(self, node: Node) -> str:
        temp = self.new_temp()
        true_label = self.new_label()
        false_label = self.new_label()
        end_label = self.new_label()

        if node.value == "&&":
            # Short-circuit AND: if any operand is 0, jump to false
            left = self.generate_expr(node.children[0])
            self.emit("J==", left, "0", false_label)
            right = self.generate_expr(node.children[1])
            self.emit("J==", right, "0", false_label)
            # Both true: fall through to true block
            self.emit("=", "1", "_", temp)
            self.emit("J", "_", "_", end_label)
            self.emit("label", "_", "_", false_label)
            self.emit("=", "0", "_", temp)
            self.emit("label", "_", "_", end_label)
        else:  # ||
            # Short-circuit OR: if any operand is non-zero, jump to true
            left = self.generate_expr(node.children[0])
            self.emit("J!=", left, "0", true_label)
            right = self.generate_expr(node.children[1])
            self.emit("J!=", right, "0", true_label)
            # Both false: fall through to false block
            self.emit("=", "0", "_", temp)
            self.emit("J", "_", "_", end_label)
            self.emit("label", "_", "_", true_label)
            self.emit("=", "1", "_", temp)
            self.emit("label", "_", "_", end_label)
        return temp

    def resolve_labels(self) -> list[Quad]:
        label_to_index: dict[str, int] = {}
        filtered: list[RawQuad] = []
        next_idx = 0
        for quad in self.raw_quads:
            if quad.op == "label":
                label_to_index[quad.result] = next_idx
            else:
                filtered.append(quad)
                next_idx += 1
        result: list[Quad] = []
        for idx, quad in enumerate(filtered):
            jump_target = quad.result
            if jump_target.startswith("@L"):
                jump_target = str(label_to_index[jump_target])
            result.append(Quad(idx=idx, op=quad.op, arg1=quad.arg1, arg2=quad.arg2, result=jump_target))
        return result


def generate_quads(root: Node | None) -> tuple[dict[str, list[Quad]], list[Diagnostic]]:
    return QuadGenerator().generate_program(root)

