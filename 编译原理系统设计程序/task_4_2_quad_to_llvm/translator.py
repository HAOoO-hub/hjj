# translator.py：把四元式基本块翻译为简化LLVM IR代码
from __future__ import annotations

from shared.basic_blocks import BasicBlock, split_basic_blocks
from shared.quads import Quad, is_number, is_temp
from task_4_2_quad_to_llvm.ir_builder import IRBuilder
from task_4_2_quad_to_llvm.symbol_collector import collect_symbols


ICMP_MAP = {
    "J>": "sgt",
    "J<": "slt",
    "J>=": "sge",
    "J<=": "sle",
    "J==": "eq",
    "J!=": "ne",
}


class LLVMTranslator:
    def __init__(self, quads: list[Quad]) -> None:
        self.quads = quads
        self.blocks = split_basic_blocks(quads)
        self.block_by_start = {block.quads[0].idx: block.label for block in self.blocks}
        self.variables, self.labels = collect_symbols(quads)
        self.builder = IRBuilder()

    def read_operand(self, token: str) -> str:
        if is_number(token):
            return token
        if is_temp(token):
            return f"%{token}"
        reg = self.builder.new_value()
        self.builder.emit(f"{reg} = load i32, ptr %{token}")
        return reg

    def write_target(self, target: str, value: str) -> None:
        if is_temp(target):
            name = f"%{target}"
            if value != name:
                self.builder.emit(f"{name} = add i32 {value}, 0")
        else:
            self.builder.emit(f"store i32 {value}, ptr %{target}")

    def translate_quad(self, quad: Quad, next_block_label: str | None) -> bool:
        if quad.op == "=":
            value = self.read_operand(quad.arg1)
            self.write_target(quad.result, value)
            return False
        if quad.op == "param":
            return False
        if quad.op == "call":
            if quad.result != "_":
                target = f"%{quad.result}"
                self.builder.emit(f"{target} = add i32 0, 0")
            return False
        if quad.op.startswith("CMP"):
            left = self.read_operand(quad.arg1)
            right = self.read_operand(quad.arg2)
            op = quad.op[3:]
            icmp_map = {"<": "slt", ">": "sgt", "<=": "sle", ">=": "sge", "==": "eq", "!=": "ne"}
            cmp_name = self.builder.new_cmp()
            self.builder.emit(f"{cmp_name} = icmp {icmp_map[op]} i32 {left}, {right}")
            target = f"%{quad.result}"
            self.builder.emit(f"{target} = zext i1 {cmp_name} to i32")
            return False
        if quad.op in {"+", "-", "*", "/"}:
            left = self.read_operand(quad.arg1)
            right = self.read_operand(quad.arg2)
            op = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}[quad.op]
            target = f"%{quad.result}" if is_temp(quad.result) else self.builder.new_value()
            self.builder.emit(f"{target} = {op} i32 {left}, {right}")
            if is_temp(quad.result):
                return False
            self.builder.emit(f"store i32 {target}, ptr %{quad.result}")
            return False
        if quad.op == "J":
            self.builder.emit(f"br label %{self.block_by_start[int(quad.result)]}")
            return True
        if quad.op in ICMP_MAP:
            left = self.read_operand(quad.arg1)
            right = self.read_operand(quad.arg2)
            cmp_name = self.builder.new_cmp()
            self.builder.emit(f"{cmp_name} = icmp {ICMP_MAP[quad.op]} i32 {left}, {right}")
            false_label = next_block_label or "exit"
            self.builder.emit(f"br i1 {cmp_name}, label %{self.block_by_start[int(quad.result)]}, label %{false_label}")
            return True
        if quad.op == "return":
            value = self.read_operand(quad.arg1)
            self.builder.emit(f"ret i32 {value}")
            return True
        if quad.op == "read":
            # Call external function to read integer
            target = f"%{quad.result}"
            self.builder.emit(f"{target} = call i32 @get_input()")
            self.builder.emit(f"store i32 {target}, ptr %{quad.result}")
            return False
        if quad.op == "write":
            # Call external function to print integer
            value = self.read_operand(quad.arg1)
            self.builder.emit(f"call void @print_int(i32 {value})")
            return False
        raise ValueError(f"不支持的四元式操作：{quad.op}")

    def translate(self) -> str:
        # Declare external I/O functions
        self.builder.emit("declare i32 @get_input()")
        self.builder.emit("declare void @print_int(i32)")
        self.builder.emit("")
        
        self.builder.emit("define i32 @main() {")
        for name in self.variables:
            if name != "_":
                self.builder.emit(f"%{name} = alloca i32")
        for i, block in enumerate(self.blocks):
            self.builder.emit(f"{block.label}:")
            next_label = self.blocks[i + 1].label if i + 1 < len(self.blocks) else None
            terminated = False
            for quad in block.quads:
                terminated = self.translate_quad(quad, next_label)
            if not terminated and next_label:
                self.builder.emit(f"br label %{next_label}")
        self.builder.emit("}")
        return self.builder.build()


def describe_blocks(blocks: list[BasicBlock]) -> str:
    lines: list[str] = []
    for block in blocks:
        lines.append(f"{block.label}:")
        for quad in block.quads:
            lines.append(f"  {quad.format()}")
    return "\n".join(lines)
