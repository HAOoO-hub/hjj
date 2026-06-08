# operations.py：实现四元式解释器的各类指令语义
from __future__ import annotations

import operator

from shared.quads import Quad, is_identifier, is_number, is_temp
from task_3_2_interpreter.runtime import RuntimeState


ARITHMETIC = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: int(a / b),
}

RELATIONS = {
    "J>": operator.gt,
    "J<": operator.lt,
    "J>=": operator.ge,
    "J<=": operator.le,
    "J==": operator.eq,
    "J!=": operator.ne,
}

CMP_OPS = {
    "CMP>": operator.gt,
    "CMP<": operator.lt,
    "CMP>=": operator.ge,
    "CMP<=": operator.le,
    "CMP==": operator.eq,
    "CMP!=": operator.ne,
}

def read_value(state: RuntimeState, token: str) -> int:
    if is_number(token):
        return int(token)
    if is_temp(token):
        if token not in state.temps:
            # Temp not initialized, default to 0
            state.temps[token] = 0
            return 0
        return state.temps[token]
    if is_identifier(token):
        if token not in state.variables:
            # Variable not initialized, default to 0 (C-like behavior)
            state.variables[token] = 0
            return 0
        return state.variables[token]
    raise ValueError(f"无法读取值：{token}")


def write_target(state: RuntimeState, target: str, value: int) -> None:
    if is_temp(target):
        state.temps[target] = value
    else:
        state.variables[target] = value


def execute_quad(state: RuntimeState, quad: Quad, index_by_idx: dict[int, int]) -> int | None:
    state.log(f"PC={quad.idx} 执行 {quad.format()}")
    if quad.op == "=":
        value = read_value(state, quad.arg1)
        write_target(state, quad.result, value)
        return None
    if quad.op in ARITHMETIC:
        left = read_value(state, quad.arg1)
        right = read_value(state, quad.arg2)
        if quad.op == "/" and right == 0:
            raise ZeroDivisionError("除零错误")
        value = ARITHMETIC[quad.op](left, right)
        write_target(state, quad.result, value)
        return None
    if quad.op in CMP_OPS:
        left = read_value(state, quad.arg1)
        right = read_value(state, quad.arg2)
        res = 1 if CMP_OPS[quad.op](left, right) else 0
        write_target(state, quad.result, res)
        return None
    if quad.op == "J":
        return index_by_idx[int(quad.result)]
    if quad.op in RELATIONS:
        left = read_value(state, quad.arg1)
        right = read_value(state, quad.arg2)
        result = RELATIONS[quad.op](left, right)
        if result:
            return index_by_idx[int(quad.result)]
        return None
    if quad.op == "param":
        value = read_value(state, quad.arg1)
        state.pending_args.append(value)
        return None
    if quad.op == "call":
        func_name = quad.arg1
        num_args = int(quad.arg2)
        args = state.pending_args[-num_args:]
        state.pending_args = state.pending_args[:-num_args]

        target_quads = None
        for pname, pquads in state.program.items():
            if pname == func_name:
                target_quads = pquads
                break

        if target_quads is None:
            raise ValueError(f"未定义函数：{func_name}")

        # Pre-allocate the result temp in caller's scope BEFORE saving frame
        if quad.result and quad.result != "_" and is_temp(quad.result):
            state.temps[quad.result] = 0

        state.call_stack.append({
            "pc": state.pc + 1,  # Return to the next instruction after call
            "quads": state.quads,
            "result_target": quad.result,
            "variables": dict(state.variables),
            "temps": dict(state.temps),  # Now includes the pre-allocated result temp
            "program": state.program,
            "index_map": state.index_map,
            "pending_args": list(state.pending_args)  # Save pending args to avoid pollution
        })

        state.quads = target_quads
        state.index_map = {q.idx: i for i, q in enumerate(state.quads)}
        state.pc = 0
        state.variables = {}
        state.temps = {}
        state.return_value = None

        params = state.params_map.get(func_name, [])
        for i, param_name in enumerate(params):
            if i < len(args):
                state.variables[param_name] = args[i]
        return None
    if quad.op == "return":
        value = read_value(state, quad.arg1) if quad.arg1 and quad.arg1 != "_" else 0
        if state.call_stack:
            frame = state.call_stack.pop()
            # Set PC to frame["pc"] - 1, because step() will add 1 after return
            state.pc = frame["pc"] - 1
            state.quads = frame["quads"]
            state.variables = frame["variables"]
            # Restore caller's temps (which include the pre-allocated result temp)
            state.temps = frame["temps"]
            # Restore pending args
            state.pending_args = frame.get("pending_args", [])
            # Write the return value into the result target
            if frame["result_target"] and frame["result_target"] != "_":
                state.temps[frame["result_target"]] = value
            state.program = frame["program"]
            state.index_map = frame["index_map"]
        else:
            state.return_value = value
            state.halted = True
        return None
    if quad.op == "read":
        # Read input from console
        var_name = quad.result
        try:
            user_input = input(f"请输入 {var_name}: ")
            value = int(user_input)
            state.variables[var_name] = value
            state.log(f"读取输入: {var_name} = {value}")
        except ValueError:
            raise ValueError(f"输入错误：'{user_input}' 不是有效的整数")
        except EOFError:
            # If no input available, default to 0
            state.variables[var_name] = 0
            state.log(f"读取输入: {var_name} = 0 (默认值)")
        return None
    if quad.op == "write":
        # Write output to console
        value = read_value(state, quad.arg1)
        print(value)
        state.log(f"输出: {value}")
        return None
    raise ValueError(f"不支持的操作符：{quad.op}")
