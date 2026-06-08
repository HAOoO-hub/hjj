# basic_blocks.py：识别Leader并划分四元式基本块
from __future__ import annotations

from dataclasses import dataclass

from shared.quads import Quad


JUMP_OPS = {"J", "J<", "J>", "J<=", "J>=", "J==", "J!="}


@dataclass#(slots=True)
class BasicBlock:
    label: str
    quads: list[Quad]


def collect_leaders(quads: list[Quad]) -> list[int]:
    if not quads:
        return []
    idx_map = {quad.idx: pos for pos, quad in enumerate(quads)}
    leaders = {quads[0].idx}
    for pos, quad in enumerate(quads):
        if quad.op in JUMP_OPS:
            if quad.result.isdigit():
                leaders.add(int(quad.result))
            if pos + 1 < len(quads):
                leaders.add(quads[pos + 1].idx)
    return sorted(leaders)


def split_basic_blocks(quads: list[Quad]) -> list[BasicBlock]:
    if not quads:
        return []
    leaders = collect_leaders(quads)
    leader_set = set(leaders)
    blocks: list[BasicBlock] = []
    current: list[Quad] = []
    current_label = "entry"
    for i, quad in enumerate(quads):
        if quad.idx in leader_set and current:
            blocks.append(BasicBlock(label=current_label, quads=current))
            current = []
            current_label = f"L{quad.idx}"
        elif not blocks and not current and quad.idx == quads[0].idx:
            current_label = "entry"
        current.append(quad)
        if i + 1 < len(quads) and quads[i + 1].idx in leader_set and current:
            blocks.append(BasicBlock(label=current_label, quads=current))
            current = []
            current_label = f"L{quads[i + 1].idx}"
    if current:
        blocks.append(BasicBlock(label=current_label, quads=current))
    return blocks

