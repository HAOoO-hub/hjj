# minimize.py：最小化DFA并输出状态划分过程日志
from __future__ import annotations

from task_4_1_log_scanner.dfa import DFA


def minimize_dfa(dfa: DFA) -> tuple[DFA, list[str]]:
    alphabet = sorted({symbol for row in dfa.transitions.values() for symbol in row})
    non_accepts = set(dfa.transitions) - dfa.accepts
    partitions = [set(dfa.accepts), non_accepts]
    partitions = [p for p in partitions if p]
    logs = [f"初始划分：{[sorted(p) for p in partitions]}"]
    changed = True
    while changed:
        changed = False
        new_partitions: list[set[str]] = []
        for part in partitions:
            buckets: dict[tuple[int, ...], set[str]] = {}
            for state in sorted(part):
                signature = []
                for symbol in alphabet:
                    target = dfa.transitions.get(state, {}).get(symbol)
                    index = next((i for i, p in enumerate(partitions) if target in p), -1)
                    signature.append(index)
                buckets.setdefault(tuple(signature), set()).add(state)
            new_partitions.extend(buckets.values())
            if len(buckets) > 1:
                changed = True
        partitions = new_partitions
        logs.append(f"细分后：{[sorted(p) for p in partitions]}")
    name_of: dict[str, str] = {}
    state_sets: dict[str, frozenset[int]] = {}
    for idx, part in enumerate(partitions):
        name = f"M{idx}"
        for state in part:
            name_of[state] = name
        merged: set[int] = set()
        for state in part:
            merged.update(dfa.state_sets[state])
        state_sets[name] = frozenset(merged)
    transitions: dict[str, dict[str, str]] = {}
    accepts: set[str] = set()
    for idx, part in enumerate(partitions):
        rep = sorted(part)[0]
        name = f"M{idx}"
        transitions[name] = {}
        if part & dfa.accepts:
            accepts.add(name)
        for symbol, target in dfa.transitions.get(rep, {}).items():
            transitions[name][symbol] = name_of[target]
    return DFA(start=name_of[dfa.start], accepts=accepts, transitions=transitions, state_sets=state_sets), logs

