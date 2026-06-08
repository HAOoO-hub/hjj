# subset.py：用子集构造法把NFA转换为DFA并记录构造过程
from __future__ import annotations

from collections import deque

from task_4_1_log_scanner.dfa import DFA
from task_4_1_log_scanner.nfa import EPSILON, NFA


def epsilon_closure(nfa: NFA, states: set[int]) -> set[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        state = stack.pop()
        for target in nfa.states[state].transitions.get(EPSILON, set()):
            if target not in closure:
                closure.add(target)
                stack.append(target)
    return closure


def move(nfa: NFA, states: set[int], symbol: str) -> set[int]:
    result: set[int] = set()
    for state in states:
        result.update(nfa.states[state].transitions.get(symbol, set()))
    return result


def nfa_alphabet(nfa: NFA) -> list[str]:
    alphabet: set[str] = set()
    for state in nfa.states.values():
        alphabet.update(symbol for symbol in state.transitions if symbol != EPSILON)
    return sorted(alphabet)


def nfa_to_dfa(nfa: NFA) -> tuple[DFA, list[str]]:
    alphabet = nfa_alphabet(nfa)
    start_set = frozenset(epsilon_closure(nfa, {nfa.start}))
    queue = deque([start_set])
    names = {start_set: "A"}
    state_sets = {"A": start_set}
    transitions: dict[str, dict[str, str]] = {}
    accepts: set[str] = set()
    logs = [f"ε-closure({{{nfa.start}}}) = {sorted(start_set)} -> A"]
    while queue:
        current = queue.popleft()
        name = names[current]
        transitions[name] = {}
        if nfa.accept in current:
            accepts.add(name)
        for symbol in alphabet:
            target = frozenset(epsilon_closure(nfa, move(nfa, set(current), symbol)))
            if not target:
                continue
            if target not in names:
                names[target] = chr(ord("A") + len(names))
                state_sets[names[target]] = target
                queue.append(target)
                logs.append(f"move({name}, '{symbol}') -> {sorted(target)} -> {names[target]}")
            transitions[name][symbol] = names[target]
    return DFA(start="A", accepts=accepts, transitions=transitions, state_sets=state_sets), logs

