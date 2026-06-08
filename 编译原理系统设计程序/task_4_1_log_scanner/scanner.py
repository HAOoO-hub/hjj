# scanner.py：使用多个最小DFA扫描日志并提取关键词
from __future__ import annotations

from dataclasses import dataclass

from task_4_1_log_scanner.dfa import DFA
from task_4_1_log_scanner.regex_spec import PRIORITY


@dataclass
class ScanMatch:
    category: str
    text: str
    line_no: int
    start: int
    end: int


def run_dfa(dfa: DFA, text: str, start: int) -> int:
    state = dfa.start
    longest = -1
    pos = start
    while pos < len(text):
        ch = text[pos]
        nxt = dfa.transitions.get(state, {}).get(ch)
        if nxt is None:
            break
        state = nxt
        pos += 1
        if state in dfa.accepts:
            longest = pos
    return longest


def previous_token(line: str, start: int) -> str:
    i = start - 1
    while i >= 0 and line[i].isspace():
        i -= 1
    end = i + 1
    while i >= 0 and not line[i].isspace():
        i -= 1
    return line[i + 1 : end]


def user_enabled(line: str, start: int) -> bool:
    return previous_token(line, start) in {"user=", "USER=", "User"}


def scan_line(line: str, line_no: int, dfas: dict[str, DFA]) -> list[ScanMatch]:
    matches: list[ScanMatch] = []
    pos = 0
    priorities = {name: i for i, name in enumerate(PRIORITY)}
    while pos < len(line):
        candidates: list[ScanMatch] = []
        for name in PRIORITY:
            if name == "USER" and not user_enabled(line, pos):
                continue
            end = run_dfa(dfas[name], line, pos)
            if end != -1 and end > pos:
                candidates.append(ScanMatch(name, line[pos:end], line_no, pos, end))
        if not candidates:
            pos += 1
            continue
        candidates.sort(key=lambda item: (-(item.end - item.start), priorities[item.category]))
        best = candidates[0]
        matches.append(best)
        pos = best.end
    return matches


def scan_text(text: str, dfas: dict[str, DFA]) -> list[ScanMatch]:
    result: list[ScanMatch] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        result.extend(scan_line(line, line_no, dfas))
    return result

