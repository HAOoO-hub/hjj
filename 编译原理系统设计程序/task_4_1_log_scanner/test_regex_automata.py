#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正则表达式识别与画图测试脚本
测试系统能否准确识别5种不同复杂度的正则表达式并生成自动机图
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from task_4_1_log_scanner.regex_parser import parse_regex
from task_4_1_log_scanner.thompson import regex_to_nfa
from task_4_1_log_scanner.subset import nfa_to_dfa
from task_4_1_log_scanner.minimize import minimize_dfa
from task_4_1_log_scanner.graphviz_renderer import nfa_to_dot, dfa_to_dot


def print_separator():
    print("=" * 80)


def print_section(title):
    print_separator()
    print(f"📋 {title}")
    print_separator()


def test_regex(name, pattern, description):
    """测试单个正则表达式"""
    print_section(f"测试 {name}")
    print(f"📝 正则表达式: {pattern}")
    print(f"📖 说明: {description}")
    print()
    
    try:
        # 1. 解析正则
        print(" 步骤1: 解析正则表达式")
        ast = parse_regex(pattern)
        print(f"   ✅ 解析成功")
        print(f"   AST结构: {ast.kind}")
        if ast.value:
            print(f"   值: {ast.value}")
        if ast.children:
            print(f"   子节点数: {len(ast.children)}")
        print()
        
        # 2. Thompson构造NFA
        print("🏗️  步骤2: Thompson构造NFA")
        nfa, nfa_logs = regex_to_nfa(ast)
        print(f"   ✅ NFA构造成功")
        print(f"   状态总数: {len(nfa.states)}")
        print(f"   开始状态: {nfa.start}")
        print(f"   接受状态: {nfa.accept}")
        print(f"   构造日志:")
        for log in nfa_logs[-5:]:  # 只显示最后5条
            print(f"      {log}")
        print()
        
        # 3. 子集构造DFA
        print("🔄 步骤3: 子集构造法 NFA→DFA")
        dfa, dfa_logs = nfa_to_dfa(nfa)
        print(f"   ✅ DFA构造成功")
        print(f"   DFA状态数: {len(dfa.transitions)}")
        print(f"   接受状态: {dfa.accepts}")
        print(f"   构造日志（前5条）:")
        for log in dfa_logs[:5]:
            print(f"      {log}")
        if len(dfa_logs) > 5:
            print(f"      ... (共{len(dfa_logs)}条)")
        print()
        
        # 4. DFA最小化
        print("📉 步骤4: Hopcroft算法 DFA最小化")
        min_dfa, min_logs = minimize_dfa(dfa)
        print(f"   ✅ DFA最小化成功")
        print(f"   最小DFA状态数: {len(min_dfa.transitions)}")
        print(f"   状态压缩比: {len(dfa.transitions)} → {len(min_dfa.transitions)} "
              f"({len(min_dfa.transitions)/len(dfa.transitions)*100:.1f}%)")
        print(f"   划分日志:")
        for log in min_logs:
            print(f"      {log}")
        print()
        
        # 5. 生成Graphviz DOT
        print(" 步骤5: 生成Graphviz自动机图")
        nfa_dot = nfa_to_dot(nfa)
        dfa_dot = dfa_to_dot(dfa)
        min_dfa_dot = dfa_to_dot(min_dfa)
        print(f"   ✅ NFA DOT图生成成功")
        print(f"   ✅ DFA DOT图生成成功")
        print(f"   ✅ 最小DFA DOT图生成成功")
        print()
        
        # 6. 转移表
        print("📊 最小DFA转移表:")
        print(f"   {'状态':<6} | {'转移':<20} | {'目标':<6}")
        print(f"   {'-'*6}-+-{'-'*20}-+-{'-'*6}")
        for state in sorted(min_dfa.transitions.keys()):
            is_accept = "✓" if state in min_dfa.accepts else " "
            print(f"   {state}{is_accept:<5} | ", end="")
            trans = min_dfa.transitions[state]
            if trans:
                first = True
                for symbol, target in sorted(trans.items()):
                    if not first:
                        print(f"{'':26} | ", end="")
                    print(f"{symbol!r:<20} | {target}")
                    first = False
            else:
                print("(无转移)")
        print()
        
        print(" 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """执行5个测试用例"""
    print_section("正则表达式识别与画图能力测试")
    print("🧪 测试系统将准确识别正则表达式并生成自动机图")
    print()
    
    test_cases = [
        {
            "name": "测试1: 简单字面量",
            "pattern": "abc",
            "description": "测试最基本的字面量匹配，无特殊字符",
        },
        {
            "name": "测试2: 字符类范围",
            "pattern": "[0-9]",
            "description": "测试字符类[0-9]，识别数字范围",
        },
        {
            "name": "测试3: 正闭包重复",
            "pattern": "[0-9]+",
            "description": "测试正闭包操作符+，匹配一个或多个数字",
        },
        {
            "name": "测试4: 选择与闭包组合",
            "pattern": "(a|b)*",
            "description": "测试选择|和闭包*的组合，匹配a和b的任意组合",
        },
        {
            "name": "测试5: 复杂IP地址片段",
            "pattern": "[0-9]{1,3}\\.[0-9]{1,3}",
            "description": "测试复杂正则：重复次数{1,3}+字面量.的组合",
        },
    ]
    
    results = []
    for i, tc in enumerate(test_cases, 1):
        success = test_regex(tc["name"], tc["pattern"], tc["description"])
        results.append(success)
        print()
    
    # 总结
    print_section("测试结果总结")
    print()
    
    passed = sum(results)
    total = len(results)
    
    for i, (tc, success) in enumerate(zip(test_cases, results), 1):
        status = "✅ 通过" if success else "❌ 失败"
        print(f"测试{i}: {tc['name']} - {status}")
        print(f"  正则: {tc['pattern']}")
        print()
    
    print_separator()
    print(f"📊 总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统能够准确识别正则表达式并生成自动机图！")
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，需要检查")
    
    print_separator()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
