# main_window.py：组装统一主窗口并切换源码编译与日志扫描工作台
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.compiler_workspace import CompilerWorkspace
from ui.log_workspace import LogWorkspace


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("编译原理课程设计综合工作台")
        self.geometry("1500x920")
        self.mode = tk.StringVar(value="compiler")
        self._build()

    def _build(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="工作模式：").pack(side="left")
        ttk.Radiobutton(toolbar, text="源码编译工作台", value="compiler", variable=self.mode, command=self.switch_mode).pack(side="left", padx=6)
        ttk.Radiobutton(toolbar, text="日志扫描工作台", value="logs", variable=self.mode, command=self.switch_mode).pack(side="left", padx=6)
        
        # 在右上角添加使用指南按钮
        ttk.Button(
            toolbar,
            text="📖 好纠结的解释器使用指南",
            command=self.show_user_guide
        ).pack(side="right", padx=10)

        self.host = ttk.Frame(self)
        self.host.pack(fill="both", expand=True)

        # Create workspaces but don't pack them yet
        self.compiler_workspace = CompilerWorkspace(self.host)
        self.log_workspace = LogWorkspace(self.host)
        
        # Pack compiler workspace by default
        self.compiler_workspace.pack(fill="both", expand=True)

    def switch_mode(self) -> None:
        for child in self.host.winfo_children():
            child.pack_forget()
        if self.mode.get() == "compiler":
            self.compiler_workspace.pack(fill="both", expand=True)
        else:
            # Check if log_workspace exists
            if hasattr(self, 'log_workspace') and self.log_workspace is not None:
                try:
                    self.log_workspace.pack(fill="both", expand=True)
                except Exception as e:
                    print(f"[Error] Failed to pack log_workspace: {e}")
            else:
                print("[Error] log_workspace not initialized properly")

    def show_user_guide(self) -> None:
        """显示使用指南对话框"""
        guide_window = tk.Toplevel(self)
        guide_window.title("📖 好纠结的解释器使用指南")
        guide_window.geometry("900x700")
        guide_window.transient(self)
        
        # 居中显示
        guide_window.update_idletasks()
        x = (guide_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (guide_window.winfo_screenheight() // 2) - (700 // 2)
        guide_window.geometry(f"+{x}+{y}")
        
        # 创建滚动框架
        main_frame = ttk.Frame(guide_window, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 绑定鼠标滚轮事件
        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        
        # 关闭对话框时解绑事件
        guide_window.protocol("WM_DELETE_WINDOW", lambda: [canvas.unbind_all("<MouseWheel>"), guide_window.destroy()])
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 添加内容
        self._add_guide_content(scrollable_frame)
        
        # 关闭按钮
        close_btn = ttk.Button(
            guide_window,
            text="关闭",
            command=lambda: [canvas.unbind_all("<MouseWheel>"), guide_window.destroy()]
        )
        close_btn.pack(pady=10)
    
    def _add_guide_content(self, parent: ttk.Frame) -> None:
        """添加使用指南内容"""
        
        # 标题
        title_label = ttk.Label(
            parent,
            text="🎯 C-Mini 编译器核心功能与巧思设计",
            font=("Microsoft YaHei UI", 14, "bold")
        )
        title_label.pack(anchor="w", pady=(0, 15))
        
        # 分隔线
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=5)
        
        # ========== 第一部分：源码编译工作台 ==========
        section1_title = ttk.Label(
            parent,
            text="══════════ 第一部分：源码编译工作台 ══════════",
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#FF6347"
        )
        section1_title.pack(anchor="w", pady=(15, 10))
        
        # 功能 1
        self._add_section(
            parent,
            "✨ 巧思 1：测试样例快速切换",
            "📁 位置：源代码编辑区顶部工具栏\n"
            "💡 功能：一键加载 8 个精心设计的测试样例\n"
            "🎨 特色：中文名称映射，双击即可加载\n"
            "📝 样例包括：\n"
            "   • 综合样例1/2/3（常量、递归、break/continue）\n"
            "   • 递归示例（阶乘计算）\n"
            "   • 错误检测样例（7种错误）\n"
            "   • break/continue 专项测试"
        )
        
        # 功能 2
        self._add_section(
            parent,
            "✨ 巧思 2：智能变量查询",
            "🔍 位置：源代码编辑区顶部工具栏\n"
            "💡 功能：输入变量名，一键查询值\n"
            "🎨 特色：\n"
            "   • 支持回车键快速查询\n"
            "   • 输入'结果'查看程序返回值（避免与result变量冲突）\n"
            "   • 智能错误提示：'输入值不存在'\n"
            "📝 使用示例：\n"
            "   输入 'a' → 显示变量 a 的值\n"
            "   输入 '结果' → 显示程序返回值"
        )
        
        # 功能 3
        self._add_section(
            parent,
            "✨ 巧思 3：四张符号表管理",
            "📊 位置：语义分析标签页\n"
            "💡 功能：总符号表、变量表、函数表、常量表\n"
            "🎨 特色：\n"
            "   • 常量表独立管理（const 声明）\n"
            "   • 作用域清晰，冲突检测准确\n"
            "   • 实时显示所有符号信息"
        )
        
        # 功能 4
        self._add_section(
            parent,
            "✨ 巧思 4：break/continue 循环控制",
            "🔄 位置：while 循环内部\n"
            "💡 功能：跳出循环 / 继续下一次迭代\n"
            "🎨 特色：\n"
            "   • 栈式管理嵌套循环上下文\n"
            "   • 语义错误检测：循环外使用会报错\n"
            "   • 无缝集成到 IR 生成和解释器"
        )
        
        # 功能 5
        self._add_section(
            parent,
            "✨ 巧思 5：递归调用支持",
            "♻️ 位置：函数定义中\n"
            "💡 功能：支持线性递归（如阶乘）\n"
            "🎨 特色：\n"
            "   • 完整的 call/return 机制\n"
            "   • 调用栈管理，上下文保存与恢复\n"
            "   • 适合教学演示（factorial(5) = 120）\n"
            "⚠️ 注意：双递归（fibonacci）存在已知限制"
        )
        
        # 功能 6
        self._add_section(
            parent,
            "✨ 巧思 6：全面的错误检测",
            "❌ 位置：词法、语法、语义三个阶段\n"
            "💡 功能：检测 20+ 种错误类型\n"
            "🎨 特色：\n"
            "   • 词法错误：非法字符\n"
            "   • 语法错误：缺少分号、括号不匹配等\n"
            "   • 语义错误：变量未定义、重复声明等\n"
            "   • 友好的错误提示和修改建议"
        )
        
        # 功能 7
        self._add_section(
            parent,
            "✨ 巧思 7：自动格式化与高亮",
            "🎨 位置：编辑器自动应用\n"
            "💡 功能：代码格式化和语法高亮\n"
            "🎨 特色：\n"
            "   • 自动缩进（花括号后增加 4 空格）\n"
            "   • 关键字、标识符、注释不同颜色\n"
            "   • 函数名特殊高亮"
        )
        
        # 功能 8
        self._add_section(
            parent,
            "✨ 巧思 8：中间代码解释执行",
            "▶️ 位置：中间代码解释标签页\n"
            "💡 功能：逐条执行四元式，模拟程序运行\n"
            "🎨 特色：\n"
            "   • 完整的运行时状态管理\n"
            "   • 变量表实时更新\n"
            "   • 执行轨迹记录\n"
            "   • 可配置最大执行步数"
        )
        
        # ========== 第二部分：日志扫描工作台 ==========
        section2_title = ttk.Label(
            parent,
            text="══════════ 第二部分：日志扫描工作台 ══════════",
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#FF6347"
        )
        section2_title.pack(anchor="w", pady=(20, 10))
        
        # 功能 9
        self._add_section(
            parent,
            "✨ 巧思 9：正则表达式 → NFA 构造",
            "🔧 位置：日志扫描工作台 - 重建自动机\n"
            "💡 功能：Thompson 构造法将正则转为 NFA\n"
            "🎨 特色：\n"
            "   • 可视化 NFA 构建过程\n"
            "   • 支持复杂正则表达式\n"
            "   • ε-转移清晰展示\n"
            "📝 应用场景：日志关键词提取、模式匹配"
        )
        
        # 功能 10
        self._add_section(
            parent,
            "✨ 巧思 10：NFA → DFA 子集构造",
            "🔄 位置：日志扫描工作台 - DFA 标签页\n"
            "💡 功能：子集构造算法将 NFA 确定化\n"
            "🎨 特色：\n"
            "   • 自动消除 ε-转移\n"
            "   • 状态合并优化\n"
            "   • 转移表直观展示"
        )
        
        # 功能 11
        self._add_section(
            parent,
            "✨ 巧思 11：DFA 最小化",
            "📉 位置：日志扫描工作台 - 最小DFA 标签页\n"
            "💡 功能：Hopcroft 算法最小化 DFA\n"
            "🎨 特色：\n"
            "   • 等价状态合并\n"
            "   • 减少状态数量\n"
            "   • 提高匹配效率\n"
            "📊 效果：通常可减少 30%-50% 的状态数"
        )
        
        # 功能 12
        self._add_section(
            parent,
            "✨ 巧思 12：字符范围压缩显示",
            "🎯 位置：自动机图转移边标签\n"
            "💡 功能：将多个连续字符合并为范围表示\n"
            "🎨 特色：\n"
            "   • '0,1,2,3,4,5,6,7,8,9' → '0-9'\n"
            "   • 'a,b,c,...,z' → 'a-z'\n"
            "   • 图形更简洁，易读性提升\n"
            "💡 价值：大幅简化自动机图的视觉复杂度"
        )
        
        # 功能 13
        self._add_section(
            parent,
            "✨ 巧思 13：交互式自动机可视化",
            "🖼️ 位置：自动机图标签页\n"
            "💡 功能：Graphviz 渲染的精美自动机图\n"
            "🎨 特色：\n"
            "   • 缩放功能：放大/缩小/重置\n"
            "   • 鼠标滚轮滚动浏览\n"
            "   • 横向和纵向滚动条\n"
            "   • 起始状态和接受状态特殊标记\n"
            "🎨 配色：浅绿（起始）、浅蓝（接受）、浅灰（普通）"
        )
        
        # 功能 14
        self._add_section(
            parent,
            "✨ 巧思 14：多类别日志扫描",
            "📋 位置：日志扫描工作台 - 类别选择\n"
            "💡 功能：支持多种日志类型分类扫描\n"
            "🎨 特色：\n"
            "   • IP 地址提取\n"
            "   • 时间戳识别\n"
            "   • 错误码匹配\n"
            "   • 优先级排序\n"
            "📝 输出：结构化关键词列表"
        )
        
        # 功能 15
        self._add_section(
            parent,
            "✨ 巧思 15：构造过程日志",
            "📜 位置：构造过程标签页\n"
            "💡 功能：详细记录自动机构建步骤\n"
            "🎨 特色：\n"
            "   • 正则解析过程\n"
            "   • NFA 构建步骤\n"
            "   • DFA 子集构造细节\n"
            "   • 最小化算法执行轨迹\n"
            "💡 价值：教学演示和调试利器"
        )
        
        # 功能 16
        self._add_section(
            parent,
            "✨ 巧思 16：转移表展示",
            "📊 位置：转移表标签页\n"
            "💡 功能：表格形式展示状态转移\n"
            "🎨 特色：\n"
            "   • 清晰的行列结构\n"
            "   • 支持滚动浏览\n"
            "   • 便于对照学习\n"
            "   • 导出方便"
        )
        
        # ========== 第三部分：通用设计 ===========
        section3_title = ttk.Label(
            parent,
            text="══════════ 第三部分：通用设计巧思 ══════════",
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#FF6347"
        )
        section3_title.pack(anchor="w", pady=(20, 10))
        
        # 功能 17
        self._add_section(
            parent,
            "✨ 巧思 17：模块化架构设计",
            "🏗️ 设计理念：高内聚低耦合\n"
            "💡 特色：\n"
            "   • 前端（lexer/parser/semantic）独立\n"
            "   • 后端（IR generator）独立\n"
            "   • 解释器独立\n"
            "   • 日志扫描器独立\n"
            "💡 价值：易于扩展和维护"
        )
        
        # 功能 18
        self._add_section(
            parent,
            "✨ 巧思 18：缓存优化机制",
            "⚡ 位置：自动机图渲染\n"
            "💡 功能：缓存已生成的 PNG 图片\n"
            "🎨 特色：\n"
            "   • 首次生成较慢，后续秒开\n"
            "   • 显著提升用户体验\n"
            "   • 智能缓存管理"
        )
        
        # 功能 19
        self._add_section(
            parent,
            "✨ 巧思 19：双工作台切换",
            "🔄 位置：顶部工具栏\n"
            "💡 功能：源码编译和日志扫描一键切换\n"
            "🎨 特色：\n"
            "   • 统一界面，风格一致\n"
            "   • 数据隔离，互不干扰\n"
            "   • 快速切换，高效工作"
        )
        
        # 功能 20
        self._add_section(
            parent,
            "✨ 巧思 20：友好的用户交互",
            "😊 设计理念：用户体验优先\n"
            "💡 特色：\n"
            "   • 自动分析（300ms 延迟）\n"
            "   • 实时高亮反馈\n"
            "   • 智能错误提示\n"
            "   • 快捷键支持（回车查询）\n"
            "   • 居中对话框\n"
            "   • 滚动支持"
        )
        
        # 快速开始
        self._add_section(
            parent,
            "🚀 快速开始（3 步上手）",
            "【源码编译工作台】\n"
            "步骤 1：点击 '📁 测试样例' 加载 demo_comprehensive_1.cmini\n"
            "步骤 2：等待自动分析完成\n"
            "步骤 3：切换到 '中间代码解释' 查看执行结果\n\n"
            "【日志扫描工作台】\n"
            "步骤 1：切换到 '日志扫描工作台' 模式\n"
            "步骤 2：点击 '重建自动机' 生成 NFA/DFA\n"
            "步骤 3：在左侧输入日志文本，点击 '扫描日志'\n\n"
            "💡 提示：\n"
            "• 在变量查询框输入 '结果' 查看返回值\n"
            "• 加载 demo_errors.cmini 学习错误检测\n"
            "• 加载 demo_recursion.cmini 体验递归功能\n"
            "• 使用缩放按钮调整自动机图大小"
        )
        
        # 底部提示
        tip_label = ttk.Label(
            parent,
            text="💖 编译原理的学习旅程中不纠结！",
            font=("Microsoft YaHei UI", 11),
            foreground="#2E8B57"
        )
        tip_label.pack(anchor="center", pady=20)
    
    def _add_section(self, parent: ttk.Frame, title: str, content: str) -> None:
        """添加一个功能章节"""
        # 标题
        title_label = ttk.Label(
            parent,
            text=title,
            font=("Microsoft YaHei UI", 11, "bold"),
            foreground="#1E90FF"
        )
        title_label.pack(anchor="w", pady=(15, 5))
        
        # 内容
        content_label = ttk.Label(
            parent,
            text=content,
            font=("Microsoft YaHei UI", 10),
            justify="left",
            wraplength=820
        )
        content_label.pack(anchor="w", padx=20)

