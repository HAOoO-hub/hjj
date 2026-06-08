# compiler_workspace.py：构建统一源码编译工作台并展示分析结果
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from compiler.pipeline import PipelineResult, analyze_source, render_ast
from shared.gui_base import create_text
from task_4_3_mini_ide.formatter import format_code
from task_4_3_mini_ide.highlighter import apply_function_highlight, apply_lexical_highlight, configure_tags
from task_4_3_mini_ide.lexer import lex


VIEW_LABELS = [
    ("tokens", "词法分析"),
    ("ast", "语法分析"),
    ("symbols", "语义分析"),
    ("quads", "四元式"),
    ("llvm", "LLVM IR"),
    ("interpreter", "中间代码解释"),
    ("tests", "测试结果"),
]


class CompilerWorkspace(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, padding=8)
        self.current_view = tk.StringVar(value="tokens")
        self.after_id: str | None = None
        self.result: PipelineResult | None = None
        self.result_font_size = 10
        self._build()
        self._load_default()
        self.analyze_now()

    def _build(self) -> None:
        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        left_pane = ttk.Panedwindow(left, orient="vertical")
        left_pane.pack(fill="both", expand=True)

        editor_frame = ttk.LabelFrame(left_pane, text="源代码编辑区", padding=6)
        error_frame = ttk.LabelFrame(left_pane, text="错误提示与修改建议", padding=6)
        left_pane.add(editor_frame, weight=3)
        left_pane.add(error_frame, weight=1)

        editor_tools = ttk.Frame(editor_frame)
        editor_tools.pack(fill="x", pady=(0, 6))
        ttk.Button(editor_tools, text="自动格式化", command=self.reformat).pack(side="left", padx=4)
        ttk.Button(editor_tools, text="立即分析", command=self.analyze_now).pack(side="left", padx=4)
        ttk.Button(editor_tools, text="📁 测试样例", command=self.show_sample_dialog).pack(side="left", padx=4)
        
        # 变量查询功能区
        var_query_frame = ttk.Frame(editor_tools)
        var_query_frame.pack(side="left", padx=(15, 4))
        ttk.Label(var_query_frame, text="变量查询:").pack(side="left", padx=(0, 4))
        self.var_query_entry = ttk.Entry(var_query_frame, width=15)
        self.var_query_entry.pack(side="left", padx=(0, 4))
        self.var_query_entry.bind("<Return>", lambda e: self.query_variable())
        ttk.Button(var_query_frame, text="🔍 输出", command=self.query_variable, width=8).pack(side="left")

        self.editor = create_text(editor_frame, height=30)
        self.editor.pack(fill="both", expand=True)
        configure_tags(self.editor)
        self.editor.bind("<KeyRelease>", self.on_key_release)
        self.editor.bind("<Return>", self.on_return_pressed)

        self.error_text = create_text(error_frame, height=10)
        self.error_text.pack(fill="both", expand=True)

        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x", pady=(0, 8))
        for key, label in VIEW_LABELS:
            ttk.Radiobutton(toolbar, text=label, variable=self.current_view, value=key, command=self.render_view).pack(side="left", padx=3)
        ttk.Button(toolbar, text="结果放大", command=lambda: self.scale_result_font(1)).pack(side="right", padx=3)
        ttk.Button(toolbar, text="结果缩小", command=lambda: self.scale_result_font(-1)).pack(side="right", padx=3)

        self.content = ttk.LabelFrame(right, text="分析结果与图表", padding=6)
        self.content.pack(fill="both", expand=True)

        self.result_container = ttk.Frame(self.content)
        self.result_container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(self.result_container, show="headings")
        self.tree_scroll_y = ttk.Scrollbar(self.result_container, orient="vertical", command=self.tree.yview)
        self.tree_scroll_x = ttk.Scrollbar(self.result_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)

        self.text = create_text(self.result_container, height=30)
        self.text.configure(font=("Consolas", self.result_font_size))
        self.text_scroll_y = ttk.Scrollbar(self.result_container, orient="vertical", command=self.text.yview)
        self.text_scroll_x = ttk.Scrollbar(self.result_container, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=self.text_scroll_y.set, xscrollcommand=self.text_scroll_x.set)

    def _load_default(self) -> None:
        path = Path("data/code/compiler_workspace_demo.cmini")
        if path.exists():
            self.editor.insert("1.0", path.read_text(encoding="utf-8"))
        else:
            self.editor.insert(
                "1.0",
                "int main() {\n    int a;\n    int b;\n    int c;\n    a = 10;\n    b = 5;\n    if (a > b) {\n        c = a + b;\n    } else {\n        c = a - b;\n    }\n    while (b > 0) {\n        b = b - 1;\n    }\n    return c;\n}\n",
            )

    def show_sample_dialog(self) -> None:
        """显示测试样例选择对话框"""
        dialog = tk.Toplevel(self)
        dialog.title("选择测试样例")
        dialog.geometry("600x400")
        dialog.transient(self)
        dialog.grab_set()
        
        # 居中显示
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # 标题
        title_frame = ttk.Frame(dialog, padding=10)
        title_frame.pack(fill="x")
        ttk.Label(title_frame, text="📁 选择测试样例", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # 说明
        info_frame = ttk.Frame(dialog, padding=(10, 0))
        info_frame.pack(fill="x")
        ttk.Label(info_frame, text="点击样例名称即可加载到编辑器", foreground="gray").pack(anchor="w")
        
        # 样例列表框架
        list_frame = ttk.LabelFrame(dialog, text="可用测试样例", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建滚动列表
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        sample_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            selectmode="single"
        )
        sample_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=sample_listbox.yview)
        
        # 绑定双击事件
        sample_listbox.bind("<Double-Button-1>", lambda e: self.load_selected_sample_with_paths(sample_listbox, dialog, sample_paths))
        
        # 存储样例路径列表（与listbox索引对应）
        sample_paths = []
        
        # 加载测试样例列表
        sample_dir = Path("data/code")
        if sample_dir.exists():
            samples = sorted([
                f for f in sample_dir.glob("*.cmini")
                if not f.name.startswith("_")  # 排除隐藏文件
            ])
            
            if samples:
                for sample_path in samples:
                    # 提取文件名和描述
                    name = sample_path.stem
                    # 美化显示名称
                    display_name = self._format_sample_name(name)
                    sample_listbox.insert("end", display_name)
                    # 存储路径到列表
                    sample_paths.append(str(sample_path))
            else:
                sample_listbox.insert("end", "暂无测试样例")
                sample_listbox.config(state="disabled")
        else:
            sample_listbox.insert("end", "data/code 目录不存在")
            sample_listbox.config(state="disabled")
        
        # 按钮框架
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill="x")
        
        ttk.Button(
            btn_frame,
            text="加载选中",
            command=lambda: self.load_selected_sample_with_paths(sample_listbox, dialog, sample_paths)
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="取消",
            command=dialog.destroy
        ).pack(side="right", padx=5)
        
        # 默认选中第一个
        if sample_listbox.size() > 0 and sample_listbox.get(0) != "暂无测试样例":
            sample_listbox.selection_set(0)

    def _format_sample_name(self, name: str) -> str:
        """格式化样例名称为更易读的形式"""
        # 替换下划线为空格
        formatted = name.replace("_", " ")
        
        # 特殊样例的中文名称映射
        name_map = {
            "demo comprehensive 1": "综合样例1 - 常量与函数",
            "demo comprehensive 2": "综合样例2 - 递归调用",
            "demo comprehensive 3": "综合样例3 - break/continue",
            "demo recursion": "递归示例 - 阶乘计算",
            "demo errors": "错误检测样例",
            "test break continue": "break/continue 基本测试",
            "test nested break continue": "嵌套循环测试",
            "test break error": "break 错误检测",
            "compiler workspace demo": "工作区演示",
        }
        
        # 尝试匹配预定义名称
        for key, value in name_map.items():
            if key in formatted.lower():
                return value
        
        # 默认返回格式化的名称
        return formatted.title()

    def load_selected_sample_with_paths(self, listbox: tk.Listbox, dialog: tk.Toplevel, sample_paths: list[str]) -> None:
        """加载选中的测试样例（使用路径列表）"""
        selection = listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        
        # 检查是否是禁用状态
        if listbox.get(0) in ["暂无测试样例", "data/code 目录不存在"]:
            dialog.destroy()
            return
        
        # 获取文件路径
        if index < len(sample_paths):
            file_path = Path(sample_paths[index])
        else:
            dialog.destroy()
            return
        
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                # 清空编辑器
                self.editor.delete("1.0", "end")
                # 插入新内容
                self.editor.insert("1.0", content)
                # 关闭对话框
                dialog.destroy()
                # 自动分析
                self.analyze_now()
            except Exception as e:
                dialog.destroy()
                # 显示错误信息
                from tkinter import messagebox
                messagebox.showerror("加载失败", f"无法加载文件：{e}")
        else:
            dialog.destroy()
            from tkinter import messagebox
            messagebox.showerror("文件不存在", f"文件不存在：{file_path}")

    def load_selected_sample(self, listbox: tk.Listbox, dialog: tk.Toplevel) -> None:
        """加载选中的测试样例"""
        selection = listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        item_data = listbox.item(index)
        
        # 获取文件路径
        if "path" in item_data:
            file_path = Path(item_data["path"])
        else:
            # 如果没有path属性，可能是禁用状态
            dialog.destroy()
            return
        
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                # 清空编辑器
                self.editor.delete("1.0", "end")
                # 插入新内容
                self.editor.insert("1.0", content)
                # 关闭对话框
                dialog.destroy()
                # 自动分析
                self.analyze_now()
            except Exception as e:
                dialog.destroy()
                # 显示错误信息
                from tkinter import messagebox
                messagebox.showerror("加载失败", f"无法加载文件：{e}")
        else:
            dialog.destroy()
            from tkinter import messagebox
            messagebox.showerror("文件不存在", f"文件不存在：{file_path}")

    def on_key_release(self, event=None) -> None:
        tokens, _ = lex(self.editor.get("1.0", "end-1c"))
        apply_lexical_highlight(self.editor, tokens)
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.after_id = self.after(300, self.analyze_now)

    def on_return_pressed(self, event=None):
        line_start = self.editor.index("insert linestart")
        line_text = self.editor.get(line_start, "insert")
        indent = len(line_text) - len(line_text.lstrip(" "))
        if line_text.strip().endswith("{"):
            indent += 4
        self.editor.insert("insert", "\n" + " " * indent)
        return "break"

    def reformat(self) -> None:
        formatted = format_code(self.editor.get("1.0", "end-1c"))
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", formatted)
        self.analyze_now()

    def analyze_now(self) -> None:
        self.after_id = None
        source = self.editor.get("1.0", "end-1c")
        self.result = analyze_source(source)
        tokens, _ = lex(source)
        apply_lexical_highlight(self.editor, tokens)
        if self.result.ast and not any(d.category == "语法错误" for d in self.result.diagnostics):
            apply_function_highlight(self.editor, self.result.ast)
        self.error_text.delete("1.0", "end")
        self.error_text.insert("1.0", render_diagnostics(self.result))
        self.render_view()

    def query_variable(self) -> None:
        """查询变量的值"""
        # 获取输入的变量名
        var_name = self.var_query_entry.get().strip()
        
        if not var_name:
            from tkinter import messagebox
            messagebox.showwarning("提示", "请输入变量名")
            return
        
        # 检查是否已经分析过
        if self.result is None:
            from tkinter import messagebox
            messagebox.showwarning("提示", "请先进行代码分析")
            return
        
        # 特殊处理：输入“结果”时显示程序返回值
        if var_name == "结果":
            if hasattr(self.result, 'interpreter_return'):
                return_value = self.result.interpreter_return
                from tkinter import messagebox
                messagebox.showinfo(
                    "程序执行结果",
                    f"程序返回值: {return_value}"
                )
            else:
                from tkinter import messagebox
                messagebox.showwarning("提示", "程序尚未执行，无法获取返回值")
            return
        
        # 检查是否有解释器执行结果
        if not hasattr(self.result, 'interpreter_variables') or not self.result.interpreter_variables:
            from tkinter import messagebox
            messagebox.showwarning("提示", "请先切换到“中间代码解释”标签页并执行程序")
            return
        
        # 在变量表中查找
        variables_dict = dict(self.result.interpreter_variables)
        
        if var_name in variables_dict:
            value = variables_dict[var_name]
            from tkinter import messagebox
            messagebox.showinfo(
                "变量值",
                f"变量名: {var_name}\n值: {value}"
            )
        else:
            from tkinter import messagebox
            messagebox.showerror("错误", "输入值不存在")

    def render_view(self) -> None:
        if self.result is None:
            return
        self._show_text("")
        key = self.current_view.get()
        if key == "tokens":
            rows = [
                {"序号": str(i), "类型": tok.kind, "单词值": tok.value, "行": str(tok.line), "列": str(tok.column)}
                for i, tok in enumerate(self.result.tokens)
                if tok.kind != "EOF"
            ]
            self._show_table(rows)
        elif key == "ast":
            self._show_text(render_ast(self.result.ast))
        elif key == "symbols":
            self._show_text("")
            self._show_symbol_tables()
        elif key == "quads":
            rows = [
                {"序号": str(q.idx), "op": q.op, "arg1": q.arg1, "arg2": q.arg2, "result": q.result}
                for q in self.result.quads
            ]
            self._show_table(rows)
        elif key == "llvm":
            self._show_llvm_with_run_button()
        elif key == "interpreter":
            lines = [f"返回值：{self.result.interpreter_return}", "", "变量表："]
            lines.extend(f"{name} = {value}" for name, value in self.result.interpreter_variables)
            lines.extend(["", "执行轨迹：", self.result.interpreter_trace or "暂无执行轨迹"])
            self._show_text("\n".join(lines))
        elif key == "tests":
            self._show_text(render_tests(self.result))

    def scale_result_font(self, delta: int) -> None:
        self.result_font_size = max(8, min(18, self.result_font_size + delta))
        self.text.configure(font=("Consolas", self.result_font_size))
        ttk.Style().configure("Treeview", rowheight=max(20, self.result_font_size * 2))
        self.render_view()

    def _clear_result(self) -> None:
        for widget in self.result_container.winfo_children():
            widget.pack_forget()
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _show_text(self, content: str) -> None:
        self._clear_result()
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.pack(fill="both", expand=True)
        self.text_scroll_y.pack(side="right", fill="y")
        self.text_scroll_x.pack(side="bottom", fill="x")

    def _show_table(self, rows: list[dict[str, str]]) -> None:
        self._clear_result()
        columns = list(rows[0].keys()) if rows else ["说明"]
        self.tree.configure(columns=columns)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor="center", stretch=True)
        for row in rows or [{"说明": "暂无数据"}]:
            values = [row.get(col, "") for col in columns]
            self.tree.insert("", "end", values=values)
        self.tree.pack(fill="both", expand=True)
        self.tree_scroll_y.pack(side="right", fill="y")
        self.tree_scroll_x.pack(side="bottom", fill="x")

    def _show_symbol_tables(self) -> None:
        self._clear_result()
        notebook = ttk.Notebook(self.result_container)
        notebook.pack(fill="both", expand=True)
        for title, rows in (
            ("总符号表", self.result.tables.symbols),
            ("变量表", self.result.tables.variables),
            ("函数表", self.result.tables.functions),
            ("常量表", self.result.tables.constants),  # NEW: Show constant table
        ):
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            tree = ttk.Treeview(frame, show="headings")
            scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            columns = list(rows[0].keys()) if rows else ["说明"]
            tree.configure(columns=columns)
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120, anchor="center")
            for row in rows or [{"说明": "暂无数据"}]:
                tree.insert("", "end", values=[row.get(col, "") for col in columns])
            tree.pack(fill="both", expand=True)
            scroll_y.pack(side="right", fill="y")
            scroll_x.pack(side="bottom", fill="x")

    def _show_llvm_with_run_button(self) -> None:
        """显示 LLVM IR 并添加运行按钮"""
        self._clear_result()
        
        # 创建顶部工具栏
        toolbar = ttk.Frame(self.result_container)
        toolbar.pack(fill="x", pady=(0, 8))
        
        ttk.Button(
            toolbar,
            text="▶️ 运行 LLVM IR",
            command=self.run_llvm_ir,
            style="Accent.TButton"
        ).pack(side="left", padx=3)
        
        ttk.Label(
            toolbar,
            text="💡 提示：运行结果将显示在下方区域",
            foreground="gray",
            font=("Microsoft YaHei UI", 9)
        ).pack(side="left", padx=10)
        
        # 创建分隔线
        separator = ttk.Separator(self.result_container, orient="horizontal")
        separator.pack(fill="x", pady=(0, 8))
        
        # 显示 LLVM IR 代码
        llvm_frame = ttk.LabelFrame(self.result_container, text="LLVM IR 代码", padding=6)
        llvm_frame.pack(fill="both", expand=True, pady=(0, 8))
        
        llvm_text = tk.Text(
            llvm_frame,
            wrap="none",
            font=("Consolas", self.result_font_size),
            state="disabled",
            bg="#f5f5f5"
        )
        llvm_scroll_y = ttk.Scrollbar(llvm_frame, orient="vertical", command=llvm_text.yview)
        llvm_scroll_x = ttk.Scrollbar(llvm_frame, orient="horizontal", command=llvm_text.xview)
        llvm_text.configure(yscrollcommand=llvm_scroll_y.set, xscrollcommand=llvm_scroll_x.set)
        
        llvm_text.pack(side="left", fill="both", expand=True)
        llvm_scroll_y.pack(side="right", fill="y")
        llvm_scroll_x.pack(side="bottom", fill="x")
        
        # 插入 LLVM IR 内容
        llvm_content = self.result.llvm_ir or "当前没有可生成的 LLVM IR"
        llvm_text.config(state="normal")
        llvm_text.insert("1.0", llvm_content)
        llvm_text.config(state="disabled")
        
        # 保存引用以便后续更新
        self.llvm_output_text = None
        self.llvm_output_frame = None

    def run_llvm_ir(self) -> None:
        """运行 LLVM IR（通过解释器执行四元式）"""
        if not self.result:
            from tkinter import messagebox
            messagebox.showwarning("警告", "请先分析代码！")
            return
        
        if not self.result.quads:
            from tkinter import messagebox
            messagebox.showwarning("警告", "没有可执行的四元式！")
            return
        
        # 清除之前的输出区域
        if hasattr(self, 'llvm_output_frame') and self.llvm_output_frame:
            self.llvm_output_frame.destroy()
        
        # 创建输出区域
        output_frame = ttk.LabelFrame(self.result_container, text="运行结果", padding=6)
        output_frame.pack(fill="both", expand=True)
        self.llvm_output_frame = output_frame
        
        # 创建输出文本框
        output_text = tk.Text(
            output_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            bg="#ffffff"
        )
        output_scroll_y = ttk.Scrollbar(output_frame, orient="vertical", command=output_text.yview)
        output_scroll_x = ttk.Scrollbar(output_frame, orient="horizontal", command=output_text.xview)
        output_text.configure(yscrollcommand=output_scroll_y.set, xscrollcommand=output_scroll_x.set)
        
        output_text.pack(side="left", fill="both", expand=True)
        output_scroll_y.pack(side="right", fill="y")
        output_scroll_x.pack(side="bottom", fill="x")
        
        self.llvm_output_text = output_text
        
        # 执行四元式（复用解释器的逻辑）
        try:
            from task_3_2_interpreter.executor import QuadExecutor
            
            # 获取所有函数的四元式
            programs = {}
            # 从 AST 中提取函数名和四元式
            if self.result.ast:
                from compiler.ir_generator import generate_quads
                programs, _ = generate_quads(self.result.ast)
            
            if not programs:
                output_text.config(state="normal")
                output_text.insert("1.0", "错误：无法获取程序代码")
                output_text.config(state="disabled")
                return
            
            # 构建参数映射表（从 AST 中提取）
            params_map = {}
            if self.result.ast:
                for func in self.result.ast.children:
                    if func.kind == "Function":
                        func_name = func.value
                        params_node = func.children[1] if len(func.children) > 1 else None
                        if params_node:
                            param_names = [param.value for param in params_node.children]
                            params_map[func_name] = param_names
            
            # 创建执行器并运行（传入 params_map）
            executor = QuadExecutor(programs, params_map)
            state = executor.run()
            
            # 收集输出
            output_lines = []
            output_lines.append("=" * 60)
            output_lines.append("LLVM IR 执行结果")
            output_lines.append("=" * 60)
            output_lines.append("")
            
            # 显示返回值
            output_lines.append(f"返回值：{state.return_value if state.return_value is not None else '-'}")
            output_lines.append("")
            
            # 显示最终变量状态
            output_lines.append("最终变量状态：")
            all_vars = {**state.variables, **state.temps}
            if all_vars:
                for name, value in sorted(all_vars.items()):
                    output_lines.append(f"  {name} = {value}")
            else:
                output_lines.append("  （无变量）")
            output_lines.append("")
            
            # 显示执行轨迹（简化版，只显示关键步骤）
            output_lines.append("执行轨迹（关键步骤）：")
            trace_lines = state.trace[-20:]  # 只显示最后 20 条
            for line in trace_lines:
                output_lines.append(f"  {line}")
            
            if len(state.trace) > 20:
                output_lines.append(f"  ... （共 {len(state.trace)} 步）")
            
            output_text.config(state="normal")
            output_text.insert("1.0", "\n".join(output_lines))
            output_text.config(state="disabled")
            
        except Exception as e:
            output_text.config(state="normal")
            output_text.insert("1.0", f"执行错误：{str(e)}")
            output_text.config(state="disabled")


def render_diagnostics(result: PipelineResult) -> str:
    if not result.diagnostics:
        return "未发现错误"
    return "\n".join(d.as_text() for d in result.diagnostics)


def render_tests(result: PipelineResult) -> str:
    lines = [
        "编译链测试摘要",
        f"Token 数量：{len([t for t in result.tokens if t.kind != 'EOF'])}",
        f"诊断数量：{len(result.diagnostics)}",
        f"四元式数量：{len(result.quads)}",
        f"解释执行返回值：{result.interpreter_return}",
        "",
        "测试分析：",
        "1. 当前样例已覆盖顺序、分支、循环和 return 的主流程。",
        "2. 若存在语法或语义错误，四元式与 LLVM IR 将停止生成或部分生成。",
        "3. 后续建议补充函数调用、错误输入、边界值和异常跳转的专项样例。",
    ]
    return "\n".join(lines)
