"""
车牌图像识别系统界面
基于现有车牌识别代码的功能封装
"""
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter import font as tkFont
import threading
import queue
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import subprocess
import atexit
import signal

# 导入现有车牌识别系统
sys.path.append(r"E:\hjj_II")
try:
    from LPR import (
        EnhancedLicensePlateSystem,
        CCPDParser,
        SimplePlateDetector,
        PlateCNNVerifier,
        visualize_comparison
    )

    print("✅ 车牌识别系统导入成功")
except ImportError as e:
    print(f"❌ 导入车牌识别系统失败: {e}")


    # 如果没有LPR模块，我们创建一个简单的占位符
    class EnhancedLicensePlateSystem:
        def __init__(self, **kwargs):
            pass

        def process_image(self, image_path):
            return None


class LicensePlateRecognitionGUI:
    """车牌识别系统GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("车牌图像识别系统 V2.0")
        self.root.geometry("1400x800")

        # 注册退出清理函数
        atexit.register(self.cleanup)

        # 设置重启标志
        self.restart_requested = False

        # 设置字体
        self.title_font = tkFont.Font(family="微软雅黑", size=20, weight="bold")
        self.heading_font = tkFont.Font(family="微软雅黑", size=14, weight="bold")
        self.normal_font = tkFont.Font(family="微软雅黑", size=11)

        # 设置颜色主题
        self.colors = {
            'primary': '#2E86C1',  # 蓝色
            'secondary': '#28B463',  # 绿色
            'accent': '#E74C3C',  # 红色
            'background': '#F2F3F4',  # 浅灰
            'text': '#2C3E50',  # 深灰
            'success': '#27AE60',  # 成功绿
            'warning': '#F39C12',  # 警告黄
            'error': '#E74C3C'  # 错误红
        }

        # 初始化系统
        self.init_systems()

        # 创建GUI
        self.setup_gui()

        # 结果队列用于线程间通信
        self.result_queue = queue.Queue()

        # 当前显示的图像
        self.current_images = {}

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_systems(self):
        """初始化车牌识别系统"""
        self.parser = CCPDParser()
        self.cnn_model_path = r"E:\CCPD 1.8\plate_verifier_best.pth"

        # I类系统：使用标签位置定位
        self.system_i = EnhancedLicensePlateSystem(use_label_position=True)

        # II类系统：使用传统+CNN定位
        self.system_ii = EnhancedLicensePlateSystem(
            use_label_position=False,
            use_cnn_verifier=True,
            cnn_model_path=self.cnn_model_path
        )

    def setup_gui(self):
        """设置GUI界面"""
        # 设置窗口背景
        self.root.configure(bg=self.colors['background'])

        # 创建顶部工具栏
        self.create_toolbar()

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 20))

        # 标题
        title_label = tk.Label(
            main_frame,
            text="🚗 车牌图像识别系统",
            font=self.title_font,
            fg=self.colors['primary'],
            bg=self.colors['background']
        )
        title_label.pack(pady=(0, 20))

        # 创建Notebook（选项卡）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # I类测试选项卡
        self.create_test_i_tab()

        # II类测试选项卡
        self.create_test_ii_tab()

        # 结果查看选项卡
        self.create_result_tab()

        # 状态栏
        self.status_bar = tk.Label(
            self.root,
            text="就绪",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=self.normal_font,
            bg=self.colors['background']
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_toolbar(self):
        """创建顶部工具栏"""
        toolbar = tk.Frame(self.root, bg=self.colors['primary'], height=40)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        # 左侧：系统标题
        title_label = tk.Label(
            toolbar,
            text="车牌识别系统 V2.0",
            font=self.normal_font,
            fg='white',
            bg=self.colors['primary']
        )
        title_label.pack(side=tk.LEFT, padx=15)

        # 右侧：工具按钮
        button_frame = tk.Frame(toolbar, bg=self.colors['primary'])
        button_frame.pack(side=tk.RIGHT, padx=15)

        # 🔥 新增：重启按钮
        restart_btn = tk.Button(
            button_frame,
            text="🔄 清理缓存",
            command=self.restart_application,
            font=self.normal_font,
            bg='#F39C12',
            fg='white',
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        restart_btn.pack(side=tk.LEFT, padx=5)

        # 帮助按钮
        help_btn = tk.Button(
            button_frame,
            text="❓ 帮助",
            command=self.show_help,
            font=self.normal_font,
            bg='#3498DB',
            fg='white',
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        help_btn.pack(side=tk.LEFT, padx=5)

        # 退出按钮
        exit_btn = tk.Button(
            button_frame,
            text="❌ 退出",
            command=self.on_closing,
            font=self.normal_font,
            bg='#E74C3C',
            fg='white',
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        exit_btn.pack(side=tk.LEFT, padx=5)

    def show_help(self):
        """显示帮助信息"""
        help_text = """🚗 车牌识别系统使用说明：

🔹 I类测试：使用数据集标签位置进行精准定位
    - 数据路径：E:\\hjj_II\\testdata\\I
    - 结果保存：E:\\hjj_II\\test result\\I
    - 特点：定位准确，速度快，支持字符分割

🔹 II类测试：使用传统颜色检测+CNN验证
    - 数据路径：E:\\hjj_II\\testdata\\II
    - 结果保存：E:\\hjj_II\\test result\\II
    - 特点：无需标签，自适应定位

🔹 结果查看：
    - 加载I/II类测试结果
    - 一键打开结果目录
    - 显示详细识别统计

🔹 清理缓存：
    - 清理所有测试状态
    - 重置系统环境
    - 解决混合测试导致的混乱

⚠️ 注意事项：
    - 确保数据目录存在且包含jpg图像
    - 测试前设置好保存路径
    - 混合测试后建议点击清理缓存再测试
        """
        messagebox.showinfo("系统帮助", help_text)

    def create_test_i_tab(self):
        """创建I类测试选项卡"""
        tab_i = ttk.Frame(self.notebook)
        self.notebook.add(tab_i, text="I类车牌集中测试")

        # 左框架：控制面板
        left_frame = ttk.LabelFrame(tab_i, text="控制面板", padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 测试数量输入
        count_frame = ttk.Frame(left_frame)
        count_frame.pack(pady=10, fill=tk.X)

        tk.Label(count_frame, text="测试图像数量:", font=self.normal_font).pack(side=tk.LEFT)
        self.count_i = tk.IntVar(value=5)
        count_spinbox = ttk.Spinbox(
            count_frame,
            from_=1,
            to=50,
            textvariable=self.count_i,
            width=10,
            font=self.normal_font
        )
        count_spinbox.pack(side=tk.LEFT, padx=10)

        # 路径显示
        path_frame_i = ttk.Frame(left_frame)
        path_frame_i.pack(pady=10, fill=tk.X)

        tk.Label(path_frame_i, text="数据路径:", font=self.normal_font).pack(anchor=tk.W)
        self.path_i_var = tk.StringVar(value=r"E:\hjj_II\testdata\I")
        path_entry_i = ttk.Entry(path_frame_i, textvariable=self.path_i_var, width=30, font=self.normal_font)
        path_entry_i.pack(pady=5, fill=tk.X)

        # 浏览按钮
        browse_btn_i = ttk.Button(
            left_frame,
            text="浏览数据目录",
            command=self.browse_path_i
        )
        browse_btn_i.pack(pady=5)

        # 保存路径显示
        save_frame_i = ttk.Frame(left_frame)
        save_frame_i.pack(pady=10, fill=tk.X)

        tk.Label(save_frame_i, text="保存路径:", font=self.normal_font).pack(anchor=tk.W)
        self.save_i_var = tk.StringVar(value=r"E:\hjj_II\test result\I")
        save_entry_i = ttk.Entry(save_frame_i, textvariable=self.save_i_var, width=30, font=self.normal_font)
        save_entry_i.pack(pady=5, fill=tk.X)

        # 开始测试按钮
        start_btn_i = ttk.Button(
            left_frame,
            text="开始I类测试",
            command=self.start_test_i,
            style="Accent.TButton"
        )
        start_btn_i.pack(pady=20)

        # 进度条
        self.progress_i = ttk.Progressbar(
            left_frame,
            length=200,
            mode='indeterminate'
        )
        self.progress_i.pack(pady=10)

        # 右框架：结果显示
        right_frame = ttk.Frame(tab_i)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建顶部框架用于显示统计信息
        top_frame = ttk.LabelFrame(right_frame, text="测试结果统计", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # 统计信息显示区域（放大）
        self.stats_text_i = scrolledtext.ScrolledText(
            top_frame,
            height=12,
            width=60,
            font=self.normal_font,
            wrap=tk.WORD
        )
        self.stats_text_i.pack(fill=tk.BOTH, expand=True)

        # 创建底部框架用于显示图像（可滚动）
        bottom_frame = ttk.LabelFrame(right_frame, text="车牌图像预览", padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # 创建Canvas和Scrollbar用于滚动
        canvas_i = tk.Canvas(bottom_frame, bg='white')
        scrollbar_i = ttk.Scrollbar(bottom_frame, orient="vertical", command=canvas_i.yview)

        # 图像显示区域（放在Canvas中）
        self.image_frame_i = ttk.Frame(canvas_i)
        self.image_frame_i.bind(
            "<Configure>",
            lambda e: canvas_i.configure(scrollregion=canvas_i.bbox("all"))
        )

        canvas_i.create_window((0, 0), window=self.image_frame_i, anchor="nw")
        canvas_i.configure(yscrollcommand=scrollbar_i.set)

        # 布局Canvas和Scrollbar
        canvas_i.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_i.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            canvas_i.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_i.bind_all("<MouseWheel>", _on_mousewheel)

        # 默认提示文本
        self.default_label_i = tk.Label(
            self.image_frame_i,
            text="点击'开始I类测试'按钮开始识别\n识别结果将显示在这里",
            font=self.normal_font,
            fg='gray'
        )
        self.default_label_i.pack(expand=True)

    def create_test_ii_tab(self):
        """创建II类测试选项卡"""
        tab_ii = ttk.Frame(self.notebook)
        self.notebook.add(tab_ii, text="II类车牌集中测试")

        # 左框架：控制面板
        left_frame = ttk.LabelFrame(tab_ii, text="控制面板", padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 测试数量输入
        count_frame = ttk.Frame(left_frame)
        count_frame.pack(pady=10, fill=tk.X)

        tk.Label(count_frame, text="测试图像数量:", font=self.normal_font).pack(side=tk.LEFT)
        self.count_ii = tk.IntVar(value=5)
        count_spinbox = ttk.Spinbox(
            count_frame,
            from_=1,
            to=50,
            textvariable=self.count_ii,
            width=10,
            font=self.normal_font
        )
        count_spinbox.pack(side=tk.LEFT, padx=10)

        # 路径显示
        path_frame_ii = ttk.Frame(left_frame)
        path_frame_ii.pack(pady=10, fill=tk.X)

        tk.Label(path_frame_ii, text="数据路径:", font=self.normal_font).pack(anchor=tk.W)
        self.path_ii_var = tk.StringVar(value=r"E:\hjj_II\testdata\II")
        path_entry_ii = ttk.Entry(path_frame_ii, textvariable=self.path_ii_var, width=30, font=self.normal_font)
        path_entry_ii.pack(pady=5, fill=tk.X)

        # 浏览按钮
        browse_btn_ii = ttk.Button(
            left_frame,
            text="浏览数据目录",
            command=self.browse_path_ii
        )
        browse_btn_ii.pack(pady=5)

        # 保存路径显示
        save_frame_ii = ttk.Frame(left_frame)
        save_frame_ii.pack(pady=10, fill=tk.X)

        tk.Label(save_frame_ii, text="保存路径:", font=self.normal_font).pack(anchor=tk.W)
        self.save_ii_var = tk.StringVar(value=r"E:\hjj_II\test result\II")
        save_entry_ii = ttk.Entry(save_frame_ii, textvariable=self.save_ii_var, width=30, font=self.normal_font)
        save_entry_ii.pack(pady=5, fill=tk.X)

        # 开始测试按钮
        start_btn_ii = ttk.Button(
            left_frame,
            text="开始II类测试",
            command=self.start_test_ii,
            style="Accent.TButton"
        )
        start_btn_ii.pack(pady=20)

        # 进度条
        self.progress_ii = ttk.Progressbar(
            left_frame,
            length=200,
            mode='indeterminate'
        )
        self.progress_ii.pack(pady=10)

        # 右框架：结果显示
        right_frame = ttk.Frame(tab_ii)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建顶部框架用于显示统计信息
        top_frame = ttk.LabelFrame(right_frame, text="测试结果统计", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # 统计信息显示区域（放大）
        self.stats_text_ii = scrolledtext.ScrolledText(
            top_frame,
            height=12,
            width=60,
            font=self.normal_font,
            wrap=tk.WORD
        )
        self.stats_text_ii.pack(fill=tk.BOTH, expand=True)

        # 创建底部框架用于显示图像（可滚动）
        bottom_frame = ttk.LabelFrame(right_frame, text="车牌图像预览", padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # 创建Canvas和Scrollbar用于滚动
        canvas_ii = tk.Canvas(bottom_frame, bg='white')
        scrollbar_ii = ttk.Scrollbar(bottom_frame, orient="vertical", command=canvas_ii.yview)

        # 图像显示区域（放在Canvas中）
        self.image_frame_ii = ttk.Frame(canvas_ii)
        self.image_frame_ii.bind(
            "<Configure>",
            lambda e: canvas_ii.configure(scrollregion=canvas_ii.bbox("all"))
        )

        canvas_ii.create_window((0, 0), window=self.image_frame_ii, anchor="nw")
        canvas_ii.configure(yscrollcommand=scrollbar_ii.set)

        # 布局Canvas和Scrollbar
        canvas_ii.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ii.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            canvas_ii.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_ii.bind_all("<MouseWheel>", _on_mousewheel)

        # 默认提示文本
        self.default_label_ii = tk.Label(
            self.image_frame_ii,
            text="点击'开始II类测试'按钮开始识别\n识别结果将显示在这里",
            font=self.normal_font,
            fg='gray'
        )
        self.default_label_ii.pack(expand=True)

    def create_result_tab(self):
        """创建结果查看选项卡 - 修改后版本"""
        tab_result = ttk.Frame(self.notebook)
        self.notebook.add(tab_result, text="结果查看")

        # 结果查看框架
        result_frame = ttk.Frame(tab_result)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部：标题和按钮区域
        top_frame = ttk.Frame(result_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # 左侧：标题
        title_label = tk.Label(
            top_frame,
            text="历史测试结果",
            font=self.heading_font,
            fg=self.colors['primary']
        )
        title_label.pack(side=tk.LEFT)

        # 右侧：导航按钮
        nav_frame = ttk.Frame(top_frame)
        nav_frame.pack(side=tk.RIGHT)

        # 添加三个导航按钮
        nav_btn_test = ttk.Button(
            nav_frame,
            text="📁 测试数据目录",
            command=lambda: self.open_directory(r"E:\hjj_II\testdata"),
            width=15
        )
        nav_btn_test.pack(side=tk.LEFT, padx=3)

        nav_btn_i = ttk.Button(
            nav_frame,
            text="📁 I类结果目录",
            command=lambda: self.open_directory(r"E:\hjj_II\test result\I"),
            width=15
        )
        nav_btn_i.pack(side=tk.LEFT, padx=3)

        nav_btn_ii = ttk.Button(
            nav_frame,
            text="📁 II类结果目录",
            command=lambda: self.open_directory(r"E:\hjj_II\test result\II"),
            width=15
        )
        nav_btn_ii.pack(side=tk.LEFT, padx=3)

        nav_btn_chars = ttk.Button(
            nav_frame,
            text="📁 所有结果目录",
            command=lambda: self.open_directory(r"E:\hjj_II\test result"),
            width=15
        )
        nav_btn_chars.pack(side=tk.LEFT, padx=3)

        # 创建Treeview显示结果
        columns = ('序号', '文件名', '真值车牌', '识别结果', '准确率', '置信度', '处理时间')
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show='headings',
            height=15
        )

        # 设置列标题和宽度
        column_widths = {
            '序号': 50,
            '文件名': 150,
            '真值车牌': 100,
            '识别结果': 120,
            '准确率': 80,
            '置信度': 80,
            '处理时间': 80
        }

        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=column_widths.get(col, 100))

        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        # 布局
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 底部：控制按钮
        btn_frame = ttk.Frame(result_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="加载I类结果",
            command=lambda: self.load_results('I')
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="加载II类结果",
            command=lambda: self.load_results('II')
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="清除结果",
            command=self.clear_results
        ).pack(side=tk.LEFT, padx=5)

        # 🔥 新增：第二行框架，用于放置新的按钮
        btn_frame2 = ttk.Frame(result_frame)
        btn_frame2.pack(fill=tk.X, pady=(5, 10))  # pady=(上边距, 下边距)

        # 🔥 修改：将按钮放在第二行并居中
        ttk.Button(
            btn_frame2,
            text="no_filename验证",
            command=self.run_hyper_no_filename,
            style="Accent.TButton"
        ).pack()  # 去掉side参数，默认居中

    def browse_path_i(self):
        """浏览I类数据目录"""
        directory = filedialog.askdirectory(initialdir=r"E:\hjj_II\testdata")
        if directory:
            self.path_i_var.set(directory)

    def browse_path_ii(self):
        """浏览II类数据目录"""
        directory = filedialog.askdirectory(initialdir=r"E:\hjj_II\testdata")
        if directory:
            self.path_ii_var.set(directory)

    def start_test_i(self):
        """开始I类测试"""
        count = self.count_i.get()
        data_path = Path(self.path_i_var.get())
        save_path = Path(self.save_i_var.get())

        # 验证路径
        if not data_path.exists():
            messagebox.showerror("错误", f"数据目录不存在: {data_path}")
            return

        # 确保保存目录存在
        save_path.mkdir(parents=True, exist_ok=True)

        # 清空显示区域
        self.clear_image_frame('I')

        # 更新状态
        self.update_status("开始I类测试...")
        self.progress_i.start()

        # 在新线程中运行测试
        thread = threading.Thread(
            target=self.run_test_i,
            args=(count, data_path, save_path),
            daemon=True
        )
        thread.start()

        # 启动结果检查
        self.check_results('I')

    def start_test_ii(self):
        """开始II类测试"""
        count = self.count_ii.get()
        data_path = Path(self.path_ii_var.get())
        save_path = Path(self.save_ii_var.get())

        # 验证路径
        if not data_path.exists():
            messagebox.showerror("错误", f"数据目录不存在: {data_path}")
            return

        # 确保保存目录存在
        save_path.mkdir(parents=True, exist_ok=True)

        # 清空显示区域
        self.clear_image_frame('II')

        # 更新状态
        self.update_status("开始II类测试...")
        self.progress_ii.start()

        # 在新线程中运行测试
        thread = threading.Thread(
            target=self.run_test_ii,
            args=(count, data_path, save_path),
            daemon=True
        )
        thread.start()

        # 启动结果检查
        self.check_results('II')

    def run_test_i(self, count, data_path, save_path):
        """运行I类测试"""
        try:
            # 获取图像文件
            image_files = list(data_path.glob('*.jpg'))[:count]
            total_images = len(image_files)

            if total_images == 0:
                self.result_queue.put(('error', 'I', '未找到图像文件'))
                return

            results = []
            success_count = 0
            total_accuracy = 0
            hyper_success_count = 0

            # 导入字符分割器
            try:
                from char_segmenter import HyperLPRCharSegmenter
                char_segmenter = HyperLPRCharSegmenter()
                segmenter_available = True
            except ImportError:
                print("⚠️ 字符分割器不可用")
                char_segmenter = None
                segmenter_available = False

            for idx, image_path in enumerate(image_files, 1):
                # 处理图像
                result = self.system_i.process_image(str(image_path))

                if result and result['success']:
                    success_count += 1
                    total_accuracy += result['accuracy']

                    if result.get('method_used') == 'hyperlpr_backup':
                        hyper_success_count += 1

                    # 保存结果
                    result_data = {
                        'filename': image_path.name,
                        'ground_truth': result['ground_truth'],
                        'predicted': result['recognition_result']['plate_number'],
                        'confidence': result['recognition_result']['confidence'],
                        'accuracy': result['accuracy'],
                        'processing_time': result['processing_time'],
                        'method_used': result.get('method_used', '')
                    }
                    results.append(result_data)

                    # 保存车牌图像
                    plate_image = result['plate_region']
                    plate_path = save_path / f"plate_{idx}_{image_path.stem}.jpg"
                    cv2.imwrite(str(plate_path), plate_image)

                    # 调用字符分割
                    segmentation_result = None
                    if segmenter_available and char_segmenter:
                        try:
                            char_seg_save_dir = Path(r"E:\hjj_II\test result\char_segments")
                            char_seg_save_dir.mkdir(parents=True, exist_ok=True)

                            segmentation_result = char_segmenter.segment_chars(
                                image_path=str(image_path),
                                save_dir=str(char_seg_save_dir)
                            )

                            if segmentation_result:
                                print(f"✅ 字符分割完成: {segmentation_result['plate_number']}")
                        except Exception as e:
                            print(f"⚠️ 字符分割失败: {e}")
                            import traceback
                            traceback.print_exc()
                            segmentation_result = None

                    # 创建可视化结果
                    fig = self.create_single_result_visualization(result, idx, segmentation_result)
                    fig_path = save_path / f"result_{idx}_{image_path.stem}.png"
                    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # 发送结果到主线程
                    self.result_queue.put(('result', 'I', {
                        'index': idx,
                        'result': result_data,
                        'image': plate_image,
                        'fig_path': fig_path,
                        'segmentation_result': segmentation_result
                    }))
                else:
                    # 发送失败信息
                    fail_data = {
                        'filename': image_path.name,
                        'ground_truth': result.get('ground_truth', "未知") if result else "未知",
                        'predicted': "识别失败",
                        'confidence': 0.0,
                        'accuracy': 0.0,
                        'processing_time': 0.0,
                        'method_used': 'failed'
                    }
                    results.append(fail_data)
                    self.result_queue.put(('result', 'I', {
                        'index': idx,
                        'result': fail_data,
                        'image': None,
                        'fig_path': None,
                        'segmentation_result': None
                    }))

                # 发送进度
                self.result_queue.put(('progress', 'I', idx / total_images * 100))

            # 计算统计信息
            success_rate = success_count / total_images * 100
            avg_accuracy = total_accuracy / success_count if success_count > 0 else 0

            # 保存详细结果到CSV
            if results:
                df = pd.DataFrame(results)
                csv_path = save_path / "test_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 发送最终统计
            stats = {
                'total_images': total_images,
                'success_count': success_count,
                'success_rate': success_rate,
                'avg_accuracy': avg_accuracy,
                'save_path': save_path,
                'hyper_success_count': hyper_success_count,
                'results': results
            }
            self.result_queue.put(('stats', 'I', stats))

        except Exception as e:
            self.result_queue.put(('error', 'I', f"测试过程中出现错误: {str(e)}"))

    def run_test_ii(self, count, data_path, save_path):
        """运行II类测试"""
        try:
            # 详细的路径检查
            if not data_path.exists():
                self.result_queue.put(('error', 'II', f'数据目录不存在: {data_path}'))
                return

            # 获取图像文件
            image_files = list(data_path.glob('*.jpg'))[:count]
            total_images = len(image_files)

            if total_images == 0:
                self.result_queue.put(('error', 'II', f'未找到图像文件。检查目录: {data_path}'))
                return

            results = []
            success_count = 0
            total_accuracy = 0
            hyper_success_count = 0

            for idx, image_path in enumerate(image_files, 1):
                # 处理图像
                result = self.system_ii.process_image(str(image_path))

                if result and result['success']:
                    success_count += 1
                    total_accuracy += result['accuracy']

                    if result.get('method_used') == 'hyperlpr_backup':
                        hyper_success_count += 1

                    # 保存结果
                    result_data = {
                        'filename': image_path.name,
                        'ground_truth': result['ground_truth'],
                        'predicted': result['recognition_result']['plate_number'],
                        'confidence': result['recognition_result']['confidence'],
                        'accuracy': result['accuracy'],
                        'processing_time': result['processing_time'],
                        'method_used': result.get('method_used', '')
                    }
                    results.append(result_data)

                    # 保存车牌图像
                    plate_image = result['plate_region']
                    plate_path = save_path / f"plate_{idx}_{image_path.stem}.jpg"
                    cv2.imwrite(str(plate_path), plate_image)

                    # 创建可视化结果
                    fig = self.create_single_result_visualization(result, idx)
                    fig_path = save_path / f"result_{idx}_{image_path.stem}.png"
                    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # 发送结果到主线程
                    self.result_queue.put(('result', 'II', {
                        'index': idx,
                        'result': result_data,
                        'image': plate_image,
                        'fig_path': fig_path
                    }))
                else:
                    # 发送失败信息
                    fail_data = {
                        'filename': image_path.name,
                        'ground_truth': result.get('ground_truth', "未知") if result else "未知",
                        'predicted': "识别失败",
                        'confidence': 0.0,
                        'accuracy': 0.0,
                        'processing_time': 0.0,
                        'method_used': 'failed'
                    }
                    results.append(fail_data)
                    self.result_queue.put(('result', 'II', {
                        'index': idx,
                        'result': fail_data,
                        'image': None,
                        'fig_path': None
                    }))

                # 发送进度
                self.result_queue.put(('progress', 'II', idx / total_images * 100))

            # 计算统计信息
            success_rate = success_count / total_images * 100
            avg_accuracy = total_accuracy / success_count if success_count > 0 else 0

            # 保存详细结果到CSV
            if results:
                df = pd.DataFrame(results)
                csv_path = save_path / "test_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 发送最终统计
            stats = {
                'total_images': total_images,
                'success_count': success_count,
                'success_rate': success_rate,
                'avg_accuracy': avg_accuracy,
                'save_path': save_path,
                'hyper_success_count': hyper_success_count,
                'results': results
            }
            self.result_queue.put(('stats', 'II', stats))

        except Exception as e:
            self.result_queue.put(('error', 'II', f"测试过程中出现错误: {str(e)}"))

    def check_results(self, test_type):
        """检查结果队列"""
        try:
            while True:
                try:
                    msg_type, msg_test_type, data = self.result_queue.get_nowait()

                    if msg_test_type == test_type:
                        if msg_type == 'result':
                            self.display_result(test_type, data)
                        elif msg_type == 'stats':
                            self.display_stats(test_type, data)
                        elif msg_type == 'error':
                            self.display_error(test_type, data)
                        elif msg_type == 'progress':
                            pass

                except queue.Empty:
                    break

        finally:
            # 继续检查
            self.root.after(100, lambda: self.check_results(test_type))

    def display_result(self, test_type, data):
        """显示单个结果"""
        idx = data['index']
        result = data['result']
        image = data['image']

        # 获取对应的显示框架
        if test_type == 'I':
            image_frame = self.image_frame_i
            stats_text = self.stats_text_i
        else:
            image_frame = self.image_frame_ii
            stats_text = self.stats_text_ii

        # 清除默认标签
        if hasattr(self, f'default_label_{test_type.lower()}'):
            getattr(self, f'default_label_{test_type.lower()}').pack_forget()

        # 创建结果显示框架
        result_frame = ttk.Frame(image_frame)
        result_frame.pack(fill=tk.X, pady=5)

        # 显示缩略图（如果有）
        if image is not None:
            thumbnail = self.create_thumbnail(image, (150, 50))
            img_label = tk.Label(result_frame, image=thumbnail)
            img_label.image = thumbnail  # 保持引用
            img_label.pack(side=tk.LEFT, padx=10)
        else:
            # 没有图像时显示占位符
            placeholder = tk.Label(result_frame, text="无图像", width=20, height=3, bg='lightgray')
            placeholder.pack(side=tk.LEFT, padx=10)

        # 显示结果文本
        result_text = f"图像 {idx}: {result['filename']}\n"
        result_text += f"真值: {result['ground_truth']}"

        if result['predicted'] != "识别失败":
            result_text += f"  →  识别: {result['predicted']}\n"
            method_desc = ""
            method_used = result.get('method_used', '')
            if method_used == 'hyperlpr_backup':
                method_desc = "（HyperLPR 整车备选识别）"
            elif method_used == 'traditional_cnn':
                method_desc = "（传统颜色 + CNN 验证）"
            elif method_used == 'traditional_only':
                method_desc = "（仅传统颜色检测）"
            elif method_used == 'label_position':
                method_desc = "（标签位置定位）"

            result_text += (
                f"准确率: {result['accuracy']:.2%}  置信度: {result['confidence']:.4f}  "
                f"时间: {result['processing_time']:.3f}s {method_desc}"
            )
        else:
            result_text += f"  →  识别: ❌ 识别失败\n"
            result_text += f"    处理时间: {result['processing_time']:.3f}s"

        text_label = tk.Label(
            result_frame,
            text=result_text,
            font=self.normal_font,
            justify=tk.LEFT,
            bg='white',
            relief=tk.RIDGE,
            padx=10,
            pady=5
        )
        text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 更新状态栏
        self.update_status(f"{test_type}类测试: 完成图像 {idx}")

    def display_stats(self, test_type, stats):
        """显示统计信息"""
        if test_type == 'I':
            stats_text = self.stats_text_i
            progress_bar = self.progress_i
        else:
            stats_text = self.stats_text_ii
            progress_bar = self.progress_ii

        # 停止进度条
        progress_bar.stop()

        # 清空统计文本框
        stats_text.delete(1.0, tk.END)

        # 添加统计信息
        stats_text.insert(tk.END, "=" * 50 + "\n")
        stats_text.insert(tk.END, f"    {test_type}类测试结果统计\n")
        stats_text.insert(tk.END, "=" * 50 + "\n\n")

        stats_text.insert(tk.END, f" 测试图像总数: {stats['total_images']}\n")
        stats_text.insert(tk.END, f" 成功识别数量: {stats['success_count']}\n")
        stats_text.insert(tk.END, f" 识别成功率: {stats['success_rate']:.1f}%\n")

        # 计算最终准确率（完全正确率）
        correct_count = 0
        if 'results' in stats and stats['results']:
            for r in stats['results']:
                if r.get('predicted') == r.get('ground_truth'):
                    correct_count += 1
            final_accuracy_rate = correct_count / stats['total_images'] if stats['total_images'] > 0 else 0.0
            stats_text.insert(tk.END,
                              f" 完全正确率: {final_accuracy_rate:.2%} ({correct_count}/{stats['total_images']})\n")

        # 计算平均准确率（字符级）
        if stats['success_count'] > 0:
            avg_accuracy = stats['avg_accuracy']
            stats_text.insert(tk.END, f" 平均字符准确率: {avg_accuracy:.2%}\n\n")
        else:
            stats_text.insert(tk.END, f" 平均字符准确率: 0.00%\n\n")

        # 显示 HyperLPR 备选方法的贡献
        if 'hyper_success_count' in stats:
            hyper_cnt = stats['hyper_success_count']
            if stats['total_images'] > 0:
                hyper_rate = hyper_cnt / stats['total_images'] * 100
            else:
                hyper_rate = 0.0
            stats_text.insert(
                tk.END,
                f" 其中由 HyperLPR 整车备选成功的数量: {hyper_cnt} "
                f"(占全部测试图像的 {hyper_rate:.1f}%)\n\n"
            )

        stats_text.insert(tk.END, f" 结果保存路径:\n")
        stats_text.insert(tk.END, f"{stats['save_path']}\n")
        stats_text.insert(tk.END, f"  - 车牌图像: plate_*.jpg\n")
        stats_text.insert(tk.END, f"  - 详细结果: result_*.png\n")
        if test_type == 'I':
            stats_text.insert(tk.END, f"  - 字符分割: E:\\hjj_II\\char_segments\\\n")
        stats_text.insert(tk.END, f"  - 统计表格: test_results.csv\n")
        if 'hyper_success_count' in stats and stats['hyper_success_count'] > 0:
            stats_text.insert(tk.END, f"  - HyperLPR备选结果: E:\\hjj_II\\hyper_location_result\\\n")

        # 格式化文本框
        stats_text.tag_configure("heading", font=self.heading_font, foreground=self.colors['primary'])
        stats_text.tag_add("heading", "2.0", "2.end")

        stats_text.tag_configure("success", foreground=self.colors['success'])
        for i in range(4, 7):
            stats_text.tag_add("success", f"{i}.0", f"{i}.end")

        # 更新状态栏
        self.update_status(f"{test_type}类测试完成! 成功率: {stats['success_rate']:.1f}%")
        messagebox.showinfo("测试完成",
                            f"{test_type}类测试完成!\n识别成功率: {stats['success_rate']:.1f}%\n完全正确率: {final_accuracy_rate:.2%}")

    def display_error(self, test_type, error_msg):
        """显示错误信息"""
        if test_type == 'I':
            progress_bar = self.progress_i
        else:
            progress_bar = self.progress_ii

        progress_bar.stop()
        messagebox.showerror("错误", error_msg)
        self.update_status(f"{test_type}类测试出错: {error_msg}")

    def create_thumbnail(self, image, size):
        """创建缩略图"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pil = Image.fromarray(image)
        image_pil.thumbnail(size, Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(image_pil)

    def create_single_result_visualization(self, result, idx, segmentation_result=None):
        """创建单个结果的详细可视化（包含字符分割）"""
        # 如果有字符分割结果，显示3个子图；否则显示2个
        if segmentation_result:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 原始图像+检测框
        img_rgb = cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)

        if result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)

        axes[0].set_title('原始图像 + 检测框', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 裁剪的车牌区域
        if result['plate_region'] is not None:
            if len(result['plate_region'].shape) == 2:
                axes[1].imshow(result['plate_region'], cmap='gray')
            else:
                plate_rgb = cv2.cvtColor(result['plate_region'], cv2.COLOR_BGR2RGB)
                axes[1].imshow(plate_rgb)

            h, w = result['plate_region'].shape[:2]
            axes[1].set_title(f'裁剪的车牌 ({w}x{h})', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 显示字符分割结果（仅I类）
        if segmentation_result:
            plate_with_box = segmentation_result.get('plate_image')
            char_regions = segmentation_result.get('char_regions', [])

            if plate_with_box is not None:
                plate_with_box_rgb = cv2.cvtColor(plate_with_box, cv2.COLOR_BGR2RGB)
                axes[2].imshow(plate_with_box_rgb)

                # 绘制字符框
                for i, (x1, y1, x2, y2) in enumerate(char_regions):
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='green', facecolor='none')
                    axes[2].add_patch(rect)
                    axes[2].text(x1, y1 - 5, f'{i + 1}', fontsize=10, color='green', weight='bold')

                axes[2].set_title(f'字符分割结果 ({len(char_regions)}个字符)', fontsize=12, fontweight='bold')
            axes[2].axis('off')

        # 添加识别结果文本
        if result['recognition_result']:
            plate_num = result['recognition_result']['plate_number']
            confidence = result['recognition_result']['confidence']
            method_desc = ""
            if result.get('method_used') == 'hyperlpr_backup':
                method_desc = " [HyperLPR备选]"
            fig.text(0.5, 0.02,
                     f"识别结果: {plate_num} | 置信度: {confidence:.4f} | 准确率: {result['accuracy']:.2%}{method_desc}",
                     ha='center', fontsize=12, fontweight='bold', color='blue')

        fig.suptitle(f"车牌识别结果 - {result['filename']}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def load_results(self, test_type):
        """加载历史结果并计算最终准确率"""
        save_path = Path(fr"E:\hjj_II\test result\{test_type}")

        if not save_path.exists():
            messagebox.showwarning("警告", f"结果目录不存在: {save_path}")
            return

        csv_path = save_path / "test_results.csv"
        if not csv_path.exists():
            messagebox.showwarning("警告", f"结果文件不存在: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)

            # 统计信息
            total_count = len(df)
            correct_count = 0
            total_accuracy_sum = 0
            success_count = 0

            # 清空现有结果
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)

            # 添加新结果并计算准确率
            for idx, row in df.iterrows():
                ground_truth = str(row['ground_truth']) if pd.notna(row['ground_truth']) else "未知"
                predicted = str(row['predicted']) if pd.notna(row['predicted']) else "识别失败"

                # 计算是否完全正确
                is_correct = (predicted != "识别失败") and (predicted == ground_truth)
                if is_correct:
                    correct_count += 1

                # 统计成功识别数量
                if predicted != "识别失败":
                    success_count += 1

                # 累加准确率
                accuracy = row.get('accuracy', 0.0)
                if pd.isna(accuracy):
                    accuracy = 0.0
                if predicted != "识别失败":
                    total_accuracy_sum += accuracy

                # 显示方法信息
                method_used = row.get('method_used', '')
                method_desc = ""
                if method_used == 'hyperlpr_backup':
                    method_desc = " [HyperLPR备选]"
                elif method_used == 'traditional_cnn':
                    method_desc = " [传统+CNN]"
                elif method_used == 'label_position':
                    method_desc = " [标签定位]"
                elif method_used == 'failed':
                    method_desc = " [识别失败]"

                # 格式化显示值
                filename_display = str(row['filename'])[:20] + "..." if len(str(row['filename'])) > 20 else str(
                    row['filename'])
                predicted_display = predicted + method_desc if predicted != "识别失败" else "❌ 识别失败"

                # 准确率和置信度显示
                if predicted != "识别失败":
                    accuracy_display = f"{accuracy:.2%}"
                    confidence_display = f"{row.get('confidence', 0):.4f}"
                else:
                    accuracy_display = "0.00%"
                    confidence_display = "0.0000"

                processing_time = f"{row.get('processing_time', 0):.3f}s"

                values = (
                    idx + 1,
                    filename_display,
                    ground_truth,
                    predicted_display,
                    accuracy_display,
                    confidence_display,
                    processing_time
                )
                self.result_tree.insert('', tk.END, values=values)

            # 计算最终准确率（完全正确率）- 分母为总测试图像数
            final_accuracy_rate = correct_count / total_count if total_count > 0 else 0.0

            # 计算平均字符准确率
            avg_accuracy = total_accuracy_sum / success_count if success_count > 0 else 0.0

            # 显示最终统计信息
            stats_msg = f"已加载{test_type}类测试结果: {total_count} 条记录\n"
            stats_msg += f"总测试图像数: {total_count}\n"
            stats_msg += f"成功识别数量: {success_count}\n"
            stats_msg += f"最终准确率（完全正确）: {final_accuracy_rate:.2%} ({correct_count}/{total_count})\n"
            stats_msg += f"平均字符准确率: {avg_accuracy:.2%}"

            self.update_status(stats_msg)

            # 显示统计对话框
            messagebox.showinfo(
                f"{test_type}类测试结果统计",
                f"总测试数量: {total_count}\n"
                f"成功识别数量: {success_count}\n"
                f"完全正确数量: {correct_count}\n"
                f"最终准确率（完全正确）: {final_accuracy_rate:.2%}\n"
                f"平均字符准确率: {avg_accuracy:.2%}"
            )

        except Exception as e:
            messagebox.showerror("错误", f"加载结果失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def clear_results(self):
        """清除结果列表"""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.update_status("已清除结果列表")

    def run_hyper_no_filename(self):
        """运行 hyper_no_filename.py 程序"""
        script_path = r"E:\hjj_II\hyper_no_filename.py"

        if not os.path.exists(script_path):
            messagebox.showerror("错误", f"脚本文件不存在:\n{script_path}")
            return

        try:
            # 运行外部Python脚本
            import subprocess
            result = subprocess.run([sys.executable, script_path],
                                    capture_output=True,
                                    text=True,
                                    cwd=r"E:\hjj_II")

            # 显示执行结果
            if result.returncode == 0:
                messagebox.showinfo("执行成功", f"程序执行成功！\n\n输出：\n{result.stdout}")
            else:
                messagebox.showerror("执行失败", f"程序执行失败！\n\n错误：\n{result.stderr}")

            self.update_status("HyperLPR分析程序已执行完成")

        except Exception as e:
            messagebox.showerror("运行错误", f"运行程序时出错：\n{str(e)}")
            self.update_status(f"运行失败：{str(e)}")

    def clear_image_frame(self, test_type):
        """清除图像显示区域"""
        if test_type == 'I':
            frame = self.image_frame_i
        else:
            frame = self.image_frame_ii

        # 移除所有子部件
        for widget in frame.winfo_children():
            widget.destroy()

        # 重新创建并显示默认标签
        default_text = f"点击'开始{test_type}类测试'按钮开始识别\n识别结果将显示在这里"
        default_label = tk.Label(
            frame,
            text=default_text,
            font=self.normal_font,
            fg='gray'
        )
        default_label.pack(expand=True)

        # 保存引用以便下次访问
        if test_type == 'I':
            self.default_label_i = default_label
        else:
            self.default_label_ii = default_label

    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.config(text=f"状态: {message}")

    def open_directory(self, directory_path):
        """打开指定目录"""
        try:
            # 检查目录是否存在
            if not os.path.exists(directory_path):
                messagebox.showwarning("目录不存在", f"目录不存在:\n{directory_path}")
                return

            # 使用系统命令打开目录
            if os.name == 'nt':  # Windows
                os.startfile(directory_path)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['xdg-open', directory_path])

            self.update_status(f"已打开目录: {directory_path}")
        except Exception as e:
            messagebox.showerror("打开目录失败", f"无法打开目录:\n{directory_path}\n错误: {str(e)}")
            self.update_status(f"打开目录失败: {str(e)}")

    def cleanup(self):
        """清理资源"""
        print("正在清理系统资源...")
        # 停止所有线程
        if hasattr(self, 'result_queue'):
            # 清空队列
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass

        # 停止进度条
        if hasattr(self, 'progress_i'):
            try:
                self.progress_i.stop()
            except:
                pass

        if hasattr(self, 'progress_ii'):
            try:
                self.progress_ii.stop()
            except:
                pass

    def restart_application(self):
        """重启应用程序"""
        if messagebox.askyesno("重启系统",
                               "确定要重启车牌识别系统吗？\n\n这将：\n1. 清理所有测试状态\n2. 重置系统环境\n3. 重新启动程序"):
            self.update_status("正在重启系统...")

            # 设置重启标志
            self.restart_requested = True

            # 清理资源
            self.cleanup()

            # 延迟关闭窗口，给清理操作一些时间
            self.root.after(100, self.perform_restart)

    def perform_restart(self):
        """执行重启操作"""
        try:
            # 获取当前脚本的路径
            script_path = os.path.abspath(__file__)

            # 关闭当前窗口
            self.root.destroy()

            # 短暂延迟，确保窗口完全关闭
            import time
            time.sleep(0.5)

            # 重新启动程序
            print("正在重新启动程序...")
            os.system(f'python "{script_path}"')

        except Exception as e:
            print(f"重启失败: {e}")
            messagebox.showerror("重启失败", f"无法重启系统: {str(e)}")

    def on_closing(self):
        """关闭窗口时的处理"""
        if self.restart_requested:
            # 如果是重启操作，直接退出
            self.root.destroy()
        else:
            if messagebox.askokcancel("退出", "确定要退出车牌识别系统吗？"):
                self.cleanup()
                self.root.destroy()


def main():
    """主函数"""
    root = tk.Tk()

    # 设置样式
    style = ttk.Style()
    style.theme_use('clam')

    # 配置样式
    style.configure('Accent.TButton',
                    foreground='white',
                    background='#3498DB',
                    font=('微软雅黑', 11, 'bold'))

    app = LicensePlateRecognitionGUI(root)

    # 设置窗口图标（可选）
    try:
        root.iconbitmap(default='icon.ico')  # 如果有图标文件的话
    except:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()