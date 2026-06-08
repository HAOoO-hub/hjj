# app.py：启动统一GUI并进入源码编译与日志扫描双工作台
from __future__ import annotations

from ui.main_window import MainWindow


if __name__ == "__main__":
    MainWindow().mainloop()
