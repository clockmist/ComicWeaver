"""UI 模块。

原始代码保留在 app.py / html_widgets.py / styles.py 中（不动）。
重构后的代码分布在 app_new.py / session.py / callbacks/ / renderers/ / composables/ 中。
"""
from .app import build_ui, main  # 原始入口
from .app_new import build_ui as build_ui_new, main as main_new  # 重构入口

__all__ = ["build_ui", "main", "build_ui_new", "main_new"]
