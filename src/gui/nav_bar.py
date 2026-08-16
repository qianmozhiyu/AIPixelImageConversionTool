"""顶部导航栏组件。

提供页面切换标签，发出 page_changed 信号通知主窗口切换 QStackedWidget。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton


class NavBar(QWidget):
    """顶部导航栏，含 Logo + 页面切换标签。

    Signals:
        page_changed(int): 页面切换信号，参数为页面索引（0=转换, 1=我的资产, 2=设置）
    """

    page_changed = Signal(int)

    PAGES = ["转换", "我的资产", "设置"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self._buttons: list[QPushButton] = []
        self._current = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        # 页面标签按钮
        for i, name in enumerate(self.PAGES):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setCursor(self.cursor())
            btn.clicked.connect(lambda checked, idx=i: self.switch_to(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch(1)
        self._update_styles()

    def _update_styles(self) -> None:
        """更新按钮样式，当前页高亮。"""
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                btn.setStyleSheet(
                    "QPushButton {"
                    "  background: transparent; color: #4a9eff; border: none;"
                    "  border-bottom: 2px solid #4a9eff;"
                    "  padding: 6px 16px; font-size: 13px; font-weight: bold;"
                    "}"
                )
                btn.setChecked(True)
            else:
                btn.setStyleSheet(
                    "QPushButton {"
                    "  background: transparent; color: #999999; border: none;"
                    "  border-bottom: 2px solid transparent;"
                    "  padding: 6px 16px; font-size: 13px;"
                    "}"
                    "QPushButton:hover { color: #cccccc; }"
                )
                btn.setChecked(False)

    def switch_to(self, idx: int) -> None:
        """切换到指定页面。"""
        if 0 <= idx < len(self.PAGES) and idx != self._current:
            self._current = idx
            self._update_styles()
            self.page_changed.emit(idx)
