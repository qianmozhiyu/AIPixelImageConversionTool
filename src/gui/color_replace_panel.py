"""可复用的像素图颜色替换面板（色卡 + HSV 色轮）。

用于在预览阶段对某张像素化结果做「全局同色替换」：显示该图像出现次数最多的
颜色色卡，点击某色块将其选为替换源色，再用内联 HSV 色轮选择目标色，点击
应用后把整图中所有与该源色相同的像素替换为目标色，并发出 ``colorReplaced``
携带替换后的新图像。

色卡与色轮均为内联控件（无模态对话框），色轮位图预渲染缓存，交互流畅不卡顿。
"""

from __future__ import annotations

from functools import partial

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QSlider,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QScrollArea,
    QFrame,
)

from ..core.color_remap import remap_color
from .color_wheel import ColorWheel


class ColorReplacePanel(QWidget):
    """色卡 + 色轮 全局颜色替换面板。"""

    colorReplaced = Signal(object)  # 全局换色后的像素图

    def __init__(self, swatch_colors: int = 16, parent=None):
        super().__init__(parent)
        self._swatch_limit = max(1, int(swatch_colors))
        self._img: np.ndarray | None = None
        self._selected_src: tuple | None = None
        self._selected_btn: QPushButton | None = None
        self._swatch_btns: dict[tuple, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- 色卡区（网格 + 限高，颜色多时也不挤压布局）---
        self._swatches_scroll = QScrollArea()
        self._swatches_scroll.setWidgetResizable(True)
        self._swatches_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._swatches_scroll.setMaximumHeight(140)
        self._swatches_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea QScrollBar:vertical { background: #2d2d2d; width: 6px;"
            "  border-radius: 3px; margin: 1px; }"
            "QScrollArea QScrollBar::handle:vertical { background: #4a9eff;"
            "  border-radius: 3px; min-height: 24px; }"
            "QScrollArea QScrollBar::add-line:vertical,"
            "QScrollArea QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollArea QScrollBar::add-page:vertical,"
            "QScrollArea QScrollBar::sub-page:vertical { background: transparent; }"
        )
        self._swatch_viewport = QWidget()
        self._swatch_grid = QGridLayout(self._swatch_viewport)
        self._swatch_grid.setContentsMargins(0, 0, 0, 0)
        self._swatch_grid.setSpacing(4)
        self._swatches_scroll.setWidget(self._swatch_viewport)
        self._swatch_rows: list[QPushButton] = []
        layout.addWidget(self._swatches_scroll)

        # --- 选择提示 + 内联色轮 ---
        self._sel_label = QLabel("点击上方色卡选择「替换源色」")
        self._sel_label.setStyleSheet("color: #c0c0c0; font-size: 12px;")
        layout.addWidget(self._sel_label)

        wheel_row = QHBoxLayout()
        wheel_row.setSpacing(10)
        self._wheel = ColorWheel(116)
        wheel_row.addWidget(self._wheel)
        side = QVBoxLayout()
        side.setSpacing(6)
        target_row = QHBoxLayout()
        target_row.setSpacing(6)
        self._target_preview = QPushButton()
        self._target_preview.setFixedSize(40, 28)
        self._target_preview.setEnabled(False)
        self._target_preview.setCursor(Qt.CursorShape.ArrowCursor)
        self._target_label = QLabel("目标色 --")
        self._target_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        target_row.addWidget(self._target_preview)
        target_row.addWidget(self._target_label)
        target_row.addStretch(1)
        side.addLayout(target_row)
        side.addWidget(QLabel("亮度"))
        self._brightness = QSlider(Qt.Orientation.Horizontal)
        self._brightness.setRange(0, 255)
        self._brightness.setValue(255)
        side.addWidget(self._brightness)
        side.addStretch(1)
        wheel_row.addLayout(side, 1)
        layout.addLayout(wheel_row)

        self._apply_btn = QPushButton("应用替换选中颜色")
        self._apply_btn.setEnabled(False)
        layout.addWidget(self._apply_btn)

        self._wheel.colorChanged.connect(self._on_wheel_changed)
        self._brightness.valueChanged.connect(self._wheel.set_brightness)
        self._apply_btn.clicked.connect(self._on_apply_replace)
        self._on_wheel_changed(self._wheel.color())

    # -- 对外接口 ----------------------------------------------------------
    def set_image(self, img) -> None:
        """设置来源图像并重建色卡。"""
        self._img = img
        self._selected_src = None
        self._selected_btn = None
        self._apply_btn.setEnabled(False)
        self._rebuild_swatches()

    def set_swatch_limit(self, n: int) -> None:
        self._swatch_limit = max(1, int(n))
        self._rebuild_swatches()

    def apply_replace(self, src, dst) -> None:
        """把图像中的 ``src`` 色替换为 ``dst``（供脚本/测试用）。"""
        if self._img is None:
            return
        new_img = remap_color(self._img, src_rgb=src, dst_rgb=dst, tol=0)
        self._img = new_img
        self._rebuild_swatches()
        if self._selected_src is not None and self._selected_src in self._swatch_btns:
            btn = self._swatch_btns[self._selected_src]
            self._selected_btn = btn
            style = btn.styleSheet()
            if "border: 2px solid #3d5a80;" not in style:
                btn.setStyleSheet(style + "border: 2px solid #3d5a80;")
        else:
            self._selected_src = None
            self._selected_btn = None
            self._apply_btn.setEnabled(False)
        self.colorReplaced.emit(new_img)

    # -- 内部 ------------------------------------------------------------------
    def _clear_swatches(self) -> None:
        while self._swatch_grid.count():
            item = self._swatch_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._swatch_rows.clear()
        self._swatch_btns.clear()
        self._selected_btn = None

    def _rebuild_swatches(self) -> None:
        self._clear_swatches()
        if self._img is None:
            return
        arr = np.asarray(self._img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return
        rgb = arr[..., :3].reshape(-1, 3)
        unique, counts = np.unique(rgb, axis=0, return_counts=True)
        if unique.shape[0] == 0:
            return
        order = np.argsort(counts)[::-1][: self._swatch_limit]
        cols = 4
        for idx, si in enumerate(order):
            src = tuple(int(c) for c in unique[si])
            count = int(counts[si])
            btn = self._make_swatch(src, count)
            self._swatch_grid.addWidget(btn, idx // cols, idx % cols)

    def _make_swatch(self, src, count) -> QPushButton:
        """生成一个紧凑色块按钮（右上提示 count，点击选为源色）。"""
        r, g, b = src
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "#ffffff" if luminance < 140 else "#000000"
        btn = QPushButton(f"#{r:02X}{g:02X}{b:02X}")
        btn.setFixedHeight(24)
        btn.setMinimumWidth(60)
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {text_color};"
            "border: 1px solid #555555; border-radius: 3px;"
            "padding: 0 2px; font-size: 11px;"
        )
        btn.setToolTip(f"RGB({r},{g},{b}) · {count} 像素\n点击选为替换源色")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(partial(self._select_swatch, src))
        self._swatch_rows.append(btn)
        self._swatch_btns[src] = btn
        return btn

    def _select_swatch(self, src, _checked=False) -> None:
        self._selected_src = src
        if self._selected_btn is not None:
            self._selected_btn.setStyleSheet(
                self._selected_btn.styleSheet().replace("border: 2px solid #3d5a80;", "")
            )
        btn = self._swatch_btns.get(src)
        self._selected_btn = btn
        if btn is not None:
            style = btn.styleSheet()
            if "border: 2px solid #3d5a80;" not in style:
                btn.setStyleSheet(style + "border: 2px solid #3d5a80;")
        self._sel_label.setText(
            f"替换源色: #{src[0]:02X}{src[1]:02X}{src[2]:02X}（点击色轮选目标色）"
        )
        self._wheel.set_color(QColor(*src))
        self._on_wheel_changed(self._wheel.color())
        self._apply_btn.setEnabled(True)

    def _on_wheel_changed(self, color) -> None:
        self._target_preview.setStyleSheet(
            f"background-color: {color.name()};"
            "border: 1px solid #555; border-radius: 3px;"
        )
        color.setAlpha(255)
        self._target_label.setText(f"目标色 {color.name()}")

    def _on_apply_replace(self) -> None:
        if self._selected_src is None:
            return
        c = self._wheel.color()
        self.apply_replace(self._selected_src, (c.red(), c.green(), c.blue()))