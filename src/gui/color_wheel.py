"""HSV 色轮控件。

用量角 = 色相、半径 = 饱和度、配合外部亮度滑杆组成的圆形取色器。颜色位图在
构造时用向量化 numpy 一次性预渲染为 QImage 并缓存，``paintEvent`` 仅绘制该
位图与指向当前色的指示圈，鼠标按下/拖动即时换算为 HSV 并发出 ``colorChanged``
——避免在 UI 线程做任何逐像素重算，保证交互流畅不卡顿。

主要接口：
- ``color()`` / ``set_color(c)``：当前颜色。
- ``set_brightness(v)``：亮度 0-255。
- ``colorChanged = Signal(QColor)``。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


def _hsv_to_rgb(hue_deg: np.ndarray, sat: np.ndarray, val: np.ndarray):
    """向量化 HSV→RGB（各参数同形状归一化数组 [0,1]/[0,360]），返回 (r,g,b) [0,1]。"""
    h = (hue_deg / 60.0) % 6.0
    c = val * sat
    x = c * (1.0 - np.abs((h % 2) - 1.0))
    m = val - c
    h2 = h
    r = np.select(
        [(h2 < 1), (h2 < 2), (h2 < 3), (h2 < 4), (h2 < 5), True],
        [c, x, 0.0, 0.0, x, c],
    )
    g = np.select(
        [(h2 < 1), (h2 < 2), (h2 < 3), (h2 < 4), (h2 < 5), True],
        [x, c, c, x, 0.0, 0.0],
    )
    b = np.select(
        [(h2 < 1), (h2 < 2), (h2 < 3), (h2 < 4), (h2 < 5), True],
        [0.0, 0.0, x, c, c, x],
    )
    return r + m, g + m, b + m


class ColorWheel(QWidget):
    """圆形 HSV 色轮（可点击/拖动取色）。"""

    colorChanged = Signal(QColor)
    picked = Signal(QColor)  # 一次取色完成（松开鼠标）

    def __init__(self, size: int = 160, parent=None):
        super().__init__(parent)
        self._size = int(size)
        self.setFixedSize(self._size, self._size)
        # 当前色（HSV）：色相/饱和度来自轮上点击位，亮度外部调节
        self._hue = 0.0
        self._sat = 255
        self._value = 255
        self._color = QColor(255, 0, 0, 255)
        self._wheel_img: QImage | None = self._render_wheel()

    def _center(self) -> float:
        return (self._size - 1) / 2.0

    def _render_wheel(self) -> QImage | None:
        """向量化预渲染 HSV 色轮位图（V=255）。"""
        N = self._size
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
        c = self._center()
        dx = xx - c
        dy = yy - c
        rad = np.hypot(dx, dy)
        rmax = c if c > 0 else 1.0
        hue = np.degrees(np.arctan2(dy, dx)) % 360.0
        sat = np.clip(rad / rmax, 0.0, 1.0)
        r, g, b = _hsv_to_rgb(hue, sat, np.full_like(sat, 1.0))
        rgb = (np.stack([r, g, b], axis=-1) * 255.0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # 轮外区域涂为面板深色，与参数面板背景融合
        outside = rad > rmax
        rgb[outside] = (43, 43, 43)
        img = QImage(rgb.data, N, N, 3 * N, QImage.Format_RGB888)
        return img.copy()

    def _pos_to_hue_sat(self, pos: QPoint) -> tuple[float, int] | None:
        c = self._center()
        dx = pos.x() - c
        dy = pos.y() - c
        rmax = c
        rad = np.hypot(dx, dy)
        if rmax <= 0 or rad > rmax:
            return None
        hue = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
        sat = int(np.clip(rad / rmax, 0.0, 1.0) * 255.0)
        return hue, sat

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        if not color.isValid():
            return
        self._color = QColor(color)
        h, s, v, _ = color.getHsv()
        if h < 0:
            h = 0
        self._hue = float(h)
        self._sat = min(255, max(0, s))
        self._value = min(255, max(0, v))
        self.update()

    def set_brightness(self, value: int) -> None:
        self._value = int(np.clip(value, 0, 255))
        self._color = QColor.fromHsv(int(self._hue), self._sat, self._value)
        self.update()
        self.colorChanged.emit(self._color)

    def _set_from_pos(self, pos: QPoint) -> None:
        hs = self._pos_to_hue_sat(pos)
        if hs is None:
            return
        hue, sat = hs
        self._hue = hue
        self._sat = sat
        self._color = QColor.fromHsv(int(round(hue)), sat, self._value)
        self.update()
        self.colorChanged.emit(self._color)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_from_pos(event.position().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_from_pos(event.position().toPoint())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(self._color)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 底色
        painter.fillRect(self.rect(), QColor(43, 43, 43))
        if self._wheel_img is not None:
            painter.drawImage(0, 0, self._wheel_img)
        # 当前色指示圈（白/黑对比外环）
        c = self._center()
        ang = np.radians(self._hue)
        rmax = c
        rad = (self._sat / 255.0) * rmax
        px = float(c + rad * np.cos(ang))
        py = float(c + rad * np.sin(ang))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPointF(px, py), 4, 4)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawEllipse(QPointF(px, py), 2, 2)
        painter.end()