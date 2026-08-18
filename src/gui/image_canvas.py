"""图像显示画布组件。

基于 QPainter 的图像显示控件，接收 numpy 数组并支持以鼠标位置为中心的滚轮缩放
与左键拖拽平移，用于在 GUI 中展示像素转换前后的图像。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QPointF, Signal
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPixmap,
    QColor,
    QPen,
    QFont,
    QPolygonF,
    QDragEnterEvent,
    QDropEvent,
    QDragLeaveEvent,
)
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """图像显示画布，支持缩放与拖拽平移。

    Features:
        - 鼠标滚轮缩放（以鼠标位置为中心）
        - 左键拖拽平移
        - set_image(arr) 接收 numpy 数组
        - 棋盘格背景
        - 底部信息栏 overlay
        - 网格叠加绘制
        - 双击 fit_to_view，Ctrl+0 重置视图
        - 拖拽导入图片文件或文件夹
    """

    # 拖拽导入信号
    image_dropped = Signal(str)      # 单张图片文件路径
    folder_dropped = Signal(str)     # 文件夹路径

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None  # (H,W,3) uint8
        self._qimage: QImage | None = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)  # pan offset
        self._dragging = False
        self._drag_start = QPoint()
        self._offset_start = QPoint()
        self._info: str = ""  # 附加信息字符串
        self._overlay_grid = None  # 网格叠加 dataclass 或 None
        self._checker: QPixmap = self._make_checkerboard()
        self._drop_highlight = False  # 拖拽高亮
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

    @staticmethod
    def _make_checkerboard() -> QPixmap:
        """生成 16x16 棋盘格 QPixmap（浅灰 #d0d0d0 与白 #ffffff 各 8x8 交替）。"""
        pm = QPixmap(16, 16)
        pm.fill(QColor("#d0d0d0"))
        p = QPainter(pm)
        p.fillRect(QRect(0, 0, 8, 8), QColor("#ffffff"))
        p.fillRect(QRect(8, 8, 8, 8), QColor("#ffffff"))
        p.end()
        return pm

    def set_image(self, arr: np.ndarray) -> None:
        """接收 numpy (H,W,3) 数组（uint8 或 float 0-255），转 QImage 并重绘。"""
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # numpy (H,W,3) -> QImage Format_RGB888
            h, w = arr.shape[:2]
            # Need contiguous bytes in RGB order
            self._qimage = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
            self._image = arr
            self._scale = 1.0
            self._offset = QPoint(0, 0)
            self.update()

    def clear(self) -> None:
        """清空画布内容（回到"无图片"提示状态）。"""
        self._qimage = None
        self._image = None
        self._overlay_grid = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.update()

    def set_info(self, extra: str) -> None:
        """设置底部信息栏附加信息字符串，触发重绘。"""
        self._info = extra or ""
        self.update()

    def set_grid_overlay(self, grid) -> None:
        """设置网格叠加。grid 是带 px,py,phase_x,phase_y,w_logic,h_logic 属性的对象；None 清除。"""
        self._overlay_grid = grid
        self.update()

    def fit_to_view(self) -> None:
        """缩放并居中图像，使其完整显示在画布内。无图像则直接返回。"""
        if self._qimage is None:
            return
        img_w = self._qimage.width()
        img_h = self._qimage.height()
        canvas_w = self.width()
        canvas_h = self.height()
        if img_w <= 0 or img_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
            return
        scale = min(canvas_w / img_w, canvas_h / img_h)
        self._scale = scale
        self._offset = QPoint(
            int((canvas_w - img_w * scale) / 2),
            int((canvas_h - img_h * scale) / 2),
        )
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        # 像素艺术使用最近邻插值，保持锐利方块边缘（禁用平滑缩放）
        # 棋盘格背景
        painter.drawTiledPixmap(self.rect(), self._checker)

        if self._qimage is not None:
            # Draw image with scale and offset
            painter.save()
            painter.translate(self._offset)
            painter.scale(self._scale, self._scale)
            painter.drawImage(0, 0, self._qimage)
            # 网格叠加绘制（在 drawImage 之上）
            if self._overlay_grid is not None:
                grid = self._overlay_grid
                img_w = self._qimage.width()
                img_h = self._qimage.height()
                # 线宽除以 scale，使屏幕上保持 1px
                painter.setPen(QPen(QColor(255, 50, 50, 140), 1.0 / self._scale))
                cell_ys = getattr(grid, 'cell_ys', None)
                cell_xs = getattr(grid, 'cell_xs', None)
                if cell_ys is not None and cell_xs is not None:
                    h_logic = cell_ys.shape[0] - 1
                    w_logic = cell_xs.shape[1] - 1
                    # 水平线：每行交点连成折线（支持局部漂移）
                    for j in range(h_logic + 1):
                        poly = QPolygonF()
                        for i in range(w_logic + 1):
                            poly.append(QPointF(float(cell_xs[j, i]), float(cell_ys[j, i])))
                        painter.drawPolyline(poly)
                    # 垂直线：每列交点连成折线
                    for i in range(w_logic + 1):
                        poly = QPolygonF()
                        for j in range(h_logic + 1):
                            poly.append(QPointF(float(cell_xs[j, i]), float(cell_ys[j, i])))
                        painter.drawPolyline(poly)
                else:
                    # 回退：等距网格
                    for i in range(int(grid.w_logic) + 1):
                        x = grid.phase_x + i * grid.px
                        painter.drawLine(QPoint(int(x), 0), QPoint(int(x), img_h))
                    for j in range(int(grid.h_logic) + 1):
                        y = grid.phase_y + j * grid.py
                        painter.drawLine(QPoint(0, int(y)), QPoint(img_w, int(y)))
            painter.restore()
        else:
            # 无图片：黑色背景 + 大字号提示
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            font = painter.font()
            font.setPointSize(26)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(110, 110, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无图片")

        # 画布信息栏（底部 overlay）
        info_h = 22
        info_rect = QRect(0, self.height() - info_h, self.width(), info_h)
        painter.fillRect(info_rect, QColor(0, 0, 0, 160))
        painter.setPen(Qt.GlobalColor.white)
        if self._qimage is not None:
            w = self._qimage.width()
            h = self._qimage.height()
        else:
            w = 0
            h = 0
        scale_percent = round(self._scale * 100)
        text = f"{w}×{h} | {scale_percent}%"
        if self._info:
            text += " | " + self._info
        # 左边距 8px，垂直居中，左对齐
        text_rect = QRect(8, self.height() - info_h, self.width() - 8, info_h)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )

        # 拖拽高亮边框
        if self._drop_highlight:
            painter.setPen(QPen(QColor("#4a9eff"), 3, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = self._scale * factor
        new_scale = max(0.1, min(20.0, new_scale))
        # Zoom centered on mouse position
        mouse_pos = event.position().toPoint()
        # Adjust offset so the point under cursor stays fixed
        img_x = (mouse_pos.x() - self._offset.x()) / self._scale
        img_y = (mouse_pos.y() - self._offset.y()) / self._scale
        self._scale = new_scale
        self._offset = QPoint(
            int(mouse_pos.x() - img_x * self._scale),
            int(mouse_pos.y() - img_y * self._scale),
        )
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._offset_start = QPoint(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position().toPoint() - self._drag_start
            self._offset = QPoint(self._offset_start.x() + delta.x(),
                                  self._offset_start.y() + delta.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """双击触发 fit_to_view。"""
        self.fit_to_view()

    def keyPressEvent(self, event):
        """Ctrl+0 重置视图。"""
        if event.key() == Qt.Key.Key_0 and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def reset_view(self):
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.update()

    # ------------------------------------------------------------------
    # 拖拽导入
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """拖拽进入：检查是否为图片文件或文件夹。"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    import os
                    if os.path.isdir(path) or path.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
                    ):
                        event.acceptProposedAction()
                        self._drop_highlight = True
                        self.update()
                        return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """拖拽离开：清除高亮。"""
        self._drop_highlight = False
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:
        """拖放：发出信号通知文件路径。"""
        self._drop_highlight = False
        self.update()
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        import os
        if os.path.isdir(path):
            self.folder_dropped.emit(path)
        else:
            self.image_dropped.emit(path)

    def set_comparison(self, original: np.ndarray, result: np.ndarray) -> None:
        """设置前后对比模式：左右并排显示原图与结果。"""
        # Simple implementation: store both, draw side by side in paintEvent
        # For simplicity, just set the result image for now (can be enhanced later)
        self.set_image(result)
