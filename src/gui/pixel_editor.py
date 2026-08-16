"""像素图编辑器。

类 Aseprite 的简易像素图编辑器，支持画笔(B)、橡皮(E)、吸管(I)、油漆桶(G)、
直线(L)、矩形(U)等工具，快捷键与 PS/Aseprite 一致。
"""

from __future__ import annotations

import numpy as np
from collections import deque

from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import (
    QImage, QPainter, QPixmap, QColor, QPen, QMouseEvent, QKeyEvent,
    QAction, QKeySequence, QIcon,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QColorDialog, QScrollArea, QFrame, QToolBar, QStatusBar,
    QToolButton, QButtonGroup, QCheckBox,
)

from ..core.asset_manager import AssetManager
from ..core.io import save_image


class PixelCanvas(QWidget):
    """像素画布组件，支持放大显示和像素级编辑。

    Signals:
        pixel_edited(): 像素被编辑（用于刷新颜色按钮等）
        stroke_started(): 一次绘制笔画开始（用于压入撤销栈）
    """

    pixel_edited = Signal()
    stroke_started = Signal()

    # 工具枚举
    TOOL_BRUSH = "brush"
    TOOL_ERASER = "eraser"
    TOOL_EYEDROPPER = "eyedropper"
    TOOL_FILL = "fill"
    TOOL_LINE = "line"
    TOOL_RECT = "rect"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None  # (H,W,4) uint8 RGBA
        self._scale = 16  # 像素放大倍数
        self._offset = QPoint(0, 0)
        self._tool = self.TOOL_BRUSH
        self._fg_color = QColor(0, 0, 0, 255)
        self._bg_color = QColor(255, 255, 255, 255)
        self._brush_size = 1
        self._drawing = False
        self._alt_picking = False  # 按住 Alt 临时吸管取色中
        self._shift_line = False   # 按住 Shift 临时直线工具中
        self._drag_start = QPoint()
        self._panning = False
        self._pan_start = QPoint()
        self._offset_start = QPoint()
        self._space_pressed = False
        self._temp_preview: np.ndarray | None = None  # 直线/矩形预览
        self._hover_pixel: QPoint | None = None  # 当前鼠标所在像素（用于十字光标）
        # 完美像素（Aseprite 风格，默认开启）：绘制时自动消除 2px 宽对角伪影
        self._perfect_pixel = True
        self._pp_prev: QPoint | None = None  # 笔画窗口：三元组中的前一点
        self._pp_last: QPoint | None = None  # 笔画窗口：待裁决的当前点
        self._pp_last_raw: QPoint | None = None  # 笔画中最后一个原始点（用于插值）
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # 隐藏系统光标，仅保留绘制的十字光标 + 前景色像素块指示位置
        self.setCursor(Qt.CursorShape.BlankCursor)

    def set_image(self, image: np.ndarray) -> None:
        """设置画布图像，自动转为 RGBA。"""
        if image.ndim == 3 and image.shape[2] == 3:
            h, w = image.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = np.clip(image, 0, 255).astype(np.uint8)
            rgba[:, :, 3] = 255
            self._image = rgba
        else:
            self._image = np.clip(image, 0, 255).astype(np.uint8)
        self._temp_preview = None
        self.update()

    def get_image(self) -> np.ndarray | None:
        """获取当前图像（RGB）。"""
        if self._image is None:
            return None
        return self._image[:, :, :3].copy()

    def get_image_rgba(self) -> np.ndarray | None:
        if self._image is None:
            return None
        return self._image.copy()

    def set_tool(self, tool: str) -> None:
        self._tool = tool

    def get_tool(self) -> str:
        return self._tool

    def set_fg_color(self, color: QColor) -> None:
        self._fg_color = color

    def get_fg_color(self) -> QColor:
        return self._fg_color

    def set_bg_color(self, color: QColor) -> None:
        self._bg_color = color

    def get_bg_color(self) -> QColor:
        return self._bg_color

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(20, size))

    def get_brush_size(self) -> int:
        return self._brush_size

    def set_perfect_pixel(self, enabled: bool) -> None:
        """启用/禁用完美像素（Aseprite 风格，默认开启）。"""
        self._perfect_pixel = bool(enabled)
        if not self._perfect_pixel:
            self._pp_prev = None
            self._pp_last = None
            self._pp_last_raw = None

    def get_perfect_pixel(self) -> bool:
        return self._perfect_pixel

    @staticmethod
    def _line_points(p0: QPoint, p1: QPoint) -> list:
        """返回 p0→p1 的 8 连通 Bresenham 点序列（含两端）。"""
        pts = []
        x0, y0 = p0.x(), p0.y()
        x1, y1 = p1.x(), p1.y()
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            pts.append(QPoint(x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    @staticmethod
    def _pp_is_redundant(prev: QPoint, cur: QPoint, next_: QPoint) -> bool:
        """Aseprite IntertwineAsPixelPerfect 规则：判断 cur 是否为冗余角点。

        当 prev↔cur 共轴、cur↔next 共轴、且 prev↔next 斜对角时，cur 是
        "L 形拐角"，直接跳过可得到 1px 宽的完美对角线条。
        """
        aligned_prev = (prev.x() == cur.x() or prev.y() == cur.y())
        aligned_next = (cur.x() == next_.x() or cur.y() == next_.y())
        diagonal = (prev.x() != next_.x() and prev.y() != next_.y())
        return aligned_prev and aligned_next and diagonal

    def _pp_append(self, pt: QPoint, color: QColor) -> None:
        """完美像素流式处理（Aseprite IntertwineAsPixelPerfect 语义）。

        维护滑动三元组 (prev, last, pt)：last 冗余则跳过并立即绘制 pt 作为
        替代（pt 成为下一组三元组的 prev）；否则绘制 last。
        """
        if self._pp_prev is not None and self._pp_last is not None:
            if self._pp_is_redundant(self._pp_prev, self._pp_last, pt):
                # last 是冗余 L 角点：跳过，pt 立即绘制并接管 prev 位置
                self._draw_pixel(pt.x(), pt.y(), color)
                self._pp_prev = QPoint(pt)
                self._pp_last = None
            else:
                self._draw_pixel(self._pp_last.x(), self._pp_last.y(), color)
                self._pp_prev = QPoint(self._pp_last)
                self._pp_last = QPoint(pt)
        elif self._pp_last is not None:
            self._pp_prev = QPoint(self._pp_last)
            self._pp_last = QPoint(pt)
        else:
            self._pp_last = QPoint(pt)

    def set_scale(self, scale: int) -> None:
        self._scale = max(1, min(64, scale))
        self.update()

    def get_scale(self) -> int:
        return self._scale

    def _widget_to_pixel(self, pos: QPoint) -> QPoint | None:
        """将控件坐标转换为像素坐标。"""
        if self._image is None:
            return None
        px = (pos.x() - self._offset.x()) // self._scale
        py = (pos.y() - self._offset.y()) // self._scale
        h, w = self._image.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            return QPoint(px, py)
        return None

    def _draw_pixel(self, x: int, y: int, color: QColor, img: np.ndarray | None = None) -> None:
        """在指定位置绘制像素（考虑笔刷大小）。"""
        target = img if img is not None else self._image
        if target is None:
            return
        h, w = target.shape[:2]
        size = self._brush_size
        half = size // 2
        for dy in range(-half, size - half):
            for dx in range(-half, size - half):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    target[ny, nx] = [color.red(), color.green(), color.blue(), color.alpha()]

    def _pick_color(self, pos: QPoint) -> None:
        """吸管取色：将像素坐标处的颜色设为前景色。"""
        if self._image is None or pos is None:
            return
        h, w = self._image.shape[:2]
        if not (0 <= pos.x() < w and 0 <= pos.y() < h):
            return
        self._fg_color = QColor(*self._image[pos.y(), pos.x()])
        self.pixel_edited.emit()
        self.update()

    def _draw_line_bresenham(self, x0: int, y0: int, x1: int, y1: int, color: QColor, img: np.ndarray) -> None:
        """Bresenham 直线算法。"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self._draw_pixel(x0, y0, color, img)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _draw_rect(self, x0: int, y0: int, x1: int, y1: int, color: QColor, img: np.ndarray) -> None:
        """绘制矩形边框。"""
        self._draw_line_bresenham(x0, y0, x1, y0, color, img)
        self._draw_line_bresenham(x1, y0, x1, y1, color, img)
        self._draw_line_bresenham(x1, y1, x0, y1, color, img)
        self._draw_line_bresenham(x0, y1, x0, y0, color, img)

    def _flood_fill(self, x: int, y: int, fill_color: QColor) -> None:
        """洪水填充连通同色区域。"""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        target = tuple(self._image[y, x])
        fill = (fill_color.red(), fill_color.green(), fill_color.blue(), fill_color.alpha())
        if target == fill:
            return
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue
            if tuple(self._image[cy, cx]) != target:
                continue
            self._image[cy, cx] = list(fill)
            queue.append((cx + 1, cy))
            queue.append((cx - 1, cy))
            queue.append((cx, cy + 1))
            queue.append((cx, cy - 1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            return
        if self._space_pressed or event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._offset_start = QPoint(self._offset)
            return

        pos = self._widget_to_pixel(event.position().toPoint())
        if pos is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                # 按住 Alt = 临时吸管工具：取色并允许拖动连续取色
                # 不修改 self._tool，松开 Alt 后自动回到上一次使用的工具
                self._drawing = True
                self._alt_picking = True
                self._temp_preview = None
                self._pick_color(pos)
                return

            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # 按住 Shift = 临时直线工具（与 Alt 吸管同理，不修改 self._tool；
                # 按下左键开始预览，松开左键提交）
                self._drawing = True
                self._shift_line = True
                self._temp_preview = self._image.copy()
                self._drag_start = QPoint(pos)
                return

            self._drawing = True
            self._drag_start = QPoint(pos)

            if self._tool == self.TOOL_BRUSH:
                # 起始点立即绘制并重置完美像素笔画状态
                self._draw_pixel(pos.x(), pos.y(), self._fg_color)
                self._pp_prev = None
                self._pp_last = QPoint(pos)
                self._pp_last_raw = QPoint(pos)
                self.pixel_edited.emit()
            elif self._tool == self.TOOL_ERASER:
                self._draw_pixel(pos.x(), pos.y(), QColor(0, 0, 0, 0))
                self._pp_prev = None
                self._pp_last = QPoint(pos)
                self._pp_last_raw = QPoint(pos)
                self.pixel_edited.emit()
            elif self._tool == self.TOOL_EYEDROPPER:
                # 吸管不改图，不记录撤销
                color = QColor(*self._image[pos.y(), pos.x()])
                self._fg_color = color
                self.pixel_edited.emit()
            elif self._tool == self.TOOL_FILL:
                self._flood_fill(pos.x(), pos.y(), self._fg_color)
                self.pixel_edited.emit()
            elif self._tool in (self.TOOL_LINE, self.TOOL_RECT):
                # 开始预览
                self._temp_preview = self._image.copy()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # 更新笔刷范围框位置（无论是否在绘制/平移都跟随光标）
        self._hover_pixel = self._widget_to_pixel(event.position().toPoint())
        self.update()

        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._offset = self._offset_start + delta
            self.update()
            return

        if self._alt_picking:
            # 按住 Alt 拖动：连续取色（先于工具分支，Alt 已松开也无妨）
            pos = self._widget_to_pixel(event.position().toPoint())
            if pos is not None:
                self._pick_color(pos)
            return

        if not self._drawing or self._image is None:
            return

        pos = self._widget_to_pixel(event.position().toPoint())
        if pos is None:
            return

        if self._shift_line and self._temp_preview is not None:
            # 按住 Shift 临时直线：从起点向当前点绘制预览（恢复原图后重画）
            self._image = self._temp_preview.copy()
            self._draw_line_bresenham(
                self._drag_start.x(), self._drag_start.y(),
                pos.x(), pos.y(), self._fg_color, self._image,
            )
            self.update()
            return

        if self._tool in (self.TOOL_BRUSH, self.TOOL_ERASER):
            color = self._fg_color if self._tool == self.TOOL_BRUSH else QColor(0, 0, 0, 0)
            # 对相邻鼠标事件间插值（8 连通 Bresenham），避免快速拖动断线
            if self._pp_last_raw is not None and pos != self._pp_last_raw:
                pts = self._line_points(self._pp_last_raw, pos)
            else:
                pts = [pos]
            if self._perfect_pixel:
                # 流式完美像素：跳过冗余角点，得到 1px 完美对角线条
                for p in pts[1:]:
                    self._pp_append(p, color)
            else:
                for p in pts:
                    self._draw_pixel(p.x(), p.y(), color)
            self._pp_last_raw = QPoint(pos)
            self.pixel_edited.emit()
        elif self._tool in (self.TOOL_LINE, self.TOOL_RECT) and self._temp_preview is not None:
            # 恢复原图，绘制预览
            self._image = self._temp_preview.copy()
            start = self._drag_start
            if self._tool == self.TOOL_LINE:
                self._draw_line_bresenham(start.x(), start.y(), pos.x(), pos.y(), self._fg_color, self._image)
            else:
                self._draw_rect(start.x(), start.y(), pos.x(), pos.y(), self._fg_color, self._image)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            return
        if self._alt_picking:
            # Alt 临时吸管结束：复位标志（不改变当前工具）
            self._alt_picking = False
            self._drawing = False
            self._temp_preview = None
            return
        if self._shift_line:
            # Shift 临时直线结束：提交当前预览（松开左键提交）
            self._shift_line = False
            self._drawing = False
            self._temp_preview = None
            self.pixel_edited.emit()
            self.stroke_started.emit()  # 操作完成后再压栈（记录操作后状态）
            return
        if self._drawing:
            self._drawing = False
            if self._tool in (self.TOOL_LINE, self.TOOL_RECT):
                self._temp_preview = None
                self.pixel_edited.emit()
            elif self._tool in (self.TOOL_BRUSH, self.TOOL_ERASER):
                # 完美像素：尾点永不被跳过，松开时补画并清空笔画状态
                if self._perfect_pixel and self._pp_last is not None:
                    last = self._pp_last
                    color = (
                        self._fg_color if self._tool == self.TOOL_BRUSH
                        else QColor(0, 0, 0, 0)
                    )
                    self._draw_pixel(last.x(), last.y(), color)
                self._pp_prev = None
                self._pp_last = None
                self._pp_last_raw = None
            # 操作完成后再压栈：撤销栈记录"操作后"状态，避免与初始状态重复
            self.stroke_started.emit()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))

        if self._image is None:
            p.setPen(QColor("#666"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无图像")
            return

        h, w = self._image.shape[:2]
        s = self._scale

        # 绘制棋盘格背景（透明区域）
        for y in range(h):
            for x in range(w):
                if self._image[y, x][3] == 0:  # 透明
                    color = QColor("#2a2a2a") if (x + y) % 2 == 0 else QColor("#333333")
                    p.fillRect(QRect(self._offset.x() + x * s, self._offset.y() + y * s, s, s), color)

        # 绘制像素
        qimg = QImage(self._image.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.drawImage(QRect(self._offset.x(), self._offset.y(), w * s, h * s), qimg)

        # 像素网格
        if s >= 8:
            p.setPen(QPen(QColor(128, 128, 128, 60), 1))
            for x in range(w + 1):
                p.drawLine(self._offset.x() + x * s, self._offset.y(),
                          self._offset.x() + x * s, self._offset.y() + h * s)
            for y in range(h + 1):
                p.drawLine(self._offset.x(), self._offset.y() + y * s,
                          self._offset.x() + w * s, self._offset.y() + y * s)

        # 光标指示：隐藏系统光标，绘制前景色像素块 + 纯十字（无箭头）
        if self._hover_pixel is not None and not self._panning:
            hx, hy = self._hover_pixel.x(), self._hover_pixel.y()
            cx = self._offset.x() + hx * s + s // 2
            cy = self._offset.y() + hy * s + s // 2

            # 光标下的像素色块：画笔时用前景色原样填充（不透明、无边框），
            # 所见即所得——与左键按下绘制的颜色完全一致
            if self._tool == self.TOOL_BRUSH:
                half = self._brush_size // 2
                bx = self._offset.x() + (hx - half) * s
                by = self._offset.y() + (hy - half) * s
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(self._fg_color))
                p.drawRect(bx, by, self._brush_size * s, self._brush_size * s)

            # 纯十字：黑色外描边（粗 3px）保证任何底色下可见，白色 1px 十字线
            arm = 14
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawLine(cx - arm, cy, cx + arm, cy)
            p.drawLine(cx, cy - arm, cx, cy + arm)
            p.setPen(QPen(QColor(255, 255, 255, 235), 1))
            p.drawLine(cx - arm, cy, cx + arm, cy)
            p.drawLine(cx, cy - arm, cx, cy + arm)

    def wheelEvent(self, event) -> None:
        """滚轮缩放。"""
        delta = event.angleDelta().y()
        if delta > 0:
            self._scale = min(64, self._scale + 2)
        else:
            self._scale = max(1, self._scale - 2)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = False
            self.setCursor(Qt.CursorShape.BlankCursor)  # 回到隐藏光标 + 十字光标模式
        elif event.key() == Qt.Key.Key_Alt and self._alt_picking:
            # 兜底：鼠标释放事件丢失时（如拖出窗口外），松开 Alt 复位取色状态
            self._alt_picking = False
            self._drawing = False
            self._temp_preview = None

    def leaveEvent(self, event) -> None:
        # 鼠标离开画布：清除笔刷范围框
        self._hover_pixel = None
        self.update()
        super().leaveEvent(event)


class PixelEditor(QMainWindow):
    """像素图编辑器主窗口。

    工具：画笔(B)、橡皮(E)、吸管(I)、油漆桶(G)、直线(L)、矩形(U)
    快捷键：[/] 笔刷大小，Ctrl+Z 撤销，Ctrl+Y 重做，Ctrl+S 保存
    """

    saved = Signal()  # 保存信号

    def __init__(self, asset_manager: AssetManager, asset_id: str, parent=None):
        super().__init__(parent)
        from ..core import config
        self.asset_manager = asset_manager
        self.asset_id = asset_id
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        # 撤销历史条数上限：从设置读取（1-500），编辑器打开时生效
        self._max_undo = max(
            1, min(500, int(config.load_preference("undo_history_limit", 50) or 50))
        )

        info = asset_manager.get_info(asset_id)
        title = f"编辑: {info.source_name}" if info else "像素编辑器"
        self.setWindowTitle(title)
        self.resize(900, 700)

        self._build_ui()
        self._load_asset()
        self._build_shortcuts()

    def _build_ui(self) -> None:
        # 中央区域
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧工具栏
        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

        # 中间画布（滚动区域）
        self.canvas = PixelCanvas()
        self.canvas.pixel_edited.connect(self._on_pixel_edited)
        # 每次绘制笔画开始时压入撤销栈（Ctrl+Z 可逐笔撤销）
        self.canvas.stroke_started.connect(self._push_undo)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")
        layout.addWidget(scroll, 1)

        # 右侧颜色面板
        color_panel = self._build_color_panel()
        layout.addWidget(color_panel)

        self.setCentralWidget(central)

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

    def _build_toolbar(self) -> QWidget:
        """构建左侧工具栏。"""
        toolbar = QFrame()
        toolbar.setFixedWidth(56)
        toolbar.setStyleSheet("QFrame { background: #2b2b2b; border-right: 1px solid #333; }")
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)

        self._tool_buttons = QButtonGroup(self)
        self._tool_buttons.setExclusive(True)

        tools = [
            (PixelCanvas.TOOL_BRUSH, "画笔 (B)", "画笔"),
            (PixelCanvas.TOOL_ERASER, "橡皮 (E)", "橡皮"),
            (PixelCanvas.TOOL_EYEDROPPER, "吸管 (I)", "吸管"),
            (PixelCanvas.TOOL_FILL, "油漆桶 (G)", "油漆桶"),
            (PixelCanvas.TOOL_LINE, "直线 (L)", "直线"),
            (PixelCanvas.TOOL_RECT, "矩形 (U)", "矩形"),
        ]

        for tool_id, tooltip, label in tools:
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(48, 48)
            btn.setStyleSheet(
                "QToolButton { background: #333; color: #e0e0e0; border: none; "
                "border-radius: 4px; font-size: 13px; }"
                "QToolButton:hover { background: #444; }"
                "QToolButton:checked { background: #3d5a80; }"
            )
            btn.clicked.connect(lambda checked, t=tool_id: self._select_tool(t))
            self._tool_buttons.addButton(btn)
            layout.addWidget(btn)

        layout.addStretch(1)
        return toolbar

    def _build_color_panel(self) -> QWidget:
        """构建右侧颜色面板。"""
        panel = QFrame()
        panel.setFixedWidth(200)
        panel.setStyleSheet("QFrame { background: #2b2b2b; border-left: 1px solid #333; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 前景色/背景色
        layout.addWidget(QLabel("前景色:"))
        self.fg_color_btn = QPushButton()
        self.fg_color_btn.setFixedSize(180, 40)
        self.fg_color_btn.clicked.connect(self._pick_fg_color)
        layout.addWidget(self.fg_color_btn)

        layout.addWidget(QLabel("背景色:"))
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedSize(180, 40)
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        layout.addWidget(self.bg_color_btn)

        # 笔刷大小
        layout.addWidget(QLabel("笔刷大小:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 20)
        self.brush_spin.setValue(1)
        self.brush_spin.valueChanged.connect(lambda v: self.canvas.set_brush_size(v))
        layout.addWidget(self.brush_spin)

        # 完美像素（Aseprite 风格，默认开启）：自动消除对角双像素伪影
        self.perfect_pixel_chk = QCheckBox("完美像素")
        self.perfect_pixel_chk.setChecked(True)
        self.perfect_pixel_chk.toggled.connect(self.canvas.set_perfect_pixel)
        layout.addWidget(self.perfect_pixel_chk)

        # 缩放
        layout.addWidget(QLabel("缩放:"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 64)
        self.scale_spin.setValue(16)
        self.scale_spin.valueChanged.connect(lambda v: self.canvas.set_scale(v))
        layout.addWidget(self.scale_spin)

        # 最近颜色
        layout.addWidget(QLabel("最近颜色:"))
        self.recent_colors_layout = QHBoxLayout()
        self.recent_colors_layout.setSpacing(4)
        self._recent_color_btns: list[QPushButton] = []
        for _ in range(8):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet("background: #333; border: 1px solid #555; border-radius: 2px;")
            self.recent_colors_layout.addWidget(btn)
            self._recent_color_btns.append(btn)
        layout.addLayout(self.recent_colors_layout)

        layout.addStretch(1)

        # 保存按钮
        self.save_btn = QPushButton("保存 (Ctrl+S)")
        self.save_btn.setStyleSheet(
            "QPushButton { background: #3d5a80; color: #fff; border: none; "
            "border-radius: 4px; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background: #4a6a90; }"
        )
        self.save_btn.clicked.connect(self._on_save)
        layout.addWidget(self.save_btn)

        return panel

    def _build_shortcuts(self) -> None:
        """构建快捷键。"""
        shortcuts = {
            "B": lambda: self._select_tool(PixelCanvas.TOOL_BRUSH),
            "E": lambda: self._select_tool(PixelCanvas.TOOL_ERASER),
            "I": lambda: self._select_tool(PixelCanvas.TOOL_EYEDROPPER),
            "G": lambda: self._select_tool(PixelCanvas.TOOL_FILL),
            "L": lambda: self._select_tool(PixelCanvas.TOOL_LINE),
            "U": lambda: self._select_tool(PixelCanvas.TOOL_RECT),
        }
        for key, handler in shortcuts.items():
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(handler)
            self.addAction(action)

        # Ctrl+Z 撤销
        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self._undo)
        self.addAction(undo_action)

        # Ctrl+Y / Ctrl+Shift+Z 重做
        redo_action = QAction(self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self._redo)
        self.addAction(redo_action)

        redo_action2 = QAction(self)
        redo_action2.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_action2.triggered.connect(self._redo)
        self.addAction(redo_action2)

        # Ctrl+S 保存
        save_action = QAction(self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._on_save)
        self.addAction(save_action)

        # [ ] 笔刷大小
        brush_down = QAction(self)
        brush_down.setShortcut(QKeySequence("["))
        brush_down.triggered.connect(lambda: self._adjust_brush(-1))
        self.addAction(brush_down)

        brush_up = QAction(self)
        brush_up.setShortcut(QKeySequence("]"))
        brush_up.triggered.connect(lambda: self._adjust_brush(1))
        self.addAction(brush_up)

    def _load_asset(self) -> None:
        """加载资产图片到画布。"""
        img = self.asset_manager.load_asset(self.asset_id)
        if img is not None:
            self.canvas.set_image(img)
            self._push_undo()

    def _select_tool(self, tool: str) -> None:
        """选择工具。"""
        self.canvas.set_tool(tool)
        idx = {
            PixelCanvas.TOOL_BRUSH: 0,
            PixelCanvas.TOOL_ERASER: 1,
            PixelCanvas.TOOL_EYEDROPPER: 2,
            PixelCanvas.TOOL_FILL: 3,
            PixelCanvas.TOOL_LINE: 4,
            PixelCanvas.TOOL_RECT: 5,
        }.get(tool, 0)
        btns = self._tool_buttons.buttons()
        if 0 <= idx < len(btns):
            btns[idx].setChecked(True)
        self.status.showMessage(f"工具: {tool}")

    def _pick_fg_color(self) -> None:
        color = QColorDialog.getColor(self.canvas.get_fg_color(), self, "选择前景色")
        if color.isValid():
            self.canvas.set_fg_color(color)
            self._update_color_buttons()
            self._add_recent_color(color)

    def _pick_bg_color(self) -> None:
        color = QColorDialog.getColor(self.canvas.get_bg_color(), self, "选择背景色")
        if color.isValid():
            self.canvas.set_bg_color(color)
            self._update_color_buttons()

    def _update_color_buttons(self) -> None:
        fg = self.canvas.get_fg_color()
        bg = self.canvas.get_bg_color()
        self.fg_color_btn.setStyleSheet(
            f"background: {fg.name()}; border: 2px solid #555; border-radius: 4px;"
        )
        self.bg_color_btn.setStyleSheet(
            f"background: {bg.name()}; border: 2px solid #555; border-radius: 4px;"
        )

    def _add_recent_color(self, color: QColor) -> None:
        """添加到最近颜色列表（去重，最近使用移到最前，超出数量截断）。"""
        btns = self._recent_color_btns
        if not btns:
            return
        color_str = color.name()
        # 取当前颜色序列（含未使用的空位 None）
        current = [btn.property("color") for btn in btns]
        # 去重：移除已存在的相同颜色，其余颜色顺延
        if color_str in current:
            current.remove(color_str)
        current.insert(0, color_str)
        current = current[: len(btns)]
        empty_style = "background: #333; border: 1px solid #555; border-radius: 2px;"
        for btn, c in zip(btns, current):
            btn.setProperty("color", c)
            btn.setStyleSheet(
                f"background: {c}; border: 1px solid #555; border-radius: 2px;"
            )
        # 超出数量的按钮恢复为空态
        for btn in btns[len(current):]:
            btn.setProperty("color", None)
            btn.setStyleSheet(empty_style)

    def _adjust_brush(self, delta: int) -> None:
        new_size = max(1, min(20, self.canvas.get_brush_size() + delta))
        self.canvas.set_brush_size(new_size)
        self.brush_spin.setValue(new_size)

    def _on_pixel_edited(self) -> None:
        """画布编辑时更新颜色按钮。"""
        self._update_color_buttons()

    def _push_undo(self) -> None:
        """压入撤销栈（记录操作完成后的状态；与上一状态相同则跳过）。"""
        img = self.canvas.get_image_rgba()
        if img is not None:
            if self._undo_stack and np.array_equal(self._undo_stack[-1], img):
                return  # 空笔画/无变化：不产生重复状态
            self._undo_stack.append(img.copy())
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            self._redo_stack.clear()

    def _undo(self) -> None:
        if len(self._undo_stack) <= 1:
            return
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        prev = self._undo_stack[-1]
        self.canvas.set_image(prev[:, :, :3])
        self.status.showMessage("撤销")

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        img = self._redo_stack.pop()
        self._undo_stack.append(img)
        self.canvas.set_image(img[:, :, :3])
        self.status.showMessage("重做")

    def _on_save(self) -> None:
        """保存回资产库。"""
        img = self.canvas.get_image()
        if img is None:
            return
        self.asset_manager.update_asset(self.asset_id, img)
        self.status.showMessage("已保存到资产库")
        self.saved.emit()
