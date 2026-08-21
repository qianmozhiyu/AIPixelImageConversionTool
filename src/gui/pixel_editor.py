"""像素图编辑器。

类 Aseprite 的简易像素图编辑器，支持画笔(B)、橡皮(E)、吸管(I)、油漆桶(G)、
直线(L)、矩形(U)等工具，快捷键与 PS/Aseprite 一致。
"""

from __future__ import annotations

import numpy as np
from collections import deque

from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QRect, QRectF
from PySide6.QtGui import (
    QImage, QPainter, QPixmap, QColor, QPen, QBrush, QMouseEvent, QKeyEvent,
    QAction, QKeySequence, QIcon,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QSlider, QScrollArea, QFrame, QToolBar, QStatusBar,
    QToolButton, QButtonGroup, QCheckBox, QComboBox,
)

from ..core.asset_manager import AssetManager
from ..core.io import save_image
from .color_wheel import ColorWheel


class PixelCanvas(QWidget):
    """像素画布组件，支持放大显示和像素级编辑。

    Signals:
        pixel_edited(): 像素被编辑（用于刷新颜色按钮等）
        stroke_started(): 一次绘制笔画开始（用于压入撤销栈）
        cursor_moved(): 光标在画布上移动/离开（用于状态栏更新）
    """

    pixel_edited = Signal()
    stroke_started = Signal()
    cursor_moved = Signal()

    # 工具枚举
    TOOL_BRUSH = "brush"
    TOOL_ERASER = "eraser"
    TOOL_EYEDROPPER = "eyedropper"
    TOOL_FILL = "fill"
    TOOL_LINE = "line"
    TOOL_RECT = "rect"
    TOOL_CIRCLE = "circle"
    TOOL_SELECT = "select"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None  # (H,W,4) uint8 RGBA
        self._scale = 16  # 像素放大倍数
        self._offset = QPoint(0, 0)
        self._tool = self.TOOL_BRUSH
        self._fg_color = QColor(0, 0, 0, 255)
        self._bg_color = QColor(255, 255, 255, 255)
        self._brush_size = 1
        self._brush_shape = "circle"  # 笔刷形状：circle=圆形（默认）/ square=方形
        self._grid_visible = True     # 是否显示像素网格
        self._symmetry = "none"       # 对称模式：none/horizontal/vertical/both
        self._drawing = False
        self._alt_picking = False  # 按住 Alt 临时吸管取色中
        self._shift_line = False   # 按住 Shift 临时直线工具中
        self._drag_start = QPoint()
        self._panning = False
        self._pan_start = QPoint()
        self._offset_start = QPoint()
        self._space_pressed = False
        self._right_picking = False  # 按住右键连续取背景色中
        self._temp_preview: np.ndarray | None = None  # 直线/矩形/圆形预览
        self._hover_pixel: QPoint | None = None  # 当前鼠标所在像素（取整，供状态栏）
        self._cursor_pos: QPointF | None = None  # 光标像素坐标（浮点，完全跟随鼠标，不取整）
        # 框选工具状态：选区（像素矩形）+ 拖动中标记 + 内部像素剪贴板
        self._selection: QRect | None = None  # 当前选区（像素坐标）；None=无选区
        self._selecting = False              # 正在拖拽框选选区
        self._selection_start = QPoint()     # 框选起始像素
        self._clipboard: np.ndarray | None = None  # 复制的选区内容 (h,w,4) RGBA
        # 选区拖动（携内容移动）：快照法保留撤回到原始
        self._select_move = False            # 正在拖动选区内容
        self._move_start = QPoint()          # 拖动起点像素
        self._move_anchor = QPoint()         # 拖动前选区左上角
        self._move_snap: np.ndarray | None = None  # 拖动前图像快照
        self._move_src: QRect | None = None        # 拖动前选区（内容来源）
        # 性能：QImage 缓存避免每帧重建；棋盘格背景用于标识透明区域
        self._qimage: QImage | None = None
        self._checker: QPixmap = self._make_checkerboard()
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
        self._qimage = None  # 图像内容变化，缓存失效
        self.update()

    def _invalidate_qimage(self) -> None:
        """标记 QImage 缓存失效（图像内容变化时调用）。"""
        self._qimage = None

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

    def set_brush_shape(self, shape: str) -> None:
        """设置笔刷形状："square"（方形）或 "circle"（圆形）。"""
        if shape in ("square", "circle"):
            self._brush_shape = shape
            self.update()

    def get_brush_shape(self) -> str:
        return self._brush_shape

    def set_grid_visible(self, visible: bool) -> None:
        """开关像素网格显示。"""
        self._grid_visible = bool(visible)
        self.update()

    def get_grid_visible(self) -> bool:
        return self._grid_visible

    @staticmethod
    def _circle_offsets(size: int) -> list:
        """返回以 (0,0) 为中心、直径 size 的圆盘内全部 (dx,dy) 偏移列表。

        覆盖面半径取 half = size//2（与 spec 一致）；size=1 时退化为单像素 (0,0)。
        """
        if size <= 1:
            return [(0, 0)]
        half = size // 2
        r2 = half * half
        return [
            (dx, dy)
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
            if dx * dx + dy * dy <= r2 + 0.5
        ]

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
    def _make_checkerboard() -> QPixmap:
        """生成 16x16 棋盘格 QPixmap（浅灰白格，透明区域用经典棋盘底色）。"""
        pm = QPixmap(16, 16)
        pm.fill(QColor("#d0d0d0"))
        p = QPainter(pm)
        p.fillRect(QRect(0, 0, 8, 8), QColor("#ffffff"))
        p.fillRect(QRect(8, 8, 8, 8), QColor("#ffffff"))
        p.end()
        return pm

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
        px = (pos.x() - self._offset.x()) // self._cell_size()
        py = (pos.y() - self._offset.y()) // self._cell_size()
        h, w = self._image.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            return QPoint(px, py)
        return None

    def _mirror_point(self, x: int, y: int, w: int, h: int) -> list:
        """返回 (x,y) 经对称轴镜像产生的额外坐标列表（不含原坐标）。

        水平 horizontal = 关于垂直中线左右镜像（x→w-1-x，Aseprite 语义），
        垂直 vertical = 关于水平中线上下镜像，both = 四种对称。
        """
        pts = []
        if self._symmetry == "horizontal":
            pts.append((w - 1 - x, y))
        elif self._symmetry == "vertical":
            pts.append((x, h - 1 - y))
        elif self._symmetry == "both":
            pts.append((w - 1 - x, y))
            pts.append((x, h - 1 - y))
            pts.append((w - 1 - x, h - 1 - y))
        return pts

    def _draw_pixel(self, x: int, y: int, color: QColor, img: np.ndarray | None = None) -> None:
        """在指定位置绘制像素（考虑笔刷大小与形状、对称）。"""
        target = img if img is not None else self._image
        if target is None:
            return
        h, w = target.shape[:2]
        size = self._brush_size
        half = size // 2

        # 根据笔刷形状收集覆盖偏移：方形用双循环矩形，圆形用圆盘偏移集
        if self._brush_shape == "circle":
            offsets = self._circle_offsets(size)
        else:
            offsets = [
                (dx, dy)
                for dy in range(-half, size - half)
                for dx in range(-half, size - half)
            ]

        # 对称：汇总原落笔点与所有镜像点作为覆盖中心
        centers = [(x, y)]
        if self._symmetry != "none":
            centers.extend(self._mirror_point(x, y, w, h))

        rgba = [color.red(), color.green(), color.blue(), color.alpha()]
        for (cx, cy) in centers:
            for (dx, dy) in offsets:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    target[ny, nx] = rgba
        if target is self._image:
            self._qimage = None  # 图像内容变化，缓存失效

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

    def _pick_bg_color(self, pos: QPoint) -> None:
        """右键取背景：将像素坐标处的颜色设为背景色。"""
        if self._image is None or pos is None:
            return
        h, w = self._image.shape[:2]
        if not (0 <= pos.x() < w and 0 <= pos.y() < h):
            return
        self._bg_color = QColor(*self._image[pos.y(), pos.x()])
        self.pixel_edited.emit()

    def swap_colors(self) -> None:
        """交换前景色与背景色（X 快捷键）。"""
        self._fg_color, self._bg_color = self._bg_color, self._fg_color
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

    def _draw_circle_midpoint(self, x0: int, y0: int, x1: int, y1: int, color: QColor, img: np.ndarray) -> None:
        """以 (x0,y0)-(x1,y1) 外接矩形中心为圆心绘制 1px 圆环。

        圆心 = 外接矩形中心，半径 = max(|dx|,|dy|)/2。
        """
        import math
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        r = max(abs(x1 - x0), abs(y1 - y0)) / 2.0
        if r < 1:
            self._draw_pixel(int(round(cx)), int(round(cy)), color, img)
            return
        r2 = r * r
        for py in range(int(cy) - int(r) - 1, int(cy) + int(r) + 2):
            for px in range(int(cx) - int(r) - 1, int(cx) + int(r) + 2):
                dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                if abs(dist - r) <= 0.5:
                    self._draw_pixel(px, py, color, img)

    def set_symmetry(self, mode: str) -> None:
        """设置对称模式："none"/"horizontal"/"vertical"/"both"。"""
        if mode in ("none", "horizontal", "vertical", "both"):
            self._symmetry = mode
            self.update()

    def get_symmetry(self) -> str:
        return self._symmetry

    def cycle_symmetry(self) -> None:
        """循环切换对称模式（Shift+S）。"""
        order = ("none", "horizontal", "vertical", "both")
        self._symmetry = order[(order.index(self._symmetry) + 1) % len(order)]
        self.update()

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
        self._qimage = None  # 图像内容变化，缓存失效

    # ------------------------------------------------------------------
    # 框选工具：选区访问 + 复制/粘贴/翻转（水平/垂直镜像）
    # ------------------------------------------------------------------
    def get_selection(self) -> QRect | None:
        """返回当前选区（像素坐标），无选区返回 None。"""
        return QRect(self._selection) if self._selection else None

    def _cell_size(self) -> int:
        """返回整数像素格尺寸（所有像素↔控件换算统一用它，避免 float scale 歧义）。"""
        return max(1, int(round(self._scale)))

    def clear_selection(self) -> None:
        """清除选区。"""
        self._selection = None
        self.update()

    def _clamped_selection(self) -> tuple[int, int, int, int] | None:
        """返回裁剪到画布边界的选区 (y0, x0, y1, x1)；无效返回 None。"""
        if self._image is None or self._selection is None:
            return None
        h, w = self._image.shape[:2]
        y0 = max(0, self._selection.y())
        x0 = max(0, self._selection.x())
        y1 = min(h, self._selection.y() + self._selection.height())
        x1 = min(w, self._selection.x() + self._selection.width())
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, x0, y1, x1)

    def copy_selection(self) -> None:
        """把选区内容复制到内部像素剪贴板（RGBA）。无选区则清除剪贴板。"""
        b = self._clamped_selection()
        if b is None:
            self._clipboard = None
            return
        y0, x0, y1, x1 = b
        self._clipboard = self._image[y0:y1, x0:x1].copy()

    def clipboard_has(self) -> bool:
        return self._clipboard is not None

    def paste_clipboard(self) -> None:
        """把剪贴板内容粘贴到当前选区左上角（无选区则贴到 (0,0)），边界裁剪。"""
        if self._image is None or self._clipboard is None:
            return
        h, w = self._image.shape[:2]
        ch, cw = self._clipboard.shape[:2]
        if self._selection is not None:
            ax, ay = self._selection.x(), self._selection.y()
        else:
            ax, ay = 0, 0
        py0 = max(0, ay)
        px0 = max(0, ax)
        py1 = min(h, ay + ch)
        px1 = min(w, ax + cw)
        if py1 <= py0 or px1 <= px0:
            return
        self._image[py0:py1, px0:px1] = self._clipboard[: py1 - py0, : px1 - px0]
        self._qimage = None
        self.pixel_edited.emit()
        self.stroke_started.emit()  # 粘贴视为一笔，可撤销
        self.update()

    def flip_selection_horizontal(self) -> None:
        """水平翻转（左右镜像）选区内容并写回。"""
        b = self._clamped_selection()
        if b is None:
            return
        y0, x0, y1, x1 = b
        self._image[y0:y1, x0:x1] = self._image[y0:y1, x0:x1][:, ::-1].copy()
        self._qimage = None
        self.pixel_edited.emit()
        self.stroke_started.emit()
        self.update()

    def flip_selection_vertical(self) -> None:
        """垂直翻转（上下镜像）选区内容并写回。"""
        b = self._clamped_selection()
        if b is None:
            return
        y0, x0, y1, x1 = b
        self._image[y0:y1, x0:x1] = self._image[y0:y1, x0:x1][::-1, :].copy()
        self._qimage = None
        self.pixel_edited.emit()
        self.stroke_started.emit()
        self.update()

    def cut_selection(self) -> None:
        """剪切：把选区内容复制到剪贴板，并将选区像素清为透明。"""
        b = self._clamped_selection()
        if b is None:
            self._clipboard = None
            return
        y0, x0, y1, x1 = b
        self._clipboard = self._image[y0:y1, x0:x1].copy()
        self._image[y0:y1, x0:x1] = [0, 0, 0, 0]
        self._qimage = None
        self.pixel_edited.emit()
        self.stroke_started.emit()  # 剪切改写图像，可撤销
        self.update()

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
        # 按下瞬间同步光标与落子格，保证笔刷预览 == 本次落子的格子（无偏移）
        self._hover_pixel = QPoint(pos)
        e = event.position()
        self._cursor_pos = QPointF(
            (e.x() - self._offset.x()) / self._cell_size(),
            (e.y() - self._offset.y()) / self._cell_size(),
        )

        if event.button() == Qt.MouseButton.RightButton:
            # 右键取背景色：按住右键拖动可连续取背景
            self._drawing = True
            self._right_picking = True
            self._pick_bg_color(pos)
            self.update()
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

            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and self._tool != self.TOOL_CIRCLE:
                # 按住 Shift = 临时直线工具（与 Alt 吸管同理，不修改 self._tool；
                # 按下左键开始预览，松开左键提交）；圆形工具下 Shift 用于锁定正圆
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
            elif self._tool in (self.TOOL_LINE, self.TOOL_RECT, self.TOOL_CIRCLE):
                # 开始预览
                self._temp_preview = self._image.copy()
            elif self._tool == self.TOOL_SELECT:
                if self._selection is not None and self._selection.contains(pos):
                    # 点击在现有选区内：开始拖动（携内容移动）
                    self._select_move = True
                    self._drawing = True
                    self._move_snap = self._image.copy()
                    self._move_src = QRect(self._selection)
                    self._move_start = QPoint(pos)
                    self._move_anchor = QPoint(self._selection.topLeft())
                    self.update()
                else:
                    # 点击在选区外/空白：开始拖拽新建选区
                    self._selecting = True
                    self._selection_start = QPoint(pos)
                    self._selection = QRect(pos.x(), pos.y(), 1, 1)
                    self.update()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # 更新笔刷范围框位置（无论是否在绘制/平移都跟随光标）
        # 记录光标像素坐标（浮点，完全跟随鼠标不取整）与取整像素（供状态栏）
        pos_px = event.position()
        self._cursor_pos = QPointF(
            (pos_px.x() - self._offset.x()) / self._cell_size(),
            (pos_px.y() - self._offset.y()) / self._cell_size(),
        )
        self._hover_pixel = self._widget_to_pixel(event.position().toPoint())
        self.cursor_moved.emit()  # 状态栏刷新（无图像时同样发一次以复位"就绪"）
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

        if self._right_picking:
            # 按住右键拖动：连续取背景色
            pos = self._widget_to_pixel(event.position().toPoint())
            if pos is not None:
                self._pick_bg_color(pos)
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

        if self._select_move:
            # 拖动选区内容：快照恢复原图后，把内容（含清除源区）画到新选区位置
            d = QPoint(pos.x() - self._move_start.x(), pos.y() - self._move_start.y())
            new_tl = QPoint(self._move_anchor.x() + d.x(), self._move_anchor.y() + d.y())
            self._selection = QRect(
                new_tl.x(), new_tl.y(),
                self._move_src.width(), self._move_src.height(),
            )
            img = self._move_snap.copy()
            h, w = img.shape[:2]
            s_ = self._move_src
            sy0 = max(0, s_.y()); sx0 = max(0, s_.x())
            sy1 = min(h, s_.y() + s_.height()); sx1 = min(w, s_.x() + s_.width())
            if sy1 > sy0 and sx1 > sx0:
                data = img[sy0:sy1, sx0:sx1].copy()
                img[sy0:sy1, sx0:sx1] = [0, 0, 0, 0]  # 清除源位置
                d_ = self._selection
                dy0 = max(0, d_.y()); dx0 = max(0, d_.x())
                dy1 = min(h, d_.y() + d_.height()); dx1 = min(w, d_.x() + d_.width())
                if dy1 > dy0 and dx1 > dx0:
                    img[dy0:dy1, dx0:dx1] = data[:dy1 - dy0, :dx1 - dx0]
            self._image = img
            self._qimage = None
            self.update()
            return

        if self._selecting:
            # 框选拖动：实时更新选区矩形（整包含起点与当前点）
            sp = self._selection_start
            x0, x1 = min(sp.x(), pos.x()), max(sp.x(), pos.x())
            y0, y1 = min(sp.y(), pos.y()), max(sp.y(), pos.y())
            self._selection = QRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
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
        elif self._tool in (self.TOOL_LINE, self.TOOL_RECT, self.TOOL_CIRCLE) and self._temp_preview is not None:
            # 恢复原图，绘制预览
            self._image = self._temp_preview.copy()
            start = self._drag_start
            if self._tool == self.TOOL_LINE:
                self._draw_line_bresenham(start.x(), start.y(), pos.x(), pos.y(), self._fg_color, self._image)
            elif self._tool == self.TOOL_RECT:
                self._draw_rect(start.x(), start.y(), pos.x(), pos.y(), self._fg_color, self._image)
            else:
                # 圆形：按住 Shift 时强制宽高相等（正圆）
                p2 = pos
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    m = max(abs(pos.x() - start.x()), abs(pos.y() - start.y()))
                    p2 = QPoint(
                        start.x() + (1 if pos.x() >= start.x() else -1) * m,
                        start.y() + (1 if pos.y() >= start.y() else -1) * m,
                    )
                self._draw_circle_midpoint(start.x(), start.y(), p2.x(), p2.y(), self._fg_color, self._image)
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
        if self._right_picking:
            # 右键取背景结束：复位标志
            self._right_picking = False
            self._drawing = False
            return
        if self._select_move:
            # 选区拖动结束：提交最终图像与选区位置（作为一笔可撤销）
            self._select_move = False
            self._drawing = False
            self._move_snap = None
            self._move_src = None
            self.pixel_edited.emit()
            self.stroke_started.emit()
            self.update()
            return
        if self._shift_line:
            # Shift 临时直线结束：提交当前预览（松开左键提交）
            self._shift_line = False
            self._drawing = False
            self._temp_preview = None
            self.pixel_edited.emit()
            self.stroke_started.emit()  # 操作完成后再压栈（记录操作后状态）
            return
        if self._selecting:
            # 框选结束：提交选区（不改图，不入撤销栈）
            self._selecting = False
            self._drawing = False
            # 单击（1×1）视为清除/回选
            if self._selection is not None and self._selection.width() <= 1 and self._selection.height() <= 1:
                self._selection = None
            self.update()
            return
        if self._drawing:
            self._drawing = False
            if self._tool in (self.TOOL_LINE, self.TOOL_RECT, self.TOOL_CIRCLE):
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
        s = int(round(self._scale))

        # 棋盘格背景铺满画布（透明区域经 drawImage 露底）
        p.drawTiledPixmap(self.rect(), self._checker)

        # 绘制像素（QImage 缓存：仅在内容变化时重建）
        if (
            self._qimage is None
            or self._qimage.width() != w
            or self._qimage.height() != h
        ):
            self._qimage = QImage(
                self._image.tobytes(), w, h, w * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.drawImage(QRect(self._offset.x(), self._offset.y(), w * s, h * s), self._qimage)

        # 像素网格
        if self._grid_visible and s >= 8:
            p.setPen(QPen(QColor(128, 128, 128, 60), 1))
            for x in range(w + 1):
                p.drawLine(self._offset.x() + x * s, self._offset.y(),
                          self._offset.x() + x * s, self._offset.y() + h * s)
            for y in range(h + 1):
                p.drawLine(self._offset.x(), self._offset.y() + y * s,
                          self._offset.x() + w * s, self._offset.y() + y * s)

        # 对称轴参考线（绘制在网格之后、光标之前）
        if self._symmetry != "none":
            if self._symmetry in ("horizontal", "both"):
                # 左右镜像（关于垂直中线）：画竖直虚线（蓝）
                xp = self._offset.x() + (w * s) // 2
                p.setPen(QPen(QColor(80, 160, 255, 210), 1, Qt.PenStyle.DashLine))
                p.drawLine(xp, self._offset.y(), xp, self._offset.y() + h * s)
            if self._symmetry in ("vertical", "both"):
                # 上下镜像（关于水平中线）：画水平虚线（红）
                yp = self._offset.y() + (h * s) // 2
                p.setPen(QPen(QColor(255, 120, 120, 210), 1, Qt.PenStyle.DashLine))
                p.drawLine(self._offset.x(), yp, self._offset.x() + w * s, yp)

        # 框选选区：双色虚线矩形（蚂蚁线，白+黑叠画保证任何底色可见）
        if self._selection is not None:
            r = self._selection
            rr = QRect(
                self._offset.x() + r.x() * s,
                self._offset.y() + r.y() * s,
                r.width() * s,
                r.height() * s,
            )
            p.setPen(QPen(QColor(255, 255, 255, 210), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rr)
            p.setPen(QPen(QColor(0, 0, 0, 170), 1, Qt.PenStyle.DashLine))
            p.drawRect(rr)

        # 光标指示：隐藏系统光标。笔刷像素块对齐像素格子（白边标示落格范围），
        # 十字准星完全跟随鼠标浮点位置（不对齐格子）
        if self._cursor_pos is not None and not self._panning:
            cxp = self._cursor_pos.x()
            cyp = self._cursor_pos.y()
            # 笔刷像素块：用与 mousePress 落子完全一致的像素映射（_hover_pixel，
            # 整除向下取整），保证预览格 == 实际落子格，绝不偏移
            if self._hover_pixel is not None:
                gx = self._hover_pixel.x()
                gy = self._hover_pixel.y()
            else:
                gx = int(cxp)
                gy = int(cyp)
            half = self._brush_size // 2
            bx = self._offset.x() + (gx - half) * s
            by = self._offset.y() + (gy - half) * s
            bw = self._brush_size * s
            bh = self._brush_size * s
            if self._tool == self.TOOL_BRUSH:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(self._fg_color))
                p.drawRect(bx, by, bw, bh)
                # 白色细边框圈出笔刷影响范围（对齐格线，帮助看清落格）
                p.setPen(QPen(QColor(255, 255, 255, 235), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(bx, by, bw, bh)

            # 纯十字准星：完全跟随鼠标（像素中心 = 鼠标像素位置，不额外偏移）
            cx = self._offset.x() + cxp * s
            cy = self._offset.y() + cyp * s
            arm = 14
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
            p.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))
            p.setPen(QPen(QColor(255, 255, 255, 235), 1))
            p.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
            p.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

    def wheelEvent(self, event) -> None:
        """滚轮以鼠标位置为锚点居中缩放（系数 1.15，范围 1-64，内部 float）。"""
        if self._image is None:
            return
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = max(1.0, min(64.0, self._scale * factor))
        mouse_pos = event.position().toPoint()
        # 保持鼠标位置下的图像像素在原处：缩放前记录图像坐标，缩放后反算 offset
        prev_scale = self._scale
        img_x = (mouse_pos.x() - self._offset.x()) / prev_scale
        img_y = (mouse_pos.y() - self._offset.y()) / prev_scale
        self._scale = new_scale
        self._offset = QPoint(
            int(mouse_pos.x() - img_x * new_scale),
            int(mouse_pos.y() - img_y * new_scale),
        )
        self.update()

    def fit_to_view(self) -> None:
        """等比缩放图像以完整显示在画布内并居中。"""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        cw, ch = self.width(), self.height()
        if w <= 0 or h <= 0 or cw <= 0 or ch <= 0:
            return
        scale = min(cw / w, ch / h)
        scale = max(1.0, min(64.0, scale))
        self._scale = scale
        self._offset = QPoint(int((cw - w * scale) / 2), int((ch - h * scale) / 2))
        self.update()

    def reset_view(self) -> None:
        """重置视图：适中缩放（视口/图像向下取整并钳制）并居中。"""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        cw, ch = self.width(), self.height()
        if w <= 0 or h <= 0 or cw <= 0 or ch <= 0:
            return
        scale = max(1, min(64, int(cw / w), int(ch / h)))
        self._scale = float(scale)
        self._offset = QPoint(int((cw - w * scale) / 2), int((ch - h * scale) / 2))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        """双击适配视图。"""
        self.fit_to_view()

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
        # 鼠标离开画布：清除光标与笔刷范围框并复位状态栏
        self._hover_pixel = None
        self._cursor_pos = None
        self.cursor_moved.emit()
        self.update()
        super().leaveEvent(event)

    def get_hover_info(self) -> str:
        """返回状态栏悬停信息字符串；无图像/未悬停时返回"就绪"。"""
        if self._image is None or self._hover_pixel is None:
            return "就绪"
        x, y = self._hover_pixel.x(), self._hover_pixel.y()
        return f"{x}, {y} | {self._fg_color.name()} | {self._cell_size()}%"


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
        self.canvas.cursor_moved.connect(self._on_cursor_moved)
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
            (PixelCanvas.TOOL_CIRCLE, "圆形 (C)", "圆形"),
            (PixelCanvas.TOOL_SELECT, "框选 (M)", "框选"),
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

        # 色轮（HSV）：位于前景色上方，直接选色替代弹窗；配合亮度滑杆调节 V
        self._active_slot = "fg"  # 色轮当前编辑的色槽："fg" 前景 / "bg" 背景
        self._wheel = ColorWheel(160)
        layout.addWidget(self._wheel)

        layout.addWidget(QLabel("亮度:"))
        self._brightness = QSlider(Qt.Orientation.Horizontal)
        self._brightness.setRange(0, 255)
        layout.addWidget(self._brightness)
        self._brightness.valueChanged.connect(self._on_brightness_changed)

        # 前景色/背景色：点击切换色轮编辑目标（不再弹窗选色）
        layout.addWidget(QLabel("前景色:"))
        self.fg_color_btn = QPushButton()
        self.fg_color_btn.setFixedSize(180, 40)
        self.fg_color_btn.clicked.connect(lambda: self._activate_color_slot("fg"))
        layout.addWidget(self.fg_color_btn)

        layout.addWidget(QLabel("背景色:"))
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedSize(180, 40)
        self.bg_color_btn.clicked.connect(lambda: self._activate_color_slot("bg"))
        layout.addWidget(self.bg_color_btn)

        # 色轮取色 → 写入当前色槽；取色结束（松开）→ 记录最近颜色
        self._wheel.colorChanged.connect(self._on_wheel_changed)
        self._wheel.picked.connect(self._on_wheel_picked)
        self._sync_wheel_from_active()

        # 交换前景/背景色（X）
        self.swap_btn = QPushButton("交换前/背景 (X)")
        self.swap_btn.clicked.connect(lambda: self.canvas.swap_colors())
        layout.addWidget(self.swap_btn)

        # 笔刷大小
        layout.addWidget(QLabel("笔刷大小:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 20)
        self.brush_spin.setValue(1)
        self.brush_spin.valueChanged.connect(lambda v: self.canvas.set_brush_size(v))
        layout.addWidget(self.brush_spin)

        # 笔刷形状（圆形/方形）
        layout.addWidget(QLabel("笔刷形状:"))
        self._shape_btns = QButtonGroup(self)
        self._shape_btns.setExclusive(True)
        shape_row = QHBoxLayout()
        for shape, label in (("circle", "圆形"), ("square", "方形")):
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setFixedSize(64, 28)
            btn.setStyleSheet(
                "QToolButton { background: #333; color: #e0e0e0; border: none; "
                "border-radius: 4px; font-size: 12px; }"
                "QToolButton:hover { background: #444; }"
                "QToolButton:checked { background: #3d5a80; }"
            )
            btn.clicked.connect(lambda checked, sh=shape: self.canvas.set_brush_shape(sh))
            self._shape_btns.addButton(btn)
            if shape == self.canvas.get_brush_shape():
                btn.setChecked(True)
            shape_row.addWidget(btn)
        layout.addLayout(shape_row)

        # 像素网格显示开关
        self.grid_visible_chk = QCheckBox("显示网格")
        self.grid_visible_chk.setChecked(True)
        self.grid_visible_chk.toggled.connect(self.canvas.set_grid_visible)
        layout.addWidget(self.grid_visible_chk)

        # 对称模式（Shift+S 循环切换）
        layout.addWidget(QLabel("对称 (Shift+S):"))
        self.symmetry_combo = QComboBox()
        self.symmetry_combo.addItems(["无", "左右镜像", "上下镜像", "四象限"])
        self.symmetry_combo.currentIndexChanged.connect(
            lambda i: self.canvas.set_symmetry(
                ["none", "horizontal", "vertical", "both"][i]
            )
        )
        layout.addWidget(self.symmetry_combo)

        # 选区操作：剪切/复制/粘贴 + 水平/垂直翻转（M 框选后用）
        layout.addWidget(QLabel("选区操作:"))
        _sel_btns = (
            ("剪切", "Ctrl+X", lambda: self.canvas.cut_selection()),
            ("复制", "Ctrl+C", lambda: self.canvas.copy_selection()),
            ("粘贴", "Ctrl+V", lambda: self.canvas.paste_clipboard()),
        )
        _sel_row1 = QHBoxLayout()
        _sel_row1.setSpacing(4)
        for text, shortcut, handler in _sel_btns:
            btn = QToolButton()
            btn.setText(f"{text}\n({shortcut})")
            btn.setToolTip(f"{text}（快捷键 {shortcut}）")
            btn.setFixedSize(56, 44)
            btn.setStyleSheet(
                "QToolButton { background: #333; color: #e0e0e0; border: none; "
                "border-radius: 4px; font-size: 11px; }"
                "QToolButton:hover { background: #444; }"
            )
            btn.clicked.connect(lambda checked, h=handler: h())
            _sel_row1.addWidget(btn)
        layout.addLayout(_sel_row1)
        _sel_row2 = QHBoxLayout()
        _sel_row2.setSpacing(4)
        for text, shortcut, handler in (
            ("水平翻转", "H", lambda: self.canvas.flip_selection_horizontal()),
            ("垂直翻转", "V", lambda: self.canvas.flip_selection_vertical()),
        ):
            btn = QToolButton()
            btn.setText(f"{text}\n({shortcut})")
            btn.setToolTip(f"{text}（快捷键 {shortcut}）")
            btn.setFixedSize(86, 44)
            btn.setStyleSheet(
                "QToolButton { background: #333; color: #e0e0e0; border: none; "
                "border-radius: 4px; font-size: 11px; }"
                "QToolButton:hover { background: #444; }"
            )
            btn.clicked.connect(lambda checked, h=handler: h())
            _sel_row2.addWidget(btn)
        layout.addLayout(_sel_row2)

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
            "C": lambda: self._select_tool(PixelCanvas.TOOL_CIRCLE),
            "M": lambda: self._select_tool(PixelCanvas.TOOL_SELECT),
            "X": lambda: self.canvas.swap_colors(),  # 交换前/背景色
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

        # Ctrl+0 重置视图（居中适中缩放）
        reset_action = QAction(self)
        reset_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_action.triggered.connect(self.canvas.reset_view)
        self.addAction(reset_action)

        # 选区操作：Ctrl+X 剪切、Ctrl+C 复制、Ctrl+V 粘贴、H 水平翻转、V 垂直翻转
        cut_action = QAction(self)
        cut_action.setShortcut(QKeySequence("Ctrl+X"))
        cut_action.triggered.connect(self.canvas.cut_selection)
        self.addAction(cut_action)

        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_action.triggered.connect(self.canvas.copy_selection)
        self.addAction(copy_action)

        paste_action = QAction(self)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_action.triggered.connect(self.canvas.paste_clipboard)
        self.addAction(paste_action)

        flip_h = QAction(self)
        flip_h.setShortcut(QKeySequence("H"))
        flip_h.triggered.connect(self.canvas.flip_selection_horizontal)
        self.addAction(flip_h)

        flip_v = QAction(self)
        flip_v.setShortcut(QKeySequence("V"))
        flip_v.triggered.connect(self.canvas.flip_selection_vertical)
        self.addAction(flip_v)

        # Shift+S 循环切换对称模式
        sym_action = QAction(self)
        sym_action.setShortcut(QKeySequence("Shift+S"))
        sym_action.triggered.connect(self.canvas.cycle_symmetry)
        self.addAction(sym_action)

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
            PixelCanvas.TOOL_CIRCLE: 6,
            PixelCanvas.TOOL_SELECT: 7,
        }.get(tool, 0)
        btns = self._tool_buttons.buttons()
        if 0 <= idx < len(btns):
            btns[idx].setChecked(True)
        self.status.showMessage(f"工具: {tool}")

    def _on_wheel_changed(self, color: QColor) -> None:
        """色轮取色：写入当前编辑的色槽（前景/背景）。"""
        if self._active_slot == "fg":
            self.canvas.set_fg_color(color)
        else:
            self.canvas.set_bg_color(color)
        self._update_color_buttons()

    def _on_wheel_picked(self, color: QColor) -> None:
        """色轮取色完成（松开鼠标）：前景色记录到最近颜色。"""
        if self._active_slot == "fg":
            self._add_recent_color(color)

    def _on_brightness_changed(self, value: int) -> None:
        """亮度滑杆：调节色轮当前色槽的 V 分量（set_brightness 会回发 colorChanged）。"""
        self._wheel.set_brightness(value)

    def _activate_color_slot(self, slot: str) -> None:
        """切换色轮编辑目标（前景/背景），色轮同步显示该槽颜色。"""
        self._active_slot = slot
        self._sync_wheel_from_active()

    def _sync_wheel_from_active(self) -> None:
        """把色轮与亮度滑杆同步到当前色槽颜色（不触发 colorChanged 回写）。"""
        c = (
            self.canvas.get_fg_color()
            if self._active_slot == "fg"
            else self.canvas.get_bg_color()
        )
        self._wheel.set_color(c)
        self._brightness.blockSignals(True)
        self._brightness.setValue(c.value())
        self._brightness.blockSignals(False)
        self._update_color_buttons()

    def _update_color_buttons(self) -> None:
        fg = self.canvas.get_fg_color()
        bg = self.canvas.get_bg_color()
        fg_border = "#3d5a80" if self._active_slot == "fg" else "#555"
        bg_border = "#3d5a80" if self._active_slot == "bg" else "#555"
        self.fg_color_btn.setStyleSheet(
            f"background: {fg.name()}; border: 2px solid {fg_border}; border-radius: 4px;"
        )
        self.bg_color_btn.setStyleSheet(
            f"background: {bg.name()}; border: 2px solid {bg_border}; border-radius: 4px;"
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
        """画布编辑时更新颜色按钮，并同步色轮到当前色槽（吸管/右键取色/交换）。"""
        self._sync_wheel_from_active()

    def _on_cursor_moved(self) -> None:
        """光标在画布移动：刷新状态栏（含工具与像素信息）。"""
        info = self.canvas.get_hover_info()
        if info == "就绪":
            self.status.showMessage("就绪")
            return
        self.status.showMessage(f"工具: {self.canvas.get_tool()} | {info}")

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
        # set_image 对 4 通道输入直接保留（含透明 alpha），透明像素可正确撤销
        self.canvas.set_image(prev)
        self.status.showMessage("撤销")

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        img = self._redo_stack.pop()
        self._undo_stack.append(img)
        self.canvas.set_image(img)
        self.status.showMessage("重做")

    def _on_save(self) -> None:
        """保存回资产库。"""
        img = self.canvas.get_image()
        if img is None:
            return
        self.asset_manager.update_asset(self.asset_id, img)
        self.status.showMessage("已保存到资产库")
        self.saved.emit()
