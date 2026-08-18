"""我的资产页面。

展示已生成的像素图资产，支持下载、删除、快速编辑、批量导出。
此为基础框架版本，Task 7-8 将完善资产网格和操作功能。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QPoint, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea,
    QGridLayout, QFrame, QLineEdit, QRubberBand,
)

import numpy as np

from ..core.asset_manager import AssetManager
from ..core.config import load_preference


class AssetCard(QFrame):
    """单个资产卡片（缩略图 + 文件名 + 像素尺寸）。

    点击选中，双击触发编辑；鼠标移入时在缩略图上显示
    编辑/下载/删除三个快捷按钮，移出隐藏。
    """

    clicked = Signal(str)   # asset_id
    double_clicked = Signal(str)  # asset_id
    edit_requested = Signal(str)  # asset_id（悬停"编辑"按钮）
    download_requested = Signal(str)  # asset_id
    delete_requested = Signal(str)    # asset_id
    # 框选拖拽信号（全局坐标）：从卡片上按下拖拽时，卡片将事件转发给页面
    drag_started = Signal(QPoint)
    drag_moved = Signal(QPoint)
    drag_ended = Signal(QPoint)

    def __init__(self, asset_info, parent=None):
        super().__init__(parent)
        self.asset_info = asset_info
        self._selected = False
        self._press_global = None
        self._dragging = False
        self.setMouseTracking(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 缩略图
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(120, 120)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("background: #1e1e1e; border-radius: 4px;")
        layout.addWidget(self.thumb_label)

        # 文件名
        self.name_label = QLabel(self.asset_info.source_name)
        self.name_label.setStyleSheet("color: #ccc; font-size: 11px;")
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(24)  # 防长文件名换行挤压下方信息
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)

        # 像素尺寸信息
        self.info_label = QLabel(f"像素 {self.asset_info.width}×{self.asset_info.height}")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.info_label)

        # 悬停快捷操作覆盖层（绝对定位在缩略图区域，默认隐藏；按钮纵向排布居中）
        self.overlay = QFrame(self)
        self.overlay.setGeometry(8, 8, 120, 120)
        self.overlay.setStyleSheet(
            "QFrame { background: rgba(20, 20, 26, 0.82); border-radius: 4px; }"
        )
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        overlay_layout.setSpacing(4)
        overlay_layout.addStretch(1)
        btn_style = (
            "QPushButton { background: rgba(45, 45, 52, 0.95); color: #e0e0e0;"
            "  border: 1px solid #4a4a52; border-radius: 3px;"
            "  font-size: 11px; padding: 2px 4px; }"
            "QPushButton:hover { background: #3d5a80; border-color: #4a9eff; }"
        )
        self.btn_edit = QPushButton("编辑", self.overlay)
        self.btn_edit.setFixedSize(44, 22)
        self.btn_edit.setStyleSheet(btn_style)
        overlay_layout.addWidget(self.btn_edit, 0, Qt.AlignmentFlag.AlignHCenter)
        self.btn_download = QPushButton("下载", self.overlay)
        self.btn_download.setFixedSize(44, 22)
        self.btn_download.setStyleSheet(btn_style)
        overlay_layout.addWidget(self.btn_download, 0, Qt.AlignmentFlag.AlignHCenter)
        self.btn_delete = QPushButton("删除", self.overlay)
        self.btn_delete.setFixedSize(44, 22)
        self.btn_delete.setStyleSheet(btn_style)
        overlay_layout.addWidget(self.btn_delete, 0, Qt.AlignmentFlag.AlignHCenter)
        overlay_layout.addStretch(1)
        self.overlay.setVisible(False)

        # 悬停按钮信号：分别发出编辑/下载/删除请求
        self.btn_edit.clicked.connect(lambda: self.edit_requested.emit(self.asset_info.id))
        self.btn_download.clicked.connect(lambda: self.download_requested.emit(self.asset_info.id))
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.asset_info.id))

        # hover 延迟隐藏：鼠标移出卡片 120ms 后再二次校验光标位置决定是否隐藏，
        # 解决"移开未隐藏"（快速移动时 leaveEvent 后光标位置判断不可靠）同时
        # 保证鼠标移到子按钮时不会提前隐藏
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(120)
        self._hover_timer.timeout.connect(self._on_hover_timeout)

        self.setFixedSize(140, 180)
        self._update_style()

    def _on_hover_timeout(self) -> None:
        from PySide6.QtGui import QCursor
        # 二次校验：光标仍在卡片矩形内（含子按钮）则不隐藏
        if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self.overlay.setVisible(False)

    def enterEvent(self, event):
        self._hover_timer.stop()
        self.overlay.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        # 延迟隐藏：给光标移动到子按钮留出时间；真正移开卡片后由
        # _on_hover_timeout 的二次校验决定隐藏
        self._hover_timer.start()
        super().leaveEvent(event)

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "AssetCard { border: 2px solid #4a9eff; border-radius: 8px; background: #252530; }"
                "AssetCard:hover { background: #2a2a35; }"
            )
        else:
            self.setStyleSheet(
                "AssetCard { border: 2px solid transparent; border-radius: 8px; background: #222228; }"
                "AssetCard:hover { background: #2a2a30; border-color: #3a3a40; }"
            )

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def is_selected(self) -> bool:
        return self._selected

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 记录按下位置，拖拽超过阈值时转为框选；未拖拽则 release 时触发选中
            self._press_global = event.globalPosition().toPoint()
            self._dragging = False
            event.accept()

    def mouseMoveEvent(self, event):
        if self._press_global is None:
            return
        if not self._dragging:
            # 拖拽距离阈值：超过 6px 判定为框选开始
            local = self.mapFromGlobal(self._press_global)
            if (event.position().toPoint() - local).manhattanLength() > 6:
                self._dragging = True
                self.drag_started.emit(self._press_global)
        if self._dragging:
            self.drag_moved.emit(event.globalPosition().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                self.drag_ended.emit(event.globalPosition().toPoint())
            else:
                self.clicked.emit(self.asset_info.id)
            self._press_global = None
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.asset_info.id)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background: #333; color: #e0e0e0; border: 1px solid #555; }"
                           "QMenu::item:selected { background: #3d5a80; }")
        download_action = menu.addAction("下载")
        edit_action = menu.addAction("快速编辑")
        delete_action = menu.addAction("删除")
        action = menu.exec(event.globalPos())
        if action == download_action:
            self.download_requested.emit(self.asset_info.id)
        elif action == edit_action:
            self.double_clicked.emit(self.asset_info.id)
        elif action == delete_action:
            self.delete_requested.emit(self.asset_info.id)


class AssetsPage(QWidget):
    """我的资产页面。"""

    edit_requested = Signal(str)  # asset_id，请求打开编辑器

    def __init__(self, asset_manager: AssetManager, parent=None):
        super().__init__(parent)
        self.asset_manager = asset_manager
        self._cards: list[AssetCard] = []
        self._selected_ids: set[str] = set()
        self._build_ui()

    def _build_ui(self) -> None:
        # 页面深色背景（普通 QWidget 默认使用系统浅色调色板，需显式设置）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("AssetsPage { background: #1a1a1a; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        title = QLabel("我的资产")
        title.setStyleSheet("color: #e0e0e0; font-size: 18px; font-weight: bold;")
        toolbar.addWidget(title)
        toolbar.addStretch(1)

        self.btn_export = QPushButton("批量导出")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_batch_export)
        toolbar.addWidget(self.btn_export)

        self.btn_delete = QPushButton("批量删除")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._on_batch_delete)
        toolbar.addWidget(self.btn_delete)

        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(self.btn_refresh)

        # 搜索框（在 btn_refresh 之后）
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索资产名称...")
        self.search_input.setFixedWidth(180)
        self.search_input.textChanged.connect(self._on_search)
        toolbar.addWidget(self.search_input)

        layout.addLayout(toolbar)

        # 滚动区域 + 网格容器（QScrollArea 的 QSS 需覆盖 viewport 层，
        # 否则网格区显示系统默认浅色背景；滚动条一并深色化）
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea { border: none; background: #1e1e1e; }"
            "QScrollArea > QWidget > QWidget { background: #1e1e1e; }"
            "QScrollArea QScrollBar:vertical { background: #1e1e1e; width: 10px; }"
            "QScrollArea QScrollBar::handle:vertical { background: #3a3a3a;"
            "  border-radius: 5px; min-height: 24px; }"
            "QScrollArea QScrollBar::add-line:vertical,"
            "QScrollArea QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollArea QScrollBar::add-page:vertical,"
            "QScrollArea QScrollBar::sub-page:vertical { background: none; }"
        )

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(8)  # 资产卡片间距（适度紧凑）
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll, 1)

        # 框选（rubber band）：空白区拖拽 + 从卡片上拖拽均支持
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.scroll.viewport())
        self._rubber_origin: QPoint | None = None
        self.grid_container.setMouseTracking(True)
        self.scroll.viewport().setMouseTracking(True)
        self.grid_container.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)

        # 空状态提示
        self.empty_label = QLabel("暂无资产，请先转换图片")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.empty_label)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def refresh(self) -> None:
        """刷新资产列表。"""
        # 清除现有卡片
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._selected_ids.clear()

        assets = self.asset_manager.list_assets()
        self.empty_label.setVisible(len(assets) == 0)

        cols = 6  # 每行 6 个
        for i, info in enumerate(assets):
            card = AssetCard(info)
            # 加载缩略图（load_thumbnail 返回 float64 0-255，先转 uint8 再喂 QImage；
            # FastTransformation=最近邻，保持像素艺术锐利，避免平滑缩放模糊）
            thumb = self.asset_manager.load_thumbnail(info.id)
            if thumb is not None:
                from PySide6.QtGui import QImage, QPixmap
                arr = np.clip(thumb, 0, 255).astype(np.uint8)
                h, w = arr.shape[:2]
                qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
                card.thumb_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                    120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                ))
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            card.edit_requested.connect(self._on_card_double_clicked)
            card.download_requested.connect(self._on_download_single)
            card.delete_requested.connect(self._on_delete_single)
            card.drag_started.connect(self._on_card_drag_started)
            card.drag_moved.connect(self._on_card_drag_moved)
            card.drag_ended.connect(self._on_card_drag_ended)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self._cards.append(card)

        self._update_toolbar_state()

    # ------------------------------------------------------------------
    # 框选（rubber band）
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        """网格容器/滚动视口上的空白区拖拽框选。"""
        t = event.type()
        if obj in (self.grid_container, self.scroll.viewport()):
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._start_rubber(self._to_viewport(obj, event.position().toPoint()))
                return True
            if t == QEvent.Type.MouseMove and self._rubber_origin is not None:
                self._update_rubber(self._to_viewport(obj, event.position().toPoint()))
                return True
            if t == QEvent.Type.MouseButtonRelease and self._rubber_origin is not None:
                self._finish_rubber(self._to_viewport(obj, event.position().toPoint()))
                return True
        return super().eventFilter(obj, event)

    def _to_viewport(self, obj, local_pos: QPoint) -> QPoint:
        """将 obj 坐标系中的点映射到滚动视口坐标系。"""
        return obj.mapTo(self.scroll.viewport(), local_pos)

    def _on_card_drag_started(self, global_pos: QPoint) -> None:
        self._start_rubber(self.scroll.viewport().mapFromGlobal(global_pos))

    def _on_card_drag_moved(self, global_pos: QPoint) -> None:
        self._update_rubber(self.scroll.viewport().mapFromGlobal(global_pos))

    def _on_card_drag_ended(self, global_pos: QPoint) -> None:
        self._finish_rubber(self.scroll.viewport().mapFromGlobal(global_pos))

    def _start_rubber(self, pos: QPoint) -> None:
        self._rubber_origin = pos
        self._rubber.setGeometry(QRect(pos, pos))
        self._rubber.show()

    def _update_rubber(self, pos: QPoint) -> None:
        if self._rubber_origin is None:
            return
        self._rubber.setGeometry(QRect(self._rubber_origin, pos).normalized())

    def _finish_rubber(self, pos: QPoint) -> None:
        if self._rubber_origin is None:
            return
        rect = QRect(self._rubber_origin, pos).normalized()
        self._rubber_origin = None
        self._rubber.hide()
        # 与矩形相交的卡片加入选中（框选替换当前选择）
        vp = self.scroll.viewport()
        new_selected = set()
        for card in self._cards:
            tl = self.grid_container.mapTo(vp, card.geometry().topLeft())
            card_rect = QRect(tl, card.size())
            if rect.intersects(card_rect):
                new_selected.add(card.asset_info.id)
        if new_selected:
            self._selected_ids = new_selected
            for card in self._cards:
                card.set_selected(card.asset_info.id in self._selected_ids)
            self._update_toolbar_state()

    def _on_card_clicked(self, asset_id: str) -> None:
        """单击切换选中状态。"""
        if asset_id in self._selected_ids:
            self._selected_ids.discard(asset_id)
        else:
            self._selected_ids.add(asset_id)
        for card in self._cards:
            card.set_selected(card.asset_info.id in self._selected_ids)
        self._update_toolbar_state()

    def _on_card_double_clicked(self, asset_id: str) -> None:
        """双击触发编辑。"""
        self.edit_requested.emit(asset_id)

    def _update_toolbar_state(self) -> None:
        has_selection = len(self._selected_ids) > 0
        self.btn_export.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)

    def _on_batch_export(self) -> None:
        """批量导出选中资产；若开启"下载后移出"则导出后删除资产。"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if not self._selected_ids:
            return
        default_dir = load_preference("default_output_dir", "")
        dest_dir = QFileDialog.getExistingDirectory(self, "选择导出文件夹", default_dir)
        if not dest_dir:
            return
        removes = load_preference("download_removes_asset", True)
        count = 0
        removed = False
        # 遍历用 list() 拷贝：删除资产时避免迭代 set 导致 RuntimeError
        for asset_id in list(self._selected_ids):
            info = self.asset_manager.get_info(asset_id)
            if info:
                dest = f"{dest_dir}/{info.source_name}.png"
                if self.asset_manager.export_asset(asset_id, dest):
                    count += 1
                    if removes in (True, "true", "True", 1, "1"):
                        self.asset_manager.delete_asset(asset_id)
                        removed = True
        if removed:
            # refresh() 会清空 _selected_ids，须在循环结束后统一刷新
            self.refresh()
        QMessageBox.information(self, "导出完成", f"已导出 {count} 个资产到 {dest_dir}")

    def _on_batch_delete(self) -> None:
        """批量删除选中资产。"""
        from PySide6.QtWidgets import QMessageBox
        if not self._selected_ids:
            return
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定要删除 {len(self._selected_ids)} 个资产吗？",
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        for asset_id in list(self._selected_ids):
            self.asset_manager.delete_asset(asset_id)
        self._selected_ids.clear()
        self.refresh()

    def _on_download_single(self, asset_id: str) -> None:
        """下载单个资产。"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        info = self.asset_manager.get_info(asset_id)
        if not info:
            return
        default_dir = load_preference("default_output_dir", "")
        default_name = f"{info.source_name}.png"
        if default_dir:
            default_path = f"{default_dir}/{default_name}"
        else:
            default_path = default_name
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", default_path, "PNG 图片 (*.png)")
        if not path:
            return
        if self.asset_manager.export_asset(asset_id, path):
            # 若设置开启"下载后移出"，则删除资产
            removes = load_preference("download_removes_asset", True)
            if removes in (True, "true", "True", 1, "1"):
                self.asset_manager.delete_asset(asset_id)
                self.refresh()
            QMessageBox.information(self, "下载完成", f"已保存到 {path}")
        else:
            QMessageBox.warning(self, "错误", "下载失败")

    def _on_delete_single(self, asset_id: str) -> None:
        """删除单个资产。"""
        from PySide6.QtWidgets import QMessageBox
        info = self.asset_manager.get_info(asset_id)
        if not info:
            return
        ret = QMessageBox.question(
            self, "确认删除", f"确定要删除 {info.source_name} 吗？",
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.asset_manager.delete_asset(asset_id)
            self._selected_ids.discard(asset_id)
            self.refresh()

    def _on_search(self, text: str) -> None:
        """搜索过滤资产。"""
        text = text.lower().strip()
        for card in self._cards:
            visible = not text or text in card.asset_info.source_name.lower()
            card.setVisible(visible)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_A:
            for card in self._cards:
                card.set_selected(True)
                self._selected_ids.add(card.asset_info.id)
            self._update_toolbar_state()
        else:
            super().keyPressEvent(event)
