"""应用主窗口。

整合菜单栏、阶段列表、图像画布、参数面板与状态栏，串联图像打开、流水线
异步执行（:class:`PipelineWorker`）、结果保存等交互流程。
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QColor, QActionGroup
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QProgressBar,
    QLabel,
    QScrollArea,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QMenuBar,
    QStatusBar,
)

from ..pipeline import Pipeline, PipelineParams, PipelineResult
from ..core import config
from ..core.asset_manager import AssetManager
from ..core.io import load_image, save_image
from .image_canvas import ImageCanvas
from .param_panel import ParamPanel
from .worker import PipelineWorker
from .assets_page import AssetsPage
from .settings_page import SettingsPage


STAGE_KEYS = ["denoise", "resize", "grid_detect", "extract", "palette_refine", "preview"]
STAGE_LABELS = {
    "denoise": "AI 降噪",
    "resize": "调整大小",
    "grid_detect": "网格检测",
    "extract": "块提取",
    "palette_refine": "调色板精炼",
    "preview": "预览",
}
# UI 阶段键 -> pipeline stages 字段名映射
STAGE_TO_PIPELINE = {
    "denoise": "denoise_global",
    "resize": "resize",
    "grid_detect": "grid_detect",
    "extract": "extract",
    "palette_refine": "palette_refine",
}

GLOBAL_QSS = """
    QMainWindow { background: #1a1a1a; }
    QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
    QListWidget {
        background: #222222; color: #e0e0e0;
        border: none; font-size: 13px;
        outline: none;
    }
    QListWidget::item { padding: 12px 16px; border-bottom: 1px solid #2a2a2a; }
    QListWidget::item:selected { background: #2d4a6f; color: #ffffff; border-left: 3px solid #4a9eff; }
    QListWidget::item:hover { background: #2a2a2a; }
    QPushButton {
        background: #2d2d2d; color: #e0e0e0;
        border: 1px solid #3a3a3a; border-radius: 6px;
        padding: 8px 18px; font-size: 13px;
    }
    QPushButton:hover { background: #353535; border-color: #4a9eff; }
    QPushButton:pressed { background: #252525; }
    QPushButton:disabled { color: #555555; background: #1e1e1e; border-color: #2a2a2a; }
    QPushButton#primaryBtn {
        background: #2d4a6f; color: #ffffff; border: none;
        font-weight: bold; border-radius: 6px;
    }
    QPushButton#primaryBtn:hover { background: #3d5a80; }
    QPushButton#primaryBtn:disabled { background: #1e2d40; color: #445566; }
    QStatusBar { background: #141414; color: #888888; border-top: 1px solid #2a2a2a; }
    QStatusBar QLabel { padding: 0 12px; }
    QProgressBar {
        background: #222222; border: 1px solid #2a2a2a;
        border-radius: 4px; text-align: center; color: #e0e0e0;
        height: 18px; font-size: 11px;
    }
    QProgressBar::chunk { background: #4a9eff; border-radius: 3px; }
    QScrollArea { border: none; background: #1a1a1a; }
    QSplitter::handle { background: #1a1a1a; }
    QSplitter::handle:hover { background: #2d4a6f; }
    QMenuBar { background: #1a1a1a; color: #e0e0e0; border-bottom: 1px solid #2a2a2a; }
    QMenuBar::item { padding: 6px 16px; background: transparent; }
    QMenuBar::item:selected { background: #2d4a6f; border-radius: 4px; }
    QMenuBar::item:checked { background: #2d4a6f; border-radius: 4px; color: #ffffff; }
    QMenu { background: #222222; color: #e0e0e0; border: 1px solid #333333; border-radius: 6px; padding: 4px; }
    QMenu::item { padding: 6px 24px; border-radius: 4px; }
    QMenu::item:selected { background: #2d4a6f; }
    /* 警告/错误/信息弹窗：信息文字用黑色（弹窗背景保持系统浅色） */
    QMessageBox { background-color: #f5f5f5; }
    QMessageBox QLabel { color: #000000; font-size: 13px; }
    QMessageBox QPushButton { min-width: 72px; }
    QCheckBox { color: #e0e0e0; font-size: 13px; spacing: 8px; }
    QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; }
    QCheckBox::indicator:unchecked { background: #2d2d2d; border: 1px solid #444; }
    QCheckBox::indicator:checked { background: #4a9eff; border: 1px solid #4a9eff; }
    QSpinBox, QDoubleSpinBox {
        background: #2d2d2d; color: #e0e0e0;
        border: 1px solid #3a3a3a; border-radius: 4px;
        padding: 4px 8px; font-size: 13px;
    }
    QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4a9eff; }
    QComboBox {
        background: #2d2d2d; color: #e0e0e0;
        border: 1px solid #3a3a3a; border-radius: 4px;
        padding: 4px 10px; font-size: 13px;
    }
    QComboBox:hover { border-color: #4a9eff; }
    QComboBox::drop-down { border: none; width: 24px; }
    QComboBox QAbstractItemView {
        background: #222222; color: #e0e0e0;
        border: 1px solid #333333; selection-background-color: #2d4a6f;
    }
    QLineEdit {
        background: #2d2d2d; color: #e0e0e0;
        border: 1px solid #3a3a3a; border-radius: 4px;
        padding: 4px 10px; font-size: 13px;
    }
    QLineEdit:focus { border-color: #4a9eff; }
    QLabel { color: #e0e0e0; }
    QGroupBox {
        color: #aaa; border: 1px solid #2a2a2a; border-radius: 6px;
        margin-top: 12px; padding-top: 16px; font-size: 13px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QFrame { color: #e0e0e0; }
"""


def _params_to_dict(p: PipelineParams) -> dict:
    """将 :class:`PipelineParams` 全部字段导出为字典。"""
    return {
        "enable_ai_denoise": p.enable_ai_denoise,
        "ai_denoise_method": p.ai_denoise_method,
        "ai_denoise_strength": p.ai_denoise_strength,
        "enable_clahe": p.enable_clahe,
        "clahe_clip_limit": p.clahe_clip_limit,
        "enable_upscale": p.enable_upscale,
        "upscale_factor": p.upscale_factor,
        "upscale_method": p.upscale_method,
        "enable_sharpen": p.enable_sharpen,
        "sharpen_strength": p.sharpen_strength,
        "min_p": p.min_p,
        "max_p": p.max_p,
        "user_hint": p.user_hint,
        "phase_step": p.phase_step,
        "outlier_reject_ratio": p.outlier_reject_ratio,
        "extract_method": p.extract_method,
        "extract_core_ratio": p.extract_core_ratio,
        "enable_palette_refine": p.enable_palette_refine,
        "palette_colors": p.palette_colors,
    }


class MainWindow(QMainWindow):
    """AI 像素图像转换工具主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 像素图像转换工具")
        self.resize(1200, 800)

        # 初始化时从 config 加载参数
        self.params = config.load_params()
        self.canvas = ImageCanvas()
        self.param_panel = ParamPanel(self.params)
        self.worker: PipelineWorker | None = None
        self._batch_worker: "BatchWorker | None" = None
        self.image = None
        self.result: PipelineResult | None = None
        self._pipeline: Pipeline | None = None
        # 资产管理器
        asset_store_dir = config.load_preference("asset_store_dir", "")
        self.asset_manager = AssetManager(asset_store_dir) if asset_store_dir else AssetManager()
        self._build_ui()
        # 参数变更仅持久化参数，不自动执行转换（用户需点击"开始转换"）
        self.param_panel.param_changed.connect(self._on_param_changed)
        self._build_menu()
        self.stage_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # QStackedWidget（页面切换由菜单栏标签控制）
        self.stack = QStackedWidget()

        # 页面 0：转换页（现有三栏 + 按钮）
        # 左侧：阶段列表（序号 + 名称 + 状态点）
        self.stage_list = QListWidget()
        self.stage_list.setMinimumWidth(200)
        for key in STAGE_KEYS:
            item = QListWidgetItem(f"{STAGE_LABELS[key]}  ●")
            item.setData(Qt.UserRole, key)
            self.stage_list.addItem(item)
        self.stage_list.currentRowChanged.connect(self._on_stage_changed)

        # 右侧：参数面板（内部为单滚动面板，所有阶段参数一起显示）

        # 中间：画布（上方显示"对比原图"按钮，所有阶段均可对比）；用 QSplitter 组织三栏
        canvas_wrap = QWidget()
        canvas_wrap_layout = QVBoxLayout(canvas_wrap)
        canvas_wrap_layout.setContentsMargins(0, 0, 0, 0)
        canvas_wrap_layout.setSpacing(4)
        self.compare_btn = QPushButton("对比原图")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setVisible(False)  # 有转换结果且选中阶段时显示
        self.compare_btn.toggled.connect(self._on_compare_toggled)
        canvas_wrap_layout.addWidget(self.compare_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        # 预览阶段工具栏（分页 + 操作，仅"预览"阶段显示）
        preview_toolbar = QHBoxLayout()
        preview_toolbar.setSpacing(6)
        self.preview_prev_btn = QPushButton("◀")
        self.preview_next_btn = QPushButton("▶")
        self.preview_page_label = QLabel("第 0 / 0 张")
        self.preview_page_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.preview_prev_btn.setFixedWidth(32)
        self.preview_next_btn.setFixedWidth(32)
        self.preview_prev_btn.clicked.connect(self._preview_prev)
        self.preview_next_btn.clicked.connect(self._preview_next)
        preview_toolbar.addWidget(self.preview_prev_btn)
        preview_toolbar.addWidget(self.preview_page_label)
        preview_toolbar.addWidget(self.preview_next_btn)
        preview_toolbar.addStretch(1)
        self.preview_regen_btn = QPushButton("重新生成")
        self.preview_add_btn = QPushButton("加入资产库")
        self.preview_del_btn = QPushButton("删除")
        self.preview_add_all_btn = QPushButton("一键加入")
        self.preview_clear_btn = QPushButton("一键清空")
        self.preview_regen_btn.clicked.connect(self._preview_regen)
        self.preview_add_btn.clicked.connect(self._preview_add_current)
        self.preview_del_btn.clicked.connect(self._preview_delete_current)
        self.preview_add_all_btn.clicked.connect(self._preview_add_all)
        self.preview_clear_btn.clicked.connect(self._preview_clear_all)
        preview_toolbar.addWidget(self.preview_regen_btn)
        preview_toolbar.addWidget(self.preview_add_btn)
        preview_toolbar.addWidget(self.preview_del_btn)
        preview_toolbar.addWidget(self.preview_add_all_btn)
        preview_toolbar.addWidget(self.preview_clear_btn)
        self.preview_toolbar_widget = QWidget()
        self.preview_toolbar_widget.setLayout(preview_toolbar)
        self.preview_toolbar_widget.setVisible(False)
        canvas_wrap_layout.addWidget(self.preview_toolbar_widget)
        canvas_wrap_layout.addWidget(self.canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.stage_list)
        splitter.addWidget(canvas_wrap)
        splitter.addWidget(self.param_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 700, 340])

        # 底部：开始/取消按钮（"导入文件夹"已移至 文件菜单）
        self.start_btn = QPushButton("开始转换")
        self.start_btn.setObjectName("primaryBtn")
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn.clicked.connect(self._on_cancel)
        # 转换状态机：idle / running / done（待放入资产库）/ batch_running
        self._conv_state = "idle"
        # 预览阶段：收集单张与批量转换结果，分页浏览确认
        self._preview_items: list[tuple[str, object, object]] = []  # (name, pixel_art, original)
        self._preview_index = 0
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch(1)

        # 连接画布拖拽信号
        self.canvas.image_dropped.connect(self._on_image_dropped)
        self.canvas.folder_dropped.connect(self._on_folder_dropped)

        convert_page = QWidget()
        convert_layout = QVBoxLayout(convert_page)
        convert_layout.setContentsMargins(0, 0, 0, 0)
        convert_layout.addWidget(splitter, 1)
        convert_layout.addLayout(btn_layout)
        self.stack.addWidget(convert_page)  # index 0

        # 页面 1：我的资产
        self.assets_page = AssetsPage(self.asset_manager)
        self.assets_page.edit_requested.connect(self._on_edit_asset)
        self.stack.addWidget(self.assets_page)  # index 1

        # 页面 2：设置
        self.settings_page = SettingsPage()
        self.settings_page.settings_changed.connect(self._on_settings_changed)
        self.stack.addWidget(self.settings_page)  # index 2

        # 主布局
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        # 状态栏：进度条水平居中（容器内双弹性占位），状态标签居左
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(320)
        self.status_label = QLabel("就绪")
        status_bar = QStatusBar()
        center_wrap = QWidget()
        center_layout = QHBoxLayout(center_wrap)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.status_label)
        center_layout.addStretch(1)
        center_layout.addWidget(self.progress_bar)
        center_layout.addStretch(1)
        status_bar.addWidget(center_wrap, 1)
        self.setStatusBar(status_bar)

        # 全局深色 QSS
        self.setStyleSheet(GLOBAL_QSS)

    def _on_page_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        # 同步菜单栏标签选中态（setChecked 只发 toggled，不回流 triggered）
        if hasattr(self, "_page_actions") and 0 <= idx < len(self._page_actions):
            self._page_actions[idx].setChecked(True)
        if idx == 1:  # 切换到资产页时刷新
            self.assets_page.refresh()

    def _on_edit_asset(self, asset_id: str) -> None:
        """打开像素编辑器编辑指定资产。"""
        from .pixel_editor import PixelEditor
        editor = PixelEditor(self.asset_manager, asset_id, self)
        editor.saved.connect(self._on_asset_edited)
        editor.show()
        self._editor = editor  # 保持引用避免被回收

    def _on_asset_edited(self) -> None:
        """资产编辑保存后刷新资产页。"""
        self.assets_page.refresh()

    def _on_settings_changed(self) -> None:
        # 设置变更后重新加载参数（"恢复默认参数"等）；
        # 需更新 param_panel 持有的引用，否则同步的仍是旧对象
        self.params = config.load_params()
        self.param_panel.set_params(self.params)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        open_action = QAction("打开图片", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        # 导入文件夹：位于"打开图片"正下方，快捷键 Ctrl+Shift+O
        self.batch_action = QAction("导入文件夹", self)
        self.batch_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.batch_action.triggered.connect(self._on_batch_import)
        file_menu.addAction(self.batch_action)

        save_action = QAction("保存结果", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # 页面切换标签（位于文件与帮助之间）
        self._build_page_actions(menubar)

        # 开始转换快捷键 Ctrl+E（窗口级 action，任意焦点生效）
        self.start_shortcut = QAction("开始转换", self)
        self.start_shortcut.setShortcut(QKeySequence("Ctrl+E"))
        self.start_shortcut.triggered.connect(self._on_start)
        self.addAction(self.start_shortcut)
        # 放入资产库快捷键 Ctrl+Y（转换完成后待确认时生效）
        self.add_asset_shortcut = QAction("放入资产库", self)
        self.add_asset_shortcut.setShortcut(QKeySequence("Ctrl+Y"))
        self.add_asset_shortcut.triggered.connect(self._on_add_asset_shortcut)
        self.addAction(self.add_asset_shortcut)

        help_menu = menubar.addMenu("帮助")
        shortcuts_action = QAction("快捷键说明", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_help)
        help_menu.addAction(shortcuts_action)
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _build_page_actions(self, menubar: QMenuBar) -> None:
        """在菜单栏加入页面切换标签（文件 | 转换 | 我的资产 | 设置 | 帮助）。

        用互斥 checkable action 替代原独立导航栏，点击标签切换 QStackedWidget 页面。
        """
        self._page_group = QActionGroup(self)
        self._page_group.setExclusive(True)
        self._page_actions = []
        for i, name in enumerate(["转换", "我的资产", "设置"]):
            act = QAction(name, self)
            act.setCheckable(True)
            act.setData(i)
            menubar.addAction(act)
            self._page_group.addAction(act)
            act.triggered.connect(lambda _c, idx=i: self._on_page_changed(idx))
            self._page_actions.append(act)
        self._page_actions[0].setChecked(True)

    # ------------------------------------------------------------------
    # 阶段切换
    # ------------------------------------------------------------------
    def _on_stage_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.stage_list.item(row)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        self.param_panel.switch_to(key)

        # 预览阶段：显示预览分页内容（单张/批量转换结果）
        if key == "preview":
            self._show_preview_current()
            return

        # 无结果或该阶段结果缺失：显示原图
        if self.result is None or STAGE_TO_PIPELINE.get(key, key) not in self.result.stages:
            self._show_original()
            self._set_compare_visible(False)
            self.preview_toolbar_widget.setVisible(False)
            return

        stages = self.result.stages
        meta = self.result.metadata

        if key == "denoise":
            arr = stages["denoise_global"]
            self.canvas.set_image(np.clip(arr, 0, 255))
            self.canvas.set_grid_overlay(None)
            self.canvas.set_info("AI 降噪")
        elif key == "resize":
            arr = stages.get("resize")
            if arr is not None:
                self.canvas.set_image(np.clip(arr, 0, 255))
                self.canvas.set_grid_overlay(None)
                self.canvas.set_info("调整大小")
            else:
                self._show_original()
                return
        elif key == "grid_detect":
            arr = stages["resize"]
            grid = stages["grid_detect"]
            self.canvas.set_image(np.clip(arr, 0, 255))
            self.canvas.set_grid_overlay(grid)
            self.canvas.set_info(
                f"网格: {meta['w_logic']}×{meta['h_logic']} 块 | "
                f"块大小: {meta['px']}×{meta['py']}px | 置信度: {meta['grid_conf']:.2f}"
            )
        elif key == "extract":
            arr = stages["extract"]
            self.canvas.set_image(np.clip(arr, 0, 255))
            self.canvas.set_grid_overlay(None)
            self.canvas.set_info("块提取")
        elif key == "palette_refine":
            arr = stages["palette_refine"]
            unique = len(np.unique(arr.reshape(-1, 3), axis=0))
            self.canvas.set_image(np.clip(arr, 0, 255))
            self.canvas.set_grid_overlay(None)
            self.canvas.set_info(f"调色板精炼 | {unique}色")

        # 每个阶段都可对比原图
        self._set_compare_visible(True)
        self.canvas.fit_to_view()

    def _set_compare_visible(self, visible: bool) -> None:
        """显示/隐藏调色板精炼阶段的"对比原图"按钮。"""
        if not hasattr(self, "compare_btn"):
            return
        self.compare_btn.blockSignals(True)
        if not visible:
            self.compare_btn.setChecked(False)
        self.compare_btn.setVisible(visible)
        self.compare_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # 预览阶段（单张/批量转换结果分页浏览、确认入库、重新生成）
    # ------------------------------------------------------------------
    def _preview_add_item(self, source_name: str, pixel_art, original) -> None:
        """将转换结果加入预览列表。"""
        self._preview_items.append((source_name, pixel_art, original))
        # 若当前正停留在预览阶段，刷新显示（切到最新一张）
        if (
            self.stage_list.currentItem() is not None
            and self.stage_list.currentItem().data(Qt.UserRole) == "preview"
        ):
            self._preview_index = len(self._preview_items) - 1
            self._show_preview_current()

    def _show_preview_current(self) -> None:
        """显示当前预览张（画布 + 基础信息 + 分页状态）。"""
        self.preview_toolbar_widget.setVisible(True)
        # 进入预览时清除残留的对比勾选态
        if self.compare_btn.isChecked():
            self.compare_btn.blockSignals(True)
            self.compare_btn.setChecked(False)
            self.compare_btn.blockSignals(False)
        if not self._preview_items:
            self.canvas.clear()
            self.canvas.set_info("暂无预览图片")
            self.preview_page_label.setText("第 0 / 0 张")
            self._set_compare_visible(False)
            self._update_preview_buttons()
            return
        # 防止索引越界（删除后调整）
        if self._preview_index >= len(self._preview_items):
            self._preview_index = len(self._preview_items) - 1
        if self._preview_index < 0:
            self._preview_index = 0
        name, art, _orig = self._preview_items[self._preview_index]
        arr = np.clip(art, 0, 255).astype(np.uint8)
        self.canvas.set_image(arr)
        self.canvas.set_grid_overlay(None)
        unique = len(np.unique(arr.reshape(-1, 3), axis=0))
        self.canvas.set_info(f"{name} | {unique}色")
        self.canvas.fit_to_view()
        self.preview_page_label.setText(
            f"第 {self._preview_index + 1} / {len(self._preview_items)} 张"
        )
        self._set_compare_visible(True)
        self._update_preview_buttons()

    def _update_preview_buttons(self) -> None:
        has = bool(self._preview_items)
        self.preview_prev_btn.setEnabled(has and self._preview_index > 0)
        self.preview_next_btn.setEnabled(has and self._preview_index < len(self._preview_items) - 1)
        for btn in (self.preview_regen_btn, self.preview_add_btn, self.preview_del_btn,
                    self.preview_add_all_btn, self.preview_clear_btn):
            btn.setEnabled(has)

    def _preview_prev(self) -> None:
        if self._preview_index > 0:
            self._preview_index -= 1
            self._show_preview_current()

    def _preview_next(self) -> None:
        if self._preview_index < len(self._preview_items) - 1:
            self._preview_index += 1
            self._show_preview_current()

    def _preview_regen(self) -> None:
        """使用当前参数重新生成当前预览张（可在参数面板调整后再点）。"""
        if not self._preview_items:
            return
        name, _art, original = self._preview_items[self._preview_index]
        try:
            pipeline = Pipeline(self.params)
            result = pipeline.run(original)
        except Exception as e:
            QMessageBox.warning(self, "重新生成失败", str(e))
            return
        self._preview_items[self._preview_index] = (name, result.pixel_art, original)
        self.status_label.setText(f"已重新生成: {name}")
        self._show_preview_current()

    def _preview_add_current(self) -> None:
        """将当前预览张加入资产库并从预览移除。"""
        if not self._preview_items:
            return
        name, art, _orig = self._preview_items[self._preview_index]
        self.asset_manager.add_asset(art, name)
        self._preview_items.pop(self._preview_index)
        self.status_label.setText(f"已加入资产库: {name}")
        self._show_preview_current()

    def _preview_delete_current(self) -> None:
        """删除当前预览张。"""
        if not self._preview_items:
            return
        name, _art, _orig = self._preview_items.pop(self._preview_index)
        self.status_label.setText(f"已删除预览: {name}")
        self._show_preview_current()

    def _preview_add_all(self) -> None:
        """一键将所有预览图片加入资产库并清空预览。"""
        if not self._preview_items:
            return
        for name, art, _orig in list(self._preview_items):
            self.asset_manager.add_asset(art, name)
        count = len(self._preview_items)
        self._preview_items.clear()
        self._preview_index = 0
        self.status_label.setText(f"已将 {count} 张预览图片加入资产库")
        self._show_preview_current()

    def _preview_clear_all(self) -> None:
        """一键清空所有预览图片。"""
        self._preview_items.clear()
        self._preview_index = 0
        self.status_label.setText("已清空预览")
        self._show_preview_current()

    def _on_compare_toggled(self, checked: bool) -> None:
        """对比原图开关：勾选时画布显示原图，取消时恢复当前阶段/预览结果。"""
        if checked:
            # 预览阶段对比当前张的原始图，其余阶段对比当前导入的原图
            current_item = self.stage_list.currentItem()
            key = current_item.data(Qt.UserRole) if current_item else None
            target = None
            if key == "preview" and self._preview_items:
                target = self._preview_items[self._preview_index][2]
            elif self.image is not None:
                target = self.image
            if target is not None:
                self.canvas.set_image(np.clip(target, 0, 255).astype(np.uint8))
                self.canvas.set_grid_overlay(None)
                self.canvas.set_info("原图（对比）")
                self.canvas.fit_to_view()
        else:
            # 取消勾选：重渲染当前阶段视图
            self._on_stage_changed(self.stage_list.currentRow())

    def _show_original(self) -> None:
        """画布回退到原图显示。"""
        if self.image is not None:
            self.canvas.set_image(self.image)
        self.canvas.set_grid_overlay(None)
        self.canvas.set_info("")
        self.canvas.fit_to_view()

    def _update_stage_list_status(self) -> None:
        """把已完成阶段的 ● 标记为绿色，未完成阶段为灰色（无结果时全灰）。"""
        done = set(self.result.stages.keys()) if self.result is not None else set()
        for i, key in enumerate(STAGE_KEYS):
            item = self.stage_list.item(i)
            if item is None:
                continue
            pipeline_key = STAGE_TO_PIPELINE.get(key, key)
            color = QColor("#4caf50") if pipeline_key in done else QColor("#9e9e9e")
            item.setForeground(color)

    def _reset_conversion_state(self) -> None:
        """导入新图片：取消进行中的转换并清除所有阶段的预览/结果。"""
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
        self.result = None
        self._pipeline = None
        self._update_stage_list_status()  # 所有阶段点变灰
        self.progress_bar.setValue(0)
        self.canvas.set_grid_overlay(None)
        self._set_compare_visible(False)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载图片失败: {e}")
            return
        self.image = img
        self._current_source_name = os.path.splitext(os.path.basename(path))[0]
        self._reset_conversion_state()  # 清除旧图片的转换结果与阶段预览
        self.canvas.set_image(img)
        self.canvas.fit_to_view()
        self.canvas.set_info("原图")
        self.status_label.setText(f"已加载: {path}")

    def _on_save(self) -> None:
        if self.result is None:
            QMessageBox.warning(self, "提示", "没有可保存的结果，请先执行转换。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG 图片 (*.png);;JPEG 图片 (*.jpg)"
        )
        if not path:
            return
        scale, ok = QInputDialog.getInt(
            self, "保存", "输入放大倍数 (1-32):", 1, 1, 32, 1
        )
        if not ok:
            return
        try:
            save_image(self.result.pixel_art, path, scale=scale)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {e}")
            return
        self.status_label.setText(f"已保存: {path}")

    # ------------------------------------------------------------------
    # 流水线执行
    # ------------------------------------------------------------------
    def _on_param_changed(self, stage: str) -> None:
        """参数面板控件变更：仅持久化参数，不自动执行转换。

        用户调整参数后需点击"开始转换"（或 Ctrl+E）才会重新执行流水线。
        """
        config.save_params(self.params)

    def _on_start(self) -> None:
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先打开一张图片。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._pipeline = Pipeline(self.params)
        self.worker = PipelineWorker(
            image=self.image,
            params=self.params,
            pipeline=self._pipeline,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()
        self._conv_state = "running"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("转换中...")

    def _on_cancel(self) -> None:
        # 转换完成待入库状态：点击"放入资产库"触发入库
        if self._conv_state == "done":
            self._add_result_to_library()
            return
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
        if (
            getattr(self, "_batch_worker", None) is not None
            and self._batch_worker.isRunning()
        ):
            self._batch_worker.cancel()

    def _on_progress(self, stage: str, percent: float) -> None:
        self.status_label.setText(f"{stage} {percent:.0f}%")
        self.progress_bar.setValue(int(percent))

    def _on_finished(self, result: PipelineResult) -> None:
        self.result = result
        self._update_stage_list_status()
        meta = result.metadata
        self.canvas.set_image(result.pixel_art)
        self.canvas.set_grid_overlay(None)
        unique = meta.get("unique_colors", "?")
        self.canvas.set_info(f"完成 | {meta['w_logic']}×{meta['h_logic']} 像素块 | {unique}色")
        self.canvas.fit_to_view()
        self.status_label.setText("完成")
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)
        # 转换结果加入"预览"阶段（单张与批量结果统一收集）
        source_name = "未命名"
        if hasattr(self, "_current_source_name"):
            source_name = self._current_source_name
        original = self.image if self.image is not None else result.pixel_art
        self._preview_add_item(source_name, result.pixel_art, original)
        # 是否自动移入资产库（设置项 auto_add_asset，默认关闭）
        auto_add = config.load_preference("auto_add_asset", False)
        if auto_add in (True, "true", "True", 1, "1"):
            self._add_result_to_library(show_message=True)
        else:
            # 待用户确认：将"取消"按钮变为"放入资产库"
            self._conv_state = "done"
            self.cancel_btn.setText("放入资产库")
            self.cancel_btn.setEnabled(True)
        # 转换完成：显示"对比原图"按钮（所有阶段可对比），清除残留的对比勾选态，
        # 保证重新转换后点击当前阶段即可对比，无需先切到其他阶段再切回
        self.compare_btn.blockSignals(True)
        self.compare_btn.setChecked(False)
        self.compare_btn.blockSignals(False)
        self._set_compare_visible(True)

    def _add_result_to_library(self, show_message: bool = False) -> None:
        """将当前转换结果加入资产库（手动"放入资产库"或自动移入时调用）。

        入库后保留各阶段预览图片，方便继续对比查看。
        """
        if self.result is None:
            return
        source_name = "未命名"
        if hasattr(self, "_current_source_name"):
            source_name = self._current_source_name
        self.asset_manager.add_asset(self.result.pixel_art, source_name)
        # 复位按钮状态
        self._conv_state = "idle"
        self.cancel_btn.setText("取消")
        self.cancel_btn.setEnabled(False)
        if show_message:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "已加入资产库", f"转换结果已保存到“我的资产”（{source_name}）")

    def _on_add_asset_shortcut(self) -> None:
        """Ctrl+Y：转换完成待确认时，将结果放入资产库。"""
        if self._conv_state == "done" and self.result is not None:
            self._add_result_to_library(show_message=True)

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "错误", msg)
        self.status_label.setText("出错")
        self._conv_state = "idle"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_cancelled(self) -> None:
        self.status_label.setText("已取消")
        self._conv_state = "idle"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # 批量文件夹处理
    # ------------------------------------------------------------------
    def _on_image_dropped(self, path: str) -> None:
        """拖拽图片文件：加载并显示。"""
        try:
            img = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载图片失败: {e}")
            return
        self.image = img
        self._current_source_name = os.path.splitext(os.path.basename(path))[0]
        self._reset_conversion_state()  # 清除旧图片的转换结果与阶段预览
        self.canvas.set_image(img)
        self.canvas.fit_to_view()
        self.canvas.set_info("原图")
        self.status_label.setText(f"已加载: {path}")

    def _on_folder_dropped(self, folder: str) -> None:
        """拖拽文件夹：触发批量转换。"""
        self._start_batch(folder)

    def _on_batch_import(self) -> None:
        """导入文件夹按钮：选择文件夹并批量转换。"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self._start_batch(folder)

    def _start_batch(self, folder: str) -> None:
        """启动批量处理。"""
        from .batch_worker import BatchWorker
        self._batch_worker = BatchWorker(folder, self.params, self.asset_manager)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.image_done.connect(lambda _: None)  # 自动移入时单张完成
        self._batch_worker.image_result.connect(self._on_batch_image_result)
        self._batch_worker.finished_all.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.cancelled.connect(self._on_batch_cancelled)
        self._batch_worker.start()
        self._conv_state = "batch_running"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(False)
        self.batch_action.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("批量处理中...")

    def _on_batch_image_result(self, source_name: str, pixel_art, original) -> None:
        """收集批量转换结果到"预览"阶段（无论是否自动移入）。"""
        self._preview_add_item(source_name, pixel_art, original)

    def _on_batch_progress(self, current: int, total: int, filename: str) -> None:
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"处理中 {current}/{total}: {filename}")

    def _on_batch_finished(self, success: int) -> None:
        self.status_label.setText(f"批量完成: {success} 张成功")
        self.progress_bar.setValue(100)
        self._conv_state = "idle"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(True)
        self.batch_action.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        # 批量结果已收集到"预览"阶段（分页浏览确认），提示用户前往查看
        if self._preview_items:
            self.status_label.setText(f"批量完成: {success} 张成功，结果已放入“预览”阶段")

    def _on_batch_error(self, filename: str, msg: str) -> None:
        self.status_label.setText(f"错误: {filename} - {msg}")

    def _on_batch_cancelled(self) -> None:
        self.status_label.setText("批量处理已取消")
        self._conv_state = "idle"
        self.cancel_btn.setText("取消")
        self.start_btn.setEnabled(True)
        self.batch_action.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # 关于
    # ------------------------------------------------------------------
    def _on_about(self) -> None:
        QMessageBox.about(self, "关于", "AI 像素图像转换工具\n基于 PySide6 实现")

    def _show_shortcuts_help(self) -> None:
        """帮助 → 快捷键说明：列出全部快捷键。"""
        lines = [
            "--- 主界面 ---",
            "Ctrl+O         打开图片",
            "Ctrl+E         开始转换",
            "Ctrl+S         保存结果",
            "Ctrl+Shift+O   导入文件夹",
            "",
            "--- 像素编辑器 ---",
            "Ctrl+Z         撤销",
            "Ctrl+Y / Ctrl+Shift+Z   重做",
            "B / E / I / G / L / U   画笔/橡皮/吸管/油漆桶/直线/矩形",
            "[ / ]          减小/增大笔刷",
            "Alt + 拖动      临时吸管取色（松开恢复）",
            "Shift + 拖动    临时直线（松开左键提交）",
            "空格 / 中键     平移画布",
            "滚轮            缩放",
        ]
        QMessageBox.information(self, "快捷键说明", "\n".join(lines))


def main() -> int:
    """启动 GUI 应用。"""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication

    app = QApplication.instance() or QApplication(sys.argv)
    # 设置组织/应用名，使无参 QSettings() 能定位到持久化存储
    # （此前未设置导致参数/偏好写入被静默丢弃，重启后回默认值）
    QCoreApplication.setOrganizationName("AIPixelTool")
    QCoreApplication.setApplicationName("AIPixelImageConversionTool")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
