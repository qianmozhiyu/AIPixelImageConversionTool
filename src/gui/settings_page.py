"""设置页面。

管理全局偏好：下载后移出资产、默认输出文件夹、资产存储位置、恢复默认参数。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QCheckBox, QLineEdit, QPushButton, QFileDialog, QGroupBox, QSpinBox,
)

from ..core import config


class SettingsPage(QWidget):
    """设置页面。"""

    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title = QLabel("设置")
        title.setStyleSheet("color: #e0e0e0; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # 资产设置组
        asset_group = QGroupBox("资产管理")
        asset_layout = QFormLayout(asset_group)

        self.cb_download_removes = QCheckBox('下载后将资产移出“我的资产”')
        self.cb_download_removes.stateChanged.connect(self._on_changed)
        asset_layout.addRow(self.cb_download_removes)

        self.le_output_dir = QLineEdit()
        self.le_output_dir.setReadOnly(True)
        self.btn_output_dir = QPushButton("选择...")
        self.btn_output_dir.clicked.connect(self._on_select_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.le_output_dir, 1)
        output_row.addWidget(self.btn_output_dir)
        asset_layout.addRow("默认输出文件夹:", output_row)

        self.le_asset_dir = QLineEdit()
        self.le_asset_dir.setReadOnly(True)
        asset_layout.addRow("资产存储位置:", self.le_asset_dir)

        layout.addWidget(asset_group)

        # 编辑器设置组
        editor_group = QGroupBox("像素编辑器")
        editor_layout = QFormLayout(editor_group)

        self.spin_undo_limit = QSpinBox()
        self.spin_undo_limit.setRange(1, 500)
        self.spin_undo_limit.setValue(50)
        self.spin_undo_limit.setToolTip("Ctrl+Z 可撤销的历史步数上限（编辑器打开时生效）")
        self.spin_undo_limit.valueChanged.connect(self._on_undo_limit_changed)
        editor_layout.addRow("撤销历史条数:", self.spin_undo_limit)

        layout.addWidget(editor_group)

        # 参数设置组
        param_group = QGroupBox("流水线参数")
        param_layout = QVBoxLayout(param_group)

        self.btn_reset_params = QPushButton("恢复默认参数")
        self.btn_reset_params.clicked.connect(self._on_reset_params)
        param_layout.addWidget(self.btn_reset_params)

        layout.addWidget(param_group)
        layout.addStretch(1)

    def _load_settings(self) -> None:
        self.cb_download_removes.setChecked(
            config.load_preference("download_removes_asset", True) is True or
            config.load_preference("download_removes_asset", True) in ("true", "True", 1, "1")
        )
        self.le_output_dir.setText(config.load_preference("default_output_dir", "") or "")
        asset_dir = config.load_preference("asset_store_dir", "") or ""
        if not asset_dir:
            from pathlib import Path
            asset_dir = str(Path.home() / ".aipixel" / "assets")
        self.le_asset_dir.setText(asset_dir)
        self.spin_undo_limit.setValue(
            int(config.load_preference("undo_history_limit", 50) or 50)
        )

    def _on_changed(self) -> None:
        config.save_preference("download_removes_asset", self.cb_download_removes.isChecked())
        self.settings_changed.emit()

    def _on_undo_limit_changed(self, value: int) -> None:
        config.save_preference("undo_history_limit", value)
        self.settings_changed.emit()

    def _on_select_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择默认输出文件夹")
        if d:
            self.le_output_dir.setText(d)
            config.save_preference("default_output_dir", d)

    def _on_reset_params(self) -> None:
        config.reset_params()
        self.settings_changed.emit()
