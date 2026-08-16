"""参数面板组件。

按流水线阶段组织参数编辑界面，使用 QStackedWidget 在降噪、放大、
网格检测、块提取、调色板精炼五个阶段页面间切换；控件变更时回写
:class:`PipelineParams` 并发出 ``param_changed`` 信号通知主窗口。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QWidget,
    QSlider,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
)

from ..pipeline import PipelineParams


class _LabeledSlider(QWidget):
    """带数值标签的水平滑块。

    通过 ``scale`` 因子在浮点业务值与整数滑块刻度间转换：滑块整数范围为
    ``[lo*scale, hi*scale]``，对外暴露的 :meth:`value` 为 ``整数刻度 / scale``。
    """

    def __init__(self, lo, hi, scale=1.0, suffix="", parent=None):
        super().__init__(parent)
        self._scale = scale
        self._suffix = suffix
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(lo * scale), int(hi * scale))
        self._label = QLabel()
        self._label.setMinimumWidth(64)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._label, 0)
        self._slider.valueChanged.connect(self._update_label)
        self._update_label(self._slider.value())

    def _update_label(self, val: int) -> None:
        v = val / self._scale
        self._label.setText(f"{v:g}{self._suffix}")

    def value(self) -> float:
        """返回当前业务浮点值（滑块整数刻度 / scale）。"""
        return self._slider.value() / self._scale

    def set_value(self, v: float) -> None:
        """以业务值设置滑块，阻塞信号以避免反馈。"""
        self._slider.blockSignals(True)
        self._slider.setValue(round(v * self._scale))
        self._slider.blockSignals(False)
        self._update_label(self._slider.value())

    def valueChanged(self):
        """返回内部滑块的 valueChanged 信号。"""
        return self._slider.valueChanged


class _StagePage(QWidget):
    """阶段参数页基类。

    子类实现 :meth:`sync_from_params` 从参数同步控件，并在控件变更时通过
    :meth:`_emit_changed` 让面板发出 ``param_changed`` 信号。
    """

    def __init__(self, panel: "ParamPanel", stage_name: str, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._stage_name = stage_name

    @property
    def _params(self) -> PipelineParams:
        return self._panel._params

    def _emit_changed(self) -> None:
        self._panel.param_changed.emit(self._stage_name)

    def sync_from_params(self) -> None:
        raise NotImplementedError


class _DenoisePage(_StagePage):
    """AI 降噪阶段页。"""

    def __init__(self, panel, stage_name, parent=None):
        super().__init__(panel, stage_name, parent)
        form = QFormLayout(self)
        title = QLabel("AI 降噪设置")
        title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 14px; padding: 4px 0;")
        form.addRow(title)
        self.method = QComboBox()
        self.method.addItem("不降噪", "none")
        self.method.addItem("NL-Means", "nl_means")
        self.method.addItem("TV-Chambolle", "tv_chambolle")
        self.method.addItem("双边滤波 (bilateral)", "bilateral")
        self.strength = _LabeledSlider(0.0, 1.0, scale=100)
        self.enable_clahe = QCheckBox("启用 CLAHE 局部对比度增强")
        self.enable_clahe.setChecked(False)
        self.clahe_clip_limit = QDoubleSpinBox()
        self.clahe_clip_limit.setRange(0.01, 0.1)
        self.clahe_clip_limit.setSingleStep(0.01)
        self.clahe_clip_limit.setDecimals(2)
        self.clahe_clip_limit.setValue(0.03)
        form.addRow("降噪方法", self.method)
        form.addRow("降噪强度", self.strength)
        form.addRow(self.enable_clahe)
        form.addRow("CLAHE 裁剪限制", self.clahe_clip_limit)
        self.method.currentIndexChanged.connect(self._on_changed)
        self.strength.valueChanged().connect(self._on_changed)
        self.enable_clahe.stateChanged.connect(self._on_changed)
        self.clahe_clip_limit.valueChanged.connect(self._on_changed)

    def _on_changed(self, *args) -> None:
        data = self.method.currentData()
        self._params.enable_ai_denoise = data != "none"
        self._params.ai_denoise_method = data
        self._params.ai_denoise_strength = self.strength.value()
        self._params.enable_clahe = self.enable_clahe.isChecked()
        self._params.clahe_clip_limit = self.clahe_clip_limit.value()
        self._emit_changed()
        self.clahe_clip_limit.setEnabled(self.enable_clahe.isChecked())

    def sync_from_params(self) -> None:
        p = self._params
        method = p.ai_denoise_method if p.enable_ai_denoise else "none"
        self.method.blockSignals(True)
        self.enable_clahe.blockSignals(True)
        self.clahe_clip_limit.blockSignals(True)
        idx = self.method.findData(method)
        self.method.setCurrentIndex(idx if idx >= 0 else 0)
        self.enable_clahe.setChecked(p.enable_clahe)
        self.clahe_clip_limit.setValue(p.clahe_clip_limit)
        self.method.blockSignals(False)
        self.enable_clahe.blockSignals(False)
        self.clahe_clip_limit.blockSignals(False)
        self.strength.set_value(p.ai_denoise_strength)
        self.clahe_clip_limit.setEnabled(self.enable_clahe.isChecked())


class _UpscalePage(_StagePage):
    """放大阶段页。"""

    def __init__(self, panel, stage_name, parent=None):
        super().__init__(panel, stage_name, parent)
        form = QFormLayout(self)
        title = QLabel("放大与锐化设置")
        title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 14px; padding: 4px 0;")
        form.addRow(title)
        self.enable = QCheckBox("启用放大（提升网格检测分辨率）")
        self.enable.setChecked(False)
        self.upscale_method = QComboBox()
        self.upscale_method.addItem("最近邻 (nearest)", "nearest")
        self.upscale_method.addItem("双线性 (bilinear)", "bilinear")
        self.upscale_method.addItem("双三次 (bicubic)", "bicubic")
        self.upscale_method.addItem("Lanczos", "lanczos")
        self.upscale_factor = QSpinBox()
        self.upscale_factor.setRange(2, 4)
        self.upscale_factor.setValue(2)
        self.enable_sharpen = QCheckBox("启用锐化（unsharp mask）")
        self.enable_sharpen.setChecked(False)
        self.sharpen_strength = QDoubleSpinBox()
        self.sharpen_strength.setRange(0.0, 1.0)
        self.sharpen_strength.setSingleStep(0.1)
        self.sharpen_strength.setDecimals(1)
        self.sharpen_strength.setValue(0.5)
        form.addRow(self.enable)
        form.addRow("放大算法", self.upscale_method)
        form.addRow("放大倍数", self.upscale_factor)
        form.addRow(self.enable_sharpen)
        form.addRow("锐化强度", self.sharpen_strength)
        self.enable.stateChanged.connect(self._on_changed)
        self.upscale_method.currentIndexChanged.connect(self._on_changed)
        self.upscale_factor.valueChanged.connect(self._on_changed)
        self.enable_sharpen.stateChanged.connect(self._on_changed)
        self.sharpen_strength.valueChanged.connect(self._on_changed)

    def _on_changed(self, *args) -> None:
        self._params.enable_upscale = self.enable.isChecked()
        self._params.upscale_method = self.upscale_method.currentData()
        self._params.upscale_factor = self.upscale_factor.value()
        self._params.enable_sharpen = self.enable_sharpen.isChecked()
        self._params.sharpen_strength = self.sharpen_strength.value()
        self._emit_changed()
        # 放大禁用时灰显放大倍数和锐化控件
        enabled = self.enable.isChecked()
        self.upscale_method.setEnabled(enabled)
        self.upscale_factor.setEnabled(enabled)
        self.enable_sharpen.setEnabled(enabled)
        self.sharpen_strength.setEnabled(enabled)

    def sync_from_params(self) -> None:
        p = self._params
        self.enable.blockSignals(True)
        self.upscale_method.blockSignals(True)
        self.upscale_factor.blockSignals(True)
        self.enable_sharpen.blockSignals(True)
        self.sharpen_strength.blockSignals(True)
        self.enable.setChecked(p.enable_upscale)
        idx = self.upscale_method.findData(p.upscale_method)
        self.upscale_method.setCurrentIndex(idx if idx >= 0 else 0)
        self.upscale_factor.setValue(p.upscale_factor)
        self.enable_sharpen.setChecked(p.enable_sharpen)
        self.sharpen_strength.setValue(p.sharpen_strength)
        self.enable.blockSignals(False)
        self.upscale_method.blockSignals(False)
        self.upscale_factor.blockSignals(False)
        self.enable_sharpen.blockSignals(False)
        self.sharpen_strength.blockSignals(False)
        enabled = self.enable.isChecked()
        self.upscale_method.setEnabled(enabled)
        self.upscale_factor.setEnabled(enabled)
        self.enable_sharpen.setEnabled(enabled)
        self.sharpen_strength.setEnabled(enabled)


class _PaletteRefinePage(_StagePage):
    """调色板精炼阶段页。"""

    def __init__(self, panel, stage_name, parent=None):
        super().__init__(panel, stage_name, parent)
        form = QFormLayout(self)
        title = QLabel("调色板精炼设置")
        title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 14px; padding: 4px 0;")
        form.addRow(title)
        self.enable = QCheckBox("启用 K-means 调色板精炼（统一全局色彩）")
        self.enable.setChecked(True)
        self.colors = _LabeledSlider(2, 64, scale=1, suffix=" 色")
        form.addRow(self.enable)
        form.addRow("颜色数", self.colors)
        self.enable.stateChanged.connect(self._on_changed)
        self.colors.valueChanged().connect(self._on_changed)

    def _on_changed(self, *args) -> None:
        self._params.enable_palette_refine = self.enable.isChecked()
        self._params.palette_colors = int(self.colors.value())
        self._emit_changed()

    def sync_from_params(self) -> None:
        p = self._params
        self.enable.blockSignals(True)
        self.enable.setChecked(p.enable_palette_refine)
        self.colors.set_value(float(p.palette_colors))
        self.enable.blockSignals(False)


class _GridDetectPage(_StagePage):
    """网格检测阶段页。"""

    def __init__(self, panel, stage_name, parent=None):
        super().__init__(panel, stage_name, parent)
        form = QFormLayout(self)
        title = QLabel("网格检测设置")
        title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 14px; padding: 4px 0;")
        form.addRow(title)
        self.min_p = QSpinBox()
        self.min_p.setRange(2, 100)
        self.max_p = QSpinBox()
        self.max_p.setRange(2, 200)
        self.snr_threshold = QDoubleSpinBox()
        self.snr_threshold.setRange(1.0, 50.0)
        self.snr_threshold.setSingleStep(0.5)
        self.snr_threshold.setDecimals(1)
        self.snr_threshold.setValue(8.0)
        form.addRow("最小周期 min_p", self.min_p)
        form.addRow("最大周期 max_p", self.max_p)
        form.addRow("SNR 阈值", self.snr_threshold)
        self.edge_tol = QSpinBox()
        self.edge_tol.setRange(1, 10)
        self.edge_tol.setValue(3)
        form.addRow("边缘搜索容差", self.edge_tol)
        self.subpixel_refine = QCheckBox("启用亚像素精炼")
        self.subpixel_refine.setChecked(True)
        form.addRow(self.subpixel_refine)
        self.smooth_strength = QDoubleSpinBox()
        self.smooth_strength.setRange(0.0, 1.0)
        self.smooth_strength.setSingleStep(0.1)
        self.smooth_strength.setDecimals(1)
        self.smooth_strength.setValue(0.5)
        form.addRow("平滑强度", self.smooth_strength)
        self.outlier_reject_ratio = QDoubleSpinBox()
        self.outlier_reject_ratio.setRange(0.0, 1.0)
        self.outlier_reject_ratio.setSingleStep(0.1)
        self.outlier_reject_ratio.setDecimals(1)
        self.outlier_reject_ratio.setValue(0.5)
        form.addRow("离群剔除比例", self.outlier_reject_ratio)
        self.fix_square = QCheckBox("正方形修正（差1时自动修正为正方形）")
        self.fix_square.setChecked(False)
        form.addRow(self.fix_square)
        self.min_p.valueChanged.connect(self._on_changed)
        self.max_p.valueChanged.connect(self._on_changed)
        self.snr_threshold.valueChanged.connect(self._on_changed)
        self.edge_tol.valueChanged.connect(self._on_changed)
        self.subpixel_refine.toggled.connect(self._on_changed)
        self.smooth_strength.valueChanged.connect(self._on_changed)
        self.outlier_reject_ratio.valueChanged.connect(self._on_changed)
        self.fix_square.stateChanged.connect(self._on_changed)

    def _on_changed(self, *args) -> None:
        self._params.min_p = self.min_p.value()
        self._params.max_p = self.max_p.value()
        self._params.snr_threshold = self.snr_threshold.value()
        self._params.edge_search_tolerance = self.edge_tol.value()
        self._params.enable_subpixel_refine = self.subpixel_refine.isChecked()
        self._params.smooth_strength = self.smooth_strength.value()
        self._params.outlier_reject_ratio = self.outlier_reject_ratio.value()
        self._params.fix_square = self.fix_square.isChecked()
        self._emit_changed()

    def sync_from_params(self) -> None:
        p = self._params
        self.min_p.blockSignals(True)
        self.max_p.blockSignals(True)
        self.snr_threshold.blockSignals(True)
        self.edge_tol.blockSignals(True)
        self.subpixel_refine.blockSignals(True)
        self.smooth_strength.blockSignals(True)
        self.outlier_reject_ratio.blockSignals(True)
        self.fix_square.blockSignals(True)
        self.min_p.setValue(p.min_p)
        self.max_p.setValue(p.max_p)
        self.snr_threshold.setValue(p.snr_threshold)
        self.edge_tol.setValue(p.edge_search_tolerance)
        self.subpixel_refine.setChecked(p.enable_subpixel_refine)
        self.smooth_strength.setValue(p.smooth_strength)
        self.outlier_reject_ratio.setValue(p.outlier_reject_ratio)
        self.fix_square.setChecked(p.fix_square)
        self.min_p.blockSignals(False)
        self.max_p.blockSignals(False)
        self.snr_threshold.blockSignals(False)
        self.edge_tol.blockSignals(False)
        self.subpixel_refine.blockSignals(False)
        self.smooth_strength.blockSignals(False)
        self.outlier_reject_ratio.blockSignals(False)
        self.fix_square.blockSignals(False)


class _ExtractPage(_StagePage):
    """块提取阶段页。"""

    def __init__(self, panel, stage_name, parent=None):
        super().__init__(panel, stage_name, parent)
        form = QFormLayout(self)
        title = QLabel("块提取设置")
        title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 14px; padding: 4px 0;")
        form.addRow(title)
        self.method = QComboBox()
        self.method.addItem("中位数 (median)", "median")
        self.method.addItem("均值 (mean)", "mean")
        self.method.addItem("众数 (mode)", "mode")
        self.method.addItem("K-means (kmeans)", "kmeans")
        self.core_ratio = _LabeledSlider(0.5, 1.0, scale=100)
        form.addRow("代表色算法", self.method)
        form.addRow("核心区比例", self.core_ratio)
        self.method.currentIndexChanged.connect(self._on_changed)
        self.core_ratio.valueChanged().connect(self._on_changed)

    def _on_changed(self, *args) -> None:
        self._params.extract_method = self.method.currentData()
        self._params.extract_core_ratio = self.core_ratio.value()
        self._emit_changed()

    def sync_from_params(self) -> None:
        p = self._params
        self.method.blockSignals(True)
        idx = self.method.findData(p.extract_method)
        self.method.setCurrentIndex(idx if idx >= 0 else 0)
        self.method.blockSignals(False)
        self.core_ratio.set_value(p.extract_core_ratio)


class ParamPanel(QWidget):
    """参数面板，按阶段切换页面并同步 :class:`PipelineParams`。

    Attributes:
        STAGES: 阶段键与中文标签的有序列表。
        param_changed: 控件变更信号，携带阶段名称。
    """

    STAGES = [
        ("denoise", "AI 降噪"),
        ("upscale", "放大"),
        ("grid_detect", "网格检测"),
        ("extract", "块提取"),
        ("palette_refine", "调色板精炼"),
    ]

    param_changed = Signal(str)

    def __init__(self, params: PipelineParams, parent=None):
        super().__init__(parent)
        self._params = params
        self.setStyleSheet("""
            ParamPanel { background: #333333; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QComboBox {
                background: #404040; color: #e0e0e0;
                border: 1px solid #555555; border-radius: 3px;
                padding: 2px 6px;
            }
            QComboBox QAbstractItemView {
                background: #404040; color: #e0e0e0;
                selection-background-color: #3d5a80;
            }
            QSpinBox, QDoubleSpinBox {
                background: #404040; color: #e0e0e0;
                border: 1px solid #555555; border-radius: 3px;
                padding: 2px 6px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #555555;
                border-bottom: 1px solid #555555;
                background: #404040;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #555555;
                background: #404040;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #505050;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e0e0e0;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e0e0e0;
            }
            QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
                border-bottom-color: #ffffff;
            }
            QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
                border-top-color: #ffffff;
            }
            QCheckBox { color: #e0e0e0; }
            QSlider::groove:horizontal {
                background: #2d2d2d; height: 4px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3d5a80; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #3d5a80; border-radius: 2px; }
        """)
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)
        self._pages: dict[str, _StagePage] = {}
        self._order: list[str] = []
        for key, _label in self.STAGES:
            page = self._make_page(key)
            self._pages[key] = page
            self._order.append(key)
            self._stack.addWidget(page)
        self.sync_from_params()

    def sync_from_params(self) -> None:
        """从参数同步各阶段页。"""
        for page in self._pages.values():
            page.sync_from_params()

    def _make_page(self, key: str) -> _StagePage:
        makers = {
            "denoise": _DenoisePage,
            "upscale": _UpscalePage,
            "grid_detect": _GridDetectPage,
            "extract": _ExtractPage,
            "palette_refine": _PaletteRefinePage,
        }
        return makers[key](self, key)

    def switch_to(self, stage: str) -> None:
        """切换到指定阶段页面并从参数同步控件。"""
        if stage not in self._pages:
            return
        idx = self._order.index(stage)
        self._stack.setCurrentIndex(idx)
        self._pages[stage].sync_from_params()
