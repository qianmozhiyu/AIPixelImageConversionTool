"""应用配置持久化。

使用 QSettings 保存/加载 PipelineParams 全部字段与全局偏好设置，
支持跨会话恢复用户参数。
"""

from __future__ import annotations

from typing import Any, Optional
from PySide6.QtCore import QSettings

from ..pipeline import PipelineParams


# QSettings 键名前缀
_PARAM_PREFIX = "pipeline/"
_PREF_PREFIX = "prefs/"

# 可序列化的参数字段列表（排除 user_hint，单独处理）
_PARAM_FIELDS = [
    "enable_ai_denoise", "ai_denoise_method", "ai_denoise_strength",
    "enable_clahe", "clahe_clip_limit",
    "enable_upscale", "upscale_factor", "upscale_method",
    "enable_sharpen", "sharpen_strength",
    "min_p", "max_p", "phase_step", "snr_threshold",
    "edge_search_tolerance", "enable_subpixel_refine", "smooth_strength",
    "outlier_reject_ratio",
    "extract_method", "extract_core_ratio", "fix_square",
    "enable_palette_refine", "palette_colors",
]

# 字段类型映射，用于从 QSettings 读取时类型转换
_FIELD_TYPES = {
    "enable_ai_denoise": bool, "ai_denoise_method": str, "ai_denoise_strength": float,
    "enable_clahe": bool, "clahe_clip_limit": float,
    "enable_upscale": bool, "upscale_factor": int, "upscale_method": str,
    "enable_sharpen": bool, "sharpen_strength": float,
    "min_p": int, "max_p": int, "phase_step": float, "snr_threshold": float,
    "edge_search_tolerance": int, "enable_subpixel_refine": bool, "smooth_strength": float,
    "outlier_reject_ratio": float,
    "extract_method": str, "extract_core_ratio": float, "fix_square": bool,
    "enable_palette_refine": bool, "palette_colors": int,
}

# 默认全局偏好
_DEFAULT_PREFS = {
    "download_removes_asset": True,   # 下载后是否将资产移出"我的资产"
    "default_output_dir": "",          # 默认输出文件夹
    "asset_store_dir": "",             # 资产存储位置（空=使用默认 ~/.aipixel/assets/）
    "undo_history_limit": 50,          # 像素编辑器撤销历史条数上限
}


def save_params(params: PipelineParams) -> None:
    """将 PipelineParams 全部字段保存到 QSettings。"""
    settings = QSettings()
    for field in _PARAM_FIELDS:
        settings.setValue(_PARAM_PREFIX + field, getattr(params, field))
    # user_hint 单独处理
    if params.user_hint is not None:
        settings.setValue(_PARAM_PREFIX + "user_hint_w", params.user_hint[0])
        settings.setValue(_PARAM_PREFIX + "user_hint_h", params.user_hint[1])
    else:
        settings.remove(_PARAM_PREFIX + "user_hint_w")
        settings.remove(_PARAM_PREFIX + "user_hint_h")


def load_params() -> PipelineParams:
    """从 QSettings 加载 PipelineParams，无保存值时使用默认值。"""
    settings = QSettings()
    defaults = PipelineParams()
    kwargs = {}
    for field in _PARAM_FIELDS:
        key = _PARAM_PREFIX + field
        if settings.contains(key):
            val = settings.value(key)
            tp = _FIELD_TYPES[field]
            # QSettings 可能返回字符串，需类型转换
            if tp is bool:
                kwargs[field] = val in (True, "true", "True", 1, "1")
            elif tp is int:
                kwargs[field] = int(val)
            elif tp is float:
                kwargs[field] = float(val)
            else:
                kwargs[field] = val
        else:
            kwargs[field] = getattr(defaults, field)
    # user_hint
    w_key = _PARAM_PREFIX + "user_hint_w"
    h_key = _PARAM_PREFIX + "user_hint_h"
    if settings.contains(w_key) and settings.contains(h_key):
        kwargs["user_hint"] = (int(settings.value(w_key)), int(settings.value(h_key)))
    return PipelineParams(**kwargs)


def reset_params() -> None:
    """清除已保存的参数，恢复默认。"""
    settings = QSettings()
    for field in _PARAM_FIELDS:
        settings.remove(_PARAM_PREFIX + field)
    settings.remove(_PARAM_PREFIX + "user_hint_w")
    settings.remove(_PARAM_PREFIX + "user_hint_h")


def save_preference(key: str, value: Any) -> None:
    """保存全局偏好设置。"""
    settings = QSettings()
    settings.setValue(_PREF_PREFIX + key, value)


def load_preference(key: str, default: Any = None) -> Any:
    """加载全局偏好设置，无值时返回 default。"""
    settings = QSettings()
    full_key = _PREF_PREFIX + key
    if settings.contains(full_key):
        return settings.value(full_key)
    return _DEFAULT_PREFS.get(key, default)


def reset_preferences() -> None:
    """清除所有偏好设置。"""
    settings = QSettings()
    for key in _DEFAULT_PREFS:
        settings.remove(_PREF_PREFIX + key)
