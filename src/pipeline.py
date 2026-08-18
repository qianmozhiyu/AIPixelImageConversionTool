"""AI 像素图像转换主流水线。

将降噪、放大、锐化、网格检测与调色板量化等核心模块串联为一条完整流水线，
支持整图一键运行以及按阶段增量执行（含前置依赖自动补齐与缓存失效重跑）。

阶段顺序：``load → denoise_global → upscale → grid_detect → extract → palette_refine``

主要接口：
- ``PipelineParams``：流水线参数。
- ``PipelineResult``：流水线运行结果。
- ``Pipeline``：流水线主体，支持 ``run`` 全量运行与 ``run_stage`` 单阶段运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .utils import normalize_image
from .core.io import to_gray
from .core.denoise import denoise_ai_noise, apply_clahe
from .core.grid_detect import (
    Grid,
    detect as _grid_detect,
    detect_with_user_grid as _grid_detect_user,
)
from skimage.transform import resize as _sk_resize
from .core.extract import extract_blocks as _extract_blocks
from .core.color_quantize import color_quantize as _color_quantize


def _unsharp_mask(img: np.ndarray, strength: float = 0.5, radius: int = 1) -> np.ndarray:
    """Unsharp mask 锐化。strength 控制锐化强度（0.0-1.0）。

    使用高斯模糊作为低通分量，将原始图像与其差值（高频细节）按强度叠加回原图。
    对 3D (H,W,C) 数组按通道分别滤波，避免沿通道轴混合。

    Args:
        img: ``(H, W, 3)`` 或 ``(H, W)`` 图像数组。
        strength: 锐化强度（0.0-1.0）。
        radius: 高斯半径（像素）。

    Returns:
        锐化后的 ``(H, W[, 3])`` float64 数组，范围裁剪到 [0, 255]。
    """
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        blurred = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            blurred[:, :, c] = gaussian_filter(arr[:, :, c], sigma=radius, mode="reflect")
    else:
        blurred = gaussian_filter(arr, sigma=radius, mode="reflect")
    sharpened = arr + strength * (arr - blurred)
    return np.clip(sharpened, 0, 255).astype(np.float64)


@dataclass
class PipelineParams:
    """流水线参数。

    Attributes:
        enable_ai_denoise: 是否启用图像级 AI 去噪。
        ai_denoise_method: 去噪方法，``"none"``/``"nl_means"``/``"tv_chambolle"``/``"bilateral"``。
        ai_denoise_strength: 去噪强度，0.0-1.0。
        enable_clahe: 降噪后是否启用 CLAHE 局部对比度增强。
        clahe_clip_limit: CLAHE 裁剪限制（0.01-0.1）。
        enable_upscale: 是否在降噪后启用双线性放大以提升网格检测分辨率。
        upscale_factor: 放大倍数（默认 2）。
        upscale_method: 放大算法，``"nearest"``/``"bilinear"``/``"bicubic"``/``"lanczos"``。
        enable_sharpen: 是否对放大结果做 unsharp mask 锐化（默认关闭，因有白边风险）。
        sharpen_strength: 锐化强度，0.0-1.0。
        min_p: 网格检测最小候选周期。
        max_p: 网格检测最大候选周期。
        user_hint: 用户给定的逻辑分辨率提示 ``(w, h)``，非 None 时优先使用。
        phase_step: 相位扫描步长。
        snr_threshold: 网格检测 SNR 阈值，低于此值判定为无网格。
        edge_search_tolerance: 共享边界搜索半径（像素），透传给 ``detect`` 的 ``edge_tol``。
        enable_subpixel_refine: 是否启用亚像素精炼。
        smooth_strength: 全局正则化混合强度（0.0=纯观测，1.0=完全用全局线性模型）。
        outlier_reject_ratio: 网格检测离群间距剔除阈值比例。
        extract_method: 块提取代表色算法，``"median"``/``"mean"``/``"mode"``/``"kmeans"``。
        extract_core_ratio: 块核心区采样比例（0.5-1.0），规避边缘杂色。
        fix_square: 当逻辑分辨率与正方形差 1 时，自动修正为正方形输出。
        enable_palette_refine: 是否对提取结果做 K-means 调色板精炼。
        palette_colors: 调色板精炼目标色数，默认 16。
    """

    # 降噪
    enable_ai_denoise: bool = True
    ai_denoise_method: str = "nl_means"        # "none"/"nl_means"/"tv_chambolle"
    ai_denoise_strength: float = 0.5           # 0.0-1.0
    enable_clahe: bool = False               # 降噪后是否启用 CLAHE 局部对比度增强
    clahe_clip_limit: float = 0.03           # CLAHE 裁剪限制（0.01-0.1）
    # 放大与锐化
    enable_upscale: bool = False               # 降噪后是否双线性放大
    upscale_factor: int = 2                    # 放大倍数
    upscale_method: str = "bilinear"          # 放大算法："nearest"/"bilinear"/"bicubic"/"lanczos"
    enable_sharpen: bool = False               # 是否启用 unsharp mask 锐化（默认关闭）
    sharpen_strength: float = 0.5              # 锐化强度 0.0-1.0
    # 网格检测
    min_p: int = 3
    max_p: int = 40
    user_hint: Optional[tuple[int, int]] = None
    phase_step: float = 0.1
    snr_threshold: float = 8.0           # 网格检测 SNR 阈值
    edge_search_tolerance: int = 3          # 共享边界搜索半径（像素）
    enable_subpixel_refine: bool = True     # 是否启用亚像素精炼
    smooth_strength: float = 0.5            # 全局平滑约束强度（0.0-1.0）
    outlier_reject_ratio: float = 0.5             # 网格检测离群间距剔除阈值比例
    # 块提取
    extract_method: str = "kmeans"              # "median"/"mean"/"mode"/"kmeans"
    extract_core_ratio: float = 0.6            # 0.5-1.0
    # 正方形修正
    fix_square: bool = False
    # 调色板精炼
    enable_palette_refine: bool = True
    palette_colors: int = 16


@dataclass
class PipelineResult:
    """流水线运行结果。

    Attributes:
        pixel_art: ``(h, w, 3)`` uint8 像素图。
        grid: 像素网格检测结果。
        metadata: 元信息字典。
        stages: 各阶段缓存结果字典。
    """

    pixel_art: np.ndarray       # (h, w, 3) uint8
    grid: Grid
    metadata: dict
    stages: dict


class Pipeline:
    """AI 像素图像转换流水线。

    按固定阶段顺序处理图像，各阶段结果缓存于 ``_cache``，已完成阶段记录于
    ``_stage_done``。支持整图运行（``run``）、单阶段运行（``run_stage``）、
    按阶段失效重跑（``reset_from``）以及用户网格覆盖（``set_user_grid``）。
    """

    STAGES = (
        "load", "denoise_global", "upscale", "grid_detect", "extract", "palette_refine",
    )
    PREREQS = {
        "load": (),
        "denoise_global": ("load",),
        "upscale": ("denoise_global",),
        "grid_detect": ("upscale",),
        "extract": ("upscale", "grid_detect"),
        "palette_refine": ("extract",),
    }

    def __init__(
        self,
        params: PipelineParams | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        self.params = params or PipelineParams()
        self._progress = progress_callback
        self._cache: dict[str, Any] = {}
        self._stage_done: set[str] = set()
        self._user_grid: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # 整图运行
    # ------------------------------------------------------------------
    def run(self, image) -> PipelineResult:
        """整图一键运行全部阶段并返回结果。

        Args:
            image: PIL Image 或 ``(H, W, 3)`` numpy 数组。

        Returns:
            PipelineResult: 流水线运行结果。
        """
        self._cache = {"image": normalize_image(image)}
        self._stage_done = set()
        # 注意：不重置 _user_grid —— set_user_grid() 是显式用户指定，
        # run() 全量重跑时应保留（B11 修复，避免用户网格静默丢失）
        n = len(self.STAGES)
        for i, stage in enumerate(self.STAGES):
            self._emit_progress(stage, i / n)
            self._run_single(stage)
        self._emit_progress("done", 1.0)
        return self._build_result()

    def _emit_progress(self, stage: str, percent: float) -> None:
        """若设置了进度回调，则上报当前阶段与进度百分比。"""
        if self._progress:
            self._progress(stage, percent)

    # ------------------------------------------------------------------
    # 各阶段执行器
    # ------------------------------------------------------------------
    def _run_load(self) -> None:
        # cache["image"] 已在 run()/run_stage(image=...) 中设置，此处仅标记完成。
        self._stage_done.add("load")

    def _run_denoise_global(self) -> None:
        img = self._cache["image"]
        p = self.params
        if p.enable_ai_denoise:
            denoised = denoise_ai_noise(
                img, method=p.ai_denoise_method, strength=p.ai_denoise_strength
            )
        else:
            denoised = img
        if p.enable_clahe and p.clahe_clip_limit > 0:
            denoised = apply_clahe(denoised, clip_limit=p.clahe_clip_limit)
        self._cache["denoise_global"] = denoised
        self._stage_done.add("denoise_global")

    def _run_upscale(self) -> None:
        img = self._cache["denoise_global"]
        p = self.params
        if p.enable_upscale and p.upscale_factor > 1:
            H, W = img.shape[:2]
            f = int(p.upscale_factor)
            if p.upscale_method == "nearest":
                # 最近邻：np.repeat 无插值开销，保留锐利边缘
                result = np.repeat(np.repeat(img, f, axis=0), f, axis=1)
            else:
                order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3, "lanczos": 5}
                order = order_map.get(p.upscale_method, 1)
                result = _sk_resize(
                    img, (H * f, W * f), order=order, anti_aliasing=False, preserve_range=True
                )
        else:
            result = img
        # 可选锐化（unsharp mask）
        if p.enable_sharpen and p.sharpen_strength > 0:
            result = _unsharp_mask(result, strength=p.sharpen_strength)
        self._cache["upscale"] = result
        self._stage_done.add("upscale")

    def _run_grid_detect(self) -> None:
        img = self._cache["upscale"]
        p = self.params

        gray = to_gray(img)

        if self._user_grid:
            w, h = self._user_grid
            grid = _grid_detect_user(gray, w, h, step=p.phase_step)
        elif p.user_hint:
            grid = _grid_detect_user(gray, *p.user_hint, step=p.phase_step)
        else:
            grid = _grid_detect(
                gray, min_p=p.min_p, max_p=p.max_p, step=p.phase_step,
                snr_threshold=p.snr_threshold,
                edge_tol=p.edge_search_tolerance,
                enable_subpixel_refine=p.enable_subpixel_refine,
                smooth_strength=p.smooth_strength,
                outlier_reject_ratio=p.outlier_reject_ratio,
            )

        # grid 坐标在放大图坐标系，extract 也用放大图，无需映射回原图
        self._cache["grid_detect"] = grid
        self._cache["gray"] = gray  # 复用已计算的灰度，避免重复 to_gray（B12）
        self._stage_done.add("grid_detect")

    def _run_extract(self) -> None:
        img = self._cache["upscale"]
        grid = self._cache["grid_detect"]
        p = self.params
        extracted = _extract_blocks(
            img, grid,
            method=p.extract_method,
            core_ratio=p.extract_core_ratio,
        )
        # 正方形修正
        if p.fix_square:
            extracted = self._apply_fix_square(extracted)
        self._cache["extract"] = extracted
        self._stage_done.add("extract")

    def _run_palette_refine(self) -> None:
        """对提取后的像素图做 K-means 调色板精炼，确保全局色彩一致。"""
        img = self._cache["extract"]
        p = self.params
        if p.enable_palette_refine:
            # 唯一色数 <= palette_colors 时跳过（避免过度量化）
            unique_colors = int(np.unique(img.reshape(-1, 3), axis=0).shape[0])
            if unique_colors <= p.palette_colors:
                refined = img
            else:
                refined = _color_quantize(img, n_colors=p.palette_colors)
        else:
            refined = img
        self._cache["palette_refine"] = refined
        self._stage_done.add("palette_refine")

    def _apply_fix_square(self, pixel_art: np.ndarray) -> np.ndarray:
        """当逻辑分辨率与正方形差 1 时，裁剪为正方形输出。

        Args:
            pixel_art: (h, w, 3) uint8 像素图。

        Returns:
            修正后的像素图（裁剪一行或一列）。
        """
        p = self.params
        if not p.fix_square:
            return pixel_art
        h, w = pixel_art.shape[:2]
        if abs(w - h) != 1:
            return pixel_art
        if w > h:
            # 宽比高多1：裁剪最后一列
            return pixel_art[:, :-1, :]
        else:
            # 高比宽多1：裁剪最后一行
            return pixel_art[:-1, :, :]

    # ------------------------------------------------------------------
    # 结果构建
    # ------------------------------------------------------------------
    def _build_result(self) -> PipelineResult:
        grid = self._cache["grid_detect"]
        pixel_art = np.clip(self._cache["palette_refine"], 0, 255).astype(np.uint8)
        # 计算唯一色数
        unique_colors = int(np.unique(pixel_art.reshape(-1, 3), axis=0).shape[0])
        metadata = {
            "w_logic": grid.w_logic,
            "h_logic": grid.h_logic,
            "px": grid.px,
            "py": grid.py,
            "grid_conf": grid.conf,
            "unique_colors": unique_colors,
            "extract_method": self.params.extract_method,
        }
        stages = {
            stage: self._cache[stage]
            for stage in self.STAGES
            if stage in self._cache
        }
        return PipelineResult(
            pixel_art=pixel_art,
            grid=grid,
            metadata=metadata,
            stages=stages,
        )

    # ------------------------------------------------------------------
    # 单阶段 / 增量运行
    # ------------------------------------------------------------------
    def build_result(self) -> PipelineResult:
        """构建当前缓存对应的流水线结果（用于增量重跑后获取结果）。"""
        return self._build_result()

    def run_stage(self, stage_name: str, image=None) -> Any:
        """运行单个阶段，必要时自动补齐其前置依赖。

        Args:
            stage_name: 阶段名称，需属于 ``STAGES``。
            image: 可选输入图像，提供时会重置缓存并以该图像为起点。

        Returns:
            该阶段的缓存结果。

        Raises:
            ValueError: 未知阶段名称。
        """
        if stage_name not in self.STAGES:
            raise ValueError(f"未知阶段: {stage_name!r}")
        if image is not None:
            self._cache = {"image": normalize_image(image)}
            self._stage_done = set()
        # 按 STAGES 顺序补齐未完成的前置阶段
        idx = self.STAGES.index(stage_name)
        for stage in self.STAGES[:idx]:
            if stage not in self._stage_done:
                self._run_single(stage)
        self._run_single(stage_name)
        return self._cache.get(stage_name)

    def get_result(self, stage_name: str) -> Any:
        """返回指定阶段的缓存结果，不存在时返回 None。"""
        return self._cache.get(stage_name)

    def reset_from(self, stage_name: str) -> None:
        """从指定阶段起失效缓存并重跑其后所有阶段。

        Args:
            stage_name: 起始失效阶段名称。

        Raises:
            ValueError: 未知阶段名称。
        """
        if stage_name not in self.STAGES:
            raise ValueError(f"未知阶段: {stage_name!r}")
        self._invalidate_from(stage_name)
        idx = self.STAGES.index(stage_name)
        for stage in self.STAGES[idx:]:
            self._run_single(stage)

    def set_user_grid(self, w: int, h: int) -> None:
        """设置用户指定的逻辑分辨率，并从 ``grid_detect`` 起失效后续缓存。

        Args:
            w: 逻辑宽（块数）。
            h: 逻辑高（块数）。
        """
        self._user_grid = (w, h)
        self._invalidate_from("grid_detect")

    def get_all_results(self) -> dict[str, Any]:
        """返回全部缓存的浅拷贝字典。"""
        return dict(self._cache)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _invalidate_from(self, stage_name: str) -> None:
        """从指定阶段起移除缓存与完成标记。"""
        idx = self.STAGES.index(stage_name)
        for stage in self.STAGES[idx:]:
            self._cache.pop(stage, None)
            self._stage_done.discard(stage)

    def _run_single(self, stage_name: str) -> None:
        """分发到对应阶段执行器。"""
        runners = {
            "load": self._run_load,
            "denoise_global": self._run_denoise_global,
            "upscale": self._run_upscale,
            "grid_detect": self._run_grid_detect,
            "extract": self._run_extract,
            "palette_refine": self._run_palette_refine,
        }
        runners[stage_name]()
