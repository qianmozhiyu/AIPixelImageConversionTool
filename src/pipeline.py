"""AI 像素图像转换主流水线。

将降噪、放大/缩小、锐化、网格检测与调色板量化等核心模块串联为一条完整流水线，
支持整图一键运行以及按阶段增量执行（含前置依赖自动补齐与缓存失效重跑）。

阶段顺序：``load → denoise_global → resize → grid_detect → extract → palette_refine``

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
from .core.aa_removal import remove_anti_aliasing as _remove_anti_aliasing
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


def _mean_grad_mag(img: np.ndarray) -> float:
    """图像梯度幅值均值（|grad| 在 x/y 方向的平均），用于边缘能量估计。"""
    a = np.asarray(img, dtype=np.float64)
    gx = np.abs(np.diff(a, axis=1))  # (H, W-1, C)
    gy = np.abs(np.diff(a, axis=0))  # (H-1, W, C)
    return float((gx.sum() + gy.sum()) / (gx.size + gy.size))


def _oklab_median_cut_palette(samples_rgb: np.ndarray, n_colors: int) -> np.ndarray:
    """在 OKLab 空间对样本做 median-cut 聚类，返回簇代表色调色板。

    与 ``core.color.extract_palette`` 的 RGB 中位切分不同，聚类在感知均匀的
    OKLab 空间进行（等亮度异色可分），每个簇的代表色取**原始 RGB 均值**，
    保证检测信号最终以 RGB 返回时色彩真实。

    Args:
        samples_rgb: ``(N, 3)`` RGB 0-255 样本数组。
        n_colors: 目标调色板颜色数（实际簇数可因方差退化而更少）。

    Returns:
        ``(M, 3)`` uint8 调色板数组，``M <= n_colors``。
    """
    from .core.color import rgb_to_oklab

    samples_rgb = np.asarray(samples_rgb, dtype=np.float64)
    n = samples_rgb.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    # OKLab 三通道统一系数缩放并平移到 [0,255] 附近（L 原始 [0,1]，
    # a/b 原始约 [-0.4,0.4]；统一系数 255 保持 OKLab 欧氏度量的各向同性，
    # 平移不影响基于方差/排序的中位切分）
    oklab = rgb_to_oklab(samples_rgb)
    scaled = oklab * 255.0 + np.array([0.0, 127.5, 127.5])

    # 迭代 median-cut：优先分割方差最大的段（每轮对半分样本数）。
    # 每段缓存各通道 sum/sumsq，方差用 E[x²]-E[x]² 增量维护，避免每轮
    # 对全部样本重新做 fancy index + var 的大样本开销
    def _seg_stats(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = scaled[seg]
        return sub.sum(axis=0), (sub * sub).sum(axis=0)

    seg0 = np.arange(n)
    s0, ss0 = _seg_stats(seg0)
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [(seg0, s0, ss0)]
    while len(segments) < n_colors:
        best_i, best_var, best_ch = -1, -1.0, 0
        for i, (seg, s, ss) in enumerate(segments):
            m = seg.shape[0]
            if m <= 1:
                continue
            var = ss / m - (s / m) ** 2
            ch = int(np.argmax(var))
            if var[ch] > best_var:
                best_i, best_var, best_ch = i, float(var[ch]), ch
        if best_i < 0 or best_var < 1e-12:
            break  # 所有段方差退化（同色簇），无法继续分割
        seg, _, _ = segments.pop(best_i)
        order = np.argsort(scaled[seg][:, best_ch])  # 快排即可，等值分割位置不影响簇质量
        seg_sorted = seg[order]
        sub_sorted = scaled[seg_sorted]
        mid = seg_sorted.shape[0] // 2
        left, right = seg_sorted[:mid], seg_sorted[mid:]
        lsub, rsub = sub_sorted[:mid], sub_sorted[mid:]
        segments.insert(best_i, (right, rsub.sum(axis=0), (rsub * rsub).sum(axis=0)))
        segments.insert(best_i, (left, lsub.sum(axis=0), (lsub * lsub).sum(axis=0)))

    # 每簇取原始 RGB 均值为代表色（色彩真实）
    palette = np.stack([
        samples_rgb[seg].mean(axis=0) for seg, _, _ in segments if seg.shape[0] > 0
    ])
    return np.clip(np.round(palette), 0, 255).astype(np.uint8)


def _quantize_detection_signal(img: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """对检测信号做 OKLab 空间小调色板量化，仅用于网格检测（不改变提取信号）。

    流程：全图 1/16 子采样 → 样本转 OKLab 做 median-cut 聚类（感知空间
    聚类使等亮度异色块可分）→ 每簇取原始 RGB 均值为调色板色 → 全图在
    OKLab 空间用 cKDTree 最近邻映射。量化把 AA 过渡带阶跃化（块边界
    色差能量集中）、块内渐变坍缩为零梯度，提升网格检测的边界可检测性。

    Args:
        img: ``(H, W, 3)`` RGB 0-255 图像数组。
        n_colors: 调色板颜色数。

    Returns:
        ``(H, W, 3)`` uint8 量化后图像；退化输入（非 3 通道/空图）原样返回。
    """
    from scipy.spatial import cKDTree

    from .core.color import rgb_to_oklab

    u8 = np.clip(np.asarray(img, dtype=np.float64), 0, 255).astype(np.uint8)
    if u8.ndim != 3 or u8.shape[2] != 3 or u8.shape[0] == 0 or u8.shape[1] == 0:
        return u8  # 退化输入安全回退
    H, W = u8.shape[:2]
    n_colors = max(1, int(n_colors))

    # 子采样（约 1/16，2K 图约 26 万样本）构建调色板
    pixels = u8.reshape(-1, 3)
    samples = pixels[::16]
    if samples.shape[0] == 0:
        samples = pixels  # 极小图回退全采样
    palette = _oklab_median_cut_palette(samples, n_colors)

    # 全图 OKLab 最近邻映射（float32 + 多线程 query 加速，最近邻判定精度足够；
    # 直接传 u8 由 rgb_to_oklab 内部一次性转 float64，避免额外 100MB 副本）
    oklab_full = rgb_to_oklab(u8).reshape(-1, 3)
    tree = cKDTree(rgb_to_oklab(palette.astype(np.float64)))
    indices = tree.query(oklab_full.astype(np.float32), workers=-1)[1]
    return palette[indices].reshape(H, W, 3)


@dataclass
class PipelineParams:
    """流水线参数。

    Attributes:
        enable_ai_denoise: 是否启用图像级 AI 去噪。
        ai_denoise_method: 去噪方法，``"none"``/``"nl_means"``/``"tv_chambolle"``/``"bilateral"``。
        ai_denoise_strength: 去噪强度，0.0-1.0。
        enable_clahe: 降噪后是否启用 CLAHE 局部对比度增强。
        clahe_clip_limit: CLAHE 裁剪限制（0.01-0.1）。
        enable_upscale: 是否在降噪后启用放大以提升网格检测分辨率。
        upscale_factor: 放大倍数（默认 2）。
        upscale_method: 放大算法，``"nearest"``/``"bilinear"``/``"bicubic"``/``"lanczos"``。
        enable_sharpen: 是否对放大结果做 unsharp mask 锐化（默认关闭，因有白边风险）。
        sharpen_strength: 锐化强度，0.0-1.0。
        enable_aa_removal: 降噪后是否启用抗锯齿消除预处理（默认关闭）。
        aa_removal_passes: AA 消除迭代次数。
        aa_removal_threshold: AA 消除两主色距离阈值。
        denoise_grid_guard: 降噪-检测耦合保护：去噪后若边缘能量衰减过度，
            自动减半强度重去噪一次（默认关闭）。
        min_p: 网格检测最小候选周期。
        max_p: 网格检测最大候选周期。
        detect_signal: 网格检测信号模式，``"gray"``（BT.601 灰度）或
            ``"oklab"``（OKLAB 感知色差，等亮度异色块边界可见），默认 gray。
        enable_pre_quantize: 网格检测前是否对检测信号做小调色板预量化（默认关闭）。
        user_hint: 用户给定的逻辑分辨率提示 ``(w, h)``，非 None 时优先使用。
        phase_step: 相位扫描步长。
        snr_threshold: 网格检测 SNR 阈值，低于此值判定为无网格。
        edge_search_tolerance: 共享边界搜索半径（像素），透传给 ``detect`` 的 ``edge_tol``。
        enable_subpixel_refine: 是否启用亚像素精炼。
        smooth_strength: 全局正则化混合强度（0.0=纯观测，1.0=完全用全局线性模型）。
        outlier_reject_ratio: 网格检测离群间距剔除阈值比例。
        enable_peak_lattice_fit: 投票周期确定后用峰值格点拟合精化为浮点周期，
            支持非整数块尺寸如 7.5px；失败自动回退投票值；附轴一致性防护
            （精化前两轴一致而精化后分裂 >2% 时整体回退投票值），默认开启。
        enable_comb_energy_score: 梳状能量集中度终审（G2）：对投影信号做
            连续 (pitch, phase) 梳状打分，原理性压制子谐波/倍频误检（子谐波
            覆盖惩罚翻倍、倍频能量减半，必然低于真周期）；低置信自动回退
            投票结果（默认开启）。
        jpeg_grid_guard: JPEG 8×8 压缩网格检测与候选降权（G5）：检测 JPEG
            8×8 DCT 网格并对 8/16/24px 附近候选降权，防护压缩伪影周期误检
            （默认开启）。
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
    enable_upscale: bool = False               # 降噪后是否放大
    upscale_factor: int = 2                    # 放大倍数
    upscale_method: str = "nearest"          # 放大算法："nearest"/"bilinear"/"bicubic"/"lanczos"
    enable_sharpen: bool = False               # 是否启用 unsharp mask 锐化（默认关闭）
    sharpen_strength: float = 0.5              # 锐化强度 0.0-1.0
    # 抗锯齿消除（默认关闭）
    enable_aa_removal: bool = False            # 降噪后是否启用 AA 消除预处理
    aa_removal_passes: int = 2                 # AA 消除迭代次数
    aa_removal_threshold: float = 0.5          # AA 消除两主色距离阈值
    # 去噪-检测耦合保护（默认关闭）
    denoise_grid_guard: bool = False           # 去噪过强时自动减半强度重去噪一次
    # 网格检测
    min_p: int = 3
    max_p: int = 40
    detect_signal: str = "gray"          # 检测信号模式："gray"/"oklab"
    enable_pre_quantize: bool = False          # 检测前对检测信号做小调色板预量化
    user_hint: Optional[tuple[int, int]] = None
    phase_step: float = 0.1
    snr_threshold: float = 8.0           # 网格检测 SNR 阈值
    edge_search_tolerance: int = 3          # 共享边界搜索半径（像素）
    enable_subpixel_refine: bool = True     # 是否启用亚像素精炼
    smooth_strength: float = 0.5            # 全局平滑约束强度（0.0-1.0）
    outlier_reject_ratio: float = 0.5             # 网格检测离群间距剔除阈值比例
    enable_peak_lattice_fit: bool = True     # 投票后峰值格点拟合周期精化（默认开启，含轴一致性防护）
    enable_comb_energy_score: bool = True    # 梳状能量集中度周期终审（默认开启）
    jpeg_grid_guard: bool = True             # JPEG 8×8 网格检测与候选降权（默认开启）
    # 块提取
    extract_method: str = "median"              # "median"/"mean"/"mode"/"kmeans"
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
        "load", "denoise_global", "resize", "grid_detect", "extract", "palette_refine",
    )
    PREREQS = {
        "load": (),
        "denoise_global": ("load",),
        "resize": ("denoise_global",),
        "grid_detect": ("resize",),
        "extract": ("resize", "grid_detect"),
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
            # 去噪-检测耦合保护：去噪过度破坏网格结构时，减半强度重去噪一次
            if p.denoise_grid_guard and p.ai_denoise_strength > 0:
                r = _mean_grad_mag(denoised) / max(_mean_grad_mag(img), 1e-9)
                if r < 0.4:
                    denoised = denoise_ai_noise(
                        img, method=p.ai_denoise_method,
                        strength=p.ai_denoise_strength / 2.0,
                    )
        else:
            denoised = img
        # 可选抗锯齿消除（默认关闭）：清理块边界/网格线 AA 混合杂色
        if p.enable_aa_removal:
            denoised = _remove_anti_aliasing(
                denoised,
                threshold=p.aa_removal_threshold,
                passes=p.aa_removal_passes,
            )
        if p.enable_clahe and p.clahe_clip_limit > 0:
            denoised = apply_clahe(denoised, clip_limit=p.clahe_clip_limit)
        self._cache["denoise_global"] = denoised
        self._stage_done.add("denoise_global")

    def _run_resize(self) -> None:
        img = self._cache["denoise_global"]
        p = self.params
        # 顺序：先放大（提升网格检测分辨率），再可选锐化
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
        # 可选锐化（unsharp mask），仍在放大后
        if p.enable_sharpen and p.sharpen_strength > 0:
            result = _unsharp_mask(result, strength=p.sharpen_strength)
        self._cache["resize"] = result
        self._stage_done.add("resize")

    def _run_grid_detect(self) -> None:
        img = self._cache["resize"]
        p = self.params

        # oklab 色差模式：RGB 直传（预量化若开启则先量化再传），色差信号在
        # detect 内部计算；2D 输入或 gray 模式保持原灰度路径不变
        use_oklab = p.detect_signal == "oklab" and img.ndim == 3
        if use_oklab:
            detect_input = _quantize_detection_signal(img, n_colors=16) if p.enable_pre_quantize else img
            # stages 缓存 "gray" 语义与灰度模式一致：缓存检测信号的 BT.601 灰度
            gray = to_gray(detect_input)
            signal = "oklab"
        else:
            gray = to_gray(img)
            # 可选预量化：仅对"检测信号"做小调色板量化，提升低对比度边缘可检测性
            if p.enable_pre_quantize:
                gray = to_gray(_quantize_detection_signal(img, n_colors=16))
            detect_input = gray
            signal = "gray"

        if self._user_grid:
            w, h = self._user_grid
            grid = _grid_detect_user(detect_input, w, h, step=p.phase_step, signal=signal)
        elif p.user_hint:
            grid = _grid_detect_user(detect_input, *p.user_hint, step=p.phase_step, signal=signal)
        else:
            grid = _grid_detect(
                detect_input, min_p=p.min_p, max_p=p.max_p, step=p.phase_step,
                snr_threshold=p.snr_threshold,
                edge_tol=p.edge_search_tolerance,
                enable_subpixel_refine=p.enable_subpixel_refine,
                smooth_strength=p.smooth_strength,
                outlier_reject_ratio=p.outlier_reject_ratio,
                signal=signal,
                enable_peak_lattice_fit=p.enable_peak_lattice_fit,
                enable_comb_energy_score=p.enable_comb_energy_score,
                jpeg_grid_guard=p.jpeg_grid_guard,
            )

        # grid 坐标在 resize 图坐标系，extract 也用 resize 图，无需映射回原图
        self._cache["grid_detect"] = grid
        self._cache["gray"] = gray  # 复用已计算的灰度，避免重复 to_gray（B12）
        self._stage_done.add("grid_detect")

    def _run_extract(self) -> None:
        img = self._cache["resize"]
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
            "low_confidence": grid.low_confidence,
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
            "resize": self._run_resize,
            "grid_detect": self._run_grid_detect,
            "extract": self._run_extract,
            "palette_refine": self._run_palette_refine,
        }
        runners[stage_name]()
