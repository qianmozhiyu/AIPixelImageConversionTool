"""像素块提取模块。

根据网格检测结果（含相位偏移）对每个块的核心区采样代表色，直接输出
逻辑分辨率的真像素图，合并原 block_refine 与 downscale 的功能。

核心区采样策略规避块边缘杂色（抗锯齿、JPEG 伪影、相邻块渗透），
代表色算法支持中位数/均值/众数/主色。
"""

from __future__ import annotations

import numpy as np

from .grid_detect import Grid

# 优先使用 OKLab 感知空间，未就绪时回退 Lab（两者均可用时 dominant 取 OKLab）
try:
    from .color import rgb_to_oklab as _rgb_to_oklab
except ImportError:  # pragma: no cover - OKLab 未就绪的回退路径
    _rgb_to_oklab = None
from .color import rgb_to_lab as _rgb_to_lab


def extract_blocks(
    img: np.ndarray,
    grid: Grid,
    method: str = "median",
    core_ratio: float = 0.6,
) -> np.ndarray:
    """将每个块的核心区代表色提取为 1 个像素，输出真像素图。

    根据网格相位对齐块边界，取块内核心区（居中 ``core_ratio`` 比例区域）
    的代表色作为该块的像素值，直接输出 ``(h_logic, w_logic, 3)`` uint8 图像。

    均匀网格（phase+等距，或等距的逐交点坐标）且 ``method`` 为
    ``"median"``/``"mean"`` 时优先走向量化快速路径（逐行带 + 按块宽分组
    ``np.take`` + 多轴一次 reduce），与循环实现逐位一致；其余场景
    （不均匀网格、``"mode"``/``"kmeans"``/``"dominant"``、边界退化）
    回退逐块循环路径 ``_extract_blocks_loop``。

    Args:
        img: ``(H, W, 3)`` float64 RGB 0-255 输入图像。
        grid: 网格检测结果，提供块尺寸、相位与逻辑分辨率。
        method: 代表色算法，``"median"`` / ``"mean"`` / ``"mode"`` / ``"kmeans"`` /
            ``"dominant"``，未知回退 median。
            ``"kmeans"`` 用中位数估计主色并以距离阈值分离渗透色，取主色像素均值。
            ``"dominant"`` 把核心区像素转到感知空间（优先 OKLab，回退 Lab），
            对小步长量化 bin 取众数，再求该 bin 内原始 RGB 像素的均值作为主色。
        core_ratio: 核心区边长占块边长比例，用于规避边缘杂色。

    Returns:
        ``(h_logic, w_logic, 3)`` uint8 像素图。
    """
    img = np.asarray(img, dtype=np.float64)
    method = method if method in ("median", "mean", "mode", "kmeans", "dominant") else "median"
    # 快速路径：均匀网格 + median/mean（两算法结果仅由像素集合决定，向量化
    # reduce 与循环逐位一致）；触发条件不满足时返回 None，回退循环路径
    if method in ("median", "mean"):
        fast = _extract_uniform_fast(img, grid, method, core_ratio)
        if fast is not None:
            return fast
    return _extract_blocks_loop(img, grid, method, core_ratio)


def _extract_uniform_fast(
    img: np.ndarray,
    grid: Grid,
    method: str,
    core_ratio: float,
) -> np.ndarray | None:
    """均匀网格 ``median``/``mean`` 的向量化快速路径。

    触发条件（全部满足，否则返回 None 回退循环路径）：

    - ``method`` 为 ``"median"`` 或 ``"mean"``；
    - 网格均匀：``cell_ys``/``cell_xs`` 为 None（phase + 等距模型），或逐交点
      坐标 round 后沿行/列方向一致（各列的 y 边界、各行的 x 边界相同）且
      相邻间距差异 < 1e-6（等距网格，含 ``_equidistant_cell_grid`` 兜底输出）；
    - round + 钳位 [0, H/W] 后的边界序列严格递增：此时循环里的边界去重
      （cummax 语义）与空块兜底分支均无操作，边界序列直接可用；任何零宽/
      负宽/重叠块（含末块越界被钳位截断为空）都整体回退循环，保守正确性优先。

    实现按行带（row band）处理：第 j 行取核心行带（y 向居中裁剪一次），
    x 向按块宽分组（等距网格 round 后通常仅 1-2 种宽度），组内各块核心列
    窗口用 ``np.take`` 拉取为 C 连续数组（浮点 reduce 顺序与循环
    ``core.reshape(-1, 3)`` 逐位一致），再对 ``(cy_h, n_g, cw_g, 3)``
    做一次多轴 ``np.median``/``np.mean``。

    Args:
        img: ``(H, W, 3)`` float64 RGB 0-255 输入图像。
        grid: 网格检测结果。
        method: 代表色算法，仅接受 ``"median"``/``"mean"``。
        core_ratio: 核心区边长占块边长比例。

    Returns:
        ``(h_logic, w_logic, 3)`` uint8 像素图；不满足触发条件时返回 None。
    """
    if method not in ("median", "mean"):
        return None
    h_logic = int(grid.h_logic)
    w_logic = int(grid.w_logic)
    if h_logic < 1 or w_logic < 1:
        return None

    H, W = img.shape[:2]
    px = float(grid.px)
    py = float(grid.py)
    use_cells = grid.cell_ys is not None and grid.cell_xs is not None

    if use_cells:
        cell_ys = np.asarray(grid.cell_ys, dtype=np.float64)
        cell_xs = np.asarray(grid.cell_xs, dtype=np.float64)
        # 形状异常（非 (h+1, w+1)）交回循环路径，保持原有行为
        if cell_ys.shape != (h_logic + 1, w_logic + 1):
            return None
        if cell_xs.shape != (h_logic + 1, w_logic + 1):
            return None
        # round 后各行/各列一致：循环逐交点取整结果与本路径提取的边界序列
        # 相同（浮点微差在 .5 边界处可能翻转 round，须显式检查而非容差近似）
        ys_round = np.rint(cell_ys)
        xs_round = np.rint(cell_xs)
        # np.array_equal 不做广播比较，此处用 ptp（极差）显式检查行/列内一致
        if np.any(np.ptp(ys_round, axis=1) != 0):
            return None
        if np.any(np.ptp(xs_round, axis=0) != 0):
            return None
        ys = cell_ys[:, 0]
        xs = cell_xs[0, :]
        # 等距检查：相邻间距差异 < 1e-6
        dy = np.diff(ys)
        dx = np.diff(xs)
        if dy.size > 0 and float(dy.max() - dy.min()) >= 1e-6:
            return None
        if dx.size > 0 and float(dx.max() - dx.min()) >= 1e-6:
            return None
    else:
        ys = float(grid.phase_y) + np.arange(h_logic + 1, dtype=np.float64) * py
        xs = float(grid.phase_x) + np.arange(w_logic + 1, dtype=np.float64) * px

    # 整数边界：与循环相同的 round（half-to-even）+ 钳位 [0, H/W]
    by = np.clip(np.rint(ys).astype(np.int64), 0, H)
    bx = np.clip(np.rint(xs).astype(np.int64), 0, W)
    # 严格递增检查：零宽/负宽/重叠（去重或空块兜底会被触发）时回退循环
    if np.any(np.diff(by) < 1) or np.any(np.diff(bx) < 1):
        return None

    # 核心区标称尺寸（与循环相同的计算：基于块标称尺寸）
    cw = max(1, int(round(px * core_ratio)))
    ch = max(1, int(round(py * core_ratio)))

    out = np.zeros((h_logic, w_logic, 3), dtype=np.uint8)

    # x 向按块宽分组（与行无关，预计算一次）：组内块共享宽度与核心列偏移，
    # 可统一 gather 后一次多轴 reduce
    x0 = int(bx[0])
    band_w = int(bx[-1]) - x0
    widths = np.diff(bx)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for w_val in np.unique(widths):
        w_i = int(w_val)
        idx = np.where(widths == w_val)[0]
        cw_g = min(cw, w_i)
        off = (w_i - cw_g) // 2
        starts = bx[idx] + off - x0
        # 核心列索引矩阵 (n_g, cw_g)：绝对列 bx[i]+off+k 的 band 内相对值
        col_idx = starts[:, None] + np.arange(cw_g, dtype=np.int64)[None, :]
        groups.append((idx, col_idx))

    # 逐行带处理：核心行带 y 向居中裁剪一次，行内所有块共享
    for j in range(h_logic):
        y0 = int(by[j])
        y1 = int(by[j + 1])
        bh_j = y1 - y0
        cy_h = min(ch, bh_j)
        cy0 = y0 + (bh_j - cy_h) // 2
        band = img[cy0 : cy0 + cy_h, x0 : x0 + band_w, :]
        for idx, col_idx in groups:
            # np.take 产出 C 连续 (cy_h, n_g, cw_g, 3)，块内元素顺序与循环
            # core.reshape(-1, 3) 相同，median/mean 多轴 reduce 逐位一致
            win = np.take(band, col_idx, axis=1)
            if method == "median":
                rep = np.median(win, axis=(0, 2))
            else:
                rep = np.mean(win, axis=(0, 2))
            out[j, idx, :] = np.clip(rep, 0, 255).astype(np.uint8)

    return out


def _extract_blocks_loop(
    img: np.ndarray,
    grid: Grid,
    method: str = "median",
    core_ratio: float = 0.6,
) -> np.ndarray:
    """块提取的逐块循环实现（原 ``extract_blocks`` 主体，供回退与对拍）。

    边界去重/钳位/核心区居中/代表色计算/空块兜底语义与历史版本完全一致；
    均匀网格 + median/mean 场景由 ``_extract_uniform_fast`` 逐位等价加速。
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape[:2]

    px = float(grid.px)
    py = float(grid.py)
    phase_x = float(grid.phase_x)
    phase_y = float(grid.phase_y)
    w_logic = int(grid.w_logic)
    h_logic = int(grid.h_logic)

    # 是否使用逐交点网格坐标（支持局部漂移）
    use_cells = grid.cell_ys is not None and grid.cell_xs is not None

    # 核心区尺寸（基于块标称尺寸）
    cw = max(1, int(round(px * core_ratio)))
    ch = max(1, int(round(py * core_ratio)))

    method = method if method in ("median", "mean", "mode", "kmeans", "dominant") else "median"

    out = np.zeros((h_logic, w_logic, 3), dtype=np.uint8)

    # y 向去重叠：记录上一行每列的 y_end，确保相邻行块不共享像素
    # （cell_ys 逐交点局部漂移可能使相邻行重叠/反向，对称处理 x 向的 prev_bx_end）
    prev_by_end = np.zeros(w_logic, dtype=np.int64)

    for j in range(h_logic):
        last_rep = None
        prev_bx_end = 0
        for i in range(w_logic):
            if use_cells:
                by_start = int(round(float(grid.cell_ys[j, i])))
                by_end = int(round(float(grid.cell_ys[j + 1, i])))
                bx_start = int(round(float(grid.cell_xs[j, i])))
                bx_end = int(round(float(grid.cell_xs[j, i + 1])))
            else:
                by_start = int(round(phase_y + j * py))
                by_end = int(round(phase_y + (j + 1) * py))
                bx_start = int(round(phase_x + i * px))
                bx_end = int(round(phase_x + (i + 1) * px))
            by_start = max(0, min(by_start, H))
            by_end = max(0, min(by_end, H))
            bx_start = max(0, min(bx_start, W))
            bx_end = max(0, min(bx_end, W))
            # 边界去重：确保相邻块不共享像素（x 与 y 双向）
            bx_start = max(bx_start, prev_bx_end)
            bx_end = max(bx_end, bx_start + 1) if bx_end <= bx_start else bx_end
            bx_end = min(bx_end, W)
            prev_bx_end = bx_end
            by_start = max(by_start, int(prev_by_end[i]))
            by_end = max(by_end, by_start + 1) if by_end <= by_start else by_end
            by_end = min(by_end, H)
            prev_by_end[i] = by_end
            if by_end <= by_start:
                if j > 0:
                    out[j, i, :] = out[j - 1, i, :]
                continue
            if bx_end <= bx_start:
                if last_rep is not None:
                    out[j, i, :] = last_rep
                elif j > 0:
                    out[j, i, :] = out[j - 1, i, :]
                continue
            block_h = by_end - by_start
            block_w = bx_end - bx_start
            # 核心区：块内居中，且不超出块边界
            cy_h = min(ch, block_h)
            cy_start = by_start + (block_h - cy_h) // 2
            cy_end = cy_start + cy_h
            cx_w = min(cw, block_w)
            cx_start = bx_start + (block_w - cx_w) // 2
            cx_end = cx_start + cx_w

            core = img[cy_start:cy_end, cx_start:cx_end, :]
            if core.size == 0:
                continue

            if method == "median":
                rep = np.median(core.reshape(-1, 3), axis=0)
            elif method == "mean":
                rep = np.mean(core.reshape(-1, 3), axis=0)
            elif method == "kmeans":
                pixels = core.reshape(-1, 3)
                if pixels.shape[0] < 4:
                    rep = np.mean(pixels, axis=0)
                else:
                    # 向量化主色分离：用中位数估计主色，距离阈值分离渗透色
                    center = np.median(pixels, axis=0)
                    diffs = pixels - center
                    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                    threshold = np.percentile(dists, 25)
                    main_mask = dists <= threshold
                    if np.any(main_mask):
                        rep = np.mean(pixels[main_mask], axis=0)
                    else:
                        rep = np.mean(pixels, axis=0)
            elif method == "dominant":
                pixels = core.reshape(-1, 3)
                if pixels.shape[0] == 0:
                    # 核心区为空/退化时回退 median
                    rep = np.median(pixels, axis=0)
                else:
                    # 感知空间小步长量化：优先 OKLab，未就绪回退 Lab。
                    # OKLab L∈[0,1]、a/b∈[-0.4,0.4]，步长需远小于 Lab 的 4-6；
                    # Lab L∈[0,100]、a/b∈[-128,128]，通道步长取 4-6 区间。
                    if _rgb_to_oklab is not None:
                        space_px = _rgb_to_oklab(pixels)
                        step = 0.02
                    else:
                        space_px = _rgb_to_lab(pixels)
                        step = 5.0
                    quantized = np.round(space_px / step).astype(np.int64)
                    uniq, counts = np.unique(quantized, axis=0, return_counts=True)
                    mode_q = uniq[int(np.argmax(counts))]
                    # 取众数 bin 内原始 RGB 像素均值，而非感知空间量化值
                    mode_mask = np.all(quantized == mode_q, axis=1)
                    if np.any(mode_mask):
                        rep = np.mean(pixels[mode_mask], axis=0)
                    else:  # 退化回退 median
                        rep = np.median(pixels, axis=0)
            else:  # mode
                pixels = core.reshape(-1, 3)
                quantized = np.round(pixels / 32.0).astype(int)
                uniq, counts = np.unique(quantized, axis=0, return_counts=True)
                mode_q = uniq[int(np.argmax(counts))]
                # 取众数 bin 内原始像素的均值，而非量化值
                mode_mask = np.all(quantized == mode_q, axis=1)
                if np.any(mode_mask):
                    rep = np.mean(pixels[mode_mask], axis=0)
                else:
                    rep = np.clip(mode_q.astype(np.float64) * 32.0, 0.0, 255.0)

            out[j, i, :] = np.clip(rep, 0, 255).astype(np.uint8)
            last_rep = out[j, i, :]

    return out
