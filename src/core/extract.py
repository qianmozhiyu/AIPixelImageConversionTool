"""像素块提取模块。

根据网格检测结果（含相位偏移）对每个块的核心区采样代表色，直接输出
逻辑分辨率的真像素图，合并原 block_refine 与 downscale 的功能。

核心区采样策略规避块边缘杂色（抗锯齿、JPEG 伪影、相邻块渗透），
代表色算法支持中位数/均值/众数。
"""

from __future__ import annotations

import numpy as np

from .grid_detect import Grid


def extract_blocks(
    img: np.ndarray,
    grid: Grid,
    method: str = "median",
    core_ratio: float = 0.6,
) -> np.ndarray:
    """将每个块的核心区代表色提取为 1 个像素，输出真像素图。

    根据网格相位对齐块边界，取块内核心区（居中 ``core_ratio`` 比例区域）
    的代表色作为该块的像素值，直接输出 ``(h_logic, w_logic, 3)`` uint8 图像。

    Args:
        img: ``(H, W, 3)`` float64 RGB 0-255 输入图像。
        grid: 网格检测结果，提供块尺寸、相位与逻辑分辨率。
        method: 代表色算法，``"median"`` / ``"mean"`` / ``"mode"`` / ``"kmeans"``，未知回退 median。
            ``"kmeans"`` 用中位数估计主色并以距离阈值分离渗透色，取主色像素均值。
        core_ratio: 核心区边长占块边长比例，用于规避边缘杂色。

    Returns:
        ``(h_logic, w_logic, 3)`` uint8 像素图。
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

    method = method if method in ("median", "mean", "mode", "kmeans") else "median"

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
