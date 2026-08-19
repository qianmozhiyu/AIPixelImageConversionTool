"""整数尺度快速检测（runs/GCD 方法）。

参考 pixel-art-downsampler / unfake.js 的 runs 方法：沿采样行/列统计
「相邻像素差值 ≤ tol 视为同色」的行程长度——像素图中每块均匀色对应一个
行程，行程长度为真实块尺寸的整数倍。取频次高的行程长度做 GCD（math.gcd）
得到整数放大尺度，作为现有 FFT/ACF 多判据投票的正交交叉验证先验。

主要接口：
- ``detect_integer_scale``：返回 x/y 整数尺度、命中率与置信度；
  无强证据时返回 ``(0, 0, 0, 0)``。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

# 视为「强证据」的命中率下限（与 detect() 集成判定口径一致）
STRONG_HIT_RATE = 0.7


def _line_runs(line: np.ndarray, tol: float) -> list[int]:
    """统计一维线的行程长度（numpy 向量化）。

    相邻像素差值 ≤ tol 视为同一颜色（同一行程），差值 > tol 处断开。
    对灰度 2D 输入，每像素仅一个通道，即 ``abs(v[i]-v[i-1]) <= tol``。

    向量化：``np.diff`` 求相邻差，``abs(diff) > tol`` 的索引 +1 即断点，
    行程长度为相邻断点（含 0 与 n）间隔。与旧逐像素 Python 循环实现
    逐位一致（浮点差值运算顺序相同：``line[i]-line[i-1]``），2K 图上
    该循环约 2M 次迭代，向量化后耗时接近归零。

    Args:
        line: 一维灰度数组。
        tol: 同色判定容差（灰度级）。

    Returns:
        行程长度列表（int，与旧实现类型一致）。
    """
    if line.size < 2:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(line)) > tol) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [line.size]))
    return (ends - starts).tolist()


def _is_multiple_of(run_len: int, scale: int, tol: float) -> bool:
    """``run_len`` 是否为 ``scale`` 的整数倍（容差 ``tol`` 内）。"""
    k = round(run_len / scale)
    if k < 1:
        return False
    return abs(run_len - k * scale) <= tol


def _best_scale_from_runs(
    runs: list[int], min_p: int, max_p: int, tol: float
) -> tuple[int, float]:
    """由行程长度直方图推导候选整数尺度并评估命中率。

    真实块尺寸是行程长度的因子：对频次最高的若干行程长度做两两 GCD
    得到候选尺度（覆盖「行程为 k*scale」的各类情况），再对每个候选计算
    命中率（行程为尺度整数倍、容差内的占比），选命中率最高者；
    命中率相同时取更大尺度（真基频优于其子谐波，避免小尺度偏向）。

    Args:
        runs: 行程长度列表（已聚合所有采样线）。
        min_p: 最小候选尺度。
        max_p: 最大候选尺度。
        tol: 整数倍判定容差。

    Returns:
        (scale, hit_rate)。无候选时返回 (0, 0.0)。
    """
    total = len(runs)
    if total == 0:
        return (0, 0.0)
    hist = Counter(runs)
    # 取出现频次前 8 的行程长度（抑制噪声长度对 GCD 的污染）
    top = [L for L, _ in hist.most_common(8)]
    cands: set[int] = set()
    for i, a in enumerate(top):
        if min_p <= a <= max_p:
            cands.add(a)
        for b in top[i + 1:]:
            g = math.gcd(a, b)
            if min_p <= g <= max_p:
                cands.add(g)
    if not cands:
        return (0, 0.0)
    best_s, best_rate = 0, 0.0
    runs_arr = np.asarray(runs, dtype=np.int64)
    for s in sorted(cands):
        # 命中率统计向量化：runs 很多（2K 图约 4.5 万条）时逐条 Python
        # 判定可观；np.rint 与 Python round 同为「四舍六入五成双」，
        # 且差值均为可精确表示的整数，结果与逐条判定逐位一致。
        ks = np.rint(runs_arr / s)
        hits = int(np.count_nonzero((ks >= 1.0) & (np.abs(runs_arr - ks * s) <= tol)))
        rate = hits / total
        if rate > best_rate + 1e-9 or (abs(rate - best_rate) < 1e-9 and s > best_s):
            best_s, best_rate = s, rate
    return best_s, best_rate


def detect_integer_scale(
    gray: np.ndarray,
    min_p: int = 3,
    max_p: int = 40,
    tol: float = 12.0,
    sample_stride: int = 4,
) -> tuple[int, int, float, float]:
    """基于像素行程长度 GCD 的整数尺度快速检测。

    每隔 ``sample_stride`` 行/列采样一条线，对每条线统计行程长度，
    跨所有线汇总行程长度直方图，用 GCD 推导候选整数尺度，并评估命中率
    与置信度（综合命中率与 x/y 一致性）。

    Args:
        gray: 灰度图数组 (H, W)。
        min_p: 最小候选尺度（默认 3，与 grid_detect 默认一致）。
        max_p: 最大候选尺度。
        tol: 判定「同色」的像素差值容差（灰度级）。
        sample_stride: 采样间隔，每隔该行/列采样一条线。

    Returns:
        (scale_x, scale_y, hit_rate, confidence)：
        - scale_x / scale_y：x/y 方向的整数尺度（像素）；无证据时为 0。
        - hit_rate：x/y 两方向命中率均值（[0,1]）。
        - confidence：综合命中率与 x/y 一致性。
        无强证据（命中率不足或尺度非法）时返回 (0, 0, 0, 0)。
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        return (0, 0, 0, 0)
    H, W = g.shape
    if H < 8 or W < 8:
        return (0, 0, 0, 0)
    # x 方向：采样行
    x_runs: list[int] = []
    for y in range(0, H, sample_stride):
        x_runs.extend(_line_runs(g[y, :], tol))
    # y 方向：采样列
    y_runs: list[int] = []
    for x in range(0, W, sample_stride):
        y_runs.extend(_line_runs(g[:, x], tol))
    sx, hx = _best_scale_from_runs(x_runs, min_p, max_p, tol)
    sy, hy = _best_scale_from_runs(y_runs, min_p, max_p, tol)
    if sx <= 0 or sy <= 0:
        return (0, 0, 0, 0)
    hit_rate = (hx + hy) / 2.0
    # 置信度：命中率为主，x/y 尺度一致性为乘子
    consistency = 1.0 if sx == sy else 0.7
    confidence = float(hit_rate * consistency)
    if hit_rate < STRONG_HIT_RATE:
        # 无强证据：返回 0 使调用方走既有投票路径
        return (0, 0, 0, 0)
    return (sx, sy, hit_rate, confidence)
