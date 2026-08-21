"""像素网格周期检测。

针对 AI 生成的伪像素图，通过梯度 + 一维 FFT 带通信噪比检测块状
周期结构，估计块尺寸（px, py）与相位偏移（phase_x, phase_y），
并据此推断逻辑分辨率（w_logic, h_logic）。

主要接口：
- ``has_pixel_grid``：判断灰度图是否包含像素网格周期。
- ``find_phase``：在已知块尺寸下扫描最佳网格相位（灰度块方差判据）。
- ``find_phase_edge``：在边缘强度图上扫描最佳网格相位（边界带能量判据）。
- ``detect``：自动检测块尺寸与相位，返回 ``Grid``。``signal="oklab"`` 时
  用 OKLAB 感知色差做检测信号，等亮度异色块边界可见（默认 ``"gray"``）。
- ``detect_with_user_grid``：用户指定逻辑分辨率，反推块尺寸并定相。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.core.scale_detect import detect_integer_scale
from src.core.color import rgb_to_oklab


@dataclass
class Grid:
    """像素网格检测结果。

    Attributes:
        w_logic: 逻辑宽（块数）。
        h_logic: 逻辑高（块数）。
        px: 块宽（像素）。
        py: 块高（像素）。
        phase_x: x 方向相位偏移（像素），全局相位，用于回退/显示。
        phase_y: y 方向相位偏移（像素），全局相位，用于回退/显示。
        conf: 置信度。
        candidates: 候选列表 [(w, h, score), ...]。
        cell_ys: 形状 ``(h_logic+1, w_logic+1)`` 的 2D 数组，
            ``cell_ys[j, i]`` 为第 ``j`` 行第 ``i`` 列网格交点的 y 像素坐标。
            第 ``(j, i)`` 个格子的区域为
            ``[cell_ys[j, i], cell_ys[j+1, i]) × [cell_xs[j, i], cell_xs[j, i+1])``。
            由 ``expand_grid_edge_guided`` 逐交点填充，支持逐交点局部漂移；
            方块检测失败时由 ``phase + 等距网格``生成兜底；未填充时为 None。
        cell_xs: 形状 ``(h_logic+1, w_logic+1)`` 的 2D 数组，
            ``cell_xs[j, i]`` 为第 ``j`` 行第 ``i`` 列网格交点的 x 像素坐标，
            含义与 ``cell_ys`` 对称；未填充时为 None。
    """

    w_logic: int
    h_logic: int
    px: float
    py: float
    phase_x: float
    phase_y: float
    conf: float
    candidates: list = field(default_factory=list)
    cell_ys: np.ndarray | None = None  # (h_logic+1, w_logic+1) 网格交点 y 坐标
    cell_xs: np.ndarray | None = None  # (h_logic+1, w_logic+1) 网格交点 x 坐标
    comb_score: float = 0.0  # Spectral Comb 周期得分（可选元信息，默认 0）
    low_confidence: bool = False  # conf < 0.4 时为 True，供 GUI/元信息提示人工确认
    comb_energy_conf: float = 0.0  # 梳状能量终审置信度（两轴均值，仅元信息，默认 0）
    jpeg_grid: tuple = ()  # JPEG 网格检测结果 (is_significant, phase, strength)，默认空 tuple


def _linear_extrap_pad(g: np.ndarray, p: int) -> np.ndarray:
    """沿两个轴用线性外推填充边界（各 p 像素）。

    用于局部对比度归一化的边界处理：相比反射/最近填充，线性外推能让
    局部滤波器在边界处仍然跟踪信号的整体趋势，从而避免纯渐变图像
    在边界产生被误判为周期结构的伪瞬态。

    Args:
        g: 2D 数组。
        p: 各轴两端填充像素数。

    Returns:
        填充后的 2D 数组，形状为 (H+2p, W+2p)；当某轴长度 < 2 时跳过该轴填充。
    """
    if p <= 0:
        return g
    arange_l = np.arange(-p, 0)
    arange_r = np.arange(1, p + 1)
    # 沿 axis=1（列）线性外推
    if g.shape[1] >= 2:
        slope_l = np.mean(np.diff(g[:, : p + 1], axis=1), axis=1)
        slope_r = np.mean(np.diff(g[:, -p - 1 :], axis=1), axis=1)
        left = g[:, 0:1] + slope_l[:, None] * arange_l[None, :]
        right = g[:, -1:] + slope_r[:, None] * arange_r[None, :]
        g = np.concatenate([left, g, right], axis=1)
    # 沿 axis=0（行）线性外推
    if g.shape[0] >= 2:
        slope_t = np.mean(np.diff(g[: p + 1, :], axis=0), axis=0)
        slope_b = np.mean(np.diff(g[-p - 1 :, :], axis=0), axis=0)
        top = g[0:1, :] + slope_t[None, :] * arange_l[:, None]
        bot = g[-1:, :] + slope_b[None, :] * arange_r[:, None]
        g = np.concatenate([top, g, bot], axis=0)
    return g


def _local_contrast_normalize(gray: np.ndarray, window: int = 33) -> np.ndarray:
    """局部对比度归一化，增强低对比度区域的弱边缘。

    减去局部均值后除以（有界钳位的）局部标准差，让低对比度区域的边缘
    也能被梯度检测到。参考 crispx 项目的预处理策略。

    做了两点稳健化处理：
    1. 边界采用线性外推填充（而非反射），避免对纯渐变图像在边界处产生
       伪瞬态被 FFT 误判为网格周期。
    2. 将局部标准差钳位到 [0.5*全局标准差, 全局标准差]：下限避免对近平坦
       区域过度放大产生低频幅度调制（会被 FFT 误读为长周期谐波），
       上限避免衰减高对比度区域。这样既能有界地增强低对比度弱边缘，
       又不破坏真实图像的周期检测。

    Args:
        gray: 灰度图数组 (H, W)。
        window: 局部窗口大小（奇数），默认 33。

    Returns:
        归一化后的灰度图，均值约为 0，局部对比度得到有界增强。
    """
    g = np.asarray(gray, dtype=np.float64)
    global_std = float(np.std(g))
    if global_std < 1e-6:
        # 常量图无对比度，归一化为零（无边缘可检测）
        return np.zeros_like(g)
    # 确保窗口为奇数
    w = int(window)
    if w % 2 == 0:
        w += 1
    p = w // 2
    # 线性外推填充，避免反射模式对纯渐变产生边界伪瞬态
    g = _linear_extrap_pad(g, p)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(g, size=w, mode="reflect")
    local_sq_mean = uniform_filter(g * g, size=w, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 1e-6))
    # 钳位局部标准差，避免近平坦区域过度放大与高对比度区域被衰减
    eff_std = np.clip(local_std, 0.5 * global_std, global_std)
    normalized = (g - local_mean) / eff_std
    return normalized[p:-p, p:-p]


def _build_integral(a: np.ndarray) -> np.ndarray:
    """构建 (H+1, W+1) float64 二维积分图。

    积分图 ``I[i+1, j+1] = sum(a[:i+1, :j+1])``（首行首列为 0），用于
    O(1) 矩形区域求和：``sum(rect) = I[y1,x1]-I[y0,x1]-I[y1,x0]+I[y0,x0]``。
    构建方式与 ``_edge_band_strength`` / ``detect_squares`` 中的积分图完全
    一致，供网格检测各阶段复用，避免同一边缘/灰度图被反复重建（大图下
    每次构建 O(H·W) 是主要热点）。

    Args:
        a: 2D 数组（float64 或可转换）。

    Returns:
        ``(H+1, W+1)`` float64 积分图。
    """
    a = np.asarray(a, dtype=np.float64)
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return I


def _fft_band_snr(sig: np.ndarray, min_p: int, max_p: int) -> tuple[float, float]:
    """一维 FFT 带通信噪比。

    在 [min_p, max_p] 周期带内寻找频谱峰值，以峰值功率与带内中位功率之比
    作为信噪比，并返回峰值对应的周期。

    Args:
        sig: 一维信号数组。
        min_p: 最小候选周期。
        max_p: 最大候选周期。

    Returns:
        (snr, period)：信噪比与峰值周期。无法计算时返回 (0.0, 0.0)。
    """
    sig = sig.astype(np.float64)
    sig = sig - sig.mean()
    if sig.std() < 1e-6:
        return (0.0, 0.0)
    n = sig.size
    if n < 4:
        return (0.0, 0.0)
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0)
    # 周期 p -> 频率 1/p；带 [min_p, max_p] -> 频率 [1/max_p, 1/min_p]
    mask = (freqs >= 1.0 / max_p) & (freqs <= 1.0 / min_p) & (freqs > 0)
    if not np.any(mask):
        return (0.0, 0.0)
    band_spec = spec[mask]
    band_freqs = freqs[mask]
    # Find local maxima within the band (single-side check at boundaries)
    m = band_spec.size
    candidates = []
    if m >= 3:
        is_peak = np.zeros(m, dtype=bool)
        is_peak[1:-1] = (band_spec[1:-1] > band_spec[:-2]) & (band_spec[1:-1] > band_spec[2:])
        is_peak[0] = band_spec[0] > band_spec[1]
        is_peak[-1] = band_spec[-1] > band_spec[-2]
        candidates = list(np.where(is_peak)[0])
    elif m == 2:
        if band_spec[0] > band_spec[1]:
            candidates.append(0)
        if band_spec[1] > band_spec[0]:
            candidates.append(1)
    if not candidates:
        # Band too short / no local maximum: fall back to global argmax
        peak_idx = int(np.argmax(band_spec))
    else:
        # Score each candidate by the larger of left-climb / right-fall and
        # pick the highest score (tie-break by amplitude).
        best = None  # (score, amp, idx)
        for c in candidates:
            c = int(c)
            amp = float(band_spec[c])
            # left_climb: walk left while strictly ascending toward the peak
            left_climb = 0.0
            k = c
            while k > 0 and band_spec[k] > band_spec[k - 1]:
                left_climb = amp - float(band_spec[k - 1])
                k -= 1
            # right_fall: walk right while strictly descending away from peak
            right_fall = 0.0
            k = c
            while k < m - 1 and band_spec[k] > band_spec[k + 1]:
                right_fall = amp - float(band_spec[k + 1])
                k += 1
            score = max(left_climb, right_fall)
            if best is None or (score, amp) > (best[0], best[1]):
                best = (score, amp, c)
        peak_idx = best[2]
    peak_amp = float(band_spec[peak_idx])
    median_amp = float(np.median(band_spec))
    if median_amp <= 1e-12:
        snr = 0.0 if peak_amp <= 1e-12 else 1e6
    else:
        snr = (peak_amp / median_amp) ** 2  # 功率比
    # Parabolic interpolation for sub-bin precision
    if 0 < peak_idx < len(band_spec) - 1:
        y0 = float(band_spec[peak_idx - 1])
        y1 = float(band_spec[peak_idx])
        y2 = float(band_spec[peak_idx + 1])
        denom_interp = y0 - 2 * y1 + y2
        if abs(denom_interp) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom_interp
            # frequency spacing between adjacent bins
            df = float(band_freqs[1] - band_freqs[0])
            refined_freq = float(band_freqs[peak_idx]) + offset * df
            if refined_freq > 0:
                period = float(1.0 / refined_freq)
            else:
                period = float(1.0 / band_freqs[peak_idx])
        else:
            period = float(1.0 / band_freqs[peak_idx])
    else:
        period = float(1.0 / band_freqs[peak_idx])
    return (float(snr), period)


def _acf_period(profile: np.ndarray, min_p: int, max_p: int) -> tuple[list[int], np.ndarray]:
    """自相关周期检测，返回 [min_p, max_p] 内的峰 lag 列表与 ACF 数组。

    计算归一化自相关函数，在 [min_p, max_p] 范围内找局部极大值峰。

    Args:
        profile: 1D 信号数组。
        min_p: 最小候选周期。
        max_p: 最大候选周期。

    Returns:
        (peaks, acf)：峰 lag 列表（按 ACF 值降序）与完整 ACF 数组。
    """
    sig = profile.astype(np.float64)
    sig = sig - sig.mean()
    n = sig.size
    if n < 4:
        return ([], np.array([]))
    # 用 FFT 加速自相关
    f = np.fft.rfft(sig, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    # 归一化
    if acf[0] > 1e-12:
        acf = acf / acf[0]
    # 在 [min_p, max_p] 找局部极大值
    peaks = []
    for p in range(min_p, min(max_p + 1, n)):
        if p < 1:
            continue
        is_peak = True
        # 检查邻域（±1）
        for d in range(-1, 2):
            idx = p + d
            if 0 <= idx < n and idx != p and acf[idx] >= acf[p]:
                is_peak = False
                break
        if is_peak and acf[p] > 0:
            peaks.append(p)
    # 按 ACF 值降序排序
    peaks.sort(key=lambda p: -acf[p])
    return (peaks, acf)


def _harmonic_interpret(peaks: list[int], tolerance: float = 0.1) -> int:
    """谐波解释：找能解释最多峰的最小基频。

    若 P2 是 P1 的整数倍（在容差内），则 P1 是基频。
    返回能解释最多峰的最小候选周期。

    Args:
        peaks: 候选峰 lag 列表（已按显著性降序）。
        tolerance: 整数倍判断容差（相对误差）。

    Returns:
        基频周期。无峰时返回 0。
    """
    if not peaks:
        return 0
    if len(peaks) == 1:
        return peaks[0]
    best_base = peaks[0]  # 默认取最显著的
    best_explained = 1
    for base in peaks:
        explained = 1  # base 自身
        for p in peaks:
            if p == base:
                continue
            ratio = p / base
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < tolerance:
                explained += 1
        if explained > best_explained:
            best_explained = explained
            best_base = base
    return best_base


def _block_variance_ratio_legacy(
    gray: np.ndarray, period: float, axis: int = 0
) -> float:
    """旧版 1D 条带块方差对比度（仅无 edge_map 的旧投票路径使用）。

    按候选周期沿一维把整幅图切成「跨全宽/全高」的条带，块内必然混入大量
    其他块的颜色，块内方差被高估、区分度不足，且系统性偏向小周期。
    该实现缺陷已由 2D 版 ``_block_variance_ratio`` 修复；此处保留旧实现
    仅为保证旧路径（无 edge_map 调用）的行为与历史版本完全一致。
    """
    g = np.asarray(gray, dtype=np.float64)
    if axis == 1:
        g = g.T  # 转置后统一按行处理
    H, W = g.shape
    p = float(period)
    if p < 1 or H < p * 2:
        return 0.0
    n_blocks = int(H / p)
    if n_blocks < 2:
        return 0.0
    # 块均值序列（转置后统一按行分块：g[start:end, :] 取行块）
    block_means = []
    block_vars = []
    for i in range(n_blocks):
        start = int(i * p)
        end = int((i + 1) * p)
        block = g[start:end, :]
        block_means.append(block.mean())
        block_vars.append(block.var())
    between_var = float(np.var(block_means))
    within_var = float(np.mean(block_vars))
    if within_var < 1e-12:
        return 0.0 if between_var < 1e-12 else 1e6
    return between_var / within_var


def _block_variance_ratio(
    gray: np.ndarray, period: float, axis: int = 0,
    integral: np.ndarray | None = None,
    integral_sq: np.ndarray | None = None,
) -> float:
    """计算 2D 真块方差对比度（块间方差 / 块内方差），真网格此值最大。

    按候选周期将图像切成 ``period×period`` 真块（而非旧实现的 1D 条带），
    以相位扫描找最优对齐，计算块均值序列方差（块间）与各块内方差均值
    （块内）之比。真网格对齐时块内同质、块间差异大，比值高。

    ``integral`` / ``integral_sq`` 为灰度及其平方的积分图：同时提供时用
    O(1) 积分图块求和（块和 = 矩形查询；均值/块内均方与旧实现同一数学，
    仅求和次序有浮点差异），避免每相位全图 reshape 的 O(H·W) 开销；
    任一为 None 时回退旧实现的逐块 reshape 计算（保持向后兼容）。
    相位扫描步长与旧实现一致（``step = max(1, p_i//4)``）。

    参数 ``axis`` 为兼容既有调用保留（2D 实现各向同性，不再分方向）。

    Args:
        gray: 灰度图数组 (H, W)。
        period: 候选周期（像素）。
        axis: 兼容参数，不影响结果。
        integral: 灰度积分图 (H+1, W+1)；None 时回退旧实现。
        integral_sq: 灰度平方积分图 (H+1, W+1)；None 时回退旧实现。

    Returns:
        方差对比度比值。越大越好。失败时返回 0.0。
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        return 0.0
    H, W = g.shape
    p = float(period)
    if p < 1 or H < p * 2 or W < p * 2:
        return 0.0
    p_i = int(round(p))
    if p_i < 1:
        return 0.0
    use_integral = integral is not None and integral_sq is not None
    if use_integral:
        I = integral
        I2 = integral_sq
        area = float(p_i * p_i)
    best = 0.0
    # 相位扫描：步长随周期缩放，限制每候选相位数（≤~16，投票用弱判据无需过密）
    step = max(1, p_i // 4)
    for py0 in range(0, p_i, step):
        n_y = (H - py0) // p_i
        if n_y < 2:
            continue
        cy = py0 + n_y * p_i
        for px0 in range(0, p_i, step):
            n_x = (W - px0) // p_i
            if n_x < 2:
                continue
            cx = px0 + n_x * p_i
            if use_integral:
                # O(1) 块求和（积分图矩形查询）：块角点 = I[py0+i*p_i, px0+j*p_i]，
                # 用积分图子块的步长切片一次性取全部角点（避免逐相位 np.ix_
                # fancy indexing 的 gather 开销，小周期大量小块时提速明显）；
                # 块均值与块内均方同旧实现数学（同一求和次序）。
                corners = I[py0:cy + 1, px0:cx + 1][::p_i, ::p_i]  # (n_y+1, n_x+1)
                s = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
                corners2 = I2[py0:cy + 1, px0:cx + 1][::p_i, ::p_i]
                sq = corners2[1:, 1:] - corners2[:-1, 1:] - corners2[1:, :-1] + corners2[:-1, :-1]
                block_means = s / area  # (n_y, n_x)
                block_sq_means = sq / area
                within = float(np.mean(block_sq_means - block_means * block_means))
                between = float(np.var(block_means))
            else:
                cell = g[py0:cy, px0:cx].reshape(n_y, p_i, n_x, p_i)
                cell_means = cell.mean(axis=(1, 3))  # (n_y, n_x)
                cell_sq = (cell * cell).mean(axis=(1, 3))
                within = float(np.mean(cell_sq - cell_means * cell_means))
                between = float(np.var(cell_means))
            if within < 1e-12:
                ratio = 0.0 if between < 1e-12 else 1e6
            else:
                ratio = between / within
            if ratio > best:
                best = ratio
    return float(best)


def _default_edge_band_phases(p: float) -> tuple[float, ...]:
    """``_edge_band_strength`` 默认相位采样集合。

    Task 6 决策记录：试验把默认相位加密到 4 相位 (0, p/4, p/2, 3p/4) 取最大
    边界带强度，并以 test_grid_detect_subharmonic 与 test_grid_detect_comb
    为验收口径（子谐波修正依赖 >1.3 的边界强度比）。实测 4 相位未导致任何
    回归（两组用例全部通过），故采纳 4 相位为默认，相位感知更鲁棒（相位偏移
    时仍能命中边界带）。`phases` 参数保留供显式覆盖（如回退 {0, p/2}）。
    """
    return (0.0, p / 4.0, p / 2.0, 3.0 * p / 4.0)


def _edge_band_strength(
    em: np.ndarray,
    period: float,
    axis: int = 0,
    phases: Sequence[float] | None = None,
    integral: np.ndarray | None = None,
) -> float:
    """边界带边缘强度：候选周期下，块边界位置 1px 条带的平均边缘强度。

    真实块边界是整幅图上的连续强边缘，块内纹理是局部弱边缘。按候选周期
    在整数边界位置取 1px 宽条带求平均边缘强度，真实周期的边界带强度显著
    高于其子谐波（块内纹理周期 P/k 的条带大部分不与真实边界对齐）。

    Task 6：``phases`` 参数提供相位感知覆盖。为 None 时用
    ``_default_edge_band_phases`` 的默认相位集合（4 相位加密采样，
    实测对子谐波/comb 回归无影响，故采纳为默认）。传入自定义相位
    （如回退 {0, p/2}）可显式覆盖。

    ``integral`` 为 ``em`` 的预构建积分图（``_build_integral`` 输出）：
    提供时直接复用，避免每次调用重建整图积分图（大图下 O(H·W) 是热点）；
    为 None 时内部构建（保持向后兼容，既有直接调用/测试不变）。

    Args:
        em: 边缘强度图 (H, W)，值域 [0,1]（``compute_edge_map`` 输出）。
        period: 候选周期（像素）。
        axis: 0=检测 y 方向周期（水平边界），1=检测 x 方向周期（垂直边界）。
        phases: 显式相位集合（相对周期起点 offset，单位像素）；None 时用默认。
        integral: ``em`` 的 (H+1, W+1) 积分图；None 时内部构建。

    Returns:
        边界带平均边缘强度。无法计算时返回 0.0。
    """
    em = np.asarray(em, dtype=np.float64)
    if em.ndim != 2:
        return 0.0
    H, W = em.shape
    p = float(period)
    if p < 1:
        return 0.0
    # 积分图 O(1) 条带求和；detect 主路径复用预构建积分图避免重复重建
    I = integral if integral is not None else _build_integral(em)
    best = 0.0
    for phase in _default_edge_band_phases(p) if phases is None else tuple(phases):
        if axis == 1:  # x 方向：垂直边界（列条带）
            n = int((W - phase) // p)
            if n < 2:
                continue
            total = 0.0
            count = 0
            for m in range(n + 1):
                x0 = int(round(phase + m * p))
                x1 = x0 + 1
                if 0 <= x0 < x1 <= W:
                    total += I[H, x1] - I[H, x0]
                    count += 1
            if count > 0:
                best = max(best, total / count)
        else:  # axis=0: y 方向：水平边界（行条带）
            n = int((H - phase) // p)
            if n < 2:
                continue
            total = 0.0
            count = 0
            for m in range(n + 1):
                y0 = int(round(phase + m * p))
                y1 = y0 + 1
                if 0 <= y0 < y1 <= H:
                    total += I[y1, W] - I[y0, W]
                    count += 1
            if count > 0:
                best = max(best, total / count)
    return float(best)


def _boundary_interior_ratio(
    em: np.ndarray,
    period: float,
    axis: int = 0,
    phases: Sequence[float] | None = None,
    integral: np.ndarray | None = None,
) -> float:
    """边界/格心边缘能量比（内部洁净度增量判据）。

    真实像素网格应有「块边界强边缘、块内干净」的结构；子谐波(P/k)的网格线
    一半落真边界、一半落入真块内部，其格心带会采到真边界边缘（内部脏），
    而真周期 P 的格心带只有块内 AA 噪声（干净）。故 ratio = 边界带均值 /
    格心带均值 在真周期处显著高于子谐波/倍频——这正是 crispx「内部洁净度」
    的负向判据，直击真实 AI 图「边界强度区分度不足(≈1.0)」的根因：纯边界带
    能量在真/假周期间区分度弱，加入格心带归一化后把「内部是否干净」变成
    判别主轴。

    采样沿用 ``_edge_band_strength`` 的积分图 O(1) 条带方式：对每个候选
    (period, phase)，边界条带取整数边界位置 1px，格心条带取块中心 1px。
    同样扫 ``_default_edge_band_phases`` 的 4 相位，取最大比值（与边界带
    强度语义一致：相位对齐处比值最高）。

    Args:
        em: 边缘强度图 (H, W)，值域 [0,1]。
        period: 候选周期（像素）。
        axis: 0=检测 y 方向周期（水平边界），1=检测 x 方向周期（垂直边界）。
        phases: 显式相位集合；None 时用默认 4 相位。
        integral: ``em`` 的 (H+1, W+1) 积分图；None 时内部构建。

    Returns:
        边界/格心边缘能量比（>0）。格心完全干净时返回大值（强偏好），
        无法计算时返回 0.0。
    """
    em = np.asarray(em, dtype=np.float64)
    if em.ndim != 2:
        return 0.0
    H, W = em.shape
    p = float(period)
    if p < 2:
        return 0.0
    I = integral if integral is not None else _build_integral(em)
    length = W if axis == 1 else H
    best = 0.0
    has_data = False
    for phase in _default_edge_band_phases(p) if phases is None else tuple(phases):
        n = int((length - phase) // p)
        if n < 2:
            continue
        b_total = i_total = 0.0
        b_count = i_count = 0
        for m in range(n + 1):
            b0 = int(round(phase + m * p))
            b1 = b0 + 1
            c0 = int(round(phase + (m + 0.5) * p))
            c1 = c0 + 1
            if axis == 1:  # x 方向：垂直条带（列）
                if 0 <= b0 < b1 <= W:
                    b_total += I[H, b1] - I[H, b0]
                    b_count += 1
                if 0 <= c0 < c1 <= W:
                    i_total += I[H, c1] - I[H, c0]
                    i_count += 1
            else:  # axis=0: y 方向：水平条带（行）
                if 0 <= b0 < b1 <= H:
                    b_total += I[b1, W] - I[b0, W]
                    b_count += 1
                if 0 <= c0 < c1 <= H:
                    i_total += I[c1, W] - I[c0, W]
                    i_count += 1
        if b_count > 0 and i_count > 0:
            b_mean = b_total / b_count
            i_mean = i_total / i_count
            ratio = b_mean / max(i_mean, 1e-9)
            best = max(best, ratio)
            has_data = True
    return float(best) if has_data else 0.0


def _spectral_comb_score(profile: np.ndarray, period: int, n_harmonics: int = 8) -> float:
    """Spectral Comb 周期得分：谐波梳状能量 / 带内中位能量。

    像素网格线是周期性脉冲序列，谐波丰富、能量集中于 k/period 频率处；
    块内纹理近似正弦波，谐波衰减快。对候选周期在谐波频率 k/period 处取
    频谱幅值求和，除以谱中位数（稀疏冲激谱加地板防退化），真实网格周期的
    得分显著高于子谐波与噪声。

    Args:
        profile: 1D 梯度投影信号。
        period: 候选周期（像素）。
        n_harmonics: 参与评分的谐波数。

    Returns:
        comb 得分（>0 表示存在谐波结构）。信号过短时返回 0.0。
    """
    sig = np.asarray(profile, dtype=np.float64)
    sig = sig - sig.mean()
    n = sig.size
    if n < 8 or period < 1:
        return 0.0
    spec = np.abs(np.fft.rfft(sig))
    total = 0.0
    for k in range(1, n_harmonics + 1):
        f = k / float(period)
        idx = f * n  # rfft bin 位置
        i0 = int(np.floor(idx))
        i1 = i0 + 1
        if i1 >= len(spec):
            break
        frac = idx - i0
        total += spec[i0] * (1.0 - frac) + spec[i1] * frac
    if total <= 1e-15:
        return 0.0
    if n > 1:
        median = float(np.median(spec[1:]))
    else:
        median = 0.0
    denom = max(median * 8.0, 1e-9)
    return float(total / denom)


def _comb_candidate_periods(
    profile: np.ndarray, min_p: int, max_p: int, top_n: int = 5
) -> list[tuple[int, float]]:
    """在 [min_p, max_p] 内按 Spectral Comb 得分取前 top_n 个候选周期。

    Args:
        profile: 1D 梯度投影信号。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        top_n: 返回候选数上限。

    Returns:
        ``[(period, score), ...]``，按 score 降序。
    """
    scored: list[tuple[int, float]] = []
    for p in range(max(min_p, 1), max_p + 1):
        s = _spectral_comb_score(profile, p)
        if s > 0:
            scored.append((p, s))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


# 投票中昂贵的 2D BVR 判据只对廉价判据（ACF/FFT/edge）排名前 N 的候选计算，
# 限制相位扫描成本（大图 + 像素格点阵输入候选可达 20+，全量计算极慢）。
VOTE_BVR_LIMIT = 10

# Task 4：runs/GCD 整数尺度交叉验证判定常量
RUNS_STRONG_HIT_RATE = 0.7    # runs 视为强证据的命中率下限
RUNS_AGREE_REL = 0.2          # runs 与 vote 一致的相对差阈值
RUNS_ADOPT_EDGE_RATIO = 0.9   # runs > vote（子谐波场景）采纳 runs 所需边界强度比
RUNS_SHRINK_EDGE_RATIO = 1.3  # runs < vote 时缩小周期所需显式边界强度比

# Task 5：高分辨率合理性门控常量
PLAUSIBILITY_MIN_PERIOD = 3   # 周期 < 该值才触发门控（1-2px 生成纹理）
PLAUSIBILITY_EDGE_RATIO = 1.3 # 采纳更大倍周期需其边界带显著更强（>1.0 防均匀格点阵）
PLAUSIBILITY_MAX_STEPS = 6    # 链式放大的最大步数（2→4→8→…）

# --- S1：长宽比守恒防护常量（AR guard）---
# 旧硬编码 0.3 过松：slice_06.jpg（683×618，真实块周期 ~9.5px 非整数）两轴
# 投票分别锁到 (10, 8)，输出 67×76 长宽比畸变 20.2% 仍被放过（用户看到
# 的"拉伸"）；正常图 round 取整误差引起的输出畸变 <5%，取 0.12 两者之间
# 留安全裕量。触发后先做两轴联合周期再搜索，无优解才回退正方形。
AR_GUARD_RATIO_DIFF = 0.12  # 触发长宽比守恒防护的输出/输入比例相对偏差阈值
AR_GUARD_ACCEPT_AR = 0.03   # 联合再搜索组合可被采纳的最大输出长宽比相对偏差
AR_GUARD_EDGE_FLOOR = 0.8   # 采纳组合的两轴边界强度之和相对原组合的下限比例

# G3：峰值格点拟合周期精化常量
PEAK_LATTICE_MIN_PEAKS = 4    # 参与拟合的最少峰数（不足直接回退投票值）
PEAK_LATTICE_MAX_FITS = 20    # k 扫描的 minimize 调用数上限（防大图过慢）
PEAK_LATTICE_ALPHA = 0.5      # 缺失格线比例的惩罚系数
PEAK_LATTICE_MAX_J = 0.15     # 接受精化结果的最大目标值 J*
PEAK_LATTICE_MAX_REL = 0.3    # 精化值相对投票值的最大相对偏移
# G3 轴一致性防护：两轴精化后的相对分裂超过该值、且精化前两轴一致
# （相对差 ≤ 5%）时判定拟合不可信，整体回退精化前投票值。正方形网格
# 两轴投影来自同一格点结构，精化成功时两轴应几乎一致（合成 7.5px 图
# 两轴同收敛 7.49/7.49）；单轴锁错格点时（image02：x 回退 20.0 /
# y 精化 20.804，分裂 3.9%）两轴周期出现无法同时成立的分裂。
PEAK_LATTICE_AXIS_GUARD_REL = 0.02
PEAK_LATTICE_AXIS_ORIG_REL = 0.05  # 精化前两轴视为"一致"的相对差上限

# G2：梳状能量集中度终审常量（连续 (pitch, phase) 梳状打分）
COMB_ACCEPT_CONF = 0.35        # 终审采纳置信度阈值（低于则保持投票链结果）
COMB_COARSE_REL_STEP = 0.02    # 粗扫 pitch 相对步长（~2%，[3,40] 约 131 个候选）
COMB_PHASE_COARSE_STEP = 0.5   # 粗扫/细网格相位步长（px）
COMB_SEG_PERIODS = 3           # 分段粗扫每段周期数（段内 2% 网格误差残余漂移 <0.5px）
COMB_SEG_MIN_LEN = 24          # 分段粗扫最小段长（px，小周期至少 ~8 齿定相）
COMB_TOPK_EXCL_REL = 0.08      # 贪心 top-K 去重窗口（≥ 分段平台宽度 ~±4%）
COMB_REFINE_TOP_K = 3          # 精化候选数（粗扫分数前 K 个不同峰）
COMB_FINE_RANGE = 0.06         # 细 pitch 网格半范围（±6%，覆盖分段平台 ±4%）
COMB_FINE_STEP_FACTOR = 0.8    # 细网格步长 = 0.8·p/n（保证落入精确盆地）
COMB_FINE_CHUNK = 128          # 细网格分块大小（防大图张量过大）
COMB_REFINE_ROUNDS = 8         # 精化抛光轮数（步长逐轮折半）
COMB_PITCH_STEP_FLOOR = 0.002  # pitch 精化步长下限（px）
COMB_PHASE_STEP_FLOOR = 0.1    # 相位精化步长下限（px）
COMB_TIE_DIVISOR = 1.15        # 平局集判定：分数 ≥ best/1.15（≈0.87×best）
COMB_REFINE_CONSISTENCY = 0.5  # 精化/粗扫一致性下限（低于视为分段平台噪声剔除）
COMB_INT_SNAP_RATIO = 0.97     # 整数吸附：整数 pitch 分数 ≥ 0.97×best_score
COMB_FAMILY_REL = 0.05         # 同族判定：相对差 <5%
COMB_FAMILY_INT_TOL = 0.02     # 同族判定：整数倍/小整数比关系容差（相对）
COMB_FAMILY_MAX_RATIO = 4      # 同族判定：小整数比 k/j 的分子分母上限（k,j ≤ 4）
COMB_MIN_PROFILE_MULT = 4      # profile 长度 < 4×min_p 时拒绝终审

# --- JPEG 8×8 压缩网格检测（G5，IPOL 2020 交叉差分法的简化实现）---
JPEG_GRID_MIN_SIZE = 32        # detect_jpeg_grid 输入最小边长（过小无统计意义）
JPEG_GRID_QUANTILE = 90.0      # 交叉差分响应参与投票的分位（仅强响应投票）
# 显著性阈值：峰 bin 占比。均匀期望 1/64≈0.0156；校准（合成 12px 网格 +
# 噪声图）：JPEG q=60-75 重载 strength≈0.082-0.111，PNG 无损对照≈0.021-0.030，
# 真实测试图（含 image02.png 的 8 对齐内容网格）≤0.057，取 0.06 兼顾两侧裕量
JPEG_GRID_STRENGTH_THRESHOLD = 0.06
JPEG_PENALTY_FACTOR = 0.6      # JPEG 网格候选的边界强度降权因子
# 受惩罚候选：8/16/24 的 ±1 邻域（JPEG 压缩伪影周期及其邻近整数候选）
JPEG_PENALTY_PERIODS = frozenset({7, 8, 9, 15, 16, 17, 23, 24, 25})


def _runs_correct_period(
    period: float,
    vote: float,
    runs: float,
    edge_map: np.ndarray,
    axis: int,
    min_p: int,
    max_p: int,
    edge_integral: np.ndarray | None = None,
) -> float:
    """runs/GCD 整数尺度对单方向周期的交叉验证修正。

    仅在 runs 证据强（detect 中已校验命中率/一致性）时调用：
    - 一致（相对差 ≤ RUNS_AGREE_REL）：周期吸附为整数 runs；
    - runs > vote 且 runs ≈ k*vote（k≥2）且 edge(runs) ≥ 0.9*edge(vote)：
      投票误检子谐波，采纳 runs（runs 指示真实大块）；
    - runs < vote：默认不缩小周期（防把真实块缩成块内纹理周期，如
      28px 块 + 7px 纹理场景 runs=7 < vote=28），除非 period ≥ 3 且
      edge(runs) > 1.3*edge(vote) 的显式边界强度证据。

    Args:
        period: 当前周期（FFT 或被投票覆盖后的值）。
        vote: 投票得出的周期（vote_px / vote_py）。
        runs: runs/GCD 检测的整数尺度。
        edge_map: 边缘强度图 (H, W)。
        axis: 0=检测 y 方向周期，1=检测 x 方向周期。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        edge_integral: ``edge_map`` 的 (H+1, W+1) 积分图；None 时内部构建。

    Returns:
        修正后的周期。无修正条件满足时返回原 ``period``。
    """
    if period <= 0 or vote <= 0 or runs <= 0:
        return period
    v = float(vote)
    r = float(runs)
    rel = abs(r - v) / max(r, v)
    if rel <= RUNS_AGREE_REL:
        # 一致：吸附为整数尺度
        return r
    if r > v:
        ratio = r / v
        k = round(ratio)
        if k >= 2 and abs(ratio - k) / k <= 0.15:
            e_runs = _edge_band_strength(edge_map, r, axis=axis, integral=edge_integral)
            e_vote = _edge_band_strength(edge_map, v, axis=axis, integral=edge_integral)
            if e_runs >= RUNS_ADOPT_EDGE_RATIO * e_vote:
                return r
        return period
    # runs < vote：默认不缩小周期
    if v >= 3:
        e_runs = _edge_band_strength(edge_map, r, axis=axis, integral=edge_integral)
        e_vote = _edge_band_strength(edge_map, v, axis=axis, integral=edge_integral)
        if e_vote > 1e-12 and e_runs > RUNS_SHRINK_EDGE_RATIO * e_vote:
            return r
    return period


def _plausibility_gate_axis(
    p: float,
    edge_map: np.ndarray,
    axis: int,
    min_p: int,
    max_p: int,
    edge_integral: np.ndarray | None = None,
) -> float:
    """高分辨率合理性门控（单轴）：小周期若为块内纹理则放大到真实块周期。

    候选周期过小（< PLAUSIBILITY_MIN_PERIOD）时，检查其整数倍 k*p
    （k=2..，直至超出 [min_p, max_p]）的边界带边缘强度，若某倍周期
    显著更强（≥ PLAUSIBILITY_EDGE_RATIO），说明小周期是块内纹理而非
    网格边界，采纳该更大周期并链式继续放大（2→16→…）直到边界带不再增强。

    ``PLAUSIBILITY_EDGE_RATIO`` 取值依据：真实 1-2px 格点阵的各整数倍
    边界带比值 ≈1.0-1.15（相位采样总能命中边界线），而「块内 2px 纹理 +
    大块网格」场景中大倍周期命中真实块边界，比值可达 1.3-1.6。阈值取 1.3
    在两者之间留有安全间隔：0.8 会把均匀 2px 格点阵误放大到 4（红线：
    真实 2px 网格必须仍检出 2），故不采用任务建议的 0.8。

    Args:
        p: 当前周期（像素）。
        edge_map: 边缘强度图 (H, W)。
        axis: 0=检测 y 方向周期，1=检测 x 方向周期。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        edge_integral: ``edge_map`` 的 (H+1, W+1) 积分图；None 时内部构建。

    Returns:
        门控后的周期。
    """
    cur = float(p)
    # 仅当周期过小（< PLAUSIBILITY_MIN_PERIOD，如 1-2px 生成纹理）时触发门控；
    # 周期已合理时不放大到其整数倍，防把正常网格误放大。
    if cur >= PLAUSIBILITY_MIN_PERIOD:
        return cur
    for _ in range(PLAUSIBILITY_MAX_STEPS):
        e_cur = _edge_band_strength(edge_map, cur, axis=axis, integral=edge_integral)
        if e_cur <= 1e-12:
            break
        next_p = cur
        # 检查 k=2.. 的倍周期，取第一个边界带显著更强的（链式继续）
        k_max = min(int(max_p // cur) + 1, 13)  # 限制检查倍数，防性能退化
        for k in range(2, k_max):
            kp = k * cur
            if kp < min_p or kp > max_p:
                continue
            e_kp = _edge_band_strength(edge_map, kp, axis=axis, integral=edge_integral)
            if e_kp >= PLAUSIBILITY_EDGE_RATIO * e_cur:
                next_p = kp
                break
        if next_p == cur:
            break
        cur = next_p
    return cur


def _refine_period_peak_lattice(
    profile: np.ndarray,
    initial_p: float,
    min_p: float,
    max_p: float,
) -> float:
    """峰值格点拟合周期精化（G3）。

    投票/runs 交叉验证输出的周期常为整数或粗估值，真实块周期常为
    非整数（如 7.45px）。参考 Pixel-Extractor 的峰值格点拟合范式：把
    投影信号的峰视为格线采样，拟合格点间距 s 使 ``峰间距 ≈ round(峰间距/s)·s``
    的残差最小，从而把周期精化为浮点值。

    流程：
    1. ``find_peaks`` 按 prominence 找峰（阈值 0.05×max 起步，峰数 <4 时
       逐级放宽到 0.02/0.01；仍 <4 峰直接回退 ``initial_p``）；
    2. 峰按 prominence 降序排序，对 k = 4..len(peaks)（峰数多时跳采样，
       限制 minimize 调用数保证单次调用 <50ms）：取前 k 峰按位置升序，
       间距 ``spacings = np.diff(positions)``；
    3. 对每个 k 用 L-BFGS-B 最小化
       ``J(s) = sqrt(mean((spacings − round(spacings/s)·s)²))/s
       + α·missing_ratio``，其中 missing_ratio 为首末峰间缺失格线比例
       （惩罚子谐波：s 减半时一半格线无峰），α≈0.5；初值 ``initial_p``，
       bounds=[min_p, max_p]；
    4. 取所有 k 中 J 最小者为 (s*, J*)；接受准则 ``J* < 0.15`` 且
       ``|s* − initial_p|/initial_p ≤ 0.3``，否则回退 ``initial_p``；
    5. 整数吸附（可选规则，已采纳）：若 ``|s*−initial_p|/initial_p ≤ 0.02``
       且 ``|s* − round(s*)| < 0.02·s*``，返回 ``round(s*)``——投票值与
       拟合值双重一致且拟合值贴近整数时判定为干净整数网格，吸附保持
       位精确；两条件同时收紧到 2% 以降低把真实非整数周期（如 7.9px）
       误吸附为整数的风险。

    Args:
        profile: 1D 投影信号（sig_x 或 sig_y）。
        initial_p: 投票得出的初始周期（>0 时才精化）。
        min_p: 周期下界（拟合 bounds）。
        max_p: 周期上界（拟合 bounds）。

    Returns:
        精化后的周期；峰不足/拟合失败/偏移过大时回退 ``initial_p``。
    """
    from scipy.optimize import minimize
    from scipy.signal import find_peaks

    prof = np.asarray(profile, dtype=np.float64)
    p0 = float(initial_p)
    if p0 <= 0 or prof.size < 8:
        return p0
    prof_max = float(prof.max())
    if prof_max <= 1e-12:
        return p0
    # 峰检测：prominence 阈值从 0.05×max 起步逐级放宽，保证典型网格图峰数 ≥4。
    # 起步值低于建议的 0.1：真实图（如 image02）约四成边界峰 prominence 在
    # 0.05-0.1×max 之间，0.1 起步会漏掉它们使 missing_ratio 惩罚超标、
    # 正确周期被拒（实测 0.1 起步时 image02 仅 y 方向精化成功）。
    peak_idx = None
    peak_prom = None
    for prom_ratio in (0.05, 0.02, 0.01):
        idx, props = find_peaks(prof, prominence=prom_ratio * prof_max)
        if idx.size >= PEAK_LATTICE_MIN_PEAKS:
            peak_idx = idx
            peak_prom = props["prominences"]
            break
    if peak_idx is None:
        return p0
    # 峰按 prominence 降序：前 k 峰构成候选子集（强峰优先，抗弱边界缺失）
    order = np.argsort(-peak_prom, kind="stable")
    pos_by_prom = peak_idx[order].astype(np.float64)
    n_peaks = int(pos_by_prom.size)

    # k 扫描序列：4..n_peaks；峰数多时跳采样（minimize 调用数有上限），
    # 但始终包含 n_peaks（全峰拟合对真周期 missing=0，是最重要的 k）
    if n_peaks <= PEAK_LATTICE_MAX_FITS + 4:
        ks: list[int] = list(range(PEAK_LATTICE_MIN_PEAKS, n_peaks + 1))
    else:
        head = list(range(PEAK_LATTICE_MIN_PEAKS, 15))
        stride = max(2, (n_peaks - 14) // (PEAK_LATTICE_MAX_FITS - len(head)))
        tail = list(range(15, n_peaks, stride))
        ks = head + tail + [n_peaks]

    best_s = None
    best_j = np.inf
    for k in ks:
        if k > n_peaks:
            continue
        pos = np.sort(pos_by_prom[:k])
        spacings = np.diff(pos)
        if spacings.size < PEAK_LATTICE_MIN_PEAKS - 1:
            continue
        first = float(pos[0])
        last = float(pos[-1])

        def _obj_jac(
            s: np.ndarray, _sp=spacings, _k=k, _first=first, _last=last
        ) -> tuple[float, np.ndarray]:
            """目标值与解析梯度（missing 为阶梯函数，区间内梯度取 0）。"""
            sv = float(np.asarray(s).ravel()[0])
            if sv <= 0:
                return (1e9, np.array([0.0]))
            # 相邻峰至少跨 1 条格线（round 下限 1，防 0 格退化放大残差）
            r = np.maximum(np.round(_sp / sv), 1.0)
            resid = _sp - r * sv
            ms = float(np.mean(resid * resid))
            rms = float(np.sqrt(ms)) / sv
            # 首末峰间缺失格线比例：惩罚"格线应有峰却无峰"的子谐波解
            n_lines = int(round((_last - _first) / sv)) + 1
            if n_lines < _k:
                n_lines = _k
            missing = max(0.0, (n_lines - _k) / n_lines)
            # d(rms)/ds = mean(resid·(-r))/(s·sqrt(ms)) − sqrt(ms)/s²
            if ms > 1e-24:
                grad = (float(np.mean(resid * (-r))) / (sv * np.sqrt(ms))
                        - np.sqrt(ms) / (sv * sv))
            else:
                grad = 0.0
            return (rms + PEAK_LATTICE_ALPHA * missing, np.array([grad]))

        try:
            res = minimize(
                _obj_jac, x0=np.array([p0]), jac=True, method="L-BFGS-B",
                bounds=[(float(min_p), float(max_p))],
            )
        except Exception:
            continue
        if res.fun < best_j:
            best_j = float(res.fun)
            best_s = float(res.x[0])

    if best_s is None or best_j >= PEAK_LATTICE_MAX_J:
        return p0
    # 安全阀：精化值相对投票值偏移过大视为不可信（可能锁到子/超谐波）
    if abs(best_s - p0) / p0 > PEAK_LATTICE_MAX_REL:
        return p0
    # 整数吸附：双证据（与投票值一致 ≤2% 且贴近整数 <2%·s*）时吸附
    if (abs(best_s - p0) / p0 <= 0.02
            and abs(best_s - round(best_s)) < 0.02 * best_s):
        return float(round(best_s))
    return best_s


def _comb_energy_score(profile: np.ndarray, pitch: float, phase: float) -> float:
    """梳状能量集中度得分（G2 终审裁决的核心判据）。

    以 (pitch, phase) 定义梳齿位置 ``pos_k = round(phase + k·pitch)``
    （k = 0..⌊(len−1−phase)/pitch⌋，仅保留 0 ≤ pos < len 的齿），每齿
    1px 采样捕获能量 ``E = Σ profile[pos_k]``，得分为::

        score = E / total_energy − n_teeth / n_positions

    惩罚项 ``n_teeth/n`` 是随机取位的期望能量占比，故 score > 0 表示梳
    齿对齐位置的能量密度高于随机基线。真周期 P 命中全部边界能量且齿数
    最少；子谐波 P/2 以双倍齿数仅多捕获块内噪声（惩罚翻倍必然更低分），
    倍频 2P 漏一半边界（能量减半），数学上均严格低于真周期——不依赖
    经验阈值，原理性压制投票链的子谐波/倍频误检。

    Args:
        profile: 1D 投影信号（建议先做 σ≈0.5 轻高斯平滑以容忍 AA 相位
            散布；平滑应由调用方做一次并缓存，不在本函数内重复）。
        pitch: 梳齿间距（候选周期，像素）。
        phase: 首齿相位（像素）。

    Returns:
        得分（< 1，可略为负）。pitch ≤ 0、无有效齿或总能量 ≈ 0 时
        返回 0.0。
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = int(prof.size)
    p = float(pitch)
    ph = float(phase)
    if n == 0 or p <= 0:
        return 0.0
    k_max = int(np.floor((n - 1 - ph) / p))
    if k_max < 0:
        return 0.0
    pos = np.rint(ph + np.arange(k_max + 1, dtype=np.float64) * p).astype(np.int64)
    pos = pos[(pos >= 0) & (pos < n)]
    if pos.size == 0:
        return 0.0
    total = float(prof.sum())
    if total <= 1e-12:
        return 0.0
    e = float(prof[pos].sum())
    return e / total - pos.size / n


def _comb_same_family(p_a: float, p_b: float) -> bool:
    """判断两个候选周期是否属同一谐波家族（G2 separation 的同族判定）。

    同族 = 相对差 < COMB_FAMILY_REL（同一峰的粗扫邻域）或小整数比关系
    hi/lo ≈ k/j（k, j ≤ COMB_FAMILY_MAX_RATIO，容差 COMB_FAMILY_INT_TOL），
    涵盖 P 与 2P/3P/4P/P/2 等整数倍及 3/2、4/3 等有理数倍（共享同一
    格点格子的不同子采样密度）。同族候选视为同一周期假设的不同表述，
    不参与次优非同族竞争。

    Args:
        p_a: 候选周期 A。
        p_b: 候选周期 B。

    Returns:
        是否同族。非正值输入视为同族（不构成有效竞争者）。
    """
    lo, hi = (float(p_a), float(p_b)) if p_a <= p_b else (float(p_b), float(p_a))
    if hi <= 0:
        return True
    if (hi - lo) / hi < COMB_FAMILY_REL:
        return True
    if lo <= 0:
        return False
    ratio = hi / lo
    # 小整数比 k/j（含整数倍 k/1）：如 2、3、4、3/2、4/3 及其倒数
    for j in range(1, COMB_FAMILY_MAX_RATIO + 1):
        for k in range(j + 1, COMB_FAMILY_MAX_RATIO + 1):
            r = k / j
            if abs(ratio - r) / r < COMB_FAMILY_INT_TOL:
                return True
    return False


def _comb_fine_grid_search(
    prof_s: np.ndarray, total: float, pitch0: float, min_p: float, max_p: float
) -> tuple[float, float, float]:
    """细 pitch 网格 × 全相位扫描（精确全局分数），返回 (pitch, phase, score)。

    分段粗扫只能把候选定位到家族平台（半宽 ~1/(3p) ≈ ±4%），而精确全局
    分数的"盆地"（全部齿命中，齿漂移 <0.5px）宽度仅 ~p/n——从平台边缘
    做步长折半会被平台内的对角线伪对齐脊线（pitch 略偏 + 相位漂移扫过
    峰列）困住（实测合成 16px 信号从 15.83 出发卡死在 15.94/相位 1.4）。
    本函数在 ±COMB_FINE_RANGE 内以步长 ``COMB_FINE_STEP_FACTOR·p/n``
    （保证网格点距真周期 ≤0.4·p/n，落入盆地）扫 pitch，每个 pitch 全
    相位（0.5px）向量化打分，取全局最优。张量 (F, n_ph, K) 按块计算防
    大图内存超标。
    """
    n = int(prof_s.size)
    p_lo = max(pitch0 * (1.0 - COMB_FINE_RANGE), min_p)
    p_hi = min(pitch0 * (1.0 + COMB_FINE_RANGE), max_p)
    if p_hi < p_lo:
        p_hi = p_lo
    step = max(COMB_FINE_STEP_FACTOR * pitch0 / n, 1e-4)
    n_steps = int(np.ceil((p_hi - p_lo) / step)) + 1
    pf = np.linspace(p_lo, p_hi, n_steps)
    n_ph = int(np.ceil(p_hi / COMB_PHASE_COARSE_STEP - 1e-9))
    phases = np.arange(n_ph) * COMB_PHASE_COARSE_STEP
    k_max = int(np.floor((n - 1) / max(p_lo, 1e-6))) + 1
    ks = np.arange(k_max, dtype=np.float64)
    best = (float(pf[0]), 0.0, -np.inf)
    for c0 in range(0, n_steps, COMB_FINE_CHUNK):
        pf_c = pf[c0:c0 + COMB_FINE_CHUNK]
        # 位置张量 (F_c, n_ph, K)：pitch_s + phase_j + m·pitch_s
        pos = pf_c[:, None, None] + phases[None, :, None] + ks[None, None, :] * pf_c[:, None, None]
        pos_r = np.rint(pos).astype(np.int64)
        valid = (pos_r >= 0) & (pos_r < n)
        idx = np.where(valid, pos_r, 0)
        vals = np.where(valid, prof_s[idx], 0.0)
        e = vals.sum(axis=2)          # (F_c, n_ph)
        nt = valid.sum(axis=2)
        scores = e / total - nt / n
        j_flat = int(np.argmax(scores))
        fi, j = np.unravel_index(j_flat, scores.shape)
        if float(scores[fi, j]) > best[2]:
            best = (float(pf_c[fi]), float(phases[j]), float(scores[fi, j]))
    return best


def _comb_seg_coarse_score(
    prof_s: np.ndarray, total: float, pitch: float
) -> float:
    """分段最优相位粗扫分数（对 2% 网格的齿漂移免疫）。

    单相位全局打分在粗扫 pitch 误差 ε 下齿位置漂移 ε·n px（n=512、
    ε=2% 时漂移 10px），真周期的粗扫分数被系统性压低、排名由网格幸运
    落点决定（实测合成 16px 周期信号：全局粗扫 top-1 是 16/3 的幸运
    网格点，真 16 附近仅 0.14 vs 精确值 0.67）。改为把 profile 切成
    ~COMB_SEG_PERIODS 个周期长的段，每段独立扫相位取最优，段内残余
    漂移仅 ε·L_seg/2 ≈ 0.3px < 0.5px——真周期在任何网格落点下都得到
    接近精确的全局分数，排名稳定；段间累积漂移被每段重新定相吸收。

    每段贡献 ``E_seg/total − n_teeth_seg/n``（与全局分数同量纲），粗扫
    分数 = Σ 段内最优。位置张量 (S, n_ph, M) 一次性 gather+sum。

    Args:
        prof_s: 平滑后的 1D 投影信号。
        total: ``prof_s`` 的总能量。
        pitch: 候选周期。

    Returns:
        分段粗扫分数。
    """
    n = int(prof_s.size)
    l_seg = max(COMB_SEG_PERIODS * pitch, float(COMB_SEG_MIN_LEN))
    n_seg = max(1, int(np.ceil(n / l_seg)))
    seg_starts = np.arange(n_seg) * l_seg
    seg_ends = np.minimum(seg_starts + l_seg, float(n))
    n_ph = int(np.ceil(pitch / COMB_PHASE_COARSE_STEP - 1e-9))
    phases = np.arange(n_ph) * COMB_PHASE_COARSE_STEP
    m_max = int(np.ceil(l_seg / pitch)) + 1
    ms = np.arange(m_max, dtype=np.float64)
    # 位置张量 (S, n_ph, M)：seg_start + phase + m·pitch
    pos_r = np.rint(
        seg_starts[:, None, None] + phases[None, :, None] + ms[None, None, :] * pitch
    ).astype(np.int64)
    lo = seg_starts[:, None, None]
    hi = seg_ends[:, None, None]
    valid = (pos_r >= lo) & (pos_r < hi) & (pos_r >= 0) & (pos_r < n)
    idx = np.where(valid, pos_r, 0)
    vals = np.where(valid, prof_s[idx], 0.0)
    e = vals.sum(axis=2)          # (S, n_ph)
    n_teeth = valid.sum(axis=2)   # (S, n_ph)
    seg_scores = e / total - n_teeth / n
    return float(seg_scores.max(axis=1).sum())


def _comb_best_period(
    profile: np.ndarray, min_p: int, max_p: int
) -> tuple[float, float, float]:
    """梳状能量集中度周期搜索（G2 终审裁决）。

    对 σ=0.5 轻高斯平滑（本函数内做一次并缓存，粗扫/精化共用）后的
    投影信号在 [min_p, max_p] 上做连续 (pitch, phase) 梳状打分搜索：

    1. 粗扫（定位家族）：pitch 以 ~2% 相对步长递增（[3,40] 约 131 个
       候选），每 pitch 做**分段最优相位**打分（``_comb_seg_coarse_score``，
       段长 ~3 周期、相位 0.5px、位置张量向量化）——单相位全局打分在
       2% 网格误差下齿漂移 ε·n px 会把真周期分数压低一个量级（排名由
       网格幸运落点决定），分段重新定相使排名对网格落点免疫；
    2. 精化（定位精确 pitch）：粗扫分数前 3 个不同峰（贪心去重，窗口
       8% ≥ 分段平台宽 ±4%）各自做**细 pitch 网格 × 全相位扫描**
       （``_comb_fine_grid_search``，±6%、步长 0.8·p/n 保证落入精确
       全局分数的盆地），再做步长折半 3×3 局部抛光（8 轮，pitch 到
       ~0.002px、相位到 0.1px）；精化分数低于自身粗扫分数 50% 的候选
       判为分段平台噪声（局部可对齐但全局不相干），从裁决池剔除；
    3. 裁决：平局集（分数 ≥ best/1.15）中取最大 pitch——谐波家族中
       最大竞争者即真值，防 p/2、p/3 子谐波因微小分数差胜出；整数吸附
       （round(pitch) 的分数 ≥ 0.97×best 时吸附为整数，干净整数网格
       保持位精确）；confidence = quality × separation，其中 quality =
       clamp(best_score, 0, 1)（能量集中度本身），separation =
       (best − 次优非同族分数)/best（同族见 ``_comb_same_family``，
       无次优非同族时为 1）。

    Args:
        profile: 1D 投影信号（sig_x / sig_y）。
        min_p: 周期下界（≥1）。
        max_p: 周期上界。

    Returns:
        (best_pitch, best_phase, confidence)。profile 过短（< 4×min_p）、
        参数非法或总能量 ≈ 0 时返回 (0.0, 0.0, 0.0)。
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = int(prof.size)
    if min_p < 1 or max_p < min_p:
        return (0.0, 0.0, 0.0)
    if n < COMB_MIN_PROFILE_MULT * int(min_p) or n < 8:
        return (0.0, 0.0, 0.0)
    # σ=0.5 轻高斯平滑（只做一次）：容忍 AA/取整造成的 ±1px 相位散布
    from scipy.ndimage import gaussian_filter1d

    prof_s = gaussian_filter1d(prof, sigma=0.5, mode="nearest")
    total = float(prof_s.sum())
    if total <= 1e-12:
        return (0.0, 0.0, 0.0)

    # --- 粗扫：pitch ~2% 相对步长；分段最优相位（漂移免疫）向量化打分 ---
    pitches: list[float] = []
    p = float(min_p)
    while p <= float(max_p) + 1e-9:
        pitches.append(p)
        p *= 1.0 + COMB_COARSE_REL_STEP
    if not pitches:
        return (0.0, 0.0, 0.0)
    coarse_score = np.array(
        [_comb_seg_coarse_score(prof_s, total, pi) for pi in pitches],
        dtype=np.float64,
    )

    # --- 精化候选：粗扫分数降序贪心选前 K 个不同峰 ---
    # 去重窗口取 COMB_TOPK_EXCL_REL（8%，≥ 分段平台宽度 ±4%）：分段打分
    # 对 pitch 误差不敏感，同一家族平台（±4%）内的粗扫点分数几乎相同，
    # 5% 窗口会让同一峰的多个平台点占满 top-K 名额。
    order = np.argsort(-coarse_score, kind="stable")
    picked: list[int] = []
    for idx in order:
        idx = int(idx)
        pi = pitches[idx]
        if any(
            abs(pi - pitches[q]) / max(pi, pitches[q]) < COMB_TOPK_EXCL_REL
            for q in picked
        ):
            continue
        picked.append(idx)
        if len(picked) >= COMB_REFINE_TOP_K:
            break
    if not picked:
        return (0.0, 0.0, 0.0)

    # --- 精化：细 pitch 网格 × 全相位扫描（精确全局分数）+ 步长折半抛光 ---
    refined: list[tuple[float, float, float]] = []
    for idx in picked:
        p_cur, ph_cur, s_cur = _comb_fine_grid_search(
            prof_s, total, pitches[idx], float(min_p), float(max_p)
        )
        # 步长折半抛光：细网格已落入精确盆地（pitch 误差 ≤0.4·p/n），此处
        # 仅在盆地内做 3×3 局部搜索微调相位/边界齿数（规格的折半结构）
        dp = max(2.0 * COMB_FINE_STEP_FACTOR * p_cur / n, COMB_PITCH_STEP_FLOOR)
        dph = COMB_PHASE_COARSE_STEP * 0.5
        for _ in range(COMB_REFINE_ROUNDS):
            best_p, best_ph, best_s = p_cur, ph_cur, s_cur
            for p_cand in (p_cur - dp, p_cur, p_cur + dp):
                if p_cand < float(min_p) or p_cand > float(max_p):
                    continue
                for ph_cand in (ph_cur - dph, ph_cur, ph_cur + dph):
                    ph_w = ph_cand % p_cand  # 相位回卷到 [0, pitch)
                    s_cand = _comb_energy_score(prof_s, p_cand, ph_w)
                    if s_cand > best_s:
                        best_p, best_ph, best_s = p_cand, ph_w, s_cand
            p_cur, ph_cur, s_cur = best_p, best_ph, best_s
            dp = max(dp * 0.5, COMB_PITCH_STEP_FLOOR)
            dph = max(dph * 0.5, COMB_PHASE_STEP_FLOOR)
        # 一致性过滤：精化分数远低于自身分段粗扫分数（<50%）说明细网格
        # 未在该粗扫点附近找到全局相干盆地——该候选只是分段平台的噪声
        # （实测干净 7.5 网格 sig_x 的 6.89 粗扫点：分段分 0.39、精化仅
        # 0.12），作为周期假设不成立，从裁决池剔除，防其拉低 separation
        if s_cur >= COMB_REFINE_CONSISTENCY * float(coarse_score[idx]):
            refined.append((p_cur, ph_cur, s_cur))
    if not refined:
        return (0.0, 0.0, 0.0)

    # --- 裁决：平局集取最大 pitch + 整数吸附 + quality×separation 置信度 ---
    best = max(refined, key=lambda t: t[2])
    tie = [t for t in refined if t[2] >= best[2] / COMB_TIE_DIVISOR]
    sel = max(tie, key=lambda t: t[0])
    best_pitch, best_phase, best_score = sel
    # 整数吸附：round(pitch) 分数 ≥ 0.97×best 时吸附（干净整数网格位精确）
    p_int = float(round(best_pitch))
    if float(min_p) <= p_int <= float(max_p):
        s_int = _comb_energy_score(prof_s, p_int, best_phase % p_int)
        if s_int >= COMB_INT_SNAP_RATIO * best_score:
            best_pitch = p_int
            best_score = s_int
    # separation：次优非同族分数（精化候选池，排除被选中者）
    second = None
    for t in refined:
        if t is sel:
            continue
        if _comb_same_family(best_pitch, t[0]):
            continue
        if second is None or t[2] > second:
            second = t[2]
    quality = min(max(best_score, 0.0), 1.0)
    if second is None:
        separation = 1.0
    else:
        separation = (best_score - second) / max(abs(best_score), 1e-9)
        separation = min(max(separation, 0.0), 1.0)
    conf = quality * separation
    return (float(best_pitch), float(best_phase), float(conf))


def _comb_top_pitches(
    profile: np.ndarray, min_p: int, max_p: int, top_n: int = 5
) -> list[float]:
    """梳状连续搜索的 top-N 周期候选（S1 联合再搜索候选池来源）。

    复用 ``_comb_best_period`` 的分段粗扫骨架（σ=0.5 平滑 + ~2% 相对
    步长 + 分段最优相位打分），按粗扫分数降序贪心取前 ``top_n`` 个不同
    峰（8% 去重窗口），每个峰再做一次细网格精化（``_comb_fine_grid_search``，
    向量化）得到非整数 pitch——真实非整数块周期（如 slice_06 的 ~9.49px）
    只能从连续搜索产出，整数化的投票/ACF/FFT-bin 候选无法覆盖。精化分数
    远低于自身粗扫分数（< COMB_REFINE_CONSISTENCY，分段平台噪声）时
    回退该粗扫平台值，保证候选仍可用。

    Args:
        profile: 1D 投影信号（sig_x / sig_y）。
        min_p: 周期下界（≥1）。
        max_p: 周期上界。
        top_n: 返回候选数上限。

    Returns:
        ``[pitch, ...]``（按粗扫分数降序，浮点非整数）。profile 过短
        （< 4×min_p）、参数非法或总能量 ≈ 0 时返回空列表。
    """
    prof = np.asarray(profile, dtype=np.float64)
    if min_p < 1 or max_p < min_p or top_n < 1:
        return []
    n = int(prof.size)
    if n < COMB_MIN_PROFILE_MULT * int(min_p) or n < 8:
        return []
    from scipy.ndimage import gaussian_filter1d

    prof_s = gaussian_filter1d(prof, sigma=0.5, mode="nearest")
    total = float(prof_s.sum())
    if total <= 1e-12:
        return []
    # 粗扫：与 _comb_best_period 相同的 ~2% 相对步长 pitch 序列 + 分段
    # 最优相位打分（对 2% 网格落点漂移免疫，真周期排名稳定）
    pitches: list[float] = []
    p = float(min_p)
    while p <= float(max_p) + 1e-9:
        pitches.append(p)
        p *= 1.0 + COMB_COARSE_REL_STEP
    if not pitches:
        return []
    coarse_score = np.array(
        [_comb_seg_coarse_score(prof_s, total, pi) for pi in pitches],
        dtype=np.float64,
    )
    # 贪心 top-N 不同峰（去重窗口同 _comb_best_period：≥ 分段平台宽 ±4%）
    order = np.argsort(-coarse_score, kind="stable")
    picked: list[int] = []
    for idx in order:
        idx = int(idx)
        pi = pitches[idx]
        if any(
            abs(pi - pitches[q]) / max(pi, pitches[q]) < COMB_TOPK_EXCL_REL
            for q in picked
        ):
            continue
        picked.append(idx)
        if len(picked) >= top_n:
            break
    # 每峰一次细网格精化（向量化、单次 ~ms 级），把平台值收敛到精确 pitch
    out: list[float] = []
    for idx in picked:
        p_ref, _ph, s_ref = _comb_fine_grid_search(
            prof_s, total, pitches[idx], float(min_p), float(max_p)
        )
        if p_ref > 0 and s_ref >= COMB_REFINE_CONSISTENCY * float(coarse_score[idx]):
            out.append(float(p_ref))
        else:
            out.append(float(pitches[idx]))
    return out


def detect_jpeg_grid(
    gray: np.ndarray, strength_threshold: float = JPEG_GRID_STRENGTH_THRESHOLD
) -> tuple[bool, tuple[int, int], float]:
    """检测 JPEG 8×8 压缩网格（G5，参考 IPOL 2020 交叉差分法）。

    原理：交叉差分 ``C(x,y) = I(x+1,y+1) − I(x+1,y) − I(x,y+1) + I(x,y)``
    的一阶差分（真实内容边缘）被成对相减抵消，仅保留量化阶跃在横纵边界
    交汇处（JPEG 块角点）产生的混合二阶差分响应。JPEG 网格的显著特征是
    全部块角点 mod 8 后落入同一相位 bin（网格相位固定），而自然图像的强
    响应位置 mod 8 近似均匀铺满 64 个 bin。

    对 64 个可能网格原点 ``(py, px) ∈ {0..7}²`` 投票：只取响应强的位置
    （|C| 超过其 90 分位）参与投票，投给 bin ``(y mod 8, x mod 8)``，
    64 bins 直方图归一化后取峰 bin 占比为 strength。

    Args:
        gray: 灰度图数组 (H, W)。
        strength_threshold: 显著性阈值（峰 bin 占比）。均匀期望 1/64≈0.0156；
            默认 0.06（约均匀的 3.8 倍），经合成测试校准：12px 网格+噪声图
            PIL JPEG q=70 保存重载 strength≈0.08-0.11（显著），同一图 PNG
            无损保存重载≈0.02-0.03（不显著）。

    Returns:
        (is_significant, phase, strength)：是否显著、峰 bin 相位 ``(py, px)``
        （各分量 0-7）、峰 bin 占比。输入过小（<32px）、全零差分或强响应
        样本不足时返回 ``(False, (0, 0), 0.0)``。
    """
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2:
        return (False, (0, 0), 0.0)
    H, W = arr.shape
    if H < JPEG_GRID_MIN_SIZE or W < JPEG_GRID_MIN_SIZE:
        return (False, (0, 0), 0.0)
    # 交叉差分（float64，全向量化：4 次移位相减）
    c = arr[1:, 1:] - arr[:-1, 1:] - arr[1:, :-1] + arr[:-1, :-1]
    abs_c = np.abs(c)
    # 仅强响应位置参与投票：JPEG 块角点是稀疏强响应源，弱响应以噪声为主、
    # 均匀铺满 64 bins 会稀释相位峰（90 分位 ≈ 前 10% 位置）
    thr = float(np.percentile(abs_c, JPEG_GRID_QUANTILE))
    if thr <= 1e-12:
        # 全零差分（常数图）或 90 分位以下无响应：无网格证据
        return (False, (0, 0), 0.0)
    ys, xs = np.nonzero(abs_c > thr)
    if ys.size < 64:
        # 强响应样本过少，相位直方图无统计意义
        return (False, (0, 0), 0.0)
    # 投票：强响应位置 (y, x) 投给网格原点 bin (y mod 8, x mod 8)
    bins = (ys % 8) * 8 + (xs % 8)
    hist = np.bincount(bins, minlength=64).astype(np.float64)
    hist /= float(ys.size)
    idx = int(np.argmax(hist))
    strength = float(hist[idx])
    phase = (idx // 8, idx % 8)
    return (bool(strength >= strength_threshold), phase, strength)


def _vote_period(
    gray: np.ndarray,
    profile: np.ndarray,
    min_p: int,
    max_p: int,
    axis: int = 0,
    edge_map: np.ndarray | None = None,
    comb_weight: float = 0.0,
    use_comb_prefilter: bool = False,
    edge_integral: np.ndarray | None = None,
    gray_integral: np.ndarray | None = None,
    gray_integral_sq: np.ndarray | None = None,
    jpeg_penalty: bool = False,
    use_interior_ratio: bool = False,
) -> tuple[float, float]:
    """多判据投票选最佳周期，返回 (period, confidence)。

    收集候选周期（FFT 主峰 + ACF 峰 + 谐波基频，可选 Spectral Comb 候选），
    对每个候选计算归一化分数并加权求和：

    - 无 ``edge_map``（旧路径，行为与旧版本完全一致）：ACF 0.4 / FFT 0.3 /
      BVR 0.3，子谐波修正仅 k=2,3（BVR/ACF 放大 1.5 倍双条件）。
    - 有 ``edge_map``（新路径）：ACF 0.3 / FFT 0.2 / BVR 0.1 / 边界强度 0.4
      （``comb_weight`` 从边界强度份额拆出），并做 k=2..6 迭代子谐波修正，
      主判据为边界强度比 >1.3，支撑判据为 ACF/BVR 严格上升（防纯网格假修正）。
      ``use_comb_prefilter`` 时修正还需 Spectral Comb 一致性
      （comb(kP) > 1.2*comb(P)），规则细线纹理（comb 高、边界弱）被边界门控拦截。

    ``edge_integral`` / ``gray_integral`` / ``gray_integral_sq`` 为预构建积分图
    （``_build_integral`` 输出），供内部 ``_edge_band_strength`` 与
    ``_block_variance_ratio`` 复用，避免同一图被重复构建（detect 主路径只构建
    一次）；均为 None 时内部惰性构建（向后兼容）。

    ``jpeg_penalty``（G5）：JPEG 8×8 压缩网格显著时（``detect_jpeg_grid``
    判定），对 c ∈ {7,8,9,15,16,17,23,24,25}（即 8/16/24 ±1）候选的 edge
    分数乘 ``JPEG_PENALTY_FACTOR``（0.6）——在归一化前施加，惩罚同样进入
    归一化基准 max_edge，实现候选间的相对降权（JPEG 伪影是固定 8px 周期，
    会在 8/16/24px 处系统性污染投票）。子谐波修正路径新增的候选分数同样
    施加惩罚，保持一致性；ACF/FFT/BVR/comb 分数不受影响。

    Args:
        gray: 灰度图数组 (H, W)。
        profile: 1D 梯度投影信号。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        axis: 0=检测 y 方向周期，1=检测 x 方向周期。
        edge_map: 边缘强度图 (H, W)，值域 [0,1]；None 时走旧路径。
        comb_weight: Spectral Comb 判据权重（0.0-1.0，从边界强度份额拆分）。
        use_comb_prefilter: 修正前是否要求 comb 与边界强度双一致。
        edge_integral: ``edge_map`` 的 (H+1, W+1) 积分图；None 时内部构建。
        gray_integral: ``gray`` 的 (H+1, W+1) 积分图；None 时内部构建。
        gray_integral_sq: ``gray*gray`` 的 (H+1, W+1) 积分图；None 时内部构建。
        jpeg_penalty: 是否对 8/16/24 ±1 候选的边界强度降权（默认 False，
            完全兼容既有行为）。

    Returns:
        (best_period, confidence)：最佳周期与置信度（0-1）。
        无候选时返回 (0.0, 0.0)。
    """
    # 收集候选周期
    candidates: set = set()
    # FFT 主峰
    snr_fft, period_fft = _fft_band_snr(profile, min_p, max_p)
    if period_fft > 0:
        candidates.add(int(round(period_fft)))
    # ACF 峰
    peaks, acf = _acf_period(profile, min_p, max_p)
    peaks_set = set(peaks)
    candidates.update(peaks)
    # 谐波解释的基频也加入
    if peaks:
        base = _harmonic_interpret(peaks)
        if base > 0:
            candidates.add(base)
    # Spectral Comb 候选（仅新路径启用时）
    if edge_map is not None and (comb_weight > 0.0 or use_comb_prefilter):
        for cp, _cs in _comb_candidate_periods(profile, min_p, max_p, top_n=5):
            candidates.add(cp)
    if not candidates:
        return (0.0, 0.0)
    # 过滤有效范围
    candidates = sorted(c for c in candidates if min_p <= c <= max_p)
    if not candidates:
        return (0.0, 0.0)
    # 预构建积分图：边缘积分图供 _edge_band_strength 复用，灰度及其平方积分图
    # 供 2D BVR 用 O(1) 块求和；detect 主路径由外部传入，仅直接调用时惰性构建
    I_edge = None
    I_gray = gray_integral
    I_gray_sq = gray_integral_sq
    if edge_map is not None:
        I_edge = edge_integral if edge_integral is not None else _build_integral(edge_map)
        if I_gray is None or I_gray_sq is None:
            I_gray = _build_integral(gray)
            I_gray_sq = _build_integral(gray * gray)
    # 对每个候选打分：先算廉价判据（ACF/FFT/边界强度），昂贵的 2D BVR
    # 只对廉价判据排名前 VOTE_BVR_LIMIT 的候选计算（BVR 是弱判据，且对
    # 像素格点阵类输入区分度差，不为低排名候选付出相位扫描成本）。
    # 旧路径（无 edge_map）的 legacy BVR 是 1D 条带、开销小，保持逐候选计算。
    # FFT 谱支撑：对 profile 频谱只计算一次，候选分数取其频率 f=1/c 处幅值
    # （相邻 bin 线性插值）/ 带内最大幅值，替代旧「距 FFT 主峰距离」近似，
    # 消除主峰邻近候选被系统性高估/低估的问题
    n_prof = int(np.size(profile))
    if n_prof > 0:
        prof_c = np.asarray(profile, dtype=np.float64)
        prof_c = prof_c - prof_c.mean()
        spec = np.abs(np.fft.rfft(prof_c))
        freqs = np.fft.rfftfreq(n_prof, d=1.0)
        band_mask = (freqs >= 1.0 / max_p) & (freqs <= 1.0 / min_p) & (freqs > 0)
        band_max = float(spec[band_mask].max()) if band_mask.any() else 0.0
    else:
        spec = np.zeros(0)
        band_max = 0.0
    scores: dict[int, dict[str, float]] = {}
    for c in candidates:
        # ACF 峰高（归一化到 [0,1]）
        acf_score = float(acf[c]) if c < len(acf) and acf[c] > 0 else 0.0
        # FFT 谱支撑分：候选频率 f=1/c 处幅值（相邻 bin 线性插值），
        # 除以带内最大幅值归一化到 [0,1]（带内无能量时为 0）
        if band_max > 1e-12:
            pos = (1.0 / c) * n_prof  # f*n 为浮点 bin 位置
            pos = min(max(pos, 0.0), spec.size - 1)
            i0 = int(pos)
            i1 = min(i0 + 1, spec.size - 1)
            amp = float(spec[i0] + (spec[i1] - spec[i0]) * (pos - i0))
            fft_score = min(max(amp / band_max, 0.0), 1.0)
        else:
            fft_score = 0.0
        # 边界带边缘强度（廉价：积分图 + 条带求和）。use_interior_ratio 时
        # 改用「边界/格心边缘能量比」（_boundary_interior_ratio）——块内干净
        # 度作为负向判据，子谐波/倍频的格心带采到真边界而内部脏，比值被压制
        if edge_map is not None:
            if use_interior_ratio:
                edge = _boundary_interior_ratio(edge_map, float(c), axis=axis, integral=I_edge)
            else:
                edge = _edge_band_strength(edge_map, float(c), axis=axis, integral=I_edge)
        else:
            edge = 0.0
        # G5 JPEG 网格防护：8/16/24 ±1 候选降权（归一化前施加，惩罚进入
        # 归一化基准 max_edge，实现候选间相对降权；JPEG 伪影固定 8px 周期
        # 会在这些候选处系统性抬高边界强度）
        if jpeg_penalty and c in JPEG_PENALTY_PERIODS:
            edge *= JPEG_PENALTY_FACTOR
        scores[c] = {"acf": acf_score, "fft": fft_score, "bvr": 0.0, "edge": edge, "comb": 0.0}
    if edge_map is None:
        # 旧路径：legacy BVR 对全部候选计算（保持与旧版本行为一致）
        for c in candidates:
            scores[c]["bvr"] = _block_variance_ratio_legacy(gray, float(c), axis=axis)
    else:
        # 新路径：廉价判据预排序，昂贵的 2D BVR 只对前 VOTE_BVR_LIMIT 名计算
        cheap_order = sorted(
            candidates,
            key=lambda c: -(0.4 * scores[c]["acf"] + 0.3 * scores[c]["fft"] + 0.3 * scores[c]["edge"]),
        )
        for c in cheap_order[:VOTE_BVR_LIMIT]:
            scores[c]["bvr"] = _block_variance_ratio(
                gray, float(c), axis=axis, integral=I_gray, integral_sq=I_gray_sq
            )
        # Spectral Comb（仅 comb_weight>0 时计入投票）
        if comb_weight > 0.0:
            for c in candidates:
                scores[c]["comb"] = _spectral_comb_score(profile, c)
    # 归一化 BVR / edge / comb（除以各自最大值）
    max_bvr = max((s["bvr"] for s in scores.values()), default=1e-12)
    if max_bvr < 1e-12:
        max_bvr = 1e-12
    max_edge = max((s["edge"] for s in scores.values()), default=1e-12)
    if max_edge < 1e-12:
        max_edge = 1e-12
    max_comb = max((s["comb"] for s in scores.values()), default=1e-12)
    if max_comb < 1e-12:
        max_comb = 1e-12
    for c in candidates:
        scores[c]["bvr"] = scores[c]["bvr"] / max_bvr
        scores[c]["edge"] = scores[c]["edge"] / max_edge
        scores[c]["comb"] = scores[c]["comb"] / max_comb
    # 加权求和
    total: dict[int, float] = {}
    for c in candidates:
        s = scores[c]
        if edge_map is None:
            # 旧路径：与旧版本加权完全一致
            total[c] = 0.4 * s["acf"] + 0.3 * s["fft"] + 0.3 * s["bvr"]
        else:
            # 新路径：ACF 0.3 / FFT 0.2 / BVR 0.1 / 边界强度 0.4，comb 从边界强度份额拆
            # BVR 权重压低：2D 真块 BVR 对「像素格点阵」类输入（最近邻放大的
            # 像素重复结构）会强烈偏向小周期（格点对齐时块内完全均匀），实测
            # 会导致 2x 放大图检出放大因子周期的子谐波；边界强度与 ACF/FFT
            # 才是主判据（edge 在格点阵上无区分度时由 ACF+FFT 决定）。
            ew = 0.4 * (1.0 - comb_weight)
            cw = 0.4 * comb_weight
            total[c] = 0.3 * s["acf"] + 0.2 * s["fft"] + 0.1 * s["bvr"] + ew * s["edge"] + cw * s["comb"]
    # 选总分最高的
    sorted_c = sorted(candidates, key=lambda c: -total[c])
    best = sorted_c[0]
    # 置信度：主峰/次峰比 × 判据一致性
    if len(sorted_c) >= 2:
        second = sorted_c[1]
        ratio = total[best] / max(total[second], 1e-6)
    else:
        ratio = 2.0  # 只有一个候选，比值为 2（高置信）
    # 判据一致性：各判据是否都偏向 best
    s_best = scores[best]
    if edge_map is None:
        consistent = 0
        if s_best["acf"] > 0.3:
            consistent += 1
        if s_best["fft"] > 0.3:
            consistent += 1
        if s_best["bvr"] > 0.3:
            consistent += 1
        consistency = consistent / 3.0
    else:
        consistent = 0
        if s_best["acf"] > 0.3:
            consistent += 1
        if s_best["fft"] > 0.3:
            consistent += 1
        if s_best["bvr"] > 0.3:
            consistent += 1
        if s_best["edge"] > 0.3:
            consistent += 1
        consistency = consistent / 4.0
    # 综合：ratio 归一化到 [0,1]（ratio>=2 视为高置信）
    conf = min(1.0, (ratio / 2.0)) * consistency
    # 子谐波修正
    best_corrected = best
    if edge_map is None:
        # 旧路径：仅 k=2,3，BVR/ACF 放大 1.5 倍双条件（保持旧行为）
        for k in (2, 3):
            kp = best * k
            if kp in peaks_set and min_p <= kp <= max_p:
                bvr_kp = _block_variance_ratio_legacy(gray, float(kp), axis=axis)
                bvr_kp_norm = bvr_kp / max_bvr
                bvr_best_norm = scores[best]["bvr"]  # 已归一化的 BVR
                # BVR 判据：真网格的块方差对比度更高
                bvr_cond = (bvr_best_norm > 1e-12
                            and bvr_kp_norm > 1.5 * bvr_best_norm)
                # ACF 判据：抗锯齿导致半周期误检时，真实周期 kP 的 ACF
                # 显著高于半周期 best（边幅度调制使 kP 处自相关更强）
                acf_best = s_best["acf"]
                acf_kp = scores[kp]["acf"] if kp in scores else 0.0
                acf_cond = (acf_best > 1e-12
                            and acf_kp > 1.5 * acf_best)
                if bvr_cond or acf_cond:
                    best_corrected = kp
                    break  # 优先取 2P，满足则不再检查 3P
    else:
        # 新路径：k=2..6 迭代修正（支持多级链如 4→8→24）
        # 主判据：边界带边缘强度比 >1.3（真实块边界为连续强边缘）
        # 支撑判据：ACF/BVR 严格上升（防纯网格因相位偏置被假修正）
        cur = best
        e_ratio_used = 0.0
        while True:
            new_best = cur
            for k in range(2, 7):
                kp = cur * k
                if not (min_p <= kp <= max_p) or kp not in peaks_set:
                    continue
                if kp not in scores:
                    # G5：修正路径新增候选同样施加 JPEG 网格降权（与主循环一致）
                    kp_edge = _edge_band_strength(edge_map, float(kp), axis=axis, integral=I_edge)
                    if jpeg_penalty and kp in JPEG_PENALTY_PERIODS:
                        kp_edge *= JPEG_PENALTY_FACTOR
                    scores[kp] = {
                        "acf": float(acf[kp]) if kp < len(acf) and acf[kp] > 0 else 0.0,
                        "fft": 0.0,
                        "bvr": _block_variance_ratio(
                            gray, float(kp), axis=axis, integral=I_gray, integral_sq=I_gray_sq
                        ) / max_bvr,
                        "edge": kp_edge / max_edge,
                        "comb": 0.0,
                    }
                    if comb_weight > 0.0:
                        scores[kp]["comb"] = _spectral_comb_score(profile, kp) / max_comb
                e_ratio = 0.0
                if scores[cur]["edge"] > 1e-12:
                    e_ratio = scores[kp]["edge"] / scores[cur]["edge"]
                if e_ratio > 1.3:
                    # 支撑判据：ACF 严格上升，或 BVR 严格上升（仅当 cur 的
                    # BVR 已实际计算时——未计算（0.0）不提供支撑，防平凡满足）
                    support = (scores[kp]["acf"] > scores[cur]["acf"]
                               or (scores[cur]["bvr"] > 0.0
                                   and scores[kp]["bvr"] > scores[cur]["bvr"]))
                    comb_ok = True
                    if use_comb_prefilter and comb_weight > 0.0:
                        comb_ok = scores[kp]["comb"] > 1.2 * scores[cur]["comb"]
                    if support and comb_ok:
                        new_best = kp
                        e_ratio_used = e_ratio
                        break
            if new_best == cur:
                break
            cur = new_best
        best_corrected = cur
        if best_corrected != best:
            # 修正成功：置信度按边界强度比上调（修正说明更长周期更可信）
            conf = min(1.0, conf * (1.0 + 0.3 * min(max(e_ratio_used - 1.3, 0.0), 1.0)))

    if best_corrected != best:
        best = best_corrected
    return (float(best), float(conf))


def has_pixel_grid(
    gray: np.ndarray, min_p: int = 3, max_p: int = 40, snr_threshold: float = 8.0,
    edge_map: np.ndarray | None = None,
    pre_normalized: bool = False,
    edge_integral: np.ndarray | None = None,
    sig_x: np.ndarray | None = None,
    sig_y: np.ndarray | None = None,
) -> tuple[bool, float, float, float, float, float, float]:
    """检测图像是否具有像素网格周期。

    对图像沿 X/Y 方向取梯度绝对值并按另一轴求和，得到一维投影信号，
    再用 ``_fft_band_snr`` 在 [min_p, max_p] 带内寻找主周期。

    ACF 谐波解释纠正 FFT octave error 时带方向保护：仅当候选基频的
    边界带边缘强度不低于当前周期 0.7 倍时才允许缩小周期，杜绝把正确
    周期缩成块内纹理周期（AI 伪像素图最常见误检）。

    Args:
        gray: 灰度图数组。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        snr_threshold: 判定为网格的 SNR 阈值。
        edge_map: 边缘强度图 (H, W)，值域 [0,1]；None 时跳过方向保护（旧行为）。
        pre_normalized: 为 True 时跳过内部 ``_local_contrast_normalize``
            （调用方已预先归一化，detect 统一计算一次避免重复归一化）。
        edge_integral: ``edge_map`` 的 (H+1, W+1) 积分图；None 时内部构建。
        sig_x: 预计算的 X 方向投影信号 (W-1,)；None 时内部计算。
            与 ``sig_y`` 配套传入可避免调用方重复计算（detect 主路径复用）。
        sig_y: 预计算的 Y 方向投影信号 (H-1,)；None 时内部计算。

    Returns:
        (has, snr, period, period_x, period_y, snr_x, snr_y)：是否为网格、
        主方向信噪比与主周期、X 方向周期、Y 方向周期、X 方向 SNR、Y 方向 SNR。
    """
    gray = np.asarray(gray, dtype=np.float64)
    if not pre_normalized:
        gray = _local_contrast_normalize(gray)
    # X 方向：列间梯度，按行求和（外部传入时直接复用，避免重复计算）
    if sig_x is None:
        dx = np.abs(np.diff(gray, axis=1))  # (H, W-1)
        sig_x = dx.sum(axis=0)  # (W-1,)
    # Y 方向：行间梯度，按列求和
    if sig_y is None:
        dy = np.abs(np.diff(gray, axis=0))  # (H-1, W)
        sig_y = dy.sum(axis=1)  # (H-1,)
    snr_x, period_x = _fft_band_snr(sig_x, min_p, max_p)
    # ACF + 谐波解释交叉验证，纠正 FFT octave error
    peaks_x, _ = _acf_period(sig_x, min_p, max_p)
    if peaks_x:
        acf_base_x = _harmonic_interpret(peaks_x)
        # 仅当 FFT 周期是 ACF 基频的整数倍（真谐波关系）时才纠正，
        # 避免 ACF 在小 lag 处的噪声峰把正确的 FFT 周期误判为倍频
        if acf_base_x > 0 and period_x > 0:
            ratio = period_x / acf_base_x
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.15:
                # 方向保护：候选基频边界强度不得显著低于当前周期（防缩成纹理周期）
                if edge_map is None or _direction_protection_ok(
                    edge_map, acf_base_x, period_x, axis=1, integral=edge_integral
                ):
                    period_x = float(acf_base_x)

    snr_y, period_y = _fft_band_snr(sig_y, min_p, max_p)
    peaks_y, _ = _acf_period(sig_y, min_p, max_p)
    if peaks_y:
        acf_base_y = _harmonic_interpret(peaks_y)
        if acf_base_y > 0 and period_y > 0:
            ratio = period_y / acf_base_y
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.15:
                if edge_map is None or _direction_protection_ok(
                    edge_map, acf_base_y, period_y, axis=0, integral=edge_integral
                ):
                    period_y = float(acf_base_y)
    if snr_x >= snr_y:
        snr, period = snr_x, period_x
    else:
        snr, period = snr_y, period_y
    has = bool(snr >= snr_threshold)
    return (has, float(snr), float(period), float(period_x), float(period_y), float(snr_x), float(snr_y))


def _direction_protection_ok(
    edge_map: np.ndarray, smaller: float, larger: float, axis: int,
    integral: np.ndarray | None = None,
) -> bool:
    """ACF 纠正方向保护：小周期（候选基频）边界强度不得显著低于大周期。

    AI 伪像素图块内纹理周期 P/k 的边界带边缘强度远低于真实周期 P；
    若候选基频（更小周期）的边缘强度不足当前周期的 0.7 倍，判定其为
    纹理周期，拒绝把检测周期缩小到它。

    Args:
        edge_map: 边缘强度图 (H, W)。
        smaller: 小周期（候选基频）。
        larger: 大周期（当前周期）。
        axis: 0=检测 y 方向周期，1=检测 x 方向周期。
        integral: ``edge_map`` 的 (H+1, W+1) 积分图；None 时内部构建。
    """
    if edge_map is None or smaller <= 0 or larger <= 0:
        return True
    e_small = _edge_band_strength(edge_map, float(smaller), axis=axis, integral=integral)
    e_large = _edge_band_strength(edge_map, float(larger), axis=axis, integral=integral)
    if e_large <= 1e-12:
        return True
    return e_small >= 0.7 * e_large


def _estimate_grid_gradient(
    gray: np.ndarray, rel_thr: float = 0.2
) -> tuple[float, float]:
    """梯度峰中位数间距法估计网格周期（FFT 回退路径）。

    用 np.diff 计算梯度投影，找局部极大值峰，取相邻峰间距的中位数
    作为块尺寸估计。适用于 FFT 检测失败但有清晰网格边缘的图像。

    Args:
        gray: 灰度图数组。
        rel_thr: 峰值阈值占最大值的比例，默认 0.2。

    Returns:
        (period_x, period_y)：x/y 方向的块尺寸估计。
        检测失败（峰数不足）时返回 (0.0, 0.0)。
    """
    gray = np.asarray(gray, dtype=np.float64)
    H, W = gray.shape
    # X 方向：列间梯度，按行求和
    dx = np.abs(np.diff(gray, axis=1))  # (H, W-1)
    sig_x = dx.sum(axis=0)  # (W-1,)
    # Y 方向：行间梯度，按列求和
    dy = np.abs(np.diff(gray, axis=0))  # (H-1, W)
    sig_y = dy.sum(axis=1)  # (H-1,)

    def _find_peaks(profile: np.ndarray, thr: float, min_interval: int = 4) -> list[int]:
        """找局部极大值峰，阈值筛选 + 最小间距。"""
        peaks = []
        for i in range(1, len(profile) - 1):
            if (profile[i] > profile[i - 1]
                and profile[i] > profile[i + 1]
                and profile[i] >= thr):
                if len(peaks) == 0 or i - peaks[-1] >= min_interval:
                    peaks.append(i)
        return peaks

    thr_x = float(rel_thr) * float(sig_x.max()) if sig_x.max() > 0 else 0.0
    thr_y = float(rel_thr) * float(sig_y.max()) if sig_y.max() > 0 else 0.0

    # 最小峰间距随图像尺寸缩放：小图（3-4px 网格）不被合并，大图抑制噪声峰
    min_interval = max(1, min(8, min(H, W) // 200))

    peaks_x = _find_peaks(sig_x, thr_x, min_interval)
    peaks_y = _find_peaks(sig_y, thr_y, min_interval)

    if len(peaks_x) < 4 or len(peaks_y) < 4:
        return (0.0, 0.0)

    # 相邻峰间距的中位数
    intervals_x = [peaks_x[i] - peaks_x[i - 1] for i in range(1, len(peaks_x))]
    intervals_y = [peaks_y[i] - peaks_y[i - 1] for i in range(1, len(peaks_y))]

    period_x = float(np.median(intervals_x))
    period_y = float(np.median(intervals_y))

    return (period_x, period_y)


def find_phase(
    gray: np.ndarray, px: float, py: float, step: float = 0.1
) -> tuple[float, float, float]:
    """扫描最佳网格相位。

    对每个候选 (phase_x, phase_y)，将图像划分为块并计算块均值的方差。
    块均值方差越大，说明网格对齐越准确（块内同质、块间差异显著）。

    为保持高效，X/Y 方向解耦：分别寻找使列条带均值方差最大、行条带
    均值方差最大的相位，置信度归一化为块间方差与全图像方差之比。

    Args:
        gray: 灰度图数组。
        px: 块宽（像素）。
        py: 块高（像素）。
        step: 相位扫描步长。

    Returns:
        (best_phase_x, best_phase_y, conf)：最佳相位与归一化置信度。
    """
    gray = np.asarray(gray, dtype=np.float64)
    H, W = gray.shape
    # 积分图 cum2d[i, j] = sum(gray[0:i, 0:j])，便于 O(1) 区域求和
    cum = np.cumsum(np.cumsum(gray, axis=0), axis=1)
    cum2d = np.zeros((H + 1, W + 1), dtype=np.float64)
    cum2d[1:, 1:] = cum
    total_var = float(np.var(gray))
    denom = total_var if total_var > 1e-12 else 1.0

    # X 方向：固定全高，按宽 px 切竖条带，最大化条带均值方差
    best_phase_x = 0.0
    best_conf_x = 0.0
    n_phases_x = max(1, int(round(px / step)))
    for phase_x in np.linspace(0.0, px, n_phases_x, endpoint=False):
        n_blocks = int((W - phase_x) // px)
        if n_blocks < 2:
            continue
        boundaries = phase_x + np.arange(n_blocks + 1) * px
        starts = boundaries[:-1].astype(int)
        ends = np.minimum(boundaries[1:].astype(int), W)
        strip_sums = cum2d[H, ends] - cum2d[H, starts]
        areas = (ends - starts) * H
        valid = areas > 0
        if valid.sum() < 2:
            continue
        strip_means = strip_sums[valid] / areas[valid]
        score = float(np.var(strip_means))
        conf_x = score / denom
        if conf_x > best_conf_x:
            best_conf_x = conf_x
            best_phase_x = float(phase_x)

    # Y 方向：固定全宽，按高 py 切横条带，最大化条带均值方差
    best_phase_y = 0.0
    best_conf_y = 0.0
    n_phases_y = max(1, int(round(py / step)))
    for phase_y in np.linspace(0.0, py, n_phases_y, endpoint=False):
        n_blocks = int((H - phase_y) // py)
        if n_blocks < 2:
            continue
        boundaries = phase_y + np.arange(n_blocks + 1) * py
        starts = boundaries[:-1].astype(int)
        ends = np.minimum(boundaries[1:].astype(int), H)
        strip_sums = cum2d[ends, W] - cum2d[starts, W]
        areas = (ends - starts) * W
        valid = areas > 0
        if valid.sum() < 2:
            continue
        strip_means = strip_sums[valid] / areas[valid]
        score = float(np.var(strip_means))
        conf_y = score / denom
        if conf_y > best_conf_y:
            best_conf_y = conf_y
            best_phase_y = float(phase_y)

    conf = min(1.0, (best_conf_x + best_conf_y) / 2.0)
    return (best_phase_x, best_phase_y, float(conf))


def find_phase_edge(
    em: np.ndarray, px: float, py: float, step: float = 0.1
) -> tuple[float, float, float]:
    """在边缘强度图上扫描最佳网格相位（``signal="oklab"`` 色差模式）。

    与灰度 ``find_phase`` 的块均值方差判据不同，本函数对每个候选相位
    直接以块边界 1px 条带捕获的总边缘能量（条带强度和，语义同
    ``_edge_band_strength`` 的单相位条带求和，积分图加速）为分数，
    取能量最大的相位为最佳。等亮度异色块边界在灰度块方差上不可见，
    但在（色差）边缘图上是连续强边缘，该判据仍可定位。

    用条带强度和而非 ``_edge_band_strength`` 返回的均值：均值会把
    图像左端无边界能量的条带计入分母，使相位系统性偏向"丢弃左端弱
    条带"的错相（相位偏移近一个周期、块数少 1），条带和则单调偏好
    覆盖全部边界带的相位，与 ``find_phase`` 的"相位 = 首条网格线"
    约定一致。

    置信度为峰中比设计：各方向 ``(峰值 - 中位数) / 峰值``（边界清晰时
    峰值远高于错相中位数 → 趋近 1；边缘图平坦时趋近 0），X/Y 两方向
    取平均，天然有界于 [0, 1]。

    Args:
        em: 边缘强度图 (H, W)，值域 [0, 1]（``compute_edge_map`` 灰度
            边缘图或 ``_oklab_signal`` 色差边缘图）。
        px: 块宽（像素）。
        py: 块高（像素）。
        step: 相位扫描步长。

    Returns:
        (best_phase_x, best_phase_y, conf)：最佳相位与峰中比置信度。
        输入非法（非 2D 或周期 <= 0）时返回 (0.0, 0.0, 0.0)。
    """
    em = np.asarray(em, dtype=np.float64)
    if em.ndim != 2 or px <= 0 or py <= 0:
        return (0.0, 0.0, 0.0)
    H, W = em.shape
    I = _build_integral(em)
    # 各列/行的边缘强度总和（积分图 O(1) 差分），条带求和退化为索引取值求和
    col_sums = I[H, 1:] - I[H, :-1]  # (W,)
    row_sums = I[1:, W] - I[:-1, W]  # (H,)

    def _scan_axis(band_sums: np.ndarray, period: float) -> tuple[float, float]:
        """单方向相位扫描：条带位置 rint(phase+m*p)，分数 = 条带强度和。"""
        n_phases = max(1, int(round(period / step)))
        phases = np.linspace(0.0, period, n_phases, endpoint=False)
        L = band_sums.size
        scores = np.zeros(n_phases, dtype=np.float64)
        for i in range(n_phases):
            ph = float(phases[i])
            n = int((L - ph) // period)
            if n < 1:
                continue
            idx = np.rint(ph + np.arange(n + 1) * period).astype(np.int64)
            valid = (idx >= 0) & (idx < L)
            if valid.any():
                scores[i] = float(band_sums[idx[valid]].sum())
        best = float(scores.max())
        if best <= 1e-12:
            return (0.0, 0.0)
        median = float(np.median(scores))
        conf = min(1.0, max(0.0, (best - median) / best))
        return (float(phases[int(np.argmax(scores))]), conf)

    best_phase_x, conf_x = _scan_axis(col_sums, px)
    best_phase_y, conf_y = _scan_axis(row_sums, py)
    conf = (conf_x + conf_y) / 2.0
    return (best_phase_x, best_phase_y, float(conf))


def _ar_joint_pick(
    cands_x: Sequence[float],
    cands_y: Sequence[float],
    W: int,
    H: int,
    edge_map: np.ndarray,
    edge_integral: np.ndarray | None,
    cur_px: float,
    cur_py: float,
) -> tuple[float, float] | None:
    """两轴周期组合的联合评分与采纳（S1 核心可测逻辑）。

    对 cands_x × cands_y 组合评分（先预计算每轴候选的逻辑块数与边界带
    强度，组合查表单次 O(1)）：

    - ``w_est = round(W/px)``、``h_est = round(H/py)``（与 _count_blocks
      的 round 容差一致，简化为纯 round）；
    - ``ar_diff = |w_est/h_est − W/H| / (W/H)``，仅 ``ar_diff <
      AR_GUARD_ACCEPT_AR`` 的组合可被采纳；
    - 主排序 ar_diff 升序（量化到 0.005 档，同档视为等价 AR），同档按
      两轴边界强度之和 ``e_x + e_y`` 降序（同档 AR 下优先边界证据强
      的组合，防 AR 巧合对齐但周期错误的候选胜出）；
    - 接受准则：最优组合的 ``e_x* + e_y*`` ≥ ``AR_GUARD_EDGE_FLOOR`` ×
      原组合（cur_px, cur_py）的边界强度之和，否则返回 None。

    Args:
        cands_x / cands_y: 两轴候选周期池（已过滤到合法范围）。
        W / H: 图像宽高（像素）。
        edge_map: 边缘强度图 (H, W)。
        edge_integral: ``edge_map`` 的预构建积分图（None 时内部构建）。
        cur_px / cur_py: 触发防护的当前（分开使用）周期，作为边界强度基准。

    Returns:
        采纳的 (px, py)，允许 px≠py（非正方形真实块组合如 (9.49, 9.51)）；
        无 AR 达标组合或边界强度不达标时返回 None。
    """
    ratio_orig = W / H
    # 每轴候选预计算：(周期, 逻辑块数, 边界带强度)。组合循环查表，避免
    # 同一候选在 ~400 个组合里重复做积分图条带求和（Python 循环热点）
    info_x: list[tuple[float, int, float]] = []
    for px in cands_x:
        px = float(px)
        if px <= 0:
            continue
        info_x.append(
            (px, int(round(W / px)),
             _edge_band_strength(edge_map, px, axis=1, integral=edge_integral))
        )
    info_y: list[tuple[float, int, float]] = []
    for py in cands_y:
        py = float(py)
        if py <= 0:
            continue
        info_y.append(
            (py, int(round(H / py)),
             _edge_band_strength(edge_map, py, axis=0, integral=edge_integral))
        )
    if not info_x or not info_y:
        return None
    # 原组合边界强度之和（接受准则的基准）
    e_cur = (
        _edge_band_strength(edge_map, float(cur_px), axis=1, integral=edge_integral)
        + _edge_band_strength(edge_map, float(cur_py), axis=0, integral=edge_integral)
    )
    best: tuple[tuple[int, float], float, float, float] | None = None
    for px, w_est, e_x in info_x:
        if w_est < 1:
            continue
        for py, h_est, e_y in info_y:
            if h_est < 1:
                continue
            ar_diff = abs(w_est / h_est - ratio_orig) / ratio_orig
            if ar_diff >= AR_GUARD_ACCEPT_AR:
                continue
            # 同 ar 档（0.005 量化）按边界强度之和降序
            key = (round(ar_diff / 0.005), -(e_x + e_y))
            if best is None or key < best[0]:
                best = (key, px, py, e_x + e_y)
    if best is None:
        return None
    if best[3] >= AR_GUARD_EDGE_FLOOR * e_cur:
        return (best[1], best[2])
    return None


def _ar_joint_research(
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    W: int,
    H: int,
    cur_px: float,
    cur_py: float,
    edge_map: np.ndarray,
    min_p: int,
    max_p: int,
    edge_integral: np.ndarray | None = None,
    vote_px: float = 0.0,
    vote_py: float = 0.0,
) -> tuple[float, float] | None:
    """长宽比失配时的两轴联合周期再搜索（S1 回退链第二级）。

    单轴投票在非整数块周期（如 ~9.5px）下两轴可分别锁到不同错误整数
    （slice_06：x=10/y=8，输出 67×76 长宽比畸变 20.2%）。本函数收集
    两轴各自的多源候选池：

    - 该轴投票值（vote_px/vote_py 原始值，detect 作用域可能已被
      runs/G3/G2 覆盖到 period_x/period_y）与当前值；
    - FFT 主峰周期（抛物线插值，非整数）；
    - ACF 峰前 5 个（整数 lag）；
    - 梳状连续搜索 top-5（``_comb_top_pitches``，非整数——真实 9.49
      这类值只能来自连续搜索）；
    - 每个候选扩展 ×2 与 ×0.5（谐波家族换算，如 9.49→18.98/4.75），
      全部过滤到 [max(min_p, 3), max_p]。

    对 cands_x × cands_y 组合（去重后 ~20×20）做长宽比守恒 + 边界强度
    联合评分（``_ar_joint_pick``），返回最优组合；无达标组合返回 None
    （调用方回退现有正方形逻辑）。

    Args:
        sig_x / sig_y: 两轴 1D 投影信号（detect 作用域已算好）。
        W / H: 图像宽高（像素）。
        cur_px / cur_py: 触发防护的当前（分开使用）周期。
        edge_map: 边缘强度图 (H, W)。
        min_p / max_p: 候选周期范围。
        edge_integral: ``edge_map`` 的预构建积分图（None 时内部构建）。
        vote_px / vote_py: 两轴投票原始值（增强候选池覆盖）。

    Returns:
        采纳的 (px, py) 或 None（无优解）。
    """
    lo = float(max(min_p, 3))
    hi = float(max_p)

    def _axis_raw(sig: np.ndarray, cur_p: float, vote_p: float) -> set[float]:
        """收集单轴直接源候选：当前/投票值、FFT 主峰、ACF 前 5 峰、梳状 top-5。"""
        raw: set[float] = set()
        for v in (float(cur_p), float(vote_p)):
            if v > 0:
                raw.add(v)
        _snr_fft, p_fft = _fft_band_snr(sig, min_p, max_p)
        if p_fft > 0:
            raw.add(float(p_fft))
        peaks, _acf = _acf_period(sig, min_p, max_p)
        for pk in peaks[:5]:
            raw.add(float(pk))
        for cp in _comb_top_pitches(sig, min_p, max_p, top_n=5):
            if cp > 0:
                raw.add(float(cp))
        return raw

    def _expand(raw: set[float]) -> list[float]:
        """扩展 ×2/×0.5 谐波换算（如 9.49→18.98/4.75）并过滤到 [lo, hi]。"""
        out: set[float] = set()
        for c in raw:
            for v in (c, 2.0 * c, 0.5 * c):
                if lo <= v <= hi:
                    out.add(round(v, 4))
        return sorted(out)

    raw_x = _axis_raw(sig_x, cur_px, vote_px)
    raw_y = _axis_raw(sig_y, cur_py, vote_py)
    # 正方形网格先验的对称交叉注入：对侧轴候选与本轴某候选同族（相对差
    # < COMB_FAMILY_REL，即同一周期的两轴表述）时注入本轴参与评分。单轴
    # 检测器在非整数周期下可整体偏移（slice_06 x 轴 comb/fft 一致锁
    # 9.93 而真值 ~9.49，y 轴 comb 锁 9.461），对侧同族候选注入后仍需
    # 通过本轴边界带强度的图像域评分验证，不引入无证据的周期假设。
    # 两侧配对都基于注入前的原始集合（防连锁注入），且只取相对差分支。
    orig_x, orig_y = set(raw_x), set(raw_y)
    raw_x |= {
        b for b in orig_y
        if any(abs(a - b) / max(a, b) < COMB_FAMILY_REL for a in orig_x)
    }
    raw_y |= {
        a for a in orig_x
        if any(abs(a - b) / max(a, b) < COMB_FAMILY_REL for b in orig_y)
    }
    cands_x = _expand(raw_x)
    cands_y = _expand(raw_y)
    if not cands_x or not cands_y:
        return None
    return _ar_joint_pick(
        cands_x, cands_y, W, H, edge_map, edge_integral, cur_px, cur_py
    )


def detect(
    gray: np.ndarray, min_p: int = 3, max_p: int = 40, step: float = 0.1,
    snr_threshold: float = 8.0,
    edge_tol: int = 3,
    enable_subpixel_refine: bool = True,
    smooth_strength: float = 0.5,
    outlier_reject_ratio: float = 0.5,
    comb_weight: float = 0.0,
    use_comb_prefilter: bool = False,
    enable_runs_crosscheck: bool = True,
    enable_plausibility_gate: bool = True,
    signal: str = "gray",
    enable_peak_lattice_fit: bool = False,
    enable_comb_energy_score: bool = False,
    jpeg_grid_guard: bool = False,
    aspect_guard_tol: float | None = None,
    enable_interior_cleanliness: bool = False,
) -> Grid:
    """自动检测像素网格。

    Args:
        gray: 灰度图数组 (H, W) 或 RGB 图数组 (H, W, 3)。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        step: 相位扫描步长。
        snr_threshold: 判定为网格的 SNR 阈值，传递给 ``has_pixel_grid``。
        edge_tol: 共享边界搜索半径（像素），透传给 ``expand_grid_edge_guided``。
        enable_subpixel_refine: 是否启用亚像素精炼，透传给 ``detect_squares``
            与 ``expand_grid_edge_guided``。
        smooth_strength: 全局正则化混合强度（0.0=纯观测，1.0=完全用全局线性模型），
            透传给 ``expand_grid_edge_guided``。
        outlier_reject_ratio: 离群间距剔除阈值比例，间距偏离 period 超过该比例的
            观测不参与全局模型拟合，透传给 ``expand_grid_edge_guided``。
        comb_weight: Spectral Comb 判据权重（0.0-1.0），透传给 ``_vote_period``，
            默认 0（comb 默认关闭，真实图安全）。
        use_comb_prefilter: 子谐波修正前是否要求 comb 与边界强度双一致，
            透传给 ``_vote_period``，默认 False。
        enable_runs_crosscheck: 是否启用 runs/GCD 整数尺度交叉验证
            （Task 4，默认开启；False 时回退到纯投票路径）。
        enable_plausibility_gate: 是否启用高分辨率合理性门控
            （Task 5，默认开启；False 时回退到纯投票路径）。
        signal: 检测信号模式，``"gray"``（默认，BT.601 灰度）或 ``"oklab"``
            （OKLAB 感知色差）。oklab 模式仅对 3D RGB 输入生效：投影信号、
            边缘图与相位搜索改用色差信号（等亮度异色块边界可见），BVR、
            runs 交叉验证、梯度回退等灰度判据仍用 BT.601 灰度；2D 输入
            恒走灰度路径（signal 参数被忽略）。
        enable_peak_lattice_fit: 是否启用峰值格点拟合周期精化（G3，默认
            关闭）。投票周期确定后对投影信号（灰度与 oklab 模式各自的
            sig_x/sig_y）做峰值格点拟合，把周期精化为浮点值（支持非整数
            块尺寸如 7.5px）；失败自动回退投票值。附轴一致性防护：精化前
            两轴周期一致（相对差 ≤ 5%）而精化后显著分裂（相对差 > 2%）
            时判定拟合锁错格点，两轴整体回退精化前投票值。
        enable_comb_energy_score: 是否启用梳状能量集中度终审（G2，默认
            关闭）。G3 精化后对投影信号做连续 (pitch, phase) 梳状打分：
            真周期齿数最少且捕获全部边界能量，子谐波以双倍齿数仅多捕获
            块内噪声（覆盖惩罚翻倍）、倍频漏一半边界（能量减半），数学
            上必然低于真周期——原理性压制投票链的子谐波/倍频误检（如
            干净硬边 7.5px 网格投票锁 30 的对齐倍数误检）。每轴独立
            接受：置信度 ≥ 0.35 且 pitch > 0 时覆盖该轴周期，低置信自动
            回退 G3/投票链结果；梳相位仅作参考，不透传 find_phase。
        jpeg_grid_guard: 是否启用 JPEG 8×8 压缩网格检测与候选降权（G5，
            默认关闭）。用 ``detect_jpeg_grid`` 对灰度做交叉差分投票（一次，
            两轴共用），检测出显著 JPEG 网格相位时向两轴 ``_vote_period``
            传 ``jpeg_penalty``，对 8/16/24 ±1 候选的边界强度降权
            （×0.6），防护 JPEG 压缩伪影的固定 8px 周期污染周期投票。
            G2 梳状终审/G3 峰值拟合为图像域独立判据，不受惩罚影响；
            检测结果 ``(is_significant, phase, strength)`` 记录进
            ``Grid.jpeg_grid`` 供 A/B 分析。
        aspect_guard_tol: 长宽比守恒防护阈值（S1）。两轴周期差异大而分开
            使用（rel_diff ≥ 0.15）时，若推导出的逻辑分辨率长宽比相对
            输入宽高比的偏差超过该阈值，先做两轴联合周期再搜索
            （``_ar_joint_research``，候选池含梳状连续搜索的非整数
            pitch，可找回 (9.49, 9.51) 这类真实非整数块组合），无优解
            再回退正方形块。None 时用模块级 ``AR_GUARD_RATIO_DIFF``
            （0.12；旧硬编码 0.3 过松，slice_06 的 20.2% 畸变被放过）。
        enable_interior_cleanliness: 是否启用「内部洁净度」边界评分（P0，
            默认 False，零回归）。True 时 ``_vote_period`` 的边界强度分
            改用边界/格心边缘能量比（``_boundary_interior_ratio``），
            以「块内是否干净」为负向判据压制子谐波/倍频——削弱真实 AI
            图纯边界能量区分度不足(≈1.0)带来的误检。默认 False 完全兼容
            既有行为。

    Returns:
        Grid: 检测结果。

    Raises:
        ValueError: 未检测到像素网格周期。
    """
    arr = np.asarray(gray, dtype=np.float64)
    # 输入分发：oklab 色差模式仅对 3D RGB 输入生效（2D 输入静默走灰度路径）；
    # 3D + signal="gray" 先转 BT.601 灰度再走原路径（向后兼容）
    use_oklab = arr.ndim == 3 and signal == "oklab"
    if use_oklab:
        # 色差路径：投影信号/边缘图来自 OKLAB 色差（_oklab_signal），
        # 灰度判据（BVR/runs/梯度回退）仍用 BT.601 灰度
        sig_x, sig_y, edge_map, gray_arr = _oklab_signal(arr)
        # has_pixel_grid 直传 sig_x/sig_y，灰度参数不参与信号计算
        norm_gray = gray_arr
    else:
        if arr.ndim == 3:
            gray_arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            gray_arr = arr
        # 边缘图统一计算一次，供方向保护 / 投票 / 方块检测 / 边缘扩展复用
        edge_map = compute_edge_map(gray_arr)
        # Task 2：归一化灰度只计算一次，同时供 has_pixel_grid 与 sig_x/sig_y
        # （两阶段使用同一信号来源，消除 has_pixel_grid(FFT) 与投票的预处理不一致）
        norm_gray = _local_contrast_normalize(gray_arr)
        # 投影信号只计算一次：has_pixel_grid 与下方投票共用同一对信号（F1 去重）
        sig_x = np.abs(np.diff(norm_gray, axis=1)).sum(axis=0)
        sig_y = np.abs(np.diff(norm_gray, axis=0)).sum(axis=1)
    H, W = gray_arr.shape
    # S1 长宽比守恒防护阈值：None 时用模块级默认（AR_GUARD_RATIO_DIFF）
    guard_tol = AR_GUARD_RATIO_DIFF if aspect_guard_tol is None else float(aspect_guard_tol)
    # 积分图统一构建一次，供 _edge_band_strength 与 2D BVR 复用，
    # 避免每个候选周期/每条路径重复重建整图积分图（大图下 O(H·W) 是热点）
    I_edge = _build_integral(edge_map)
    I_gray = _build_integral(gray_arr)
    I_gray_sq = _build_integral(gray_arr * gray_arr)

    # --- G5：JPEG 8×8 压缩网格检测（一次，两轴共用）---
    # 交叉差分投票检测 JPEG 网格相位；显著时向两轴 _vote_period 传
    # jpeg_penalty 对 8/16/24 ±1 候选降权（JPEG 伪影固定 8px 周期，会在
    # 8/16/24px 处系统性污染投票）。检测结果记录进 Grid.jpeg_grid 供
    # A/B 分析；G2/G3 为图像域独立判据，不受惩罚影响。
    jpeg_grid_info: tuple = ()
    jpeg_penalty_flag = False
    if jpeg_grid_guard:
        j_sig, j_phase, j_strength = detect_jpeg_grid(gray_arr)
        jpeg_grid_info = (j_sig, j_phase, j_strength)
        jpeg_penalty_flag = j_sig

    def _phase_search(p_x: float, p_y: float) -> tuple[float, float, float]:
        """相位搜索：oklab 模式在色差边缘图上找边界带能量峰，灰度模式用块方差。"""
        if use_oklab:
            return find_phase_edge(edge_map, p_x, p_y, step)
        return find_phase(gray_arr, p_x, p_y, step)

    has, snr, period, period_x, period_y, snr_x, snr_y = has_pixel_grid(
        norm_gray, min_p, max_p, snr_threshold=snr_threshold, edge_map=edge_map,
        pre_normalized=True, edge_integral=I_edge, sig_x=sig_x, sig_y=sig_y,
    )
    has_fft = bool(has)  # 保存 FFT 原始判定，用于低置信度拒绝
    if not has:
        # FFT 检测失败，回退到梯度峰中位数间距法
        grad_px, grad_py = _estimate_grid_gradient(gray_arr)
        if grad_px <= 0 or grad_py <= 0:
            raise ValueError("未检测到像素网格周期，输入可能不是AI生成的伪像素图")
        # 用梯度回退的周期构建检测参数
        period_x = grad_px
        period_y = grad_py
        period = (grad_px + grad_py) / 2.0
        snr = 0.0
        snr_x = 0.0
        snr_y = 0.0
        has = True

    # 多判据投票：ACF + FFT + 块方差对比度 + 边界强度综合选周期
    # 投影信号已在上方与 has_pixel_grid 共用计算（同一信号来源）；
    # BVR 判据仍用原始 gray_arr（只换 profile，不改变 BVR 语义）
    vote_px, conf_x = _vote_period(
        gray_arr, sig_x, min_p, max_p, axis=1, edge_map=edge_map,
        comb_weight=comb_weight, use_comb_prefilter=use_comb_prefilter,
        edge_integral=I_edge, gray_integral=I_gray, gray_integral_sq=I_gray_sq,
        jpeg_penalty=jpeg_penalty_flag, use_interior_ratio=enable_interior_cleanliness,
    )
    vote_py, conf_y = _vote_period(
        gray_arr, sig_y, min_p, max_p, axis=0, edge_map=edge_map,
        comb_weight=comb_weight, use_comb_prefilter=use_comb_prefilter,
        edge_integral=I_edge, gray_integral=I_gray, gray_integral_sq=I_gray_sq,
        jpeg_penalty=jpeg_penalty_flag, use_interior_ratio=enable_interior_cleanliness,
    )
    # 投票结果覆盖 FFT：置信度高，或边界强度显著更高
    # （FFT 主峰易被块内纹理污染，而边界带边缘强度判据不受其影响；
    #   纹理场景投票置信度偏低，但边界强度仍可靠，故用 edge 无条件覆盖）
    if vote_px > 0:
        e_vote = _edge_band_strength(edge_map, vote_px, axis=1, integral=I_edge)
        e_fft = _edge_band_strength(edge_map, period_x, axis=1, integral=I_edge) if period_x > 0 else 0.0
        if conf_x > 0.3 or (e_vote > 1e-12 and e_fft > 1e-12 and e_vote > 1.2 * e_fft):
            period_x = float(vote_px)
    if vote_py > 0:
        e_vote = _edge_band_strength(edge_map, vote_py, axis=0, integral=I_edge)
        e_fft = _edge_band_strength(edge_map, period_y, axis=0, integral=I_edge) if period_y > 0 else 0.0
        if conf_y > 0.3 or (e_vote > 1e-12 and e_fft > 1e-12 and e_vote > 1.2 * e_fft):
            period_y = float(vote_py)
    # 综合置信度
    vote_conf = (conf_x + conf_y) / 2.0

    # --- Task 4：runs/GCD 整数尺度交叉验证（最终 px/py 选择前）---
    # 基于像素行程长度 GCD 的整数尺度作为投票的正交交叉验证先验：
    # 证据强（命中率达标 + x/y 一致 + 尺度在范围）时按规则修正周期，
    # 失败/无证据时静默回退到纯投票结果（默认行为零回归）。
    if enable_runs_crosscheck:
        runs_x, runs_y, runs_hit, _runs_conf = detect_integer_scale(
            gray_arr, min_p=min_p, max_p=max_p,
        )
        if (runs_x > 0 and runs_y > 0 and runs_x == runs_y
                and runs_hit >= RUNS_STRONG_HIT_RATE):
            runs_scale = float(runs_x)
            period_x = _runs_correct_period(
                period_x, vote_px, runs_scale, edge_map, axis=1,
                min_p=min_p, max_p=max_p, edge_integral=I_edge,
            )
            period_y = _runs_correct_period(
                period_y, vote_py, runs_scale, edge_map, axis=0,
                min_p=min_p, max_p=max_p, edge_integral=I_edge,
            )

    # --- G3：峰值格点拟合周期精化（runs/GCD 交叉验证之后、px/py 选择之前）---
    # 投票/runs 输出的周期常为整数或粗估值（真实案例 7.45/7.48px 非整数周期
    # 未被精化）；对投影信号做峰值格点拟合把周期精化为浮点值。灰度与 oklab
    # 两种信号模式都生效（profile 来源不同而已）。失败自动回退投票值，
    # 精化成功则后续 px/py 选择（含统一正方形分支、长宽比校验）、门控、
    # 相位与方块检测全部自然使用精化值。
    if enable_peak_lattice_fit:
        orig_px, orig_py = period_x, period_y
        if period_x > 0:
            period_x = _refine_period_peak_lattice(sig_x, period_x, min_p, max_p)
        if period_y > 0:
            period_y = _refine_period_peak_lattice(sig_y, period_y, min_p, max_p)
        # 轴一致性防护：正方形网格两轴投影来自同一格点结构，精化成功时
        # 两轴周期应几乎一致；若投票周期本就两轴一致、精化后反而显著
        # 分裂（单轴锁错格点，如 image02 x 回退 20.0 / y 精化 20.804），
        # 判定拟合不可信——两轴整体回退精化前投票值，行为与关闭 G3 一致。
        # 原本两轴就不一致的非方格（相对差 > 5%，如 slice_01 的 8/7）
        # 精化各自独立进行，不适用本防护；两轴同时精化失败时
        # refined == orig，回退为无操作，同样不改变结果。
        if orig_px > 0 and orig_py > 0:
            orig_diff = abs(orig_px - orig_py) / max(orig_px, orig_py)
            refined_diff = abs(period_x - period_y) / max(period_x, period_y)
            if (orig_diff <= PEAK_LATTICE_AXIS_ORIG_REL
                    and refined_diff > PEAK_LATTICE_AXIS_GUARD_REL):
                period_x, period_y = orig_px, orig_py

    # --- G2：梳状能量集中度终审（G3 之后、px/py 选择之前）---
    # 连续 (pitch, phase) 梳状打分对两轴投影信号做周期终审：子谐波以双倍
    # 齿数仅多捕获块内噪声（覆盖惩罚翻倍）、倍频漏一半边界（能量减半），
    # 数学上必然低于真周期——原理性压制投票链的子谐波/倍频误检。每轴独立
    # 接受：置信度 ≥ COMB_ACCEPT_CONF 且 pitch > 0 时覆盖该轴周期，否则
    # 保持 G3/投票链结果（低置信自动回退）。梳相位仅作参考不透传
    # find_phase（find_phase/find_phase_edge 有自己的相位搜索）；终审
    # 覆盖后 px/py 选择、门控、相位与方块检测全部自然使用新值。
    comb_conf_x = 0.0
    comb_conf_y = 0.0
    if enable_comb_energy_score:
        pitch_cx, _phase_cx, conf_cx = _comb_best_period(sig_x, min_p, max_p)
        comb_conf_x = float(conf_cx)
        if conf_cx >= COMB_ACCEPT_CONF and pitch_cx > 0:
            period_x = float(pitch_cx)
        pitch_cy, _phase_cy, conf_cy = _comb_best_period(sig_y, min_p, max_p)
        comb_conf_y = float(conf_cy)
        if conf_cy >= COMB_ACCEPT_CONF and pitch_cy > 0:
            period_y = float(pitch_cy)

    # --- px/py 选择策略 ---
    # 1. 某方向检测失败（period 为 0）：使用非零值作为统一 px=py
    if period_x <= 0 and period_y <= 0:
        px = py = float(period)
    elif period_x <= 0:
        px = py = float(period_y)
    elif period_y <= 0:
        px = py = float(period_x)
    else:
        # 2. 两者都有效：判断相对差异
        rel_diff = abs(period_x - period_y) / max(period_x, period_y)
        if rel_diff < 0.15:
            # 容差内：按 SNR 加权平均作为统一正方形块
            total_snr = snr_x + snr_y
            if total_snr > 0:
                unified = (period_x * snr_x + period_y * snr_y) / total_snr
            else:
                unified = (period_x + period_y) / 2.0
            px = py = float(unified)
        else:
            # 3. 差异大：分别使用，但做长宽比守恒防护（S1 回退链：
            #    阈值收紧 → 两轴联合再搜索 → 正方形回退保底）
            px = float(period_x)
            py = float(period_y)
            phase_x_t, phase_y_t, conf_t = _phase_search(px, py)
            w_logic_t = int(round((W - phase_x_t) / px))
            h_logic_t = int(round((H - phase_y_t) / py))
            if w_logic_t > 0 and h_logic_t > 0:
                ratio_out = w_logic_t / h_logic_t
                ratio_orig = W / H
                ratio_diff = abs(ratio_out - ratio_orig) / ratio_orig
                if ratio_diff > guard_tol:
                    # 长宽比畸变：先做两轴联合周期再搜索（候选池含梳状连续
                    # 搜索的非整数 pitch，找回非整数真实块周期组合如
                    # (9.49, 9.51)，其输出 AR 守恒且边界强度达标）
                    joint = _ar_joint_research(
                        sig_x, sig_y, W, H, px, py, edge_map, min_p, max_p,
                        edge_integral=I_edge, vote_px=vote_px, vote_py=vote_py,
                    )
                    if joint is not None:
                        px, py = joint
                    else:
                        # 再搜索无优解：回退为正方形块（取边界强度较高方向的
                        # 周期，SNR 同样受块内纹理污染，边界强度判据更可靠）
                        e_x = _edge_band_strength(edge_map, px, axis=1, integral=I_edge)
                        e_y = _edge_band_strength(edge_map, py, axis=0, integral=I_edge)
                        if e_x >= e_y:
                            px = py = float(period_x)
                        else:
                            px = py = float(period_y)
            # 若回退了，下面会重新计算相位；若未回退，也重新计算（统一流程）

    # --- Task 5：高分辨率合理性门控（确定最佳周期后、find_phase 前）---
    # 防止把 1-2px 生成纹理误当真实网格：若最佳周期 < 3px 且其整数倍
    # 边界带显著更强（说明小周期是块内纹理而非网格边界），采纳更大周期。
    if enable_plausibility_gate:
        if abs(px - py) <= 0.15 * max(px, py):
            # 统一正方形块：x/y 两轴分别门控后取一致结果，保持 px == py
            gx = _plausibility_gate_axis(px, edge_map, axis=1, min_p=min_p, max_p=max_p, edge_integral=I_edge)
            gy = _plausibility_gate_axis(py, edge_map, axis=0, min_p=min_p, max_p=max_p, edge_integral=I_edge)
            if gx >= gy:
                px = py = gx
            else:
                px = py = gy
        else:
            px = _plausibility_gate_axis(px, edge_map, axis=1, min_p=min_p, max_p=max_p, edge_integral=I_edge)
            py = _plausibility_gate_axis(py, edge_map, axis=0, min_p=min_p, max_p=max_p, edge_integral=I_edge)

    phase_x, phase_y, conf = _phase_search(px, py)
    # 综合投票置信度与相位置信度
    conf = min(1.0, max(conf, vote_conf))

    def _count_blocks(length: float, period: float, phase: float) -> int:
        """由图像尺寸/周期/相位推逻辑块数。

        round 避免 (length-phase)/period = 7.997 时被 int() 截断丢行；
        但 round 向上取整可能使最后一条网格线越界（如 113.72 → 114），
        超出小容差（2% 块尺寸，容纳亚像素相位估计误差）时回退一行。
        """
        n = int(round((length - phase) / period))
        if n > 1 and phase + n * period > length + 0.02 * period:
            n -= 1
        return n

    w_logic_phase = min(max(1, _count_blocks(W, px, phase_x)), W)
    h_logic_phase = min(max(1, _count_blocks(H, py, phase_y)), H)

    # --- 新流程：方块检测 + BFS 分配 + 边缘引导扩展 ---
    # edge_map 已在函数开头统一计算，这里直接复用
    squares = detect_squares(edge_map, px, py, enable_subpixel_refine=enable_subpixel_refine)
    placed, grid_bounds = assign_grid_bfs(squares, px, py, W, H)
    if placed:
        cell_ys, cell_xs = expand_grid_edge_guided(
            edge_map, placed, grid_bounds, px, py,
            edge_tol=edge_tol,
            enable_subpixel_refine=enable_subpixel_refine,
            smooth_strength=smooth_strength,
            outlier_reject_ratio=outlier_reject_ratio,
        )
        raw_w = cell_xs.shape[1] - 1
        raw_h = cell_ys.shape[0] - 1
        # 与相位法得到的逻辑尺寸取较小，保证最后网格线不越界
        w_logic = max(1, min(raw_w, w_logic_phase))
        h_logic = max(1, min(raw_h, h_logic_phase))
        cell_ys = cell_ys[: h_logic + 1, : w_logic + 1]
        cell_xs = cell_xs[: h_logic + 1, : w_logic + 1]
    else:
        # 兜底：方块检测失败，由全局相位 + 等距网格生成
        w_logic = w_logic_phase
        h_logic = h_logic_phase
        cell_ys, cell_xs = _equidistant_cell_grid(phase_x, phase_y, px, py, w_logic, h_logic)

    # 候选分辨率：主分辨率及其邻近项
    candidates: list[tuple[int, int, float]] = [(w_logic, h_logic, conf)]
    for dw, dh, weight in (
        (1, 0, 0.9),
        (0, 1, 0.9),
        (-1, 0, 0.8),
        (0, -1, 0.8),
        (1, 1, 0.85),
        (-1, -1, 0.85),
    ):
        cw = w_logic + dw
        ch = h_logic + dh
        if cw > 0 and ch > 0:
            candidates.append((cw, ch, conf * weight))
    # 低置信度拒绝（仅在极端情况下）
    if conf < 0.05 and not has_fft:
        raise ValueError("网格检测置信度低，建议人工指定逻辑分辨率（user_hint）")
    # Spectral Comb 元信息：对最终周期在主投影信号上取分（0 表示无谐波结构）
    comb_score = 0.0
    if px > 0:
        comb_score = max(
            _spectral_comb_score(sig_x, int(round(px))),
            _spectral_comb_score(sig_y, int(round(py))),
        )
    return Grid(
        w_logic, h_logic, px, py, phase_x, phase_y, conf, candidates,
        cell_ys=cell_ys, cell_xs=cell_xs, comb_score=comb_score,
        low_confidence=bool(conf < 0.4),
        comb_energy_conf=(comb_conf_x + comb_conf_y) / 2.0,
        jpeg_grid=jpeg_grid_info,
    )


def detect_with_user_grid(
    gray: np.ndarray, w: int, h: int, step: float = 0.1, signal: str = "gray"
) -> Grid:
    """用户指定逻辑分辨率，反推块尺寸并定相。

    Args:
        gray: 灰度图数组 (H, W) 或 RGB 图数组 (H, W, 3)。
        w: 用户指定逻辑宽。
        h: 用户指定逻辑高。
        step: 相位扫描步长。
        signal: 检测信号模式，``"gray"``（默认）或 ``"oklab"``。oklab 模式
            仅对 3D RGB 输入生效：相位搜索改在 OKLAB 色差边缘图上进行
            （等亮度异色块边界可见）；2D 输入恒走灰度路径。

    Returns:
        Grid: 检测结果。
    """
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim == 3 and signal == "oklab":
        # 色差路径：相位搜索用 OKLAB 色差边缘图（仅需边缘图，无需投影信号）
        gray_arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        edge_map = _oklab_edge_map(rgb_to_oklab(arr))
    else:
        if arr.ndim == 3:
            gray_arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            gray_arr = arr
        edge_map = None
    H, W = gray_arr.shape
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ValueError(f"逻辑分辨率必须为正整数，得到 w={w}, h={h}")
    if w > W or h > H:
        raise ValueError(f"逻辑分辨率不能超过图像尺寸：w={w} > W={W} 或 h={h} > H={H}")
    px = gray_arr.shape[1] / w
    py = gray_arr.shape[0] / h
    if edge_map is not None:
        phase_x, phase_y, conf = find_phase_edge(edge_map, px, py, step)
    else:
        phase_x, phase_y, conf = find_phase(gray_arr, px, py, step)
    # 用户指定分辨率路径：由全局相位 + 等距网格生成 cell_ys/cell_xs
    cell_ys, cell_xs = _equidistant_cell_grid(phase_x, phase_y, px, py, w, h)
    return Grid(
        w, h, px, py, phase_x, phase_y, conf, [(w, h, conf)],
        cell_ys=cell_ys, cell_xs=cell_xs, low_confidence=bool(conf < 0.4),
    )


def _gaussian_smooth_1d(profile: np.ndarray, kernel_size: int) -> np.ndarray:
    """对 1D 剖面做高斯平滑。

    Args:
        profile: 1D 信号数组。
        kernel_size: 高斯核大小（取奇数）。

    Returns:
        平滑后的 1D 数组，长度与输入一致。
    """
    k = int(kernel_size)
    if k < 3:
        return profile
    if k % 2 == 0:
        k += 1
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = ker / (ker.sum() + 1e-8)
    return np.convolve(profile, ker, mode="same")


def _normalize_combine_edge(mag: np.ndarray) -> np.ndarray:
    """边缘幅值图的归一化组合，缩放到 [0, 1]。

    ``compute_edge_map`` 的组合核心：``max(幅值/全局最大值, 33×33 局部
    对比度归一化值)`` 后整体缩放到 [0, 1]。抽出为独立函数供灰度 Sobel
    路径（``compute_edge_map``）与 OKLAB 色差路径（``_oklab_edge_map``）
    复用，保证两条路径的边缘图组合逻辑一致（重构自 compute_edge_map，
    灰度路径行为逐位不变）。

    Args:
        mag: 2D 边缘幅值图。

    Returns:
        ``(H, W)`` float64 边缘强度图，值域 [0, 1]；全零输入返回全零。
    """
    mag = np.asarray(mag, dtype=np.float64)
    global_max = float(mag.max())
    if global_max < 1e-12:
        return np.zeros_like(mag)
    mag_norm = mag / global_max
    local_norm = _local_contrast_normalize(mag, window=33)
    combined = np.maximum(mag_norm, local_norm)
    cmax = float(combined.max())
    if cmax > 1e-12:
        combined = combined / cmax
    return combined


def compute_edge_map(img: np.ndarray) -> np.ndarray:
    """计算 2D 边缘强度图。

    对输入图像（灰度 2D 或 RGB 3D）计算 Sobel 梯度幅值，再做 33×33
    局部归一化（复用 ``_local_contrast_normalize`` 的钳位局部 std 思路），
    组合 ``max(原始幅值/全局最大值, 局部归一化值)`` 后整体缩放到 [0, 1]
    （组合部分由 ``_normalize_combine_edge`` 实现）。

    Args:
        img: ``(H, W)`` 灰度图或 ``(H, W, 3)`` RGB 图（float64）。

    Returns:
        ``(H, W)`` float64 边缘强度图，值域 [0, 1]。
    """
    from scipy.ndimage import sobel

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    mag = np.sqrt(gx * gx + gy * gy)
    return _normalize_combine_edge(mag)


def _oklab_edge_map(oklab: np.ndarray) -> np.ndarray:
    """由 OKLAB 数组构造色差边缘强度图。

    对原始（非归一化）OKLAB 取相邻像素前向差分的 L2 范数作为逐像素
    色差幅值：Mdx[j] 记录第 j 与 j+1 列的色差、Mdy[i] 记录第 i 与 i+1
    行的色差，均计入右/下侧像素（缺失侧补 0 对齐到 ``(H, W)``），组合
    ``sqrt(Mdx^2 + Mdy^2)`` 后经 ``_normalize_combine_edge`` 缩放到
    [0, 1]。等亮度异色块边界（BT.601 灰度梯度为 0）在此图上仍是
    连续强边缘。

    边界能量计入右/下块首像素的约定与 ``find_phase`` 的"相位 = 首条
    网格线"块起点语义对齐：块边界条带落在边界列/行本身，使
    ``_edge_band_strength`` 默认相位 0 恰好命中对齐网格的边界带。

    Args:
        oklab: ``(H, W, 3)`` OKLab 数组（``rgb_to_oklab`` 输出）。

    Returns:
        ``(H, W)`` float64 色差边缘强度图，值域 [0, 1]。
    """
    oklab = np.asarray(oklab, dtype=np.float64)
    H, W = oklab.shape[:2]
    # 逐通道差分平方累加得各方向色差幅值（与 3D 差分 ||·||₂ 逐位同构，
    # 求和次序一致），避免 (H, W, 3) 大临时数组
    sq_x = np.zeros((H, W), dtype=np.float64)
    sq_y = np.zeros((H, W), dtype=np.float64)
    for c in range(oklab.shape[2]):
        ch = oklab[:, :, c]
        ddx = ch[:, 1:] - ch[:, :-1]
        sq_x[:, 1:] += ddx * ddx
        ddy = ch[1:, :] - ch[:-1, :]
        sq_y[1:, :] += ddy * ddy
    Mdx = np.sqrt(sq_x, out=sq_x)
    Mdy = np.sqrt(sq_y, out=sq_y)
    mag = np.sqrt(Mdx * Mdx + Mdy * Mdy)
    return _normalize_combine_edge(mag)


def _oklab_signal(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 OKLAB 色差检测信号（``signal="oklab"`` 模式的信号原语）。

    等亮度异色块边界（BT.601 灰度梯度 ≈ 0）在灰度路径的边缘图、投影
    信号与相位搜索中全部不可见。本函数改用 OKLAB 感知色差构造检测
    信号，供 ``detect(signal="oklab")`` 的全链路（FFT/ACF 判据、投票、
    方块检测、相位搜索）使用：

    - 投影信号 ``sig_x``/``sig_y``：对 OKLAB 每通道独立做局部对比度
      归一化（与灰度路径"先归一化再差分"同构，弱色度边缘被有界放大），
      再取相邻像素差分的 L2 范数沿另一轴求和，与灰度路径投影信号
      同形同义；
    - 边缘图 ``edge_map``：原始 OKLAB 前向差分的逐像素幅值
      （``_oklab_edge_map``），组合逻辑与灰度 ``compute_edge_map`` 一致；
    - ``gray``：BT.601 灰度。BVR、runs 交叉验证、梯度回退等灰度判据
      仍基于它（语义与灰度路径一致）。

    Args:
        img_rgb: ``(H, W, 3)`` RGB 0-255 浮点数组。

    Returns:
        (sig_x, sig_y, edge_map, gray)：
        ``sig_x`` 形状 ``(W-1,)``、``sig_y`` 形状 ``(H-1,)``（投影信号），
        ``edge_map`` 形状 ``(H, W)`` 值域 [0, 1]（色差边缘图），
        ``gray`` 形状 ``(H, W)``（BT.601 灰度）。
    """
    img = np.asarray(img_rgb, dtype=np.float64)
    oklab = rgb_to_oklab(img)
    H, W = img.shape[:2]
    # 每通道独立局部对比度归一化（放大弱色度边缘），逐通道差分平方累加得
    # L2 范数（与 3D 差分 ||·||₂ 逐位同构，求和次序一致），避免 (H, W, 3)
    # 大临时数组，降低 2048×2048 级输入的峰值内存
    sq_x = np.zeros((H, W - 1), dtype=np.float64)
    sq_y = np.zeros((H - 1, W), dtype=np.float64)
    for c in range(oklab.shape[2]):
        nc = _local_contrast_normalize(oklab[:, :, c])
        dxc = nc[:, 1:] - nc[:, :-1]
        sq_x += dxc * dxc
        dyc = nc[1:, :] - nc[:-1, :]
        sq_y += dyc * dyc
    np.sqrt(sq_x, out=sq_x)
    np.sqrt(sq_y, out=sq_y)
    sig_x = sq_x.sum(axis=0)  # (W-1,)
    sig_y = sq_y.sum(axis=1)  # (H-1,)
    edge_map = _oklab_edge_map(oklab)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return (sig_x, sig_y, edge_map, gray)


def _equidistant_cell_grid(
    phase_x: float,
    phase_y: float,
    px: float,
    py: float,
    w_logic: int,
    h_logic: int,
) -> tuple[np.ndarray, np.ndarray]:
    """由全局相位 + 等距网格生成 cell_ys / cell_xs（兜底路径）。"""
    ys = phase_y + np.arange(h_logic + 1, dtype=np.float64) * py
    xs = phase_x + np.arange(w_logic + 1, dtype=np.float64) * px
    cell_ys = np.broadcast_to(ys[:, None], (h_logic + 1, w_logic + 1)).copy()
    cell_xs = np.broadcast_to(xs[None, :], (h_logic + 1, w_logic + 1)).copy()
    return cell_ys, cell_xs


def _best_col(
    em: np.ndarray,
    y0: int,
    y1: int,
    x_center: float,
    edge_tol: int,
    enable_subpixel: bool,
) -> tuple[float | None, float | None]:
    """在 ``[x_center-tol, x_center+tol]`` 范围内找列边缘强度和最大的位置。

    双向搜索（±tol），返回亚像素位置与峰强度。

    Args:
        em: ``(H, W)`` 边缘强度图。
        y0: 列起点 y（含）。
        y1: 列终点 y（不含）。
        x_center: 搜索中心 x。
        edge_tol: 搜索半径（像素）。
        enable_subpixel: 是否做抛物线插值。

    Returns:
        (位置, 峰强度)。峰强度 <= ``0.03*边长`` 时返回 ``(None, None)``。
        ``enable_subpixel=False`` 时返回整数位置。
    """
    H, W = em.shape
    xlo = max(0, int(x_center) - edge_tol)
    xhi = min(W, int(x_center) + edge_tol + 1)
    if xhi <= xlo or y1 <= y0:
        return (None, None)
    seg = em[y0:y1, xlo:xhi]
    col_sum = seg.sum(axis=0)
    side_len = y1 - y0
    thr = 0.03 * side_len
    best = int(np.argmax(col_sum))
    peak = float(col_sum[best])
    if peak <= thr:
        return (None, None)
    if not enable_subpixel:
        return (float(xlo + best), peak)
    # 抛物线插值
    if 0 < best < col_sum.size - 1:
        y_km1 = float(col_sum[best - 1])
        y_k = float(col_sum[best])
        y_kp1 = float(col_sum[best + 1])
        denom = y_km1 - 2.0 * y_k + y_kp1
        if abs(denom) > 1e-6:
            offset = 0.5 * (y_km1 - y_kp1) / denom
            offset = max(-1.0, min(1.0, offset))
            return (float(xlo + best + offset), peak)
    return (float(xlo + best), peak)


def _best_row(
    em: np.ndarray,
    x0: int,
    x1: int,
    y_center: float,
    edge_tol: int,
    enable_subpixel: bool,
) -> tuple[float | None, float | None]:
    """在 ``[y_center-tol, y_center+tol]`` 范围内找行边缘强度和最大的位置。

    双向搜索（±tol），返回亚像素位置与峰强度。语义与 ``_best_col`` 对称。

    Args:
        em: ``(H, W)`` 边缘强度图。
        x0: 行起点 x（含）。
        x1: 行终点 x（不含）。
        y_center: 搜索中心 y。
        edge_tol: 搜索半径（像素）。
        enable_subpixel: 是否做抛物线插值。

    Returns:
        (位置, 峰强度)。峰强度 <= ``0.03*边长`` 时返回 ``(None, None)``。
        ``enable_subpixel=False`` 时返回整数位置。
    """
    H, W = em.shape
    ylo = max(0, int(y_center) - edge_tol)
    yhi = min(H, int(y_center) + edge_tol + 1)
    if yhi <= ylo or x1 <= x0:
        return (None, None)
    seg = em[ylo:yhi, x0:x1]
    row_sum = seg.sum(axis=1)
    side_len = x1 - x0
    thr = 0.03 * side_len
    best = int(np.argmax(row_sum))
    peak = float(row_sum[best])
    if peak <= thr:
        return (None, None)
    if not enable_subpixel:
        return (float(ylo + best), peak)
    if 0 < best < row_sum.size - 1:
        y_km1 = float(row_sum[best - 1])
        y_k = float(row_sum[best])
        y_kp1 = float(row_sum[best + 1])
        denom = y_km1 - 2.0 * y_k + y_kp1
        if abs(denom) > 1e-6:
            offset = 0.5 * (y_km1 - y_kp1) / denom
            offset = max(-1.0, min(1.0, offset))
            return (float(ylo + best + offset), peak)
    return (float(ylo + best), peak)


def _weighted_median(observations: list[tuple[float, float]]) -> float:
    """加权中位数。

    Args:
        observations: ``[(value, weight), ...]`` 列表。

    Returns:
        按权重排序、累积权重达到总权重 50% 时对应的观测值。
        空列表返回 0.0；权重总和 <=0 时退化为普通中位数。
    """
    if not observations:
        return 0.0
    if len(observations) == 1:
        return float(observations[0][0])
    sorted_obs = sorted(observations, key=lambda x: x[0])
    vals = np.array([o[0] for o in sorted_obs], dtype=np.float64)
    ws = np.array([o[1] for o in sorted_obs], dtype=np.float64)
    total = float(ws.sum())
    if total <= 0.0:
        return float(vals[len(vals) // 2])
    cumw = np.cumsum(ws)
    idx = int(np.searchsorted(cumw, 0.5 * total))
    if idx >= len(vals):
        idx = len(vals) - 1
    return float(vals[idx])


def detect_squares(
    edge_map: np.ndarray,
    px: float,
    py: float,
    side_avg_threshold: float = 0.06,
    per_pixel_threshold: float = 0.03,
    min_coherence: float = 0.5,
    interior_ratio: float = 0.5,
    nms_overlap: float = 0.30,
    enable_subpixel_refine: bool = True,
) -> list[dict]:
    """在 px×py 尺寸下滑窗评分检测方块候选，NMS 去重。

    用积分图 O(1) 查询每个候选方块的 4 边平均边缘强度、连贯性
    （边上 edge > per_pixel_threshold 的像素占比）与内部洁净度，
    接受至少 2 条相邻有界边的候选，按总分 NMS。候选位置步长采样
    ``step = max(1, int(min(px,py)*0.3))``，评分全部向量化。

    NMS 选中候选后，在 ``(y0, x0)`` 的 ±step 范围内用 4 边边缘强度和
    找最优整数位置，再做抛物线插值得亚像素位置（受 ``enable_subpixel_refine``
    控制）。

    Args:
        edge_map: ``(H, W)`` 边缘强度图（值域 [0,1]）。
        px: 块宽（像素）。
        py: 块高（像素）。
        side_avg_threshold: 边"有界"的平均强度阈值。
        per_pixel_threshold: 单像素边缘强度阈值，用于连贯性计数。
        min_coherence: 边"有界"的连贯性阈值。
        interior_ratio: 内部洁净度比例（interior_avg <= boundary_avg*ratio）。
        nms_overlap: NMS 重叠面积比例阈值。
        enable_subpixel_refine: NMS 后是否做局部亚像素精炼。

    Returns:
        方块字典列表，每项含 ``y0/x0/y1/x1/score/bounded_sides``。
    """
    em = np.asarray(edge_map, dtype=np.float64)
    if em.ndim != 2:
        raise ValueError(f"edge_map 必须为 2D，得到 shape={em.shape}")
    H, W = em.shape
    px_i = max(1, int(round(px)))
    py_i = max(1, int(round(py)))
    if py_i >= H or px_i >= W:
        return []

    # 积分图 I（边缘强度求和）与 B（edge>thr 的像素计数）
    I = np.zeros((H + 1, W + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(em, axis=0), axis=1)
    binmap = (em > per_pixel_threshold).astype(np.float64)
    B = np.zeros((H + 1, W + 1), dtype=np.float64)
    B[1:, 1:] = np.cumsum(np.cumsum(binmap, axis=0), axis=1)

    step = max(1, int(min(px, py) * 0.3))
    ys = np.arange(0, H - py_i + 1, step, dtype=np.int64)
    xs = np.arange(0, W - px_i + 1, step, dtype=np.int64)
    if ys.size == 0 or xs.size == 0:
        return []
    Y0, X0 = np.meshgrid(ys, xs, indexing="ij")  # (ny, nx)
    Y1 = Y0 + py_i
    X1 = X0 + px_i

    def iq(integ: np.ndarray, y0: np.ndarray, y1: np.ndarray,
           x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]

    # 4 边（各 1px 宽）：top/bottom 为水平边（长 px_i），left/right 为垂直边（长 py_i）
    top_sum = iq(I, Y0, Y0 + 1, X0, X1)
    bottom_sum = iq(I, Y1 - 1, Y1, X0, X1)
    left_sum = iq(I, Y0, Y1, X0, X0 + 1)
    right_sum = iq(I, Y0, Y1, X1 - 1, X1)
    top_cnt = iq(B, Y0, Y0 + 1, X0, X1)
    bottom_cnt = iq(B, Y1 - 1, Y1, X0, X1)
    left_cnt = iq(B, Y0, Y1, X0, X0 + 1)
    right_cnt = iq(B, Y0, Y1, X1 - 1, X1)
    interior_sum = iq(I, Y0 + 1, Y1 - 1, X0 + 1, X1 - 1)

    top_avg = top_sum / px_i
    bottom_avg = bottom_sum / px_i
    left_avg = left_sum / py_i
    right_avg = right_sum / py_i
    top_coh = top_cnt / px_i
    bottom_coh = bottom_cnt / px_i
    left_coh = left_cnt / py_i
    right_coh = right_cnt / py_i
    interior_area = max(0, (py_i - 2) * (px_i - 2))
    if interior_area > 0:
        interior_avg = interior_sum / interior_area
    else:
        interior_avg = np.zeros_like(interior_sum)
    boundary_avg = (top_avg + bottom_avg + left_avg + right_avg) / 4.0

    top_b = (top_avg >= side_avg_threshold) & (top_coh >= min_coherence)
    bottom_b = (bottom_avg >= side_avg_threshold) & (bottom_coh >= min_coherence)
    left_b = (left_avg >= side_avg_threshold) & (left_coh >= min_coherence)
    right_b = (right_avg >= side_avg_threshold) & (right_coh >= min_coherence)

    bounded_count = (
        top_b.astype(np.float64) + bottom_b.astype(np.float64)
        + left_b.astype(np.float64) + right_b.astype(np.float64)
    )
    adjacent = (top_b & right_b) | (right_b & bottom_b) | (bottom_b & left_b) | (left_b & top_b)
    clean = interior_avg <= boundary_avg * interior_ratio

    score = (
        top_avg + bottom_avg + left_avg + right_avg
        + bounded_count * 0.5 + clean.astype(np.float64) * 0.5
    )

    acc_mask = adjacent
    acc_idx = np.where(acc_mask.ravel())[0]
    if acc_idx.size == 0:
        return []

    Y0f = Y0.ravel()[acc_idx].astype(np.int64)
    X0f = X0.ravel()[acc_idx].astype(np.int64)
    scoref = score.ravel()[acc_idx]
    top_bf = top_b.ravel()[acc_idx]
    bottom_bf = bottom_b.ravel()[acc_idx]
    left_bf = left_b.ravel()[acc_idx]
    right_bf = right_b.ravel()[acc_idx]

    # 每个真实格保留最高分候选（粗 NMS），将候选数降到格数级别
    cell_gy = np.floor((Y0f + py_i / 2.0) / py_i).astype(np.int64)
    cell_gx = np.floor((X0f + px_i / 2.0) / px_i).astype(np.int64)
    n_gx = W // px_i + 3
    cell_key = (cell_gy + 2) * n_gx + (cell_gx + 2)
    order = np.argsort(-scoref, kind="stable")
    keys_sorted = cell_key[order]
    _, first_pos = np.unique(keys_sorted, return_index=True)
    best_local = order[first_pos]

    bY0 = Y0f[best_local]
    bX0 = X0f[best_local]
    bscore = scoref[best_local]
    btop = top_bf[best_local]
    bbottom = bottom_bf[best_local]
    bleft = left_bf[best_local]
    bright = right_bf[best_local]

    # 贪心 NMS（空间分箱，仅检查 3×3 邻域已选方块）
    order2 = np.argsort(-bscore, kind="stable")
    bin_size = max(1, int(min(px_i, py_i)))
    bins: dict = {}
    selected: list[dict] = []
    area = float(px_i * py_i)
    for k in order2:
        y0 = int(bY0[k]); x0 = int(bX0[k])
        y1 = y0 + py_i; x1 = x0 + px_i
        by = y0 // bin_size; bx = x0 // bin_size
        ok = True
        for ddy in (-1, 0, 1):
            if not ok:
                break
            for ddx in (-1, 0, 1):
                for sidx in bins.get((by + ddy, bx + ddx), []):
                    s = selected[sidx]
                    iy0 = y0 if y0 > s["y0"] else s["y0"]
                    iy1 = y1 if y1 < s["y1"] else s["y1"]
                    ix0 = x0 if x0 > s["x0"] else s["x0"]
                    ix1 = x1 if x1 < s["x1"] else s["x1"]
                    ov = max(0, iy1 - iy0) * max(0, ix1 - ix0)
                    if area > 0 and ov / area > nms_overlap:
                        ok = False
                        break
                if not ok:
                    break
        if not ok:
            continue
        sides: list[str] = []
        if btop[k]:
            sides.append("top")
        if bbottom[k]:
            sides.append("bottom")
        if bleft[k]:
            sides.append("left")
        if bright[k]:
            sides.append("right")
        bins.setdefault((by, bx), []).append(len(selected))
        selected.append({
            "y0": float(y0), "x0": float(x0),
            "y1": float(y1), "x1": float(x1),
            "score": float(bscore[k]), "bounded_sides": sides,
        })

    # NMS 后局部亚像素精炼：在 (y0, x0) 的 ±step 范围内用 4 边边缘强度和
    # 找最优整数位置，再做抛物线插值。
    if enable_subpixel_refine and selected:
        radius = max(1, step)
        for s in selected:
            y0 = int(s["y0"])
            x0 = int(s["x0"])
            y_lo = max(0, y0 - radius)
            y_hi = min(H - py_i, y0 + radius)
            x_lo = max(0, x0 - radius)
            x_hi = min(W - px_i, x0 + radius)
            if y_hi < y_lo or x_hi < x_lo:
                continue
            ys_cand = np.arange(y_lo, y_hi + 1, dtype=np.int64)
            xs_cand = np.arange(x_lo, x_hi + 1, dtype=np.int64)
            Y, X = np.meshgrid(ys_cand, xs_cand, indexing="ij")
            top = iq(I, Y, Y + 1, X, X + px_i)
            bottom = iq(I, Y + py_i - 1, Y + py_i, X, X + px_i)
            left = iq(I, Y, Y + py_i, X, X + 1)
            right = iq(I, Y, Y + py_i, X + px_i - 1, X + px_i)
            s4 = top + bottom + left + right
            flat_idx = int(np.argmax(s4))
            ny_idx, nx_idx = np.unravel_index(flat_idx, s4.shape)
            best_y = int(ys_cand[ny_idx])
            best_x = int(xs_cand[nx_idx])
            # 抛物线插值（y 方向）
            offset_y = 0.0
            if 0 < ny_idx < s4.shape[0] - 1:
                yv0 = float(s4[ny_idx - 1, nx_idx])
                yv1 = float(s4[ny_idx, nx_idx])
                yv2 = float(s4[ny_idx + 1, nx_idx])
                denom = yv0 - 2.0 * yv1 + yv2
                if abs(denom) > 1e-6:
                    offset_y = 0.5 * (yv0 - yv2) / denom
                    offset_y = max(-1.0, min(1.0, offset_y))
            # 抛物线插值（x 方向）
            offset_x = 0.0
            if 0 < nx_idx < s4.shape[1] - 1:
                xv0 = float(s4[ny_idx, nx_idx - 1])
                xv1 = float(s4[ny_idx, nx_idx])
                xv2 = float(s4[ny_idx, nx_idx + 1])
                denom = xv0 - 2.0 * xv1 + xv2
                if abs(denom) > 1e-6:
                    offset_x = 0.5 * (xv0 - xv2) / denom
                    offset_x = max(-1.0, min(1.0, offset_x))
            new_y0 = float(best_y) + offset_y
            new_x0 = float(best_x) + offset_x
            s["y0"] = new_y0
            s["x0"] = new_x0
            s["y1"] = new_y0 + py_i
            s["x1"] = new_x0 + px_i
    return selected


def assign_grid_bfs(
    squares: list[dict],
    px: float,
    py: float,
    img_w: int,
    img_h: int,
    adjacency_tol: float = 3.0,
) -> tuple[dict, tuple]:
    """从中心种子 BFS 分配整数网格坐标，未连通方块按位置回退放置。

    Args:
        squares: ``detect_squares`` 返回的方块列表。
        px: 块宽（像素）。
        py: 块高（像素）。
        img_w: 图像宽（像素）。
        img_h: 图像高（像素）。
        adjacency_tol: 邻接判定容差（像素）。

    Returns:
        (placed, grid_bounds)：``placed`` 为 ``dict[(gx,gy)] = (y0,x0,y1,x1)``，
        ``grid_bounds=(min_gx, max_gx, min_gy, max_gy)``。
        ``squares`` 为空时返回空 dict 与基于图像中心的默认 grid_bounds。
    """
    cx_img = img_w / 2.0
    cy_img = img_h / 2.0

    def _default_bounds(scx: float, scy: float) -> tuple:
        return (
            -int(np.floor(scx / px)),
            int(np.floor((img_w - scx) / px)),
            -int(np.floor(scy / py)),
            int(np.floor((img_h - scy) / py)),
        )

    if not squares:
        return ({}, _default_bounds(cx_img, cy_img))

    cxs = np.array([(s["x0"] + s["x1"]) / 2.0 for s in squares], dtype=np.float64)
    cys = np.array([(s["y0"] + s["y1"]) / 2.0 for s in squares], dtype=np.float64)
    scores = np.array([s["score"] for s in squares], dtype=np.float64)
    dist = np.abs(cxs - cx_img) + np.abs(cys - cy_img)
    # 用图像中心构建临时 coord_map 计算连通度（与种子选择解耦）
    agx_tmp = np.rint((cxs - cx_img) / px).astype(np.int64)
    agy_tmp = np.rint((cys - cy_img) / py).astype(np.int64)
    coord_map_tmp: dict = {}
    for i in range(len(squares)):
        coord_map_tmp.setdefault((int(agx_tmp[i]), int(agy_tmp[i])), []).append(i)
    # 种子选择：分数归一化到 [0,1]，主键 = norm_score * connectivity，次键 = dist
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max - s_min > 1e-9:
        norm_score = (scores - s_min) / (s_max - s_min)
        connectivity = np.zeros(len(squares), dtype=np.float64)
        for i in range(len(squares)):
            cnt = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (int(agx_tmp[i]) + dx, int(agy_tmp[i]) + dy) in coord_map_tmp:
                    cnt += 1
            connectivity[i] = float(cnt)
        primary = norm_score * connectivity
        order = np.lexsort((dist, -primary))
    else:
        # 退化为原逻辑（分数全相同）
        order = np.lexsort((dist, -scores))
    seed_idx = int(order[0])
    seed_cx = float(cxs[seed_idx])
    seed_cy = float(cys[seed_idx])

    grid_bounds = _default_bounds(seed_cx, seed_cy)

    # 各方块相对种子的近似网格坐标（用于 O(1) 邻居查找）
    agx = np.rint((cxs - seed_cx) / px).astype(np.int64)
    agy = np.rint((cys - seed_cy) / py).astype(np.int64)
    coord_map: dict = {}
    for i in range(len(squares)):
        coord_map.setdefault((int(agx[i]), int(agy[i])), []).append(i)

    def sq_tuple(i: int) -> tuple:
        s = squares[i]
        return (float(s["y0"]), float(s["x0"]), float(s["y1"]), float(s["x1"]))

    placed: dict = {(0, 0): sq_tuple(seed_idx)}
    visited = {seed_idx}
    queue = [(0, 0, seed_idx)]
    head = 0
    while head < len(queue):
        gx, gy, idx = queue[head]
        head += 1
        cur_cx = float(cxs[idx])
        cur_cy = float(cys[idx])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ng = (gx + dx, gy + dy)
            if ng in placed:
                continue
            for ci in coord_map.get(ng, ()):
                if ci in visited:
                    continue
                ddx = float(cxs[ci]) - cur_cx
                ddy = float(cys[ci]) - cur_cy
                if dx == 1:
                    ok = abs(ddx - px) <= adjacency_tol and abs(ddy) <= adjacency_tol
                elif dx == -1:
                    ok = abs(ddx + px) <= adjacency_tol and abs(ddy) <= adjacency_tol
                elif dy == 1:
                    ok = abs(ddy - py) <= adjacency_tol and abs(ddx) <= adjacency_tol
                else:
                    ok = abs(ddy + py) <= adjacency_tol and abs(ddx) <= adjacency_tol
                if ok:
                    placed[ng] = sq_tuple(ci)
                    visited.add(ci)
                    queue.append((ng[0], ng[1], ci))
                    break

    # 未连通方块按近似坐标回退放置，冲突跳过
    for i in range(len(squares)):
        if i in visited:
            continue
        g = (int(agx[i]), int(agy[i]))
        if g not in placed:
            placed[g] = sq_tuple(i)
            visited.add(i)

    return (placed, grid_bounds)


def _fit_global_regularization(
    cell_coords: np.ndarray,
    counts: np.ndarray,
    period: float,
    min_g: int,
    regularity_strength: float,
    outlier_reject_ratio: float,
    limit: float,
) -> np.ndarray:
    """对单轴交点坐标做全局正则化混合。

    用加权线性回归拟合 ``pos = phase + idx * period``（全局模型），
    再按观测置信度将观测位置与全局模型混合：
    ``final = observed * w + global_model * (1-w)``，
    其中 ``w = min(1.0, counts/4.0) * (1 - regularity_strength)``。
    高置信观测保留局部位置，低置信观测回归全局模型，消除累积漂移。

    拟合前剔除离群间距：相邻交点间距偏离 period 超过
    ``outlier_reject_ratio * period`` 的观测在拟合时权重置 0。

    Args:
        cell_coords: ``(N, M)`` 单轴交点坐标数组（cell_ys 或 cell_xs 的转置）。
            沿 axis=1（列方向）是网格索引递增方向。
        counts: ``(N, M)`` 各交点的观测计数。
        period: 理想间距（px_i 或 py_i）。
        min_g: 网格起始索引（用于全局模型的 index 偏移）。
        regularity_strength: 正则化强度（0=纯观测，1=纯全局模型）。
        outlier_reject_ratio: 离群间距剔除阈值比例。
        limit: 坐标上界（图像宽或高），用于钳位。

    Returns:
        正则化后的 ``(N, M)`` 交点坐标数组。
    """
    coords = np.asarray(cell_coords, dtype=np.float64).copy()
    cnts = np.asarray(counts, dtype=np.float64)
    N, M = coords.shape
    if regularity_strength <= 0.0 or M < 2:
        return coords

    # 构建 (index, position) 观测对，index = min_g + 列号
    # 沿 axis=1 方向（每行）拟合线性模型
    indices = min_g + np.arange(M, dtype=np.float64)

    # 计算拟合权重：基础权重 = 观测计数；离群观测权重置 0
    fit_weights = cnts.copy()

    # 离群间距剔除：对每行，检查相邻间距是否偏离 period 过多
    if M >= 2:
        spacing = np.diff(coords, axis=1)  # (N, M-1)
        dev = np.abs(spacing - period)
        is_outlier = dev > (outlier_reject_ratio * period)
        # 离群间距涉及左右两个交点，两者拟合权重都置 0
        for j in range(M - 1):
            mask = is_outlier[:, j]
            if np.any(mask):
                fit_weights[mask, j] = 0.0
                fit_weights[mask, j + 1] = 0.0

    # 对每行做加权线性回归：pos = phase + idx * period
    # 使用所有行的观测联合拟合全局 phase 和 period（更稳健）
    # 收集有效观测
    valid = fit_weights > 0
    if not np.any(valid):
        # 全部离群，退化为理想等距网格
        for i in range(N):
            coords[i, :] = coords[i, 0] + np.arange(M) * period
        return np.clip(coords, 0.0, limit)

    # 联合所有行拟合全局 phase 和 period
    all_idx = np.broadcast_to(indices[None, :], (N, M))[valid]
    all_pos = coords[valid]
    all_w = fit_weights[valid]

    # 加权线性回归：pos = a + b * idx，其中 a=phase, b=period
    # 用 period 作为 b 的先验，约束 b 接近 period（正则化）
    # 简化：固定 b=period，只拟合 a（phase），这样最稳健
    # 因为 period 已经由 FFT/投票确定，全局模型只需确定 phase
    # phase = weighted_mean(pos - idx * period)
    residuals = all_pos - all_idx * period
    total_w = float(all_w.sum())
    if total_w > 1e-12:
        global_phase = float((residuals * all_w).sum() / total_w)
    else:
        global_phase = 0.0

    # 全局模型位置
    global_model = global_phase + indices * period  # (M,)
    global_model = np.broadcast_to(global_model[None, :], (N, M))

    # 混合权重：高观测计数 + 低正则化强度 → 保留观测
    w = np.minimum(1.0, cnts / 4.0) * (1.0 - regularity_strength)
    w = np.broadcast_to(w, (N, M))

    # 混合
    blended = coords * w + global_model * (1.0 - w)
    # 扩展范围软钳位：允许越界一个块（[-period, limit+period]），防止极端外推
    blended = np.clip(blended, -period, limit + period)
    # 单调性约束：确保位置严格递增（最小间距 0.3*period），防止边界处网格线坍缩
    min_spacing = 0.3 * period
    for i in range(1, M):
        blended[:, i] = np.maximum(blended[:, i], blended[:, i - 1] + min_spacing)
    return blended


def expand_grid_edge_guided(
    edge_map: np.ndarray,
    placed: dict,
    grid_bounds: tuple,
    px: float,
    py: float,
    edge_tol: int = 3,
    enable_subpixel_refine: bool = True,
    smooth_strength: float = 0.5,
    outlier_reject_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """从已放置种子 BFS 向外填满网格，边缘引导精炼位置，聚合 cell_ys/cell_xs。

    对每个未放置邻居，在共享边界 ``[中心±edge_tol]`` 范围内（双向搜索）搜索
    边缘强度和最大的列/行作为新方块边界，并通过抛物线插值得到亚像素位置
    （受 ``enable_subpixel_refine`` 控制）；无边缘时用邻居位置 + px/py 偏移外推。
    另一轴坐标从邻居继承。

    聚合所有格子四角时按边缘强度加权的中位数（权重来自 ``_best_col``/``_best_row``
    返回的峰强度，理想外推格子权重 1.0）生成 ``(h_logic+1, w_logic+1)`` 的
    ``cell_ys`` / ``cell_xs``。聚合后做全局正则化混合：用加权线性回归拟合
    全局 ``phase + idx * period`` 模型，按观测置信度将观测位置与全局模型混合，
    消除累积漂移。

    Args:
        edge_map: ``(H, W)`` 边缘强度图。
        placed: ``assign_grid_bfs`` 返回的已放置方块 dict。
        grid_bounds: ``(min_gx, max_gx, min_gy, max_gy)``。
        px: 块宽（像素）。
        py: 块高（像素）。
        edge_tol: 共享边界搜索半径（像素）。
        enable_subpixel_refine: 是否启用亚像素抛物线插值。
        smooth_strength: 全局正则化混合强度（0.0=纯观测，1.0=完全用全局线性模型）。
        outlier_reject_ratio: 离群间距剔除阈值比例，间距偏离 period 超过该比例的
            观测不参与全局模型拟合。

    Returns:
        (cell_ys, cell_xs)：形状 ``(h_logic+1, w_logic+1)`` 的交点坐标数组。
        ``placed`` 为空时由 phase=0 + 等距网格兜底生成。
    """
    em = np.asarray(edge_map, dtype=np.float64)
    H, W = em.shape
    min_gx, max_gx, min_gy, max_gy = grid_bounds
    w_logic = max_gx - min_gx + 1
    h_logic = max_gy - min_gy + 1

    if not placed:
        # 等距兜底网格：直接用浮点周期，避免取整导致的累积漂移
        cell_ys = np.broadcast_to(
            ((np.arange(h_logic + 1, dtype=np.float64) + min_gy) * py)[:, None],
            (h_logic + 1, w_logic + 1),
        ).copy()
        cell_xs = np.broadcast_to(
            ((np.arange(w_logic + 1, dtype=np.float64) + min_gx) * px)[None, :],
            (h_logic + 1, w_logic + 1),
        ).copy()
        return (cell_ys, cell_xs)

    placed = dict(placed)
    # weights 字典：记录每个 placed 格子的观测权重
    # 初始 placed 格子（来自 assign_grid_bfs）权重 1.0；边缘引导扩展的格子
    # 权重 = best_col/best_row 返回的峰强度；理想外推格子权重 1.0
    weights: dict = {k: 1.0 for k in placed.keys()}
    seed = placed.get((0, 0))
    if seed is not None:
        seed_x0 = float(seed[1])
        seed_y0 = float(seed[0])
    else:
        seed_x0 = 0.0
        seed_y0 = 0.0

    queue = list(placed.keys())
    head = 0
    while head < len(queue):
        gx, gy = queue[head]
        head += 1
        cy0, cx0, cy1, cx1 = placed[(gx, gy)]
        iy0 = int(cy0); iy1 = int(cy1); ix0 = int(cx0); ix1 = int(cx1)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ng = (gx + dx, gy + dy)
            if not (min_gx <= ng[0] <= max_gx and min_gy <= ng[1] <= max_gy):
                continue
            if ng in placed:
                continue
            if dx == 1:  # 向右扩展，共享垂直边界在 cx1 附近
                bc, peak = _best_col(em, iy0, iy1, cx1, edge_tol, enable_subpixel_refine)
                if bc is not None:
                    nx0 = float(bc)
                    weights[ng] = max(0.0, float(peak))
                else:
                    nx0 = cx1
                    weights[ng] = 1.0
                nx1 = nx0 + px  # 浮点周期外推，避免非整数周期累积取整漂移
                placed[ng] = (cy0, nx0, cy1, nx1)
            elif dx == -1:  # 向左扩展，共享垂直边界在 cx0 附近
                bc, peak = _best_col(em, iy0, iy1, cx0, edge_tol, enable_subpixel_refine)
                if bc is not None:
                    nx1 = float(bc)
                    weights[ng] = max(0.0, float(peak))
                else:
                    nx1 = cx0
                    weights[ng] = 1.0
                nx0 = nx1 - px
                placed[ng] = (cy0, nx0, cy1, nx1)
            elif dy == 1:  # 向下扩展，共享水平边界在 cy1 附近
                br, peak = _best_row(em, ix0, ix1, cy1, edge_tol, enable_subpixel_refine)
                if br is not None:
                    ny0 = float(br)
                    weights[ng] = max(0.0, float(peak))
                else:
                    ny0 = cy1
                    weights[ng] = 1.0
                ny1 = ny0 + py
                placed[ng] = (ny0, cx0, ny1, cx1)
            else:  # dy == -1，向上扩展，共享水平边界在 cy0 附近
                br, peak = _best_row(em, ix0, ix1, cy0, edge_tol, enable_subpixel_refine)
                if br is not None:
                    ny1 = float(br)
                    weights[ng] = max(0.0, float(peak))
                else:
                    ny1 = cy0
                    weights[ng] = 1.0
                ny0 = ny1 - py
                placed[ng] = (ny0, cx0, ny1, cx1)
            queue.append(ng)

    # 无邻居可达的剩余格用理想位置填充（基于种子相位）
    for gy in range(min_gy, max_gy + 1):
        for gx in range(min_gx, max_gx + 1):
            if (gx, gy) not in placed:
                nx0 = seed_x0 + gx * px
                ny0 = seed_y0 + gy * py
                placed[(gx, gy)] = (ny0, nx0, ny0 + py, nx0 + px)
                weights[(gx, gy)] = 1.0

    # 加权中位数聚合：每个格子贡献 4 个角观测
    # 角 (jj, ii) <- y0, x0；角 (jj+1, ii) <- y1；角 (jj, ii+1) <- x1
    ys_cnt = np.zeros((h_logic + 1, w_logic + 1), dtype=np.float64)
    xs_cnt = np.zeros((h_logic + 1, w_logic + 1), dtype=np.float64)
    cell_ys = np.zeros((h_logic + 1, w_logic + 1), dtype=np.float64)
    cell_xs = np.zeros((h_logic + 1, w_logic + 1), dtype=np.float64)

    y_obs_dict: dict[tuple[int, int], list[tuple[float, float]]] = {}
    x_obs_dict: dict[tuple[int, int], list[tuple[float, float]]] = {}

    for (gx, gy), (y0, x0, y1, x1) in placed.items():
        w = float(weights.get((gx, gy), 1.0))
        jj_k = int(gy - min_gy)
        ii_k = int(gx - min_gx)
        y_obs_dict.setdefault((jj_k, ii_k), []).append((float(y0), w))
        x_obs_dict.setdefault((jj_k, ii_k), []).append((float(x0), w))
        y_obs_dict.setdefault((jj_k + 1, ii_k), []).append((float(y1), w))
        x_obs_dict.setdefault((jj_k, ii_k + 1), []).append((float(x1), w))

    for (jj_k, ii_k), obs in y_obs_dict.items():
        if 0 <= jj_k < h_logic + 1 and 0 <= ii_k < w_logic + 1:
            ys_cnt[jj_k, ii_k] = float(len(obs))
            cell_ys[jj_k, ii_k] = _weighted_median(obs)

    for (jj_k, ii_k), obs in x_obs_dict.items():
        if 0 <= jj_k < h_logic + 1 and 0 <= ii_k < w_logic + 1:
            xs_cnt[jj_k, ii_k] = float(len(obs))
            cell_xs[jj_k, ii_k] = _weighted_median(obs)

    # 无观测的交点用理想外推
    ideal_y = seed_y0 + (np.arange(h_logic + 1) + min_gy) * py
    ideal_x = seed_x0 + (np.arange(w_logic + 1) + min_gx) * px
    ideal_ys = np.broadcast_to(ideal_y[:, None], (h_logic + 1, w_logic + 1))
    ideal_xs = np.broadcast_to(ideal_x[None, :], (h_logic + 1, w_logic + 1))
    cell_ys = np.where(ys_cnt == 0, ideal_ys, cell_ys)
    cell_xs = np.where(xs_cnt == 0, ideal_xs, cell_xs)

    # 全局正则化混合：用加权线性回归拟合全局 phase+period，
    # 按观测置信度混合观测位置与全局模型，消除累积漂移
    if smooth_strength > 0.0:
        # x 轴：cell_xs 沿 axis=1（列方向）是网格 x 索引递增
        cell_xs = _fit_global_regularization(
            cell_xs, xs_cnt, float(px), min_gx,
            smooth_strength, outlier_reject_ratio, float(W),
        )
        # y 轴：cell_ys 沿 axis=0（行方向）是网格 y 索引递增，需转置
        cell_ys_t = _fit_global_regularization(
            cell_ys.T, ys_cnt.T, float(py), min_gy,
            smooth_strength, outlier_reject_ratio, float(H),
        )
        cell_ys = cell_ys_t.T

    return (cell_ys, cell_xs)
