"""像素网格周期检测。

针对 AI 生成的伪像素图，通过梯度 + 一维 FFT 带通信噪比检测块状
周期结构，估计块尺寸（px, py）与相位偏移（phase_x, phase_y），
并据此推断逻辑分辨率（w_logic, h_logic）。

主要接口：
- ``has_pixel_grid``：判断灰度图是否包含像素网格周期。
- ``find_phase``：在已知块尺寸下扫描最佳网格相位。
- ``detect``：自动检测块尺寸与相位，返回 ``Grid``。
- ``detect_with_user_grid``：用户指定逻辑分辨率，反推块尺寸并定相。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    gray: np.ndarray, period: float, axis: int = 0
) -> float:
    """计算 2D 真块方差对比度（块间方差 / 块内方差），真网格此值最大。

    按候选周期将图像切成 ``period×period`` 真块（而非旧实现的 1D 条带），
    以相位扫描找最优对齐，计算块均值序列方差（块间）与各块内方差均值
    （块内）之比。真网格对齐时块内同质、块间差异大，比值高。

    参数 ``axis`` 为兼容既有调用保留（2D 实现各向同性，不再分方向）。

    Args:
        gray: 灰度图数组 (H, W)。
        period: 候选周期（像素）。
        axis: 兼容参数，不影响结果。

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


def _edge_band_strength(
    em: np.ndarray, period: float, axis: int = 0
) -> float:
    """边界带边缘强度：候选周期下，块边界位置 1px 条带的平均边缘强度。

    真实块边界是整幅图上的连续强边缘，块内纹理是局部弱边缘。按候选周期
    在整数边界位置取 1px 宽条带求平均边缘强度，真实周期的边界带强度显著
    高于其子谐波（块内纹理周期 P/k 的条带大部分不与真实边界对齐）。

    Args:
        em: 边缘强度图 (H, W)，值域 [0,1]（``compute_edge_map`` 输出）。
        period: 候选周期（像素）。
        axis: 0=检测 y 方向周期（水平边界），1=检测 x 方向周期（垂直边界）。

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
    # 积分图 O(1) 条带求和
    I = np.zeros((H + 1, W + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(em, axis=0), axis=1)
    best = 0.0
    for phase in (0.0, p / 2.0):
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


def _vote_period(
    gray: np.ndarray,
    profile: np.ndarray,
    min_p: int,
    max_p: int,
    axis: int = 0,
    edge_map: np.ndarray | None = None,
    comb_weight: float = 0.0,
    use_comb_prefilter: bool = False,
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

    Args:
        gray: 灰度图数组 (H, W)。
        profile: 1D 梯度投影信号。
        min_p: 最小候选周期。
        max_p: 最大候选周期。
        axis: 0=检测 y 方向周期，1=检测 x 方向周期。
        edge_map: 边缘强度图 (H, W)，值域 [0,1]；None 时走旧路径。
        comb_weight: Spectral Comb 判据权重（0.0-1.0，从边界强度份额拆分）。
        use_comb_prefilter: 修正前是否要求 comb 与边界强度双一致。

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
    # 对每个候选打分：先算廉价判据（ACF/FFT/边界强度），昂贵的 2D BVR
    # 只对廉价判据排名前 VOTE_BVR_LIMIT 的候选计算（BVR 是弱判据，且对
    # 像素格点阵类输入区分度差，不为低排名候选付出相位扫描成本）。
    # 旧路径（无 edge_map）的 legacy BVR 是 1D 条带、开销小，保持逐候选计算。
    scores: dict[int, dict[str, float]] = {}
    for c in candidates:
        # ACF 峰高（归一化到 [0,1]）
        acf_score = float(acf[c]) if c < len(acf) and acf[c] > 0 else 0.0
        # FFT SNR（用 _fft_band_snr 对该周期附近评估）
        # 简化：用主峰 SNR 作为所有候选的参考，候选离主峰越近分越高
        if period_fft > 0:
            dist = abs(c - period_fft) / max(period_fft, 1e-6)
            fft_score = max(0.0, 1.0 - dist)
        else:
            fft_score = 0.0
        # 边界带边缘强度（廉价：积分图 + 条带求和）
        edge = _edge_band_strength(edge_map, float(c), axis=axis) if edge_map is not None else 0.0
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
            scores[c]["bvr"] = _block_variance_ratio(gray, float(c), axis=axis)
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
                    scores[kp] = {
                        "acf": float(acf[kp]) if kp < len(acf) and acf[kp] > 0 else 0.0,
                        "fft": 0.0,
                        "bvr": _block_variance_ratio(gray, float(kp), axis=axis) / max_bvr,
                        "edge": _edge_band_strength(edge_map, float(kp), axis=axis) / max_edge,
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

    Returns:
        (has, snr, period, period_x, period_y, snr_x, snr_y)：是否为网格、
        主方向信噪比与主周期、X 方向周期、Y 方向周期、X 方向 SNR、Y 方向 SNR。
    """
    gray = np.asarray(gray, dtype=np.float64)
    gray = _local_contrast_normalize(gray)
    # X 方向：列间梯度，按行求和
    dx = np.abs(np.diff(gray, axis=1))  # (H, W-1)
    sig_x = dx.sum(axis=0)  # (W-1,)
    # Y 方向：行间梯度，按列求和
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
                    edge_map, acf_base_x, period_x, axis=1
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
                    edge_map, acf_base_y, period_y, axis=0
                ):
                    period_y = float(acf_base_y)
    if snr_x >= snr_y:
        snr, period = snr_x, period_x
    else:
        snr, period = snr_y, period_y
    has = bool(snr >= snr_threshold)
    return (has, float(snr), float(period), float(period_x), float(period_y), float(snr_x), float(snr_y))


def _direction_protection_ok(
    edge_map: np.ndarray, smaller: float, larger: float, axis: int
) -> bool:
    """ACF 纠正方向保护：小周期（候选基频）边界强度不得显著低于大周期。

    AI 伪像素图块内纹理周期 P/k 的边界带边缘强度远低于真实周期 P；
    若候选基频（更小周期）的边缘强度不足当前周期的 0.7 倍，判定其为
    纹理周期，拒绝把检测周期缩小到它。
    """
    if edge_map is None or smaller <= 0 or larger <= 0:
        return True
    e_small = _edge_band_strength(edge_map, float(smaller), axis=axis)
    e_large = _edge_band_strength(edge_map, float(larger), axis=axis)
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


def detect(
    gray: np.ndarray, min_p: int = 3, max_p: int = 40, step: float = 0.1,
    snr_threshold: float = 8.0,
    edge_tol: int = 3,
    enable_subpixel_refine: bool = True,
    smooth_strength: float = 0.5,
    outlier_reject_ratio: float = 0.5,
    comb_weight: float = 0.0,
    use_comb_prefilter: bool = False,
) -> Grid:
    """自动检测像素网格。

    Args:
        gray: 灰度图数组。
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

    Returns:
        Grid: 检测结果。

    Raises:
        ValueError: 未检测到像素网格周期。
    """
    gray_arr = np.asarray(gray, dtype=np.float64)
    H, W = gray_arr.shape
    # 边缘图统一计算一次，供方向保护 / 投票 / 方块检测 / 边缘扩展复用
    edge_map = compute_edge_map(gray_arr)
    has, snr, period, period_x, period_y, snr_x, snr_y = has_pixel_grid(
        gray_arr, min_p, max_p, snr_threshold=snr_threshold, edge_map=edge_map,
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
    sig_x = np.abs(np.diff(gray_arr, axis=1)).sum(axis=0)
    sig_y = np.abs(np.diff(gray_arr, axis=0)).sum(axis=1)
    vote_px, conf_x = _vote_period(
        gray_arr, sig_x, min_p, max_p, axis=1, edge_map=edge_map,
        comb_weight=comb_weight, use_comb_prefilter=use_comb_prefilter,
    )
    vote_py, conf_y = _vote_period(
        gray_arr, sig_y, min_p, max_p, axis=0, edge_map=edge_map,
        comb_weight=comb_weight, use_comb_prefilter=use_comb_prefilter,
    )
    # 投票结果覆盖 FFT：置信度高，或边界强度显著更高
    # （FFT 主峰易被块内纹理污染，而边界带边缘强度判据不受其影响；
    #   纹理场景投票置信度偏低，但边界强度仍可靠，故用 edge 无条件覆盖）
    if vote_px > 0:
        e_vote = _edge_band_strength(edge_map, vote_px, axis=1)
        e_fft = _edge_band_strength(edge_map, period_x, axis=1) if period_x > 0 else 0.0
        if conf_x > 0.3 or (e_vote > 1e-12 and e_fft > 1e-12 and e_vote > 1.2 * e_fft):
            period_x = float(vote_px)
    if vote_py > 0:
        e_vote = _edge_band_strength(edge_map, vote_py, axis=0)
        e_fft = _edge_band_strength(edge_map, period_y, axis=0) if period_y > 0 else 0.0
        if conf_y > 0.3 or (e_vote > 1e-12 and e_fft > 1e-12 and e_vote > 1.2 * e_fft):
            period_y = float(vote_py)
    # 综合置信度
    vote_conf = (conf_x + conf_y) / 2.0

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
            # 3. 差异大：分别使用，但做长宽比校验
            px = float(period_x)
            py = float(period_y)
            phase_x_t, phase_y_t, conf_t = find_phase(gray_arr, px, py, step)
            w_logic_t = int(round((W - phase_x_t) / px))
            h_logic_t = int(round((H - phase_y_t) / py))
            if w_logic_t > 0 and h_logic_t > 0:
                ratio_out = w_logic_t / h_logic_t
                ratio_orig = W / H
                ratio_diff = abs(ratio_out - ratio_orig) / ratio_orig
                if ratio_diff > 0.3:
                    # 长宽比畸变：回退为正方形块（取边界强度较高方向的周期，
                    # SNR 同样受块内纹理污染，边界强度判据更可靠）
                    e_x = _edge_band_strength(edge_map, px, axis=1)
                    e_y = _edge_band_strength(edge_map, py, axis=0)
                    if e_x >= e_y:
                        px = py = float(period_x)
                    else:
                        px = py = float(period_y)
            # 若回退了，下面会重新计算相位；若未回退，也重新计算（统一流程）

    phase_x, phase_y, conf = find_phase(gray_arr, px, py, step)
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
    )


def detect_with_user_grid(
    gray: np.ndarray, w: int, h: int, step: float = 0.1
) -> Grid:
    """用户指定逻辑分辨率，反推块尺寸并定相。

    Args:
        gray: 灰度图数组。
        w: 用户指定逻辑宽。
        h: 用户指定逻辑高。
        step: 相位扫描步长。

    Returns:
        Grid: 检测结果。
    """
    gray_arr = np.asarray(gray, dtype=np.float64)
    H, W = gray_arr.shape
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ValueError(f"逻辑分辨率必须为正整数，得到 w={w}, h={h}")
    if w > W or h > H:
        raise ValueError(f"逻辑分辨率不能超过图像尺寸：w={w} > W={W} 或 h={h} > H={H}")
    px = gray_arr.shape[1] / w
    py = gray_arr.shape[0] / h
    phase_x, phase_y, conf = find_phase(gray_arr, px, py, step)
    # 用户指定分辨率路径：由全局相位 + 等距网格生成 cell_ys/cell_xs
    cell_ys, cell_xs = _equidistant_cell_grid(phase_x, phase_y, px, py, w, h)
    return Grid(
        w, h, px, py, phase_x, phase_y, conf, [(w, h, conf)],
        cell_ys=cell_ys, cell_xs=cell_xs,
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


def compute_edge_map(img: np.ndarray) -> np.ndarray:
    """计算 2D 边缘强度图。

    对输入图像（灰度 2D 或 RGB 3D）计算 Sobel 梯度幅值，再做 33×33
    局部归一化（复用 ``_local_contrast_normalize`` 的钳位局部 std 思路），
    组合 ``max(原始幅值/全局最大值, 局部归一化值)`` 后整体缩放到 [0, 1]。

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
    global_max = float(mag.max())
    if global_max < 1e-12:
        return np.zeros_like(gray)
    mag_norm = mag / global_max
    local_norm = _local_contrast_normalize(mag, window=33)
    combined = np.maximum(mag_norm, local_norm)
    cmax = float(combined.max())
    if cmax > 1e-12:
        combined = combined / cmax
    return combined


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
    px_i = max(1, int(round(px)))
    py_i = max(1, int(round(py)))
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
                nx1 = nx0 + px_i
                placed[ng] = (cy0, nx0, cy1, nx1)
            elif dx == -1:  # 向左扩展，共享垂直边界在 cx0 附近
                bc, peak = _best_col(em, iy0, iy1, cx0, edge_tol, enable_subpixel_refine)
                if bc is not None:
                    nx1 = float(bc)
                    weights[ng] = max(0.0, float(peak))
                else:
                    nx1 = cx0
                    weights[ng] = 1.0
                nx0 = nx1 - px_i
                placed[ng] = (cy0, nx0, cy1, nx1)
            elif dy == 1:  # 向下扩展，共享水平边界在 cy1 附近
                br, peak = _best_row(em, ix0, ix1, cy1, edge_tol, enable_subpixel_refine)
                if br is not None:
                    ny0 = float(br)
                    weights[ng] = max(0.0, float(peak))
                else:
                    ny0 = cy1
                    weights[ng] = 1.0
                ny1 = ny0 + py_i
                placed[ng] = (ny0, cx0, ny1, cx1)
            else:  # dy == -1，向上扩展，共享水平边界在 cy0 附近
                br, peak = _best_row(em, ix0, ix1, cy0, edge_tol, enable_subpixel_refine)
                if br is not None:
                    ny1 = float(br)
                    weights[ng] = max(0.0, float(peak))
                else:
                    ny1 = cy0
                    weights[ng] = 1.0
                ny0 = ny1 - py_i
                placed[ng] = (ny0, cx0, ny1, cx1)
            queue.append(ng)

    # 无邻居可达的剩余格用理想位置填充（基于种子相位）
    for gy in range(min_gy, max_gy + 1):
        for gx in range(min_gx, max_gx + 1):
            if (gx, gy) not in placed:
                nx0 = seed_x0 + gx * px_i
                ny0 = seed_y0 + gy * py_i
                placed[(gx, gy)] = (ny0, nx0, ny0 + py_i, nx0 + px_i)
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
    ideal_y = seed_y0 + (np.arange(h_logic + 1) + min_gy) * py_i
    ideal_x = seed_x0 + (np.arange(w_logic + 1) + min_gx) * px_i
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
