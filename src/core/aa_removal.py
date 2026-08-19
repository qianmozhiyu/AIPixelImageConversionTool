"""可选抗锯齿（AA）消除预处理模块。

针对 AI 伪像素图块边界/网格线上的 1px 线性混合（抗锯齿）杂色，在
OKLAB 感知空间用「三角形不等式」判定当前像素颜色是否位于邻域两主色之间，
若是则吸附到较近的主色，从而把混合色清理为纯块色。

模块内自包含（自带 OKLAB 转换，约 15 行向量化实现），不依赖 ``color.py``，
避免与并行改动该文件的代理耦合。

主要接口：
- ``remove_anti_aliasing``：对 (H, W, 3) RGB 图执行 AA 消除。

参考：pixfix（lovelaced/pixfix）的 OKLAB 三角形不等式 AA 消除思路。
"""

from __future__ import annotations

import numpy as np

# 8 邻域偏移：行偏移、列偏移
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# 三角形不等式相对容差：p 恰在线段 a-b 上（理想 AA 混合）时该值≈0；
# OKLAB 非线性使饱和色对的 RGB 中点 tri 可达 ~0.25，故主路径取 0.4
_TRIANGLE_TOL = 0.4
# 回退路径（p 自身类为邻域最多）使用更严格的容差，防纯色角点被误判
_STRICT_TOL = 0.05


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """RGB(0-255) -> OKLAB 向量化转换（模块内自包含）。

    标准 OKLAB 公式：sRGB 非线性压缩 -> 线性化 -> LMS -> 立方根 -> OKLAB。
    """
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # sRGB 线性化
    r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    g = np.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
    b = np.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
    # LMS 三刺激值（线性 RGB）
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    # 正交化到 OKLAB 通道
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, A, B], axis=-1)


def remove_anti_aliasing(
    img_rgb: np.ndarray,
    threshold: float = 0.5,
    passes: int = 2,
) -> np.ndarray:
    """消除像素图块边界/网格线上的抗锯齿混合杂色。

    算法（参考 pixfix 的三角形不等式 AA 消除）：
    1. 转到 OKLAB 感知空间，并对 8 邻域颜色按 RGB 粗量化归为若干类。
    2. 找出出现频次最高的两个主要邻域色 a、b（8 邻域内的前两类）。
    3. 仅当两主色足够不同（d(a,b) > ``threshold``）时继续，避免破坏
       有意保留的细节。
    4. 用三角形不等式判定当前像素颜色 p 是否"位于"a、b 之间：
       ``|d(p,a)+d(p,b)-d(a,b)| / d(a,b)`` 相对较小（< 内部容差）则视为
       AA 混合色；且要求 p 与 a、b 均明显不同（真混合色而非纯块色），
       然后吸附到更近的 a 或 b。
    5. 重复 ``passes`` 次，以处理更宽的 AA 过渡带。

    Args:
        img_rgb: ``(H, W, 3)`` RGB 0-255 图像数组（float64 或 uint8）。
        threshold: 两主色 OKLAB 距离阈值，低于该距离不做吸附。
        passes: 迭代次数（≥1），重复处理以收敛宽 AA 过渡。

    Returns:
        ``(H, W, 3)`` float64 RGB 0-255 处理后的数组副本。
    """
    arr = np.asarray(img_rgb, dtype=np.float64)
    if arr.ndim == 2:
        # 灰度输入按三通道复制，保持接口统一
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"输入必须是 (H, W, 3) RGB 数组，实际形状 {arr.shape}")

    oklab = _rgb_to_oklab(arr)
    H, W = arr.shape[:2]
    n_off = len(_OFFSETS)

    def _neigh(pad: np.ndarray, k: int) -> np.ndarray:
        """取第 k 个 8 邻域对应的 (H, W[, 3]) 切片。"""
        dy, dx = _OFFSETS[k]
        return pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]

    for _ in range(max(1, int(passes))):
        pad_rgb = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
        pad_ok = np.pad(oklab, ((1, 1), (1, 1), (0, 0)), mode="edge")
        # 1) 收集 8 邻域并量化 RGB 为类别标签（每通道 32 级，至多 512 类）
        lab = np.empty((H, W, n_off), dtype=np.int32)
        for k in range(n_off):
            n_rgb = _neigh(pad_rgb, k)
            q = (n_rgb / 32.0).astype(np.int32)
            lab[..., k] = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        # 中心像素自身类别：判断 AA 混合色时应基于周围「纯色」而非其它 AA 像素
        q_c = (arr / 32.0).astype(np.int32)
        lab_c = q_c[..., 0] * 64 + q_c[..., 1] * 8 + q_c[..., 2]
        # 2) 统计各类频次（8 样本逐对比较，内存友好）
        counts = np.zeros((H, W, n_off), dtype=np.int8)
        for k in range(n_off):
            labk = lab[..., k]
            for l in range(n_off):
                counts[..., k] += lab[..., l] == labk
        # 全部邻域的前两类主色
        arg0 = np.argmax(counts, axis=-1)
        lab0 = np.take_along_axis(lab, arg0[..., None], axis=-1)[..., 0]
        counts1 = np.where(lab != lab0[..., None], counts, -1)
        arg1 = np.argmax(counts1, axis=-1)
        lab1 = np.take_along_axis(lab, arg1[..., None], axis=-1)[..., 0]
        # 排除中心色后的前两类主色（交叉 AA 模式下的回退判定）
        selfmask = lab != lab_c[..., None]
        counts_ns = np.where(selfmask, counts, -1)
        arg0n = np.argmax(counts_ns, axis=-1)
        lab0n = np.take_along_axis(lab, arg0n[..., None], axis=-1)[..., 0]
        counts1n = np.where((lab != lab0n[..., None]) & selfmask, counts_ns, -1)
        arg1n = np.argmax(counts1n, axis=-1)
        lab1n = np.take_along_axis(lab, arg1n[..., None], axis=-1)[..., 0]
        # 3) 每类取邻域均值作为主色（RGB 与 OKLAB），按类累计省内存
        a_ok = np.zeros((H, W, 3))
        b_ok = np.zeros((H, W, 3))
        a_rgb = np.zeros((H, W, 3))
        b_rgb = np.zeros((H, W, 3))
        an_ok = np.zeros((H, W, 3))
        bn_ok = np.zeros((H, W, 3))
        an_rgb = np.zeros((H, W, 3))
        bn_rgb = np.zeros((H, W, 3))
        a_cnt = np.zeros((H, W))
        b_cnt = np.zeros((H, W))
        an_cnt = np.zeros((H, W))
        bn_cnt = np.zeros((H, W))
        for k in range(n_off):
            n_rgb = _neigh(pad_rgb, k)
            n_ok = _neigh(pad_ok, k)
            m0 = lab[..., k] == lab0
            m1 = lab[..., k] == lab1
            m0n = lab[..., k] == lab0n
            m1n = lab[..., k] == lab1n
            a_ok += np.where(m0[..., None], n_ok, 0.0)
            b_ok += np.where(m1[..., None], n_ok, 0.0)
            a_rgb += np.where(m0[..., None], n_rgb, 0.0)
            b_rgb += np.where(m1[..., None], n_rgb, 0.0)
            an_ok += np.where(m0n[..., None], n_ok, 0.0)
            bn_ok += np.where(m1n[..., None], n_ok, 0.0)
            an_rgb += np.where(m0n[..., None], n_rgb, 0.0)
            bn_rgb += np.where(m1n[..., None], n_rgb, 0.0)
            a_cnt += m0
            b_cnt += m1
            an_cnt += m0n
            bn_cnt += m1n
        a_cnt = np.maximum(a_cnt, 1.0)[..., None]
        b_cnt = np.maximum(b_cnt, 1.0)[..., None]
        an_cnt = np.maximum(an_cnt, 1.0)[..., None]
        bn_cnt = np.maximum(bn_cnt, 1.0)[..., None]
        a_ok = a_ok / a_cnt
        b_ok = b_ok / b_cnt
        a_rgb = a_rgb / a_cnt
        b_rgb = b_rgb / b_cnt
        an_ok = an_ok / an_cnt
        bn_ok = bn_ok / bn_cnt
        an_rgb = an_rgb / an_cnt
        bn_rgb = bn_rgb / bn_cnt
        # 4) 三角形不等式：p 是否位于 a-b 之间（AA 混合色）
        d_ab = np.sqrt(((a_ok - b_ok) ** 2).sum(axis=-1))
        d_pa = np.sqrt(((oklab - a_ok) ** 2).sum(axis=-1))
        d_pb = np.sqrt(((oklab - b_ok) ** 2).sum(axis=-1))
        tri = np.abs(d_pa + d_pb - d_ab) / np.maximum(d_ab, 1e-12)
        d_abn = np.sqrt(((an_ok - bn_ok) ** 2).sum(axis=-1))
        d_pan = np.sqrt(((oklab - an_ok) ** 2).sum(axis=-1))
        d_pbn = np.sqrt(((oklab - bn_ok) ** 2).sum(axis=-1))
        trin = np.abs(d_pan + d_pbn - d_abn) / np.maximum(d_abn, 1e-12)
        # 主路径：p 明显不同于两主色（普通 AA 混合色）
        p_solid = (lab_c == lab0) | (lab_c == lab1)
        main_mask = (
            (~p_solid)
            & (d_ab > threshold)
            & (tri < _TRIANGLE_TOL)
            & (d_pa > 1e-4)
            & (d_pb > 1e-4)
        )
        # 回退路径：交叉 AA 模式下 p 自身类最多（但为混合色），用非中心类
        # 严格判定（容差更紧，避免纯色角点被误判）
        ns_mask = (
            p_solid
            & (d_abn > threshold)
            & (trin < _STRICT_TOL)
            & (d_pan > 1e-4)
            & (d_pbn > 1e-4)
        )
        mask = main_mask | ns_mask
        # 5) 吸附到更近的主色（平局取 a）
        main_near_rgb = np.where((d_pa <= d_pb)[..., None], a_rgb, b_rgb)
        main_near_ok = np.where((d_pa <= d_pb)[..., None], a_ok, b_ok)
        ns_near_rgb = np.where((d_pan <= d_pbn)[..., None], an_rgb, bn_rgb)
        ns_near_ok = np.where((d_pan <= d_pbn)[..., None], an_ok, bn_ok)
        nearer_rgb = np.where(main_mask[..., None], main_near_rgb, ns_near_rgb)
        nearer_ok = np.where(main_mask[..., None], main_near_ok, ns_near_ok)
        arr = np.where(mask[..., None], nearer_rgb, arr)
        oklab = np.where(mask[..., None], nearer_ok, oklab)

    return np.clip(arr, 0.0, 255.0)
