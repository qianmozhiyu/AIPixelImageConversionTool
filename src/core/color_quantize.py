"""K-means 色彩聚类量化模块。

将图像颜色通过 K-means 聚类量化为有限簇心，每个像素吸附到最近簇心。
渐变过渡区域自动变为硬边界（阶跃），无截断伪影。

与 unsharp mask 的区别：
- unsharp mask 放大高频，高强度时产生白边（截断失真）
- K-means 是替换操作（像素值→簇心值），不会超出 [0,255]，无白边

主要接口：
- ``color_quantize``：K-means 色彩聚类量化。
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def color_quantize(
    img: np.ndarray,
    n_colors: int = 16,
) -> np.ndarray:
    """对图像进行 K-means 色彩聚类量化。

    将图像所有像素在 RGB 空间聚类为 ``n_colors`` 个簇，每个像素替换为
    其所属簇的中心颜色。渐变过渡区域被量化为离散簇心，自动变为硬边界。

    Args:
        img: ``(H, W, 3)`` 图像数组（uint8 或 float64，范围 0-255）。
        n_colors: 聚类簇数（调色板大小），默认 16。

    Returns:
        量化后图像，dtype 与输入一致，形状不变。
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"输入必须是 (H, W, 3) 图像，实际 shape={arr.shape}")

    orig_dtype = arr.dtype
    H, W, C = arr.shape

    # 转为 float64 用于聚类
    pixels = arr.reshape(-1, 3).astype(np.float64)

    # 像素数少于簇数时直接返回
    if pixels.shape[0] < n_colors:
        return arr.copy()

    # 聚合唯一色与计数，避免对重复像素重复计算（大幅加速 KMeans）。
    # float64 输入先取整：0-255 色阶下的浮点值几乎全唯一，不取整则聚合退化
    # 为全量 KMeans（B9 修复）。
    unique_colors, inverse, counts = np.unique(
        np.round(pixels), axis=0, return_inverse=True, return_counts=True
    )

    # 唯一色数 <= n_colors 时跳过聚类直接返回
    if unique_colors.shape[0] <= n_colors:
        return arr.copy()

    # K-means 聚类（对唯一色加权聚类，sample_weight=频次）
    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=3,
    )
    kmeans.fit(unique_colors, sample_weight=counts)
    centers = kmeans.cluster_centers_

    # 仅对唯一色预测簇标签，再通过 inverse 索引映射回所有像素（避免重复预测）
    unique_labels = kmeans.predict(unique_colors)
    all_labels = unique_labels[inverse]
    quantized = centers[all_labels].reshape(H, W, 3)

    # 还原 dtype
    if np.issubdtype(orig_dtype, np.integer):
        max_val = np.iinfo(orig_dtype).max
        quantized = np.clip(np.round(quantized), 0, max_val).astype(orig_dtype)
    else:
        quantized = np.clip(quantized, 0, 255).astype(orig_dtype)

    return quantized
