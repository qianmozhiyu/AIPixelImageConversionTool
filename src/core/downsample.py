"""图像块降采样模块。

对图像按 2x2 块取中位数降采样，用于平滑 AI 生成图像中像素块交界处的杂色噪声。
各块独立取像素，天然避免跨块颜色混合。奇数尺寸通过边缘复制补齐到 2 的整数倍后再降采样。

主要接口：
- ``downsample_2x``：2x2 中位数降采样，返回尺寸约为原图一半的图像。
"""

from __future__ import annotations

import numpy as np


def downsample_2x(img: np.ndarray) -> np.ndarray:
    """对图像进行 2x2 中位数降采样。

    将图像按 2x2 块取中位数，输出尺寸约为原图一半。若宽或高为奇数，
    先用边缘复制方式补 1 像素到偶数再降采样。

    Args:
        img: ``(H, W, 3)`` 或 ``(H, W)`` 的图像数组（float64 或 uint8）。

    Returns:
        降采样后图像，dtype 与输入一致。偶数尺寸时输出为 ``(H//2, W//2[, 3])``，
        奇数尺寸时输出为 ``((H+1)//2, (W+1)//2[, 3])``。
    """
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError(f"输入必须是 2D (H,W) 或 3D (H,W,3) 数组，实际 ndim={arr.ndim}")

    orig_dtype = arr.dtype
    H, W = arr.shape[:2]

    # 奇数尺寸边缘补齐到偶数
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        pad_width = [(0, pad_h), (0, pad_w)]
        if arr.ndim == 3:
            pad_width.append((0, 0))
        arr = np.pad(arr, pad_width, mode="edge")

    H2, W2 = arr.shape[:2]
    # 确保 H2, W2 是偶数
    H2_even = H2 - (H2 % 2)
    W2_even = W2 - (W2 % 2)
    arr = arr[:H2_even, :W2_even]

    # 2x2 中位数降采样：reshape 后取中位数（避免跨块颜色混合）
    if arr.ndim == 3:
        # (H//2, 2, W//2, 2, C) -> (H//2, W//2, C, 4) -> median over last axis
        blocks = arr[:H2_even, :W2_even, :].reshape(H2_even // 2, 2, W2_even // 2, 2, -1)
        # 将 (2,2) 的 4 个值展平到最后一维
        blocks = blocks.transpose(0, 2, 4, 1, 3).reshape(H2_even // 2, W2_even // 2, -1, 4)
        out = np.median(blocks, axis=-1)
    else:
        blocks = arr[:H2_even, :W2_even].reshape(H2_even // 2, 2, W2_even // 2, 2)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(H2_even // 2, W2_even // 2, 4)
        out = np.median(blocks, axis=-1)

    # 还原 dtype
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(np.round(out), 0, np.iinfo(orig_dtype).max).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype)

    return out

