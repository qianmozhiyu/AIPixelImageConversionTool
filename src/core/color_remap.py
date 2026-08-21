"""纯颜色替换模块。

提供 ``remap_color``：将图像中与指定源色匹配（精确或容差内）的像素，
仅替换其前 3 通道（RGB）为目标色，alpha 通道保持不变。全程不原地修改输入，
任何退化输入均安全返回输入数组的拷贝。

主要接口：
- ``remap_color``：颜色掩码 + 向量化替换。
"""

from __future__ import annotations

import numpy as np


def remap_color(img: np.ndarray, src_rgb, dst_rgb, tol: float = 0) -> np.ndarray:
    """将图像中匹配 ``src_rgb``（容差 ``tol`` 内）的像素前3通道替换为 ``dst_rgb``。

    判定仅基于前 3 通道：对每个像素，若三通道各自的绝对差
    ``abs(img[...,i] - src[i]) <= tol`` 均成立则判定匹配（``tol=0`` 表示精确相等）。
    匹配像素的前 3 通道被替换为目标色，alpha 通道（若存在）保持原值。

    Args:
        img: ``(H, W, 3)`` 或 ``(H, W, 4)`` 的 numpy 数组（uint8 或 float）。
        src_rgb: 长度为 3 的源色序列（0-255 整数或浮点）。
        dst_rgb: 长度为 3 的目标色序列（0-255 整数或浮点）。
        tol: 容差，每个通道独立判定。默认 0 表示精确匹配。

    Returns:
        新数组，shape / dtype 与输入一致。``img`` 不会被修改。

    退化输入安全：
      - ``img`` 为空图、非 2D（ndim != 3）、通道数 < 3，返回输入拷贝。
      - ``src_rgb == dst_rgb`` 时输出与输入逐位一致。
    """
    arr = np.asarray(img)

    # 退化输入安全：仅处理 (H,W,C>=3)。
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr.copy()
    H, W, C = arr.shape
    if H == 0 or W == 0:
        return arr.copy()

    src = np.asarray(src_rgb, dtype=np.float64).reshape(-1)[:3]
    dst = np.asarray(dst_rgb, dtype=np.float64).reshape(-1)[:3]
    if src.size < 3 or dst.size < 3:
        return arr.copy()

    out = arr.copy()

    # 基于前 3 通道向量化判定匹配掩码
    rgb = arr[..., :3].astype(np.float64)
    mask = np.all(np.abs(rgb - src) <= tol, axis=-1)

    if np.any(mask):
        # 目标色 clip 到 0-255
        dst_clip = np.clip(dst, 0, 255)
        # 仅替换前 3 通道，alpha 通道保持不变
        out[..., :3] = np.where(mask[..., np.newaxis], dst_clip, out[..., :3])

    return out