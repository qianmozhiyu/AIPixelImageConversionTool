"""通用图像工具函数。

提供将 PIL Image 或 numpy 数组统一归一化为 (H, W, 3) float64 RGB 0-255
表示的便捷接口，供后续调色板映射、颜色空间转换等模块复用。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_image(image) -> np.ndarray:
    """将输入图像归一化为 (H, W, 3) float64 RGB 0-255 数组。

    接受 PIL Image 或 numpy 数组：
    - PIL Image: 通过 ``convert("RGB")`` 转换后转为 float64 数组。
    - numpy 数组: 直接转为 float64；若 dtype 为 object 则视为无效输入。

    若结果为浮点且最大值不超过 1.0，则按 0-1 区间放大至 0-255。

    Args:
        image: PIL Image 或 (H, W, 3) 的 numpy 数组。

    Returns:
        np.ndarray: 形状为 (H, W, 3) 的 float64 RGB 0-255 数组。

    Raises:
        ValueError: 输入无法转为有效图像数组，或形状不符合 (H, W, 3)。
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        if not isinstance(image, np.ndarray):
            arr = np.asarray(image)
        else:
            arr = image
        if arr.dtype == object:
            raise ValueError(
                "输入无法转为有效图像数组，期望 PIL Image 或 (H,W,3) numpy 数组"
            )
        arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA：丢弃 alpha 通道转 RGB（透明区域按原 RGB 保留，不预乘）
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"输入必须是 (H,W,3) RGB 或 PIL Image，实际形状 {arr.shape}"
        )

    if (
        np.issubdtype(arr.dtype, np.floating)
        and arr.size > 0
        and arr.min() >= 0.0
        and arr.max() <= 1.0
    ):
        arr = arr * 255.0

    return arr
