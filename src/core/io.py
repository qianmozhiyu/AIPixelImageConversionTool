"""图像输入输出与灰度转换工具。

提供基于 PIL 的图像加载/保存，以及标准亮度系数的 RGB -> 灰度转换，
统一以 (H, W, 3) float64 RGB 0-255 数组作为内部表示。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..utils import normalize_image


def load_image(path: str | Path) -> np.ndarray:
    """从磁盘加载图像并归一化为 (H, W, 3) float64 RGB 0-255 数组。

    处理 EXIF 方向（手机/相机竖拍照片自动转正），文件不存在或损坏时
    抛出带文件路径的友好 ValueError（而非 PIL 原始异常）。

    Args:
        path: 图像文件路径。

    Returns:
        np.ndarray: 形状为 (H, W, 3) 的 float64 RGB 0-255 数组。

    Raises:
        ValueError: 文件不存在、无法解码或不是图像文件。
    """
    path = Path(path)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # 应用 EXIF orientation
            img = im.convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"图像文件不存在: {path}") from None
    except OSError as e:
        raise ValueError(f"无法解码图像文件 {path}: {e}") from None
    except Exception as e:  # 兜底：非图像文件等 PIL 各类异常
        raise ValueError(f"读取图像文件失败 {path}: {e}") from None
    return normalize_image(img)


def save_image(arr: np.ndarray, path: str | Path, scale: int = 1) -> None:
    """将 (H, W, 3) RGB 数组保存为图像文件。

    输入数组会被裁剪到 [0, 255] 并转为 uint8 后保存。
    当 scale > 1 时，使用最近邻插值放大（保持像素艺术锐利边缘）。

    Args:
        arr: 形状为 (H, W, 3) 的 RGB 数组（0-255）。
        path: 输出图像文件路径。
        scale: 放大倍数，1=原始分辨率。使用最近邻插值避免模糊。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="RGB")
    if scale > 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    img.save(path)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """按标准亮度系数将 RGB 图像转换为灰度图像。

    使用 ITU-R BT.601 系数：0.299 R + 0.587 G + 0.114 B。

    Args:
        rgb: 形状为 (H, W, 3) 的 RGB 数组。

    Returns:
        np.ndarray: 形状为 (H, W) 的 float64 灰度数组。
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
