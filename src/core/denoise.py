"""AI 像素图像去噪模块。

提供基于 scikit-image 的图像级去噪功能 ``denoise_ai_noise``，用于收敛 AI
生成图像中的扩散色偏与 JPEG DCT 块效应。支持 ``nl_means``（非局部均值）、
``tv_chambolle``（全变分 Chambolle）与 ``bilateral``（双边滤波）三种方法，
以及 ``none`` 直通选项。

此外提供 ``apply_clahe`` 函数，对去噪后的图像做 CLAHE 局部对比度增强，
以提升后续网格检测对低对比度区域的识别能力。
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_bilateral


def denoise_ai_noise(
    img_rgb: np.ndarray,
    method: str = "nl_means",
    strength: float = 0.5,
) -> np.ndarray:
    """对 AI 生成图像进行图像级去噪。

    用于收敛扩散色偏与 JPEG DCT 块效应。支持三种模式：

    - ``"nl_means"``：非局部均值去噪（``denoise_nl_means``），以
      ``h = strength * 0.03`` 作为滤波强度，``patch_size=2``、``patch_distance=3``，
      适合去除平滑区域中的低频色偏与 DCT 伪影。
    - ``"tv_chambolle"``：全变分 Chambolle 去噪（``denoise_tv_chambolle``），以
      ``weight = strength * 0.1`` 作为正则权重，能在去噪的同时较好地保留边缘。
    - ``"bilateral"``：双边滤波去噪（``denoise_bilateral``），以
      ``sigma_color = strength * 0.1``、``sigma_spatial = 2`` 为参数，
      在平滑色偏的同时保留像素块边缘。
    - ``"none"`` 或 ``strength <= 0``：不做处理，返回输入数组的副本。
    - 其他未知方法：同样返回输入数组的副本。

    内部统一将输入归一化到 ``[0, 1]`` 区间调用 skimage，再还原回 ``[0, 255]``。

    Args:
        img_rgb: 形状为 (H, W, 3) 的 RGB 0-255 float64 数组。
        method: 去噪方法，``"nl_means"``、``"tv_chambolle"``、``"bilateral"`` 或 ``"none"``。
        strength: 去噪强度，``[0, 1]`` 区间内的浮点数，越大去噪越强。

    Returns:
        np.ndarray: 形状为 (H, W, 3) 的 float64 RGB 0-255 去噪后数组，
        已裁剪到 ``[0, 255]``。
    """
    arr = np.asarray(img_rgb, dtype=np.float64)
    if method == "none" or strength <= 0:
        return arr.copy()

    img_01 = arr / 255.0

    if method == "nl_means":
        h = strength * 0.03
        denoised = denoise_nl_means(
            img_01, patch_size=2, patch_distance=3, h=h, channel_axis=2
        )
    elif method == "tv_chambolle":
        weight = strength * 0.1
        denoised = denoise_tv_chambolle(img_01, weight=weight, channel_axis=2)
    elif method == "bilateral":
        sigma_color = strength * 0.1
        denoised = denoise_bilateral(
            img_01, sigma_color=sigma_color, sigma_spatial=2, channel_axis=2
        )
    else:
        return arr.copy()

    return np.clip(denoised * 255.0, 0.0, 255.0)


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """对图像做 CLAHE 局部对比度增强。

    使用 ``skimage.exposure.equalize_adapthist``（对比度受限自适应直方图均衡化）
    增强弱边缘对比度，提升网格检测对低对比度区域的识别能力。

    按通道分别处理，避免部分 skimage 版本中 ``equalize_adapthist`` 经
    ``adapt_rgb`` 适配器转发 ``channel_axis`` 参数到灰度滤镜时引发
    ``TypeError``。

    Args:
        img_rgb: 形状为 (H, W, 3) 的 RGB 0-255 float64 数组。
        clip_limit: CLAHE 裁剪限制（0.01-0.1），越大增强越强。

    Returns:
        np.ndarray: 形状为 (H, W, 3) 的 float64 RGB 0-255 增强后数组。
    """
    from skimage.exposure import equalize_adapthist
    arr = np.asarray(img_rgb, dtype=np.float64)
    img_01 = arr / 255.0
    enhanced = np.empty_like(img_01)
    for c in range(img_01.shape[2]):
        enhanced[:, :, c] = equalize_adapthist(img_01[:, :, c], clip_limit=clip_limit)
    return np.clip(enhanced * 255.0, 0.0, 255.0)
