"""AI 像素图转换工具命令行入口。

基于 ``argparse`` 构建命令行界面，将输入图像经 :class:`Pipeline` 转换为像素图
并保存为 PNG。支持 ``python -m src.cli image.jpg [options]`` 形式调用。

主要接口：
- ``main``：解析参数并运行流水线，返回退出码。
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import Pipeline, PipelineParams
from .core.io import load_image, save_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI 像素图转换工具")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("-o", "--output", default="output.png", help="输出 PNG 路径")
    parser.add_argument(
        "--extract-method",
        choices=["median", "mean", "mode", "kmeans"],
        default="median",
        help="块提取代表色算法：median（默认，与 GUI 一致）/mean/mode/kmeans",
    )
    parser.add_argument(
        "--extract-core-ratio",
        type=float,
        default=0.6,
        help="块核心区采样比例（0.5-1.0，规避边缘杂色，默认 0.6）",
    )
    parser.add_argument(
        "--denoise",
        choices=["none", "nl_means", "tv_chambolle", "bilateral"],
        default="nl_means",
        help="AI 去噪方法：nl_means（默认）/tv_chambolle/bilateral/none",
    )
    parser.add_argument("--strength", type=float, default=0.5, help="去噪强度")
    parser.add_argument(
        "--no-denoise", action="store_true", help="禁用 AI 去噪（等价于 --denoise none）"
    )
    parser.add_argument("--grid-min", type=int, default=3, help="网格检测最小候选周期")
    parser.add_argument("--grid-max", type=int, default=40, help="网格检测最大候选周期")
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=8.0,
        help="网格检测 SNR 阈值（默认 8.0，低对比度图像可降低）",
    )
    parser.add_argument(
        "--edge-tol",
        type=int,
        default=3,
        help="网格检测共享边界搜索半径（像素，默认3）",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="禁用亚像素精炼（默认启用）",
    )
    parser.add_argument(
        "--smooth-strength",
        type=float,
        default=0.5,
        help="全局平滑约束强度（0.0-1.0，默认0.5）",
    )
    parser.add_argument(
        "--outlier-reject-ratio",
        type=float,
        default=0.5,
        help="网格检测离群间距剔除阈值比例（0.0-1.0，默认0.5）",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="输出放大倍数（最近邻插值，保持像素锐利，默认 1=原始逻辑分辨率）",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="启用降噪后的双线性放大（默认关闭，与 GUI/库默认一致）",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="禁用放大（兼容参数；默认已关闭，保留以兼容旧脚本）",
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=2,
        help="放大倍数（默认 2）",
    )
    parser.add_argument(
        "--upscale-method",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
        help="放大算法（默认nearest）",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="禁用放大后的 unsharp mask 锐化（默认关闭，启用锐化需显式开启）",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="启用放大后的 unsharp mask 锐化（默认关闭）",
    )
    parser.add_argument(
        "--sharpen-strength",
        type=float,
        default=0.5,
        help="锐化强度（0.0-1.0，默认 0.5）",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="启用 CLAHE 局部对比度增强（默认禁用）",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="禁用 CLAHE（显式指定，默认禁用）",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=0.03,
        help="CLAHE 裁剪限制（0.01-0.1，默认0.03）",
    )
    parser.add_argument(
        "--no-palette-refine",
        action="store_true",
        help="禁用调色板精炼（默认启用）",
    )
    parser.add_argument(
        "--palette-colors",
        type=int,
        default=16,
        help="调色板精炼簇数（默认 16）",
    )
    parser.add_argument(
        "--fix-square",
        action="store_true",
        help="当检测到的逻辑分辨率与正方形差 1 时，自动修正为正方形输出",
    )
    args = parser.parse_args(argv)

    import time

    t0 = time.time()

    try:
        img = load_image(args.input)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    enable_clahe = args.clahe and not args.no_clahe
    params = PipelineParams(
        extract_method=args.extract_method,
        extract_core_ratio=args.extract_core_ratio,
        ai_denoise_method=("none" if args.no_denoise else args.denoise),
        ai_denoise_strength=args.strength,
        enable_clahe=enable_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        min_p=args.grid_min,
        max_p=args.grid_max,
        snr_threshold=args.snr_threshold,
        edge_search_tolerance=args.edge_tol,
        enable_subpixel_refine=not args.no_subpixel,
        smooth_strength=args.smooth_strength,
        outlier_reject_ratio=args.outlier_reject_ratio,
        enable_upscale=args.upscale and not args.no_upscale,
        upscale_factor=args.upscale_factor,
        upscale_method=args.upscale_method,
        enable_sharpen=args.sharpen and not args.no_sharpen,
        sharpen_strength=args.sharpen_strength,
        enable_palette_refine=not args.no_palette_refine,
        palette_colors=args.palette_colors,
        fix_square=args.fix_square,
    )
    pipeline = Pipeline(params)
    try:
        result = pipeline.run(img)
    except (ValueError, OSError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    try:
        save_image(result.pixel_art, args.output, scale=args.scale)
    except OSError as e:
        print(f"错误: 保存输出失败: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0
    print(f"转换完成: {args.output}")
    print(f"逻辑分辨率: {result.grid.w_logic} × {result.grid.h_logic}")
    print(f"唯一色数: {result.metadata['unique_colors']}")
    print(f"耗时: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
