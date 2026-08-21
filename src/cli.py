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
        choices=["median", "mean", "mode", "kmeans", "dominant"],
        default="kmeans",
        help="块提取代表色算法：kmeans（默认，与 GUI 一致）/median/mean/mode/dominant",
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
        help="启用降噪后的放大（默认算法 nearest，默认关闭，与 GUI/库默认一致）",
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
        choices=["bilinear", "bicubic", "lanczos"],
        default="bilinear",
        help="放大算法（默认bilinear，最近邻已移除）",
    )
    parser.add_argument(
        "--detect-max-size",
        type=int,
        default=2048,
        help=(
            "网格检测输入图的最大边长（像素，默认2048）：放大后图像超过该值时，"
            "网格检测改在放大前的图像上执行，检测完成后网格坐标按缩放比映射回"
            "放大坐标系，保证大图放大路径的检测性能；未超限时仍在放大图上检测"
        ),
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
    parser.add_argument(
        "--aa-removal",
        action="store_true",
        help="启用抗锯齿消除预处理（默认关闭）",
    )
    parser.add_argument(
        "--aa-passes",
        type=int,
        default=2,
        help="抗锯齿消除迭代次数（默认 2）",
    )
    parser.add_argument(
        "--aa-threshold",
        type=float,
        default=0.5,
        help="抗锯齿消除两主色距离阈值（默认 0.5）",
    )
    parser.add_argument(
        "--detect-signal",
        choices=["gray", "oklab"],
        default="gray",
        help="网格检测信号模式：gray（默认，BT.601 灰度）/oklab（OKLAB 感知色差，等亮度异色块边界可见）",
    )
    parser.add_argument(
        "--pre-quantize",
        action="store_true",
        help="启用网格检测前预量化（默认关闭）",
    )
    parser.add_argument(
        "--denoise-guard",
        action="store_true",
        help="启用去噪-检测耦合保护（默认关闭）",
    )
    parser.add_argument(
        "--no-peak-lattice-fit",
        action="store_true",
        help="禁用峰值格点拟合周期精化（默认开启）：投票周期确定后拟合格点间距"
        "精化为浮点周期，支持非整数块尺寸如 7.5px，失败自动回退投票值；"
        "轴一致性防护下精化前两轴一致而精化后分裂时整体回退投票值",
    )
    parser.add_argument(
        "--no-comb-energy-score",
        action="store_true",
        help="禁用梳状能量集中度终审（默认开启）：对投影信号做连续 (pitch, phase) "
        "梳状打分，原理性压制子谐波/倍频误检（子谐波覆盖惩罚翻倍、倍频能量减半），"
        "低置信自动回退投票结果",
    )
    parser.add_argument(
        "--no-jpeg-grid-guard",
        action="store_true",
        help="禁用 JPEG 8×8 压缩网格检测与候选降权（默认开启）：交叉差分投票检测"
        " JPEG DCT 网格相位，显著时对 8/16/24px 附近候选降权，防护压缩伪影周期误检",
    )
    parser.add_argument(
        "--interior-cleanliness",
        action="store_true",
        help="启用内部洁净度边界评分（P0，默认关闭）：周期投票的边界强度分改用"
        "边界/格心边缘能量比，以块内干净度压制子谐波/倍频误检——直击真实 AI 图"
        "边界区分度不足问题",
    )
    args = parser.parse_args(argv)

    import time

    t0 = time.time()

    try:
        img = load_image(args.input)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    params = PipelineParams(
        extract_method=args.extract_method,
        extract_core_ratio=args.extract_core_ratio,
        ai_denoise_method=("none" if args.no_denoise else args.denoise),
        ai_denoise_strength=args.strength,
        min_p=args.grid_min,
        max_p=args.grid_max,
        detect_signal=args.detect_signal,
        snr_threshold=args.snr_threshold,
        edge_search_tolerance=args.edge_tol,
        enable_subpixel_refine=not args.no_subpixel,
        smooth_strength=args.smooth_strength,
        outlier_reject_ratio=args.outlier_reject_ratio,
        enable_upscale=args.upscale and not args.no_upscale,
        upscale_factor=args.upscale_factor,
        upscale_method=args.upscale_method,
        detect_max_size=args.detect_max_size,
        enable_sharpen=args.sharpen and not args.no_sharpen,
        sharpen_strength=args.sharpen_strength,
        enable_palette_refine=not args.no_palette_refine,
        palette_colors=args.palette_colors,
        fix_square=args.fix_square,
        enable_aa_removal=args.aa_removal,
        aa_removal_passes=args.aa_passes,
        aa_removal_threshold=args.aa_threshold,
        enable_pre_quantize=args.pre_quantize,
        denoise_grid_guard=args.denoise_guard,
        enable_peak_lattice_fit=not args.no_peak_lattice_fit,
        enable_comb_energy_score=not args.no_comb_energy_score,
        jpeg_grid_guard=not args.no_jpeg_grid_guard,
        enable_interior_cleanliness=args.interior_cleanliness,
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
