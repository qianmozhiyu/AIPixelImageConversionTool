# 像素图转化精度与准确度优化调研（网格检测 + 预处理）Spec

## Why
工具对部分真实 AI 像素图仍存在网格误检（既有文档已记录隐藏翻倍案例：真实 30px 块被检出 16px，且真实图边界强度区分度仅 1.08，远低于合成图的 1.7-2.1）与转换误差；需在 2K 图全流程 ≤10s 的性能约束下，通过代码审查 + 外部调研（网络/GitHub）找到可落地的精度提升路径。

## What Changes
- **不改动任何代码**：本变更的唯一交付物是一份优化方案文档
- 全面审查现有代码（重点：网格检测 `grid_detect.py` 与预处理链），输出问题与优化空间清单
- 在 TestImage 测试集上建立性能与检测精度基线
- 调研网络与 GitHub 上可提升转化精度/准确度的方法（网格检测策略 + 预处理技术）
- 在项目根目录生成 `优化方案.md`：分析结论 + 可落地改进建议（含性能预算与优先级）

## Impact
- Affected specs: 无（纯调研与文档产出）
- Affected code: 无代码改动；文档将引用 `src/core/grid_detect.py`、`src/core/scale_detect.py`、`src/core/denoise.py`、`src/core/aa_removal.py`、`src/core/extract.py`、`src/core/color_quantize.py`、`src/pipeline.py`
- 交付物：`优化方案.md`（项目根目录）
- 参考基线：`docs/GRID_PRECISION_UPGRADE.md`（上一轮调研，Spectral Comb/GPA/cornerSubPix/仿射外推已实测并在真实图上否定或默认关闭，本轮不得重复推荐已否定方案，但可重新审视其失败根因）

## ADDED Requirements

### Requirement: 现有代码审查
系统 SHALL 对现有转换流水线做全面代码审查，覆盖全部核心模块（pipeline、grid_detect、scale_detect、denoise、aa_removal、extract、color_quantize、io/utils），重点评估网格检测的准确性/鲁棒性/性能与预处理链的有效性，并在 TestImage 测试集上建立基线。

#### Scenario: 基线测量
- **WHEN** 对 `src/images/TestImage` 全部 5 张测试图（image01.jpg 573×577、image02.png 2048×2048、slice_01.jpg 480×853、slice_06.jpg 451×447、slice_09.jpg 486×903）以默认参数运行流水线
- **THEN** 记录每张图的各阶段耗时与网格检测结果（px/py、逻辑分辨率、置信度），形成基线数据；2K 图（image02.png）全流程 ≤10s 为达标线

#### Scenario: 问题识别
- **WHEN** 审查 `grid_detect.py`（约 2400 行多判据投票链路）与预处理模块（denoise/aa_removal/resize/sharpen/pre_quantize）
- **THEN** 输出分级问题清单（正确性 / 鲁棒性 / 性能三类），每项含代码位置与影响分析；特别复核既有文档记录的未解决问题（真实图隐藏翻倍、边界强度区分度不足）

### Requirement: 外部方法调研
系统 SHALL 通过网络搜索与 GitHub 检索（使用 trae-remote-official:github 插件与浏览器），调研提升像素图转化精度与准确度的方法，覆盖网格检测与预处理两个方向，并与既有调研结论（`docs/GRID_PRECISION_UPGRADE.md`）对照。

#### Scenario: 网格检测调研
- **WHEN** 检索网格/周期检测方法（学术论文与开源项目，含已知项目 pixfix、crispx、unfake.js、pixel-art-downsampler 及新项目）
- **THEN** 每个方法记录：来源（可访问链接）、核心思路、与本项目现状的差异、可落地性评估

#### Scenario: 预处理调研
- **WHEN** 检索面向像素图/网格检测的预处理技术（降噪、边缘增强、色彩量化、抗锯齿消除、超分等）
- **THEN** 同上记录，并特别关注能增强真实图网格信号、解决"边界强度区分度不足（1.08）"问题的预处理手段

### Requirement: 优化方案文档
系统 SHALL 在项目根目录生成 `优化方案.md`，给出可落地的改进建议（每条含改法、预期收益、风险、性能预算），性能约束为 2K 图全流程 ≤10s。

#### Scenario: 文档生成
- **WHEN** 代码审查与外部调研完成
- **THEN** `优化方案.md` 包含五个部分：① 代码审查结论（分级问题清单）；② 外部调研结论（方法对比 + 来源链接）；③ 网格检测改进方案；④ 预处理改进方案；⑤ 落地优先级与性能预算

#### Scenario: 建议可执行性
- **WHEN** 阅读任一条改进建议
- **THEN** 能明确：改什么（模块/函数）、怎么改（算法思路）、为何有效（针对哪个已识别问题）、性能代价多少、风险与回滚方式

## MODIFIED Requirements
无

## REMOVED Requirements
无
