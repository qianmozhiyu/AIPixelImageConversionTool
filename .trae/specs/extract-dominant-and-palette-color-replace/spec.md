# 提取默认主色算法 + 调色板/编辑器全局换色 Spec

## Why
当前块提取默认代表色为 `kmeans`，且保留了 `mode`/`mean` 低区分度算法，用户预期默认更贴近"主色"（感知空间众数）以获得更干净的像素色块。同时，转化结果与编辑器缺少"拾取某色→全局替换所有同色像素"的便捷能力，用户希望能直接从色卡或编辑器一键统一某颜色。

## What Changes
- **默认代表色换为主色**：`PipelineParams.extract_method` 默认由 `kmeans` 改为 `dominant`。
- **删除 `mode`(众数) 与 `mean`(均值) 提取算法**：从 extract 实现、UI 下拉、CLI choices、config 允许集中移除。
- **调色板精炼页新增色卡与全局换色**：启用"统一全局色彩"时，下方列出结果图像的唯一色卡；选中某色后可将其全局替换为指定目标色。
- **编辑器新增"统一改变颜色"**：在像素编辑器中提供对某颜色做全图像素替换的功能。

## Impact
- Affected specs: improve-pixel-editor-ux（编辑器新增改色入口）
- Affected code:
  - `src/core/extract.py`（移除 mode/mean 分支）
  - `src/pipeline.py`（extract_method 默认值、允许集）
  - `src/core/config.py`（无字段增删，仅允许值变化不影响序列化）
  - `src/cli.py`（--extract-method choices/默认）
  - `src/gui/param_panel.py`（提取下拉移除默认/两项；调色板页加色卡+换色控件）
  - `src/gui/pixel_editor.py`（新增统一改色工具/动作）
  - `src/core/color.py` 或新增 `src/core/color_remap.py`（颜色替换工具函数）
  - `tests/`（extract/pipeline/cli/gui 相应示例更新，新增色卡与编辑器改色测试）

## ADDED Requirements

### Requirement: 全局颜色替换核心函数
系统 SHALL 提供将图像中所有与指定源色（容差内）匹配的像素替换为目标色的函数，供色卡换色与编辑器共用以保证语义一致。

#### Scenario: 精确源色替换
- **WHEN** 输入 `(H,W,3)` 图像与源色、目标色，源色为图像中真实存在的颜色
- **THEN** 输出中所有与原源色完全相等的像素变为目标色，其余像素不变

### Requirement: 调色板精炼页色卡与换色
当"启用 K-means 调色板精炼（统一全局色彩）"勾选时，调色板页下方 SHALL 显示结果图（extract 后、精炼前或精炼后视实现）的唯一色卡（每个色块可点击选中）；用户为选中色指定目标色后，SHALL 在像素图上全局替换所有同色像素并刷新预览。

#### Scenario: 勾选后显示色卡
- **WHEN** 用户勾选"启用统一全局色彩"
- **THEN** 下方出现该图像唯一色的色卡列表（颜色样本 + 数量），数量上限以配置的 `palette_colors` 为准展示
- **THEN** 取消勾选时色卡隐藏

#### Scenario: 选择色卡统一换色
- **WHEN** 用户点击某个色卡，并选定一个目标色
- **THEN** 像素图里所有与该色匹配的像素被替换为目标色，画布预览随之更新
- **THEN** 该操作可撤销/可再次选择其他颜色

### Requirement: 编辑器全局统一改色
像素编辑器 SHALL 提供"统一改变颜色"能力：用户在编辑器中取某源色（或当前前景色），并指定目标色后，将画布图像中所有同色像素替换为目标色。

#### Scenario: 编辑器内一键换色
- **WHEN** 用户在编辑器中触发统一改色，确认源色与目标色
- **THEN** 画布内所有与源色匹配的像素变为目标色，且可撤销
- **THEN** 与既有撤销/恢复机制兼容

## MODIFIED Requirements

### Requirement: 块提取代表色算法（默认与允许集）
系统 SHALL 将块提取默认代表色算法改为 `dominant`（主色，感知空间众数），并将允许算法收敛为 `dominant`/`kmeans`/`median`——移除 `mode` 与 `mean`。

#### Scenario: 默认主色
- **WHEN** 使用 `PipelineParams()` 默认参数运行流水线
- **THEN** `extract_method == "dominant"`，输出为感知空间主色块

#### Scenario: CLI 与配置收敛
- **WHEN** 运行 `python -m src.cli --help` 或加载 GUI 提取下拉
- **THEN** `--extract-method` 与下拉仅暴露 `dominant/kmeans/median`
- **THEN** 传入已被移除的 `mode`/`mean` 时回退到 `median`（与既有"未知回退 median"语义一致）

## REMOVED Requirements

### Requirement: 众数(mode) / 均值(mean) 提取算法
**Reason**: 对真像素色块区分度低，且与默认主色语义重叠，用户要求删除以简化。
**Migration**: 既有 `extract_method="mode"`/`"mean"` 的配置或调用将按"未知方法回退 median"处理；UI/CLI 不再提供选项。