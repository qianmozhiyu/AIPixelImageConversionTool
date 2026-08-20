# 转换拉伸修复与放大性能优化 Spec

## Why
部分测试图转换出现拉伸畸变（slice_06.jpg 宽高比畸变达 20.2%：真实块周期 ~9.5px 非整数，两轴投票分别锁到错误周期 10/8，而现有长宽比校验阈值 0.3 过宽松放过了畸变），用户只能靠放大 2x 缓解（周期 ×2 后接近整数、检测变准），但 2K 图放大后 grid_detect 达 10.7s（4096² 检测面积 ×4）、全流程 12.3s 超出 10s 预算——两个问题互为锁死，需同时修复。

## What Changes
- **S1 拉伸修复（网格检测）**：
  - 长宽比校验阈值收紧：`ratio_diff > 0.3` 回退正方形块 → 失配阈值收紧至 0.12（20% 畸变不再放过）
  - 失配时回退策略升级：不再简单取单方向周期作正方形（虽 AR 守恒但逻辑分辨率仍可能错），增加**两轴联合周期再搜索**——候选池（两轴投票值、FFT/ACF 峰、梳状搜索 top 值及其 2 倍/半值，含非整数）组合中选「AR 守恒 + 边界带强度最高」的组合；搜索无显著更优解时回退保底正方形
- **S2 放大性能修复（流水线架构）**：
  - 新增 `detect_max_size: int = 2048`：grid_detect 的输入图超过该尺寸时，改在 resize 前的原图（denoise 后）上检测，网格坐标（px/py/phase/cell_ys/cell_xs）按放大倍率统一映射回放大坐标系供 extract 消费
  - 放大后 ≤2048（小图）保持现状（放大图上检测，保留精度收益）；>2048（2K 图）检测回到原图尺寸，grid_detect 恒 ≤~2.5s
- **验证**：TestImage 5 图双配置验收 + 回归集新增拉伸用例 + 文档同步

## Impact
- Affected specs: implement-conversion-optimization（其 G2/G3/G7 判据均在 detect() 内，S1 再搜索复用其积分图与候选机制）
- Affected code:
  - `src/core/grid_detect.py`（S1：detect() 的 px/py 选择策略区长宽比校验与再搜索）
  - `src/pipeline.py`（S2：PipelineParams 新增 detect_max_size、_run_grid_detect 检测图选择与坐标映射；S1 联动）
  - `src/cli.py`（--detect-max-size 可选暴露）
  - `tests/`（新增拉伸修复用例 + 放大性能/映射正确性用例；既有测试若 pin 旧阈值行为需更新）
- 性能预算：任何配置下 image02.png 全流程 ≤10s（放大 2x 时 grid_detect ≤3s）；默认路径维持 ~4.1s 不变
- 兼容性：Grid/PipelineResult API 不变；默认 detect_max_size=2048 仅改变放大后 >2048 图的检测执行域（结果应一致或更优——坐标映射为精确缩放，nearest 放大下数学等价）

## ADDED Requirements

### Requirement: 长宽比守恒防护（S1a 保底）
detect() 的 px/py 选择策略中，当两轴周期相对差 ≥0.15 分开使用时，逻辑分辨率长宽比校验阈值从 0.3 收紧至 0.12；失配时先尝试 S1b 联合再搜索，无显著更优解时回退「边界强度较高方向的周期作统一正方形块」（正方形块的 AR 天然守恒）。

#### Scenario: 明显畸变被拦截
- **WHEN** 检测两轴周期 (10, 8) 导致输出 AR 与原图 AR 相对差 >12%（slice_06 场景）
- **THEN** 不直接采纳该组合，触发再搜索或正方形回退，最终输出 AR 相对差 ≤5%

### Requirement: 两轴联合周期再搜索（S1b 进阶）
长宽比失配触发时，系统 SHALL 在两轴候选周期组合中联合搜索：候选池含两轴各自投票值、FFT/ACF 峰、梳状搜索 top-N pitch（不受 conf 门槛限制、含非整数）、及其 2 倍与 0.5 倍值（过滤至 [min_p, max_p]）；组合评分 = AR 守恒项（|AR(round(W/px)/round(H/py)) − W/H|/(W/H)）为主 + 两轴边界带边缘强度（复用已构建积分图）为辅；最优组合的 AR 失配 <3% 且边界强度不低于原组合 0.8 倍时采纳，否则回退 S1a 正方形。

#### Scenario: 非整数周期组合被找回
- **WHEN** slice_06.jpg（683×618，真实 ~9.5px 块）默认参数检测
- **THEN** 最终逻辑分辨率宽高比与原图相对差 ≤3%（对照：修复前 20.2%），且 px/py 落在 [8.5, 10.5] 区间的周期（组合或统一解均可）

#### Scenario: 正常图零扰动
- **WHEN** 对 AR 本就守恒的图（image02 等其余 4 张 TestImage）运行
- **THEN** 再搜索不触发（或触发后原组合胜出），检测结果与修复前完全一致

### Requirement: 检测尺寸上限与坐标映射（S2）
系统 SHALL 提供 PipelineParams.detect_max_size（默认 2048）：grid_detect 输入图（resize 后）任一边超过该值时，检测改在 resize 前图像上执行，检测完成后将 Grid 的 px/py/phase_x/phase_y/cell_ys/cell_xs 按缩放比（放大图尺寸/检测图尺寸）统一映射至放大坐标系；w_logic/h_logic 不变；extract 阶段不变（仍消费放大图与映射后网格）。

#### Scenario: 2K 图放大性能达标
- **WHEN** image02.png（2048²）以 enable_upscale=True, upscale_factor=2 运行
- **THEN** grid_detect 阶段耗时 ≤3s，全流程 ≤10s，检测结果（px/py/w_logic/h_logic）与映射前量级一致（周期 ≈2× 原图周期）

#### Scenario: 小图放大保留精度
- **WHEN** 放大后图 max 边 ≤ detect_max_size（如 slice 系列 683→1366）
- **THEN** 仍在放大图上检测（现状不变），slice_06 放大路径仍检出 ~18.76px 正确网格

#### Scenario: 映射正确性
- **WHEN** 在原图检测后映射至放大坐标系（nearest 放大）
- **THEN** 映射后网格落在真实块边界上（合成图断言：映射后 cell 边界列的图像梯度显著高于块内）

## MODIFIED Requirements

### Requirement: 长宽比校验（detect() px/py 选择策略）
现有「rel_diff < 0.15 统一 / 否则分开 + ratio_diff > 0.3 回退正方形」调整为：统一分支不变；分开分支的回退阈值 0.3 → 0.12，且回退动作改为「S1b 再搜索 → S1a 正方形保底」两级。

### Requirement: PipelineParams 参数集
新增 `detect_max_size: int = 2048`（docstring：网格检测输入图的最大边长，超出时检测在放大前图像执行并映射坐标，保证大图放大路径性能）；CLI 暴露 `--detect-max-size`。

## REMOVED Requirements
无
