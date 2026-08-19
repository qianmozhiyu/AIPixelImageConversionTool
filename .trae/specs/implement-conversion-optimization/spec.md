# 转换流水线精度与性能优化实施 Spec

## Why
按《优化方案.md》（2026-08-19 调研）落地第一、二梯队改进：现有检测链路存在灰度化丢失等亮度色边界（C1）、AA 过渡带下边界带能量漏采（C2，真实图区分度仅 1.08）、整数倍修正链无法捕捉非整数倍误检（C3，30px→16px 案例）、JPEG 8×8 伪影无处理（C7）等根因问题；需在 2K 图全流程 ≤10s 约束下提升转化准确度并保持现有测试零回归。

## What Changes
- **G7 一致性修复**：detect() 统一投影信号计算消除重复 O(H·W)（F1）；_vote_period 的 FFT 分数改为候选周期自身谱支撑（C4）；expand_grid_edge_guided 邻居偏移与理想外推改用浮点 px/py（C5）；Grid 元信息新增 low_confidence 标记（R1）
- **G1 感知空间检测信号**：新增 OKLAB 相邻像素色差信号模式（edge_map、投影剖面、相位评分），`detect_signal="gray"|"oklab"` 可选
- **P2 OKLab 预量化**：`_quantize_detection_signal` 升级为 OKLab 空间 + 子采样建调色板 + 最近色 LUT 快速路径，仅作用于检测分支
- **G3 峰值格点拟合**：投票周期确定后用 find_peaks + 间距最小二乘（L-BFGS-B）精化 px/py 为浮点值
- **G2 梳状能量集中度裁决**：连续 (pitch, phase) 梳状打分（捕获能量分数 − 覆盖分数）作为周期终审，含平局集取大周期、整数吸附（≥0.97×浮点最优）、quality×separation 置信度
- **G5 JPEG 网格防护**：交叉差分 + 64 相位投票检测 JPEG 8×8 网格，显著时对 8/16/24px 附近候选降权
- **E1 提取向量化**：extract_blocks 对均匀网格（median/mean）提供 reshape 向量化快速路径，输出与循环实现逐位一致
- **E2 工程清理**：cli.py `--upscale` help 文案修正（Q2）；io.save_image 改用 `Image.Resampling.NEAREST`（Q3）。注：Q1（删除 downsample.py）经核实**不执行**——`tests/test_grid_detect.py` L237 引用 `downsample_2x` 作测试工具，非死代码
- **验证与默认值定型**：新增合成精度回归测试集；TestImage 5 图 A/B；验证通过后切换默认参数并更新受影响测试
- 所有新路径带独立开关；实施阶段默认关闭保证零回归，验证阶段后定型默认值

## Impact
- Affected specs: research-conversion-precision-optimization（调研结论 → 本实施）
- Affected code:
  - `src/core/grid_detect.py`（主要改动：G7/G1/G3/G2/G5）
  - `src/pipeline.py`（PipelineParams 新参数 + 检测分支 + 透传）
  - `src/core/extract.py`（E1 快速路径）
  - `src/core/io.py`（Q3）、`src/cli.py`（Q2 + 新开关暴露）
  - `tests/`（新增 test_optimization_regression.py；默认值定型后更新受影响用例）
- 性能预算：image02.png（2048²）全流程 ≤10s（目标 ≤7s，基线 4.66s）；新增检测路径合计 ≤2s
- 默认行为变化（验证后定型）：detect_signal="oklab"、enable_pre_quantize=True、enable_peak_lattice_fit=True、enable_comb_energy_score/G5 依验证结果——全部可参数回退
- GUI 不新增控件（默认值即推荐配置）；CLI 暴露新开关

## ADDED Requirements

### Requirement: OKLAB 色差检测信号（G1）
系统 SHALL 支持基于 OKLAB 相邻像素色差的检测信号模式：色差图经局部对比度归一化生成 edge_map、色差沿轴求和生成投影剖面、相位评分基于边界带色差能量；通过 `PipelineParams.detect_signal="oklab"` 启用，默认 `"gray"` 保持现行为；detect() 接受 RGB 3D 输入（gray 2D 输入走原路径不变）。

#### Scenario: 等亮度异色边界可见
- **WHEN** 输入含等亮度异色块边界的合成图（灰度梯度≈0）且 detect_signal="oklab"
- **THEN** edge_map 在该边界处有显著响应（gray 模式下无响应），网格检测逻辑分辨率与真值一致

#### Scenario: 灰度模式零回归
- **WHEN** detect_signal="gray"（默认）运行现有全部测试
- **THEN** 检测结果与改动前完全一致

### Requirement: OKLab 检测前预量化（P2）
系统 SHALL 在网格检测前对检测分支图像做 OKLab 空间量化：子采样像素（约 1/16）构建 median-cut 调色板 + KDTree 最近色 LUT 全图映射；量化结果仅作为检测信号源，提取路径继续使用未量化图像。

#### Scenario: 量化不污染提取
- **WHEN** enable_pre_quantize=True 运行流水线
- **THEN** pixel_art 的颜色采样来自未量化图像（量化仅改变检测信号）

#### Scenario: 边界阶跃化提升区分度
- **WHEN** 对含 AA 过渡带（1-3px 渐变）的合成图测量 edge_band(真周期)/edge_band(子谐波) 区分度
- **THEN** oklab 预量化开启后区分度显著高于关闭时（记录具体数值于验证报告）

### Requirement: 峰值格点拟合周期精化（G3）
系统 SHALL 在投票周期确定后、find_phase 之前，对投影剖面做峰值格点拟合：scipy.signal.find_peaks（prominence 排序）取前 k 峰（k 扫描），L-BFGS-B 最小化 `RMSE(间距 − round(间距/s)·s)/s + 缺失格比例惩罚` 拟合浮点 spacing；拟合失败或偏离投票周期 >30% 时回退投票值；`enable_peak_lattice_fit` 开关。

#### Scenario: 非整数周期
- **WHEN** 输入 7.5px 块周期合成图（如 64 逻辑格 × 7.5 = 480px）
- **THEN** 检测 px/py ∈ [7.3, 7.7]，w_logic×h_logic = 64×64

### Requirement: 梳状能量集中度终审裁决（G2）
系统 SHALL 实现连续 (pitch, phase) 梳状能量集中度打分作为周期终审：pitch 浮点粗扫（自适应步长，全轴漂移 ≤0.5px）+ 步长折半精化，相位粗扫 0.5px；`score = 捕获能量分数 − 捕获权重分数`；平局集（最高分 ×1.15 内）取最大 pitch；整数 pitch 得分 ≥0.97× 浮点最优才吸附；置信度 = quality × separation，低于 0.35 拒绝吸附保持投票结果；`enable_comb_energy_score` 开关。

#### Scenario: 子谐波必然低分
- **WHEN** 对真周期 P 的合成图计算 score(P) 与 score(P/2)
- **THEN** score(P) > score(P/2)（双倍覆盖惩罚保证）

#### Scenario: 终审覆盖投票
- **WHEN** 投票结果为纹理周期 T 而梳状终审以足够置信度支持真周期 P
- **THEN** px/py 采纳 P，Grid.conf 反映终审置信度

### Requirement: JPEG 网格检测与降权（G5）
系统 SHALL 通过交叉差分 `C(x,y)=I(x+1,y+1)−I(x+1,y)−I(x,y+1)+I(x,y)` + 64 相位直方图投票 + 峰显著性检验检测 JPEG 8×8 网格；显著存在时对投票中 8/16/24px ±1 邻域候选的 edge 分数乘惩罚系数（约 0.6）；`jpeg_grid_guard` 开关。

#### Scenario: JPEG 压缩不引入 8px 误检
- **WHEN** 输入块周期非 8 的整数倍的干净像素图经 JPEG q=70 压缩
- **THEN** 检测逻辑分辨率与未压缩时一致

### Requirement: 合成精度回归测试集
系统 SHALL 提供 tests/test_optimization_regression.py：程序生成干净像素图（已知逻辑分辨率与块周期）+ 受控退化（JPEG q=70/85、AA 模拟、高斯噪声、非整数缩放如 7.5×、等亮度异色块、非整数倍纹理周期），断言 px/py/w_logic/h_logic 在容差内。

#### Scenario: 回归集可重复验证
- **WHEN** 运行该测试文件（默认参数定型后）
- **THEN** 全部用例通过，覆盖优化方案 1.2 节 C1/C2/C3/C7 对应场景

### Requirement: 提取快速路径（E1）
系统 SHALL 对均匀网格（cell 坐标等距，容差内）的 median/mean 提取提供向量化快速路径；非均匀网格或 mode/kmeans/dominant 方法回退现有循环实现。

#### Scenario: 输出一致且提速
- **WHEN** 对 image02.png（102×102 均匀网格，默认 median）提取
- **THEN** 输出与循环实现逐位一致，extract 阶段耗时 <0.2s（基线 0.56s）

### Requirement: 性能预算达标
系统 SHALL 在全部默认参数定型后满足：image02.png（2048×2048）全流水线 ≤10s（目标 ≤7s），其中新增检测路径（G1 色差 + P2 量化 + G3 拟合 + G2 裁决 + G5 防护）合计增量 ≤2s。

#### Scenario: 2K 性能验收
- **WHEN** 以定型默认参数对 image02.png 全流程计时（含加载）
- **THEN** 总耗时 ≤10s

## MODIFIED Requirements

### Requirement: 网格检测主流程（G7 修复）
detect() 内投影信号 sig_x/sig_y 统一计算一次并传递给 has_pixel_grid（消除重复 O(H·W) diff+sum）；_vote_period 的 FFT 分数改为候选周期处谱幅值（谐波频率线性插值归一化），删除"距主峰距离"近似；expand_grid_edge_guided 的邻居扩展偏移（`nx1 = nx0 + px_i` 等）与理想外推（`gx * px_i`）改用浮点 px/py；Grid 元信息新增 low_confidence 标记（conf < 0.4 时为 True，记入 PipelineResult.metadata）。

#### Scenario: 非整数周期网格线定位
- **WHEN** 对 px=7.48 的图（slice_01 类）做边缘引导扩展
- **THEN** 相邻网格线间距按浮点周期递推，不产生整数累积漂移

### Requirement: PipelineParams 参数集
新增参数（实施阶段默认关闭/gray，验证阶段定型）：`detect_signal: str = "gray"`、`enable_peak_lattice_fit: bool = False`、`enable_comb_energy_score: bool = False`、`jpeg_grid_guard: bool = False`；CLI 同步暴露对应开关；detect() 签名向后兼容（既有灰度调用不变）。

### Requirement: 文档同步
docs/CORE_PIPELINE_ALGORITHMS.md 增补新参数、新判据（G1/G2/G3/G5）与快速路径说明；README 处理流程不变。

## REMOVED Requirements
无（Q1 downsample 删除经核实不执行：`tests/test_grid_detect.py` L237 引用该模块，非死代码）
