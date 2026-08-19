# Checklist

- [x] G7 修复完成：F1 投影信号单次计算、C4 候选自身谱支撑 FFT 分数、C5 浮点周期扩展偏移、R1 low_confidence 标记
- [x] detect_signal="oklab" 色差信号模式实现：等亮度异色合成图检出正确网格；默认 "gray" 模式全量测试零回归
- [x] OKLab 预量化升级实现：子采样建调色板 + LUT，量化仅作用检测分支（提取不受影响的单测通过）
- [x] 峰值格点拟合实现：7.5px 非整数周期合成图 px∈[7.3,7.7] 且逻辑分辨率正确
- [x] 梳状能量集中度终审实现：score(P) > score(P/2) 性质单测通过；非整数倍纹理误检用例通过
- [x] JPEG 网格防护实现：非 8 整数倍周期图 JPEG q=70 压缩后逻辑分辨率不变
- [x] extract 均匀网格快速路径：输出与循环实现逐位一致，image02.png 提取耗时 <0.2s
- [x] Q2/Q3 工程清理完成（cli help 文案、Image.Resampling.NEAREST）
- [x] tests/test_optimization_regression.py 合成退化回归集完成且通过
- [x] TestImage 5 图 A/B 对比报告产出（检测结果 + 各阶段耗时，含基线对照）
- [x] 默认参数定型：验证通过的开关切换为默认开，全部既有测试更新后全量 pytest 通过
- [x] 性能验收：image02.png 全流程（定型默认参数）≤10s，目标 ≤7s
- [x] CLI 暴露全部新开关；docs/CORE_PIPELINE_ALGORITHMS.md 已同步新参数与算法说明
- [x] 向后兼容：detect() 灰度 2D 调用路径、Grid 既有字段语义、PipelineResult 结构不变
