# Checklist

- [x] 代码审查覆盖全部核心模块（pipeline、grid_detect、scale_detect、denoise、aa_removal、extract、color_quantize、io/utils）
- [x] 网格检测问题清单完成：按正确性/鲁棒性/性能分级，每项含代码位置与影响分析
- [x] 预处理链问题清单完成：覆盖降噪/AA 消除/放大/锐化/预量化/CLAHE，并指出缺失环节
- [x] TestImage 5 张测试图基线测量完成（各阶段耗时 + px/py/逻辑分辨率/置信度）
- [x] 2K 图（image02.png）全流程耗时已测量并与 ≤10s 达标线对比（实测 4.662s，达标）
- [x] GitHub 调研完成：覆盖已知项目（pixfix/crispx/unfake.js/pixel-art-downsampler 等）与新检索项目，含来源链接
- [x] 网络调研完成：网格检测与预处理两方向均有方法与来源链接
- [x] 调研结论与 GRID_PRECISION_UPGRADE.md 既有结论对照：未重复推荐已否定方案，或已明确说明重新审视的理由
- [x] 优化方案.md 已生成于项目根目录
- [x] 优化方案.md 结构完整：审查结论、调研结论、网格检测改进、预处理改进、落地优先级与性能预算五部分
- [x] 每条改进建议包含：改法（模块/函数/算法思路）、预期收益、针对的问题、性能预算、风险与回滚
- [x] 2K ≤10s 性能约束在每条建议中已评估（或有全局性能预算章节统一评估）
