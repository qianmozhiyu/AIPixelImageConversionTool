# Tasks
- [x] Task 1: 代码审查与基线测量
  - [x] SubTask 1.1: 审查网格检测模块（src/core/grid_detect.py、src/core/scale_detect.py）：多判据投票链路的正确性、鲁棒性（子谐波/纹理/格点阵场景）、性能热点
  - [x] SubTask 1.2: 审查预处理链（src/core/denoise.py、src/core/aa_removal.py、src/pipeline.py 的 resize/sharpen/pre_quantize/clahe）：各步骤对网格检测信号的实际贡献、默认关闭项的问题、缺失的预处理环节
  - [x] SubTask 1.3: 审查其余模块（src/core/extract.py、src/core/color_quantize.py、src/core/color.py、src/core/io.py、src/utils.py）对最终精度的影响
  - [x] SubTask 1.4: 在 TestImage 测试集（5 张图）上运行基线测量：各阶段耗时、网格检测结果（px/py/逻辑分辨率/置信度）、2K 图（image02.png）全流程是否 ≤10s
- [x] Task 2: 外部调研（网络 + GitHub）
  - [x] SubTask 2.1: GitHub 检索像素图转换/网格检测开源项目：已知项目（pixfix、crispx、unfake.js、pixel-art-downsampler 等）的最新实现 + 新项目搜索，分析其网格检测与预处理策略
  - [x] SubTask 2.2: 网络检索网格/周期检测方法（学术与工程：周期信号检测、图像网格/棋盘格检测、透视网格估计等）
  - [x] SubTask 2.3: 网络检索预处理技术（降噪、边缘增强、色彩量化、AA 消除、像素图专用预处理等）
  - [x] SubTask 2.4: 调研结论与 docs/GRID_PRECISION_UPGRADE.md 既有结论对照：标记已否定方案（Spectral Comb 主导/GPA/cornerSubPix/仿射外推），分析其失败根因是否有新的解法（如从预处理侧增强信号）
- [x] Task 3: 生成优化方案.md（项目根目录）
  - [x] SubTask 3.1: 汇总代码审查结论：分级问题清单（正确性/鲁棒性/性能，含代码位置与影响）
  - [x] SubTask 3.2: 汇总外部调研结论：方法对比表（来源链接、思路、与本项目差异、可落地性）
  - [x] SubTask 3.3: 撰写具体改进建议：网格检测方向与预处理方向各若干条，每条含改法、预期收益、针对的问题、性能预算、风险与回滚
  - [x] SubTask 3.4: 复核文档：结构完整（审查结论/调研结论/网格检测改进/预处理改进/优先级与性能预算）、每条建议可落地、2K≤10s 约束已评估

# Task Dependencies
- Task 2、Task 3 依赖 Task 1 的基线数据与问题清单（Task 1 先行完成）
- Task 2 的 SubTask 2.1/2.2/2.3 相互独立，可并行执行
- Task 3 依赖 Task 1 与 Task 2 全部完成
