# Tasks
- [x] Task 1: S1 拉伸修复——长宽比守恒防护与联合再搜索（src/core/grid_detect.py）
  - [x] SubTask 1.1: 收紧长宽比校验阈值（ratio_diff 0.3 → AR_GUARD_RATIO_DIFF 0.12）并调整回退链：失配时先联合再搜索、无优解回退边界强度较高方向的正方形块；detect() 新增 aspect_guard_tol 可选参数覆盖阈值
  - [x] SubTask 1.2: 实现 _ar_joint_research 联合再搜索——候选池（当前值/两轴投票原始值/FFT 主峰插值/ACF 前 5 峰/_comb_top_pitches 梳状 top-5 连续 pitch 含非整数/及 ×2 ×0.5 扩展，正方形先验对称交叉注入）；_ar_joint_pick 组合评分（ar_diff 0.005 档量化主排序 + e_x+e_y 辅助）；接受准则 ar_diff<0.03 且 edge≥0.8×
  - [x] SubTask 1.3: 单测 tests/test_grid_detect_ar_guard.py 12 个全过（含真实 slice_06 复现：stretch 20.2%→0.4%、px=py=9.5、71×64；合成 9.5px 非整数网格；阈值边界与函数级回退；整数网格零扰动）——注：该测试文件在用户"不用做测试"指示前已完成，予以保留
- [x] Task 2: S2 放大性能——检测尺寸上限与坐标映射（src/pipeline.py）
  - [x] SubTask 2.1: PipelineParams 新增 detect_max_size=2048 + docstring；CLI --detect-max-size（实现已存在于工作区，经核实功能完整：_run_grid_detect 大图回退逻辑 + _map_grid_to_scale 坐标映射 + CLI 透传）
  - [x] SubTask 2.2: _run_grid_detect 改造——放大后 max 边 > detect_max_size 时在 denoise 后原图检测（含 user_grid/user_hint 路径，signal 透传），Grid 坐标按缩放比映射回放大坐标系；≤上限现状不变；"gray" 缓存语义保持
  - [x] SubTask 2.3: ~~单测~~ 按用户指示（"不用做测试"）跳过新测试编写，以实测验收替代（见 Task 3.1 实测数据：image02 放大 grid 10.7s→3.8s、全流程 12.3s→6.9s，检测结果 40px/102×102 正确；slice_06 小图放大不触发仍 18.76px/72×65）
- [x] Task 3: 验收与文档
  - [x] SubTask 3.1: TestImage 5 图双配置实测验收——默认参数：stretch 全部 ≤3%（slice_06 0.38%、image01 2.90%、image02 0%、slice_01 0.33%、slice_09 1.93%）；放大 2x：stretch 全部 ≤3%，image02 全流程 6.87s ≤10s、grid_detect 3.76s（原 10.7s），其余 4 图 grid ≤1.6s
  - [x] SubTask 3.2: ~~全量 pytest 零回归~~ 按用户指示跳过全量测试；以实测功能验收替代（S1 代理在其完成时点已跑过全量：466 通过/2 存量失败，含 test_pre_quantize_oklab 的 KeyError:'upscale' 存量问题，属工作区既有改动非本轮引入）
  - [x] SubTask 3.3: docs/CORE_PIPELINE_ALGORITHMS.md 同步——§3.1 伪代码 px/py 策略更新（0.3→AR_GUARD 联合再搜索）、§3.12 新增 S1/S2 小节、§2.4 速查表与附录 A.1 补 detect_max_size 行
- [x] Task 4: ~~独立验证代理核对 checklist~~ 按用户指示（"不用做测试"）取消独立验证，由主代理以实测数据直接核对并勾选 checklist

# Task Dependencies
- Task 1 与 Task 2 相互独立（均已落地）
- Task 3 依赖 Task 1 + Task 2 全部完成
- Task 4 已按用户指示取消
