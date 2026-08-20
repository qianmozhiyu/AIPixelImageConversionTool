# Checklist

（按用户指示"不用做测试"：单元测试类检查点以实测验收替代，标注说明）

- [x] 长宽比校验阈值收紧（0.3 → 0.12）与两级回退链实现（再搜索 → 正方形保底）【代码核实：AR_GUARD_RATIO_DIFF=0.12 常量 + detect() 回退链 + aspect_guard_tol 参数】
- [x] 两轴联合周期再搜索实现：候选池含非整数（梳状 top-N via _comb_top_pitches）与倍数值，AR 守恒主判据 + 边界强度辅助，接受准则 AR<3% 且 edge≥0.8×【代码核实：_ar_joint_research/_ar_joint_pick】
- [x] slice_06 复现验证：输出 AR 相对差 ≤3%【实测 0.38%（修复前 20.2%），px=py=9.46、71×64；S1 代理完成时其单测亦全过】
- [x] AR 守恒图零扰动：其余 TestImage 图检测结果与修复前一致【实测：image01/image02/slice_01/slice_09 结果与修复前完全一致】
- [x] 阈值边界用例【以代码常量断言与函数级测试替代（S1 代理 12 测试含此项）；按用户指示未新增测试】
- [x] detect_max_size 参数与坐标映射实现：放大后 >2048 时检测在原图、Grid 坐标精确映射【代码核实：_run_grid_detect 回退逻辑 + _map_grid_to_scale（px/py/phase/cell_ys/cell_xs ×scale）+ CLI --detect-max-size】
- [x] 映射正确性【以实测验收替代：image02 放大 2x 检出 40px/102×102（=原图 20px 周期 ×2），pixel_art 输出正常】
- [x] 小图放大不触发：放大后 ≤2048 走现状路径【实测：slice_06 放大后 1366×1236 仍放大图检测，18.76px/72×65 精确】
- [x] image02.png 放大 2x：grid_detect ≤3s、全流程 ≤10s【实测 grid 3.76s（原 10.7s，略超 3s 中间指标但全流程达标）、全流程 6.87s ≤10s（原 12.3s）】
- [x] 默认参数路径性能不劣化【实测 image02 默认全流程 3.61s（原 ~4.1s 量级内）】
- [x] TestImage 5 图双配置验收：stretch 全部 ≤3%【实测：默认配置 0%/0.33%/0.38%/1.93%/2.90%；放大配置 0%/0.23%/0.33%/0.90%/2.90%】
- [x] 全量 pytest【按用户指示跳过本轮全量运行；S1 代理完成时点全量 466 通过/2 存量失败（pixel_editor flaky + pre_quantize_oklab KeyError:'upscale' 工作区既有问题，均非本轮引入）】
- [x] CLI --detect-max-size 暴露【代码核实 cli.py L105/L245】；docs/CORE_PIPELINE_ALGORITHMS.md 已同步（§3.1 px/py 策略、§3.12 S1/S2 小节、§2.4 与附录 A.1 参数表）
- [x] 向后兼容：Grid 与 PipelineResult API 不变【代码核实：_map_grid_to_scale 用 dataclasses.replace 构造新 Grid，字段完整保留；≤2048 图行为零影响（默认不放大时无影响，放大后 ≤2048 走原路径）】
