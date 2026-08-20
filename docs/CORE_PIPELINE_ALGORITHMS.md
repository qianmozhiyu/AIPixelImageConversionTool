# 核心转换流程与算法文档

> 本文档描述 AI 像素图转换工具的核心转换引擎：从输入图像到规范像素图的完整
> 流水线、每个阶段的算法原理、关键阈值与默认参数、以及设计改进动机。
> 适用于二次开发、算法调优与代码维护参考。UI 界面不在本文档范围内。

## 1 文档说明

### 1.1 目标读者

- 需要理解转换引擎内部原理的开发者；
- 需要调整网格检测、块提取等算法参数的使用者；
- 计划扩展或替换某个阶段的二次开发者。

### 1.2 术语约定

| 术语 | 含义 |
| --- | --- |
| 逻辑分辨率（w_logic × h_logic） | 输出像素图中"像素块"的数量，即规范像素图的宽×高 |
| 块周期（px / py） | 输入图像中每个像素块在 x / y 方向的尺寸（像素） |
| 相位（phase_x / phase_y） | 网格在图像中的偏移起点 |
| 网格（Grid） | 检测结果：块周期 + 相位 + 逻辑分辨率 + 逐交点坐标 |
| 子谐波 | 真实块周期的整数分之一（如真实周期 32 的子谐波 16/8/4） |
| 边缘强度图（edge_map） | 对灰度图做 Sobel + 对比度归一化得到的 [0,1] 边缘响应图 |
| 增量重跑 | 参数变更后仅从受影响阶段起重新计算，而非全量重跑 |

### 1.3 源码对应关系

| 模块 | 职责 |
| --- | --- |
| `src/pipeline.py` | 流水线编排（阶段依赖、缓存、增量重跑）、调整大小（放大）与锐化内联实现 |
| `src/gui/worker.py` | 流水线异步线程封装（QThread） |
| `src/core/grid_detect.py` | 网格检测（本文档核心，约 2000 行） |
| `src/core/scale_detect.py` | runs/GCD 整数尺度检测（网格检测的正交交叉验证先验，见 3.11） |
| `src/core/extract.py` | 块提取（逐块代表色采样） |
| `src/core/color_quantize.py` | K-means 调色板精炼 |
| `src/core/denoise.py` | 降噪（NL-Means / TV / 双边）与 CLAHE |
| `src/core/aa_removal.py` | 抗锯齿消除预处理（OKLAB 三角形不等式吸附，默认关闭） |
| `src/core/io.py` | 图像读写与灰度转换 |
| `src/utils.py` | 图像归一化工具 |

---

## 2 流水线总览

### 2.1 阶段依赖与数据流

转换引擎按固定阶段顺序处理图像，每一阶段消费上一阶段的输出，结果缓存复用：

```
load → denoise_global → resize → grid_detect → extract → palette_refine
```

阶段前置依赖（`Pipeline.PREREQS`）：

| 阶段 | 前置 | 输出（缓存 key） |
| --- | --- | --- |
| `load` | 无 | `image`（(H,W,3) float64 RGB 0-255） |
| `denoise_global` | load | `denoise_global` |
| `resize` | denoise_global | `resize` |
| `grid_detect` | resize | `grid_detect`（Grid 对象）、`gray`（灰度图，供提取复用） |
| `extract` | resize, grid_detect | `extract`（(h_logic, w_logic, 3) uint8） |
| `palette_refine` | extract | `palette_refine`（最终 pixel_art） |

数据流要点：

- 输入统一经 `normalize_image` 归一化为 `(H, W, 3)` float64 RGB 0-255；
- `grid_detect` 阶段同时缓存灰度图（`to_gray`，BT.601 系数），供后续阶段复用，
  避免重复灰度化；
- `extract` 作用于调整大小后图像的坐标系（若启用了放大），与网格坐标一致，无需映射回原图；
- 最终输出 `pixel_art` 为 `(h_logic, w_logic, 3)` uint8 的规范像素图。

### 2.2 缓存与增量重跑机制

`Pipeline` 内部维护两个状态：

- `_cache: dict[str, Any]`：各阶段结果缓存；
- `_stage_done: set[str]`：已完成的阶段标记。

支持四种运行模式：

| 接口 | 行为 |
| --- | --- |
| `run(image)` | 整图一键运行全部阶段；**不重置 `_user_grid`**（用户显式网格应保留，见 B11 修复） |
| `run_stage(stage)` | 仅运行指定阶段，自动补齐其前置依赖 |
| `reset_from(stage)` | 从指定阶段起 `_invalidate_from` 后重跑（参数变更后的增量重跑） |
| `set_user_grid(w, h)` | 设置用户网格覆盖，并从 `grid_detect` 起失效重跑 |

`_invalidate_from(stage)` 会删除该阶段及之后所有阶段的缓存与完成标记。
`run()` 每阶段前通过进度回调上报 `(stage, i/n)`，全部完成后上报 `("done", 1.0)`。

### 2.3 PipelineWorker 异步执行

`src/gui/worker.py` 的 `PipelineWorker(QThread)` 将流水线放到后台线程执行，避免
阻塞界面，通过信号与主线程通信：

- `progress(str, float)`：阶段名 + 进度百分比；
- `finished_result(object)`：完成，携带 `PipelineResult`；
- `error(str)`：异常信息；
- `cancelled()`：用户取消。

运行有三种模式：

1. 传入 `pipeline` 且 `reset_stage` 非空：`pipeline.reset_from(stage)` 增量重跑；
2. 传入 `pipeline`：`pipeline.run(image)`；
3. 均未传入：内部新建 `Pipeline(params)` 全量运行。

进度回调内检查取消标志（`_cancelled`），取消时抛出 `InterruptedError` 并发出
`cancelled` 信号。

### 2.4 PipelineParams 默认参数速查

`src/pipeline.py::PipelineParams` 为 dataclass，全量参数见附录 A。关键默认值：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `enable_ai_denoise` | True | 启用图像级去噪 |
| `ai_denoise_method` | "nl_means" | 去噪方法 |
| `enable_upscale` | False | 默认不放大（B4 修复：最近邻放大引入像素格点伪周期） |
| `upscale_factor` | 2 | 放大倍数 |
| `min_p` / `max_p` | 3 / 40 | 网格候选周期搜索范围（像素） |
| `snr_threshold` | 8.0 | FFT 网格判定 SNR 阈值 |
| `detect_max_size` | 2048 | 检测输入图最大边长：放大后超限则检测回退原图并映射坐标（S2） |
| `detect_signal` | "gray" | 检测信号模式："gray"/"oklab"（G1，等亮度异色块需 oklab） |
| `enable_pre_quantize` | False | 检测前对检测信号做小调色板预量化（P2，默认关） |
| `enable_peak_lattice_fit` | True | 峰值格点拟合周期精化（G3，非整数块尺寸如 7.5px） |
| `enable_comb_energy_score` | True | 梳状能量集中度周期终审（G2，压制子谐波/倍频误检） |
| `jpeg_grid_guard` | True | JPEG 8×8 网格检测与候选降权（G5，防压缩伪影误检） |
| `extract_method` | "median" | 块代表色算法 |
| `extract_core_ratio` | 0.6 | 核心区采样比例 |
| `enable_palette_refine` | True | 启用 K-means 调色板精炼 |
| `palette_colors` | 16 | 调色板目标色数 |

---

## 3 网格检测（核心算法）

网格检测是转换引擎的核心：从像素图中估计每个"像素块"的尺寸（周期）与位置
（相位），并输出逐交点网格坐标。AI 生成像素图通常带有块内纹理、渐变、JPEG
DCT 伪影、抗锯齿等干扰，检测必须对这些伪影鲁棒——尤其要避免把大块误拆成
多个小块（网格数量翻倍问题）。

### 3.1 detect() 主流程

`grid_detect.py::detect(gray, min_p=3, max_p=40, step=0.1, snr_threshold=8.0,
edge_tol=3, enable_subpixel_refine=True, smooth_strength=0.5,
outlier_reject_ratio=0.5, comb_weight=0.0, use_comb_prefilter=False,
enable_runs_crosscheck=True, enable_plausibility_gate=True, signal="gray",
enable_peak_lattice_fit=False, enable_comb_energy_score=False,
jpeg_grid_guard=False)`（函数级默认保守关闭；`PipelineParams` 层的
G2/G3/G5 默认开启并透传），伪代码：

```python
# 1. 边缘强度图（一次性计算，后续多处复用）
edge_map = compute_edge_map(gray)

# 2. 统一信号：归一化灰度只计算一次，FFT 粗判与投票共用同一信号来源，
#    消除 has_pixel_grid(FFT) 与投票的预处理不一致
norm_gray = _local_contrast_normalize(gray)
sig_x = |diff(norm_gray, axis=1)|.sum(axis=0)   # 垂直边缘投影
sig_y = |diff(norm_gray, axis=0)|.sum(axis=1)   # 水平边缘投影

# 3. FFT 周期粗判：has_pixel_grid 带 ACF 谐波纠正与方向保护
#    （pre_normalized=True 跳过内部重复归一化；BVR 判据仍用原始 gray）
has, snr, period, period_x, period_y, snr_x, snr_y = \
    has_pixel_grid(norm_gray, min_p, max_p, snr_threshold, edge_map, pre_normalized=True)
if not has:
    # FFT 失败回退：梯度投影峰的中位数间距法
    px, py = _estimate_grid_gradient(gray)

# 4. 双轴多判据投票（轴 1 = x 方向、轴 0 = y 方向）
#    G5（jpeg_grid_guard）：投票前 detect_jpeg_grid 交叉差分检测 JPEG 8×8
#    网格相位，显著（strength ≥ 0.06）时向两轴 _vote_period 传
#    jpeg_penalty，对 8/16/24 ±1 候选的 edge 分数降权 ×0.6（见 3.12）
vote_px, conf_x = _vote_period(gray, sig_x, min_p, max_p, axis=1, edge_map=edge_map)
vote_py, conf_y = _vote_period(gray, sig_y, min_p, max_p, axis=0, edge_map=edge_map)

# 5. 投票结果覆盖 FFT（置信度高，或投票周期的边缘强度显著更优；
#    FFT 主峰易被块内纹理污染，edge 判据不受其影响，故可用 edge 无条件覆盖）
if conf_x > 0.3 or (e_vote_x > 1e-12 and e_fft_x > 1e-12 and e_vote_x > 1.2 * e_fft_x):
    period_x = vote_px
# ... y 轴同理

# 6. runs/GCD 整数尺度交叉验证（Task 4，见 3.11）：
#    仅当 runs 证据强（runs_x == runs_y > 0 且命中率 >= RUNS_STRONG_HIT_RATE）
#    时对两轴分别调 _runs_correct_period：
#      - 与 vote 一致（相对差 <= RUNS_AGREE_REL）：周期吸附为整数 runs
#      - runs > vote 且 runs ≈ k*vote（k>=2）：投票误检子谐波，
#        edge(runs) >= RUNS_ADOPT_EDGE_RATIO(0.9)*edge(vote) 时采纳 runs（真实大块）
#      - runs < vote：默认不缩小周期（防把真实块缩成块内纹理周期），
#        除非 period >= 3 且 edge(runs) > RUNS_SHRINK_EDGE_RATIO(1.3)*edge(vote)
#    无证据/不满足时静默回退纯投票结果（零回归）

# 6.5 G3 峰值格点拟合精化（enable_peak_lattice_fit，见 3.12）：
#     runs 之后、px/py 选择之前，对两轴投影信号做峰值格点拟合，把投票
#     周期精化为浮点值（支持 7.5px 等非整数块尺寸）；失败自动回退投票值；
#     轴一致性防护：精化前两轴一致（相对差 ≤ 5%）而精化后分裂（> 2%）
#     时判定拟合锁错格点，两轴整体回退精化前投票值
#
# 6.6 G2 梳状能量终审（enable_comb_energy_score，见 3.12）：
#     G3 之后、px/py 选择之前，对两轴投影信号做连续 (pitch, phase) 梳状
#     打分搜索（_comb_best_period）：置信度 ≥ COMB_ACCEPT_CONF(0.35) 且
#     pitch > 0 时覆盖该轴周期；低置信自动回退 G3/投票链结果

# 7. px/py 策略：
#    单方向失败 → 统一为另一方向；
#    相对差 < 0.15 → 按 SNR 加权平均；
#    否则分开，且长宽比差异 > AR_GUARD_RATIO_DIFF(0.12) 时先做两轴联合
#    周期再搜索（_ar_joint_research，见 3.12），无优解再回退为边界强度
#    更高方向的周期（正方块）——防止非整数周期两轴锁不同错误周期导致的
#    拉伸畸变（真实案例：9.5px 块两轴锁 10/8，输出 AR 畸变 20%）

# 8. 高分辨率合理性门控（Task 5，见 3.11）：确定最佳周期后、find_phase 前
#    周期 < PLAUSIBILITY_MIN_PERIOD(3px) 时链式检查整数倍 k*p：
#    若某倍周期边界带 >= PLAUSIBILITY_EDGE_RATIO(1.3)*当前边界带，说明小周期
#    是块内纹理而非网格边界，采纳更大周期并继续放大（2→4→8→…，
#    至多 PLAUSIBILITY_MAX_STEPS=6 步）；统一正方形块时两轴分别门控后
#    取一致结果，保持 px == py

# 9. 相位搜索
phase_x, phase_y, conf = find_phase(gray, px, py, step)   # step=0.1

# 10. 逻辑分辨率：round + 越界容差回退
w_logic = _count_blocks(W, px, phase_x)   # 见 3.9

# 11. 方块检测 + BFS 分配 + 边缘引导扩展（新流程，优于等距假设）
squares   = detect_squares(edge_map, px, py, enable_subpixel_refine=...)
placed    = assign_grid_bfs(squares, px, py, ...)
cell_ys, cell_xs = expand_grid_edge_guided(edge_map, placed, bounds, px, py,
                                           edge_tol, smooth_strength, ...)
# placed 为空时兜底：phase + 等距网格 _equidistant_cell_grid

# 12. 候选分辨率主选 + 邻域（±1）加权，低置信拒绝：
#     conf < 0.05 且非 FFT 命中 → ValueError
# 13. 元信息：comb_score = max(comb(sig_x, px), comb(sig_y, py))
```

`Grid` 数据结构字段：

```python
Grid(
    w_logic, h_logic,        # 逻辑分辨率（像素块数量）
    px, py,                  # 块周期（像素）
    phase_x, phase_y,        # 网格相位
    conf,                    # 置信度 0-1
    candidates=[],           # 候选周期列表
    cell_ys=None,            # (h_logic+1, w_logic+1) 逐交点 y 坐标
    cell_xs=None,            # (h_logic+1, w_logic+1) 逐交点 x 坐标
    comb_score=0.0,          # 谐波梳得分（元信息）
    low_confidence=False,    # conf < 0.4 时为 True，供 GUI/元信息提示人工确认
    comb_energy_conf=0.0,    # G2 梳状终审置信度（两轴均值，仅元信息）
    jpeg_grid=(),            # G5 JPEG 网格检测结果 (is_significant, phase, strength)
)
```

`cell_ys/cell_xs` 约定：格子 `(j, i)` 的像素区域为
`[cell_ys[j,i], cell_ys[j+1,i]) × [cell_xs[j,i], cell_xs[j,i+1])`，支持局部
漂移（非严格等距网格）。

### 3.2 compute_edge_map 边缘强度图

```python
def compute_edge_map(img) -> (H, W) float64 [0,1]:
    # 1. Sobel 梯度幅值
    mag = hypot(sobel_x, sobel_y)
    # 2. 两种归一化取 max：
    #    全局归一化：mag / global_max
    #    局部对比度归一化：_local_contrast_normalize(mag, window=33)
    #      （局部标准差钳位到 [0.5*global_std, global_std]，线性外推边界）
    # 3. 整体缩放到 [0, 1]
```

设计要点：局部对比度归一化对低对比度区域的边缘（AI 生成图中常见的弱边缘）
敏感，全局归一化对强边缘保留，二者取 max 兼顾两种场景。

### 3.3 has_pixel_grid 周期判定

粗判阶段：对图像沿两轴取梯度绝对值并按另一轴求和得到一维投影信号，用
FFT + ACF 交叉验证。

```python
def has_pixel_grid(gray, min_p=3, max_p=40, snr_threshold=8.0, edge_map=None,
                   pre_normalized=False):
    if not pre_normalized:
        gray = _local_contrast_normalize(gray)   # 局部对比度归一化
    sig_x = |diff(gray, axis=1)|.sum(axis=0)     # 垂直边缘投影
    sig_y = |diff(gray, axis=0)|.sum(axis=1)     # 水平边缘投影

    snr_x, period_x = _fft_band_snr(sig_x, min_p, max_p)
    peaks_x = _acf_period(sig_x, min_p, max_p)
    acf_base_x = _harmonic_interpret(peaks_x)   # ACF 谐波基频
    # 方向保护：仅当 FFT 周期是 ACF 基频的整数倍（真谐波关系）时才纠正
    # 为基频，且纠正后的周期需通过 _direction_protection_ok：
    #   小周期边界强度 >= 大周期边界强度的 0.7 倍才允许缩小周期
    if 整数倍关系 and 方向保护通过:
        period_x = acf_base_x
    # ... y 轴同理
    has = snr >= snr_threshold
```

`pre_normalized=True` 时跳过内部 `_local_contrast_normalize`：`detect` 已统一
计算 `norm_gray` 一次，FFT 粗判与投票使用同一信号来源，避免重复归一化导致
两阶段预处理不一致。

方向保护的意义：ACF 在小 lag 处的噪声峰可能把正确的 FFT 周期误判为倍频，
只有"FFT 周期 = ACF 基频 × 整数"且缩小后边界强度不显著下降时才允许纠正。

### 3.4 _vote_period 四判据投票

`_vote_period(gray, profile, min_p, max_p, axis, edge_map=None,
comb_weight=0.0, use_comb_prefilter=False, edge_integral=None,
gray_integral=None, gray_integral_sq=None)` 是周期选择的最终裁决器。

**候选收集**：FFT 主峰 + ACF 峰 + 谐波解释基频 +（新路径启用时）谐波梳
top-5 候选。

**打分与加权**：

| 路径 | ACF | FFT | BVR | 边界强度 edge | 谐波梳 comb |
| --- | --- | --- | --- | --- | --- |
| 旧路径（无 edge_map） | 0.4 | 0.3 | 0.3 | — | — |
| 新路径（有 edge_map） | 0.3 | 0.2 | 0.1 | 0.4×(1-cw) | 0.4×cw |

各判据先除以自身最大值归一化。BVR 权重压低（0.2→0.1）的原因：2D 真块 BVR
对"像素格点阵"类输入（最近邻放大产生的像素重复结构）会强烈偏向小周期——
格点对齐时块内完全均匀、BVR 极高，实测导致 2x 放大图误检出放大因子的子谐波；
边界强度与 ACF/FFT 才是主判据。

**性能优化**：2D BVR 需逐周期相位扫描，开销大。新路径先用廉价判据
（`0.4*acf + 0.3*fft + 0.3*edge`）预排序，只对前 `VOTE_BVR_LIMIT = 10` 名
候选计算 BVR；旧路径的 legacy BVR（1D 条带）开销小，保持逐候选计算。

**积分图复用**：`detect()` 主路径一次性预构建边缘积分图（`_build_integral`）
与灰度（及其平方）积分图，经 `edge_integral` / `gray_integral` /
`gray_integral_sq` 传入 `_vote_period`，`_edge_band_strength`（O(1) 条带求和）
与 2D BVR（O(1) 块求和）全程复用，消除每个候选周期重复重建整图积分图的
O(H·W) 开销；参数为 None 时内部惰性构建（向后兼容）。实测 2K 测试图（2048²）
的 grid_detect 阶段耗时从约 4.9s 降至约 2s，输出逐位一致。

**置信度**：

```python
ratio = total[best] / max(total[second], 1e-6)   # 主/次总分比；单候选=2.0
consistency = 归一化后各判据(>0.3)的个数 / 判据数   # 判据一致性
conf = min(1.0, ratio / 2.0) * consistency
```

**子谐波修正**（解决"大网格被拆成小网格"的核心，新路径）：

```python
cur = best
e_ratio_used = 0.0
while True:
    new_best = cur
    for k in range(2, 7):                    # k=2..6，支持多级链（如 4→8→24）
        kp = cur * k
        if kp 不在 ACF 峰集 or kp 超出 [min_p, max_p]: continue
        e_ratio = edge(kp) / edge(cur)        # 边界强度比（主判据）
        if e_ratio > 1.3:                     # 真实块边界是连续强边缘
            support = (acf(kp) > acf(cur))    # 支撑：ACF 严格上升
                      or (bvr(cur) > 0 and bvr(kp) > bvr(cur))  # 或 BVR 严格上升
            if use_comb_prefilter and comb_weight > 0:
                support &= comb(kp) > 1.2 * comb(cur)   # comb 一致性预筛
            if support:
                new_best = kp; e_ratio_used = e_ratio; break
    if new_best == cur: break
    cur = new_best
# 修正成功：置信度上调（修正说明更长周期更可信）
conf = min(1.0, conf * (1.0 + 0.3 * clamp(e_ratio_used - 1.3, 0.0, 1.0)))
```

支撑判据的作用：防止"纯网格"（无块内纹理）因相位偏置被假修正。旧路径（无
edge_map）仅支持 k=2,3，条件为 BVR 或 ACF 放大 1.5 倍。

### 3.5 五大判据算法细节

**FFT 带通 SNR（`_fft_band_snr`）**：

```python
# 去均值 → rfft → 频带掩码（freq ∈ [1/max_p, 1/min_p]）
# 带内取局部极大候选：按 max(左爬升, 右下降) 打分，平局比幅值
# snr = (peak / median)^2     # 峰值与带内中位数的功率比
# 抛物线插值得到亚 bin 的周期估计
```

**ACF 周期（`_acf_period`）**：用 FFT 加速自相关（Wiener-Khinchin），
`acf/acf[0]` 归一化，在 `[min_p, max_p]` 内取局部极大（±1 邻域）按 ACF 值降序。

**谐波解释（`_harmonic_interpret`）**：寻找能解释最多 ACF 峰的最小基频，
要求 `|ratio - round(ratio)| / round(ratio) < 0.1`（峰周期为基频的整数倍）。

**2D 块方差对比度（`_block_variance_ratio`）**：

```python
def _block_variance_ratio(gray, period, axis, integral=None, integral_sq=None):
    # 2D 真块：按 period 将图像切为 period×period 的块（轴无关，各向同性）
    # integral/integral_sq 为灰度及其平方的积分图：同时提供时用 O(1) 矩形查询
    # 求各块和/块平方和（块均值与块内均方同旧实现同一数学，仅求和次序有浮点
    # 差异），替代旧实现每相位全图 reshape 的 O(H·W) 开销；任一为 None 时回退
    # 旧实现（保持向后兼容）；_vote_period 复用灰度积分图
    best = 0
    for phase in 相位扫描(步长 step = max(1, period // 4)):
        cell_means = 各块均值            # 块间（积分图 O(1) 块和 / area）
        within     = 各块内方差的均值      # 块内（O(1) 块平方和 - 均值²）
        between    = var(cell_means)
        best = max(best, between / within)
    return best
```

真网格对齐时块内同质（within 小）、块间差异大（between 大），比值最高。
相位扫描步长随周期缩放，限制每候选相位数（≤~16），平衡精度与开销。

**边界带边缘强度（`_edge_band_strength`）**：

```python
def _edge_band_strength(edge_map, period, axis, phases=None, integral=None):
    # 积分图 O(1) 求条带和：axis=0 取水平边界（行条带），axis=1 取垂直边界
    # integral 为 edge_map 的预构建积分图（None 时内部构建，向后兼容）：
    # detect()/_vote_period 主路径只构建一次并复用，消除每候选 O(H·W) 重建
    # 默认相位采样为 4 相位 {0, p/4, p/2, 3p/4}（Task 6 加密，经子谐波/comb 回归
    # 验收无回归），1px 宽条带取最大强度；phases 参数可显式覆盖（如回退 {0, p/2}）
    # 返回：条带平均边缘强度
```

真实周期的边界是连续强边缘（块与块之间），而块内纹理是局部弱边缘——实测
真实周期边界强度为子谐波的 1.7~2.1 倍，这是子谐波修正主判据（>1.3 倍）的依据。

**谐波梳（`_spectral_comb_score`）**：

```python
def _spectral_comb_score(profile, period):
    # 在谐波频率 k/period（k=1..8）处取频谱幅值（线性插值到非整数 bin）
    total = Σ 谐波幅值
    denom = max(median(谱[1:]) * 8, 1e-9)   # 带内中位数基准
    return total / denom
```

网格线是脉冲序列（谐波丰富），而近正弦的块内纹理谐波衰减快——因此谐波梳能
区分二者，识别真实的网格周期。

### 3.6 find_phase 相位搜索

```python
def find_phase(gray, px, py, step=0.1):
    # X / Y 轴解耦，分别搜索
    # 对每个候选相位：取周期对齐的条带（积分图 O(1) 求条带均值）
    #   分数 = 条带均值的方差        # 相位对齐时条带间差异最大
    # conf_x = score / 总方差
    # conf  = min(1, (conf_x + conf_y) / 2)
    # 相位扫描：linspace(0, period, n=round(period/step), endpoint=False)
```

相位对齐是后续块提取正确采样的前提：相位偏移会导致块边界错位、相邻块颜色
混叠。

### 3.7 detect_squares + assign_grid_bfs 方块检测

**`detect_squares(edge_map, px, py)`**：在边缘强度图上检测可信的网格方块
（作为 BFS 扩展的种子）。

```python
# 积分图 I（边缘和）与 B（edge > 0.03 的计数）
# 搜索步长 step = max(1, int(min(px, py) * 0.3))
# 对每个候选方块计算四边：
#   平均边缘强度 avg、连贯性 coh
#   有界边：avg >= 0.06 且 coh >= 0.5
#   接受条件：>= 2 条相邻有界边
#   内部洁净：interior_avg <= boundary_avg * 0.5
#   得分 = 四边强度之和 + 有界边数*0.5 + 洁净*0.5
# 双重 NMS：
#   粗 NMS：按 cell_key 分箱，每箱保留最高分
#   贪心 NMS：3×3 邻域，overlap > 0.30 拒绝
# 亚像素精炼（enable_subpixel_refine）：
#   在 ±step 内搜索四边和最大，抛物线插值
```

**`assign_grid_bfs(squares, px, py, adjacency_tol=3.0)`**：从最优种子方块
开始 BFS 扩展，将检测到的方块组织成网格。

```python
# 种子选择：lexsort(距离, -主评分)，主评分 = 归一化分数 × 连通性
# BFS 四邻域扩展：
#   邻接容差 adjacency_tol = 3.0：
#     左右邻：|ddx - ±px| <= tol 且 |ddy| <= tol
#     上下邻：|ddy - ±py| <= tol 且 |ddx| <= tol
# 未连通方块按近似网格坐标回退放置
```

### 3.8 expand_grid_edge_guided 网格扩展与全局正则化

在已分配方块之间/之外，用边缘引导外推补齐整张网格的逐交点坐标。

```python
def expand_grid_edge_guided(edge_map, placed, bounds, px, py, edge_tol=3, ...):
    # BFS 从已放置方块向外扩
    # 每个新网格线位置：
    #   共享边界搜索：在 [中心 ± edge_tol] 内找 _best_col/_best_row
    #     （边缘强度峰值，阈值 0.03 × 边长，抛物线亚像素，偏移钳位 [-1,1]）
    #   无边缘时：用邻居位置 + px/py 外推（权重 = 1.0）
    #   权重 = 边缘峰强度（边缘引导的观测更可信）
    # 四角观测经 _weighted_median 聚合为 cell_ys / cell_xs
    # 无观测的交点用理想等距外推填充
    # 若 smooth_strength > 0：_fit_global_regularization 全局正则化
```

**`_fit_global_regularization`**（全局线性模型混合）：

```python
# 加权线性回归：pos = phase + idx * period
#   （固定周期，只拟合 phase）
# 离群剔除：间距偏差 > outlier_reject_ratio * period 的观测权重置 0
# 混合权重：w = min(1, counts/4) * (1 - smooth_strength)
#   smooth_strength=0 → 纯观测；1 → 完全用全局线性模型
# 结果钳位 [-period, limit + period]，单调性约束 min_spacing = 0.3 * period
```

### 3.9 逻辑分辨率、候选分辨率与低置信拒绝

**逻辑分辨率（`_count_blocks`）**：

```python
def _count_blocks(length, period, phase):
    n = round((length - phase) / period)
    # 越界容差回退：亚像素相位误差允许 2% 块宽的越界
    if n > 1 and phase + n * period > length + 0.02 * period:
        n -= 1
    return n
```

设计动机：早期实现用 `int()` 截断，相位偏移使 `(H-phase)/py = 7.997` 时被截成
7，错误丢失最后一行；`round()` 又可能让最后一根网格线越界 1px。round + 2%
越界容差回退兼顾两者。

**候选分辨率**：主选分辨率外，将 ±1 邻域分辨率加入候选，按权重
`0.9 / 0.8 / 0.85` 打分择优，防止单点误差。

**低置信拒绝**：`conf < 0.05` 且非 FFT 命中时抛 `ValueError`，提示输入可能
不是 AI 伪像素图。

### 3.10 改进动机与设计说明

本节的算法设计源于真实 AI 像素图的伪影特征，记录如下供后续维护参考。

**问题 1：网格数量翻倍（核心问题）**

AI 生成像素图的块内常有次级结构（纹理/渐变/JPEG DCT 伪影），其周期为真实块
周期 P 的 1/k，FFT/ACF 会误检到 P/k，导致逻辑分辨率翻 k 倍（实测 32px 块 +
16px 纹理被检出 8px，10×8 图像被识别成 40×31）。

修复组合：
- 新增**边界带边缘强度**判据（真实周期边界是连续强边缘，子谐波是局部弱边缘，
  比值实测 1.7~2.1 倍）；
- `_vote_period` 升级为四判据加权（ACF/FFT/BVR/edge），edge 权重最高 0.4；
- 子谐波修正扩展为 k=2..6 迭代，主判据 `e_ratio > 1.3`，辅以 ACF/BVR 严格
  上升支撑（防纯网格假修正）；
- 新增**谐波梳**（`_spectral_comb_score`）区分"网格线脉冲序列"与"近正弦纹理"，
  作为可选的 comb 预筛与投票判据（默认关闭）。

**问题 2：2D BVR 在像素格点阵上失效**

最近邻放大产生 2px 周期格点时，2D BVR 强烈偏向小周期（格点对齐时块内完全
均匀），导致 2x 放大图误检 4px。修复：BVR 权重 0.2→0.1（份额给 edge），且
BVR 只对廉价判据前 10 名候选计算。

**问题 3：逻辑分辨率截断丢行**

见 3.9 的 `_count_blocks` round + 越界回退。

**问题 4：像素格点阵回归与投票性能**

格点阵输入导致 ACF 峰候选激增（~20 个），每个候选都跑 2D BVR 相位扫描。
修复：廉价判据预筛 + BVR 限量计算（`VOTE_BVR_LIMIT=10`），nearest 放大全流程
耗时从 7.4s 降至 4.7s。

**问题 5：投票缺乏「整数放大尺度」的正交先验**

块尺寸的整数尺度（runs/GCD，见 3.11）与 FFT/ACF 投票相互独立，可作为交叉
验证先验：证据强时吸附/修正投票周期（大块场景的 runs>vote 采纳、防缩小的
runs<vote 默认不缩小），且无强证据时静默回退纯投票结果（零回归）。

**问题 6：1-2px 生成纹理被误当真实网格**

AI 生成的细小纹理周期过小（<3px），FFT/投票可能误检为真实网格。修复：
高分辨率合理性门控（见 3.11），周期过小时按整数倍链式检查边界带强度，
采纳显著更强的更大周期；阈值 1.3 与均匀 2px 格点阵的整数倍比值（≈1.0-1.15）
保持安全间隔，避免把真实小网格误放大。

### 3.11 runs/GCD 整数尺度检测与高分辨率合理性门控

**runs/GCD 整数尺度检测（`src/core/scale_detect.py::detect_integer_scale`）**

基于像素行程长度 GCD 的整数尺度快速检测，参考 pixel-art-downsampler /
unfake.js 的 runs 方法：沿采样行/列统计「相邻像素差值 ≤ tol 视为同色」的
行程长度——像素图中每块均匀色对应一个行程，行程长度为真实块尺寸的整数倍。
取频次高的行程长度做 GCD 得到整数放大尺度，作为 FFT/ACF 多判据投票的正交
交叉验证先验（网格检测的第三道防线，区别于 FFT 与投票）。

```python
def detect_integer_scale(gray, min_p=3, max_p=40, tol=12.0, sample_stride=4):
    # 每隔 sample_stride 行/列采样一条线，统计行程长度（相邻差值 > tol 处断开）
    x_runs = 所有采样行的行程长度;  y_runs = 所有采样列的行程长度
    # 对频次前 8 的行程长度做两两 GCD 得候选尺度（覆盖「行程 = k*scale」）
    # 对每个候选计算命中率（行程为尺度整数倍、容差内的占比），选命中率最高者；
    # 命中率相同时取更大尺度（真基频优于其子谐波，避免小尺度偏向）
    sx, hx = _best_scale_from_runs(x_runs, min_p, max_p, tol)
    sy, hy = _best_scale_from_runs(y_runs, min_p, max_p, tol)
    hit_rate = (hx + hy) / 2.0
    confidence = hit_rate * (1.0 if sx == sy else 0.7)   # x/y 一致性乘子
    if hit_rate < STRONG_HIT_RATE(0.7): return (0, 0, 0, 0)  # 无强证据
    return (sx, sy, hit_rate, confidence)
```

**集成规则（`detect` 内，`enable_runs_crosscheck` 默认开启）**：仅当
`runs_x > 0 and runs_y > 0 and runs_x == runs_y` 且命中率
`>= RUNS_STRONG_HIT_RATE(0.7)` 时，对两轴分别调 `_runs_correct_period`
（常量见附录 A.2）：

| 情形 | 条件 | 动作 |
| --- | --- | --- |
| 一致 | \|runs − vote\| / max(runs, vote) ≤ `RUNS_AGREE_REL`(0.2) | 周期吸附为整数 runs |
| runs > vote | runs ≈ k·vote（k≥2，容差 0.15）且 `edge(runs) ≥ 0.9·edge(vote)` | 投票误检子谐波，采纳 runs（真实大块） |
| runs < vote | `vote ≥ 3` 且 `edge(runs) > 1.3·edge(vote)` | 缩小周期；否则默认不缩小 |

runs < vote 默认不缩小的动机：块内纹理（如 28px 块 + 7px 纹理）场景下
runs=7 < vote=28，无显式边界强度证据时不允许把真实块缩成块内纹理周期。

**高分辨率合理性门控（`_plausibility_gate_axis`）**

候选周期过小（< `PLAUSIBILITY_MIN_PERIOD` = 3px，如 1-2px 生成纹理）时，
检查其整数倍 k·p（k=2..，直至超出 [min_p, max_p]，至多 13 倍）的边界带
边缘强度：

```python
def _plausibility_gate_axis(p, edge_map, axis, min_p, max_p):
    if p >= PLAUSIBILITY_MIN_PERIOD(3): return p      # 周期已合理不放大
    for _ in range(PLAUSIBILITY_MAX_STEPS(6)):        # 链式放大 2→4→8→…
        e_cur = _edge_band_strength(edge_map, cur, axis)
        if e_cur <= 1e-12: break
        for k in range(2, k_max):                     # 取第一个显著更强的倍周期
            kp = k * cur
            if kp < min_p or kp > max_p: continue
            if _edge_band_strength(edge_map, kp, axis) >= PLAUSIBILITY_EDGE_RATIO * e_cur:
                cur = kp; break
    return cur
```

`PLAUSIBILITY_EDGE_RATIO` = 1.3 的依据：真实 1-2px 格点阵的各整数倍边界带
比值 ≈1.0-1.15（相位采样总能命中边界线），而「块内 2px 纹理 + 大块网格」
场景大倍周期命中真实块边界，比值可达 1.3-1.6；阈值取 1.3 在两者之间留有
安全间隔（0.8 会把均匀 2px 格点阵误放大到 4——红线是真实 2px 网格必须仍
检出 2）。统一正方形块时两轴分别门控后取一致结果（gx ≥ gy 取 gx），
保持 px == py。

### 3.12 色差信号、周期精化与终审判据（G1/P2/G3/G2/G5）

本节为网格检测的增强判据层：G1 改换检测信号、P2 预处理检测信号、G3 精化
周期数值、G2 周期终审、G5 压缩伪影防护。除 G1/P2 外均默认开启
（`PipelineParams` 层；`detect()` 函数级默认保守关闭），低置信/失败场景
一律自动回退投票链结果，对真实图的扰动为零。

**G1：OKLAB 色差检测信号（`detect_signal="oklab"`，默认 "gray"）**

等亮度异色块（如纯红块与纯绿块灰度相同）边界在 BT.601 灰度上梯度 ≈ 0，
灰度路径的边缘图/投影信号/相位搜索全部不可见。`_oklab_signal` 改用 OKLAB
感知色差构造检测信号：投影信号对 OKLAB 每通道独立做局部对比度归一化
（弱色度边缘被有界放大）后取相邻像素差分的 L2 范数沿另一轴求和，与灰度
路径投影信号同形同义；边缘图为原始 OKLAB 前向差分的逐像素幅值。BVR、
runs 交叉验证、梯度回退等灰度判据仍基于 BT.601 灰度（语义不变）；2D 输入
恒走灰度路径。默认保持 "gray"：真实图上混合结果（个别图 conf 提升但
slice_09 发散），仅等亮度异色场景显式开启。

**P2：OKLab 预量化（`enable_pre_quantize`，默认关）**

`pipeline.py::_quantize_detection_signal` 对检测信号做小调色板量化（全图
1/16 子采样 → 样本转 OKLab 做 median-cut 聚类 → 全图 cKDTree 最近邻映射），
把 AA 过渡带阶跃化、块内渐变坍缩为零梯度，提升边界可检测性。仅改变
**检测信号**（不改变提取/输出信号）；gray 模式下实测 conf 下降
（image01 0.633→0.478），默认关闭，仅 oklab 模式配合使用。

**G3：峰值格点拟合周期精化（`enable_peak_lattice_fit=True`）**

投票/runs 输出的周期常为整数或粗估值（真实案例 7.45/7.48px 非整数）。
`_refine_period_peak_lattice` 参考 Pixel-Extractor 的峰值格点拟合范式：
投影信号按 prominence 找峰（阈值 0.05×max 起步，峰 <4 逐级放宽到
0.02/0.01），把峰视为格线采样，L-BFGS-B 最小化
`J(s) = sqrt(mean((spacings − round(spacings/s)·s)²))/s + α·missing_ratio`
（α=0.5，missing_ratio 为首末峰间缺失格线比例，惩罚子谐波）。接受准则
`J* < 0.15` 且 `|s* − initial_p|/initial_p ≤ 0.3`，否则回退投票值；
投票值与拟合值双重一致（偏移 ≤ 2%）且贴近整数（< 0.02·s*）时吸附整数，
保持干净整数网格位精确。

**轴一致性防护**：正方形网格两轴投影来自同一格点结构，精化成功时两轴
周期应几乎一致。若投票周期本就两轴一致（相对差 ≤
`PEAK_LATTICE_AXIS_ORIG_REL` = 5%）而精化后显著分裂（相对差 >
`PEAK_LATTICE_AXIS_GUARD_REL` = 2%，如 image02 的 x 回退 20.0 / y 精化
20.804，分裂 3.9%），判定单轴拟合锁错格点，两轴整体回退精化前投票值。
两轴同时精化失败（refined == orig）时回退为无操作；原本两轴就不一致的
非方格（如 slice_01 的 8/7）精化各自独立进行，不适用本防护。

**G2：梳状能量集中度终审（`enable_comb_energy_score=True`）**

对投影信号做连续 (pitch, phase) 梳状打分，原理性压制投票链的子谐波/倍频
误检。打分公式（`_comb_energy_score`）：

```python
# 梳齿位置 pos_k = round(phase + k·pitch)，每齿 1px 采样捕获能量
E = Σ profile[pos_k]
score = E / total_energy − n_teeth / n_positions
# 惩罚项 n_teeth/n 为随机取位的期望能量占比（score > 0 即高于随机基线）
```

真周期 P 命中全部边界能量且齿数最少；子谐波 P/2 以双倍齿数仅多捕获块内
噪声（惩罚翻倍必然更低分）、倍频 2P 漏一半边界（能量减半），数学上均
严格低于真周期——不依赖经验阈值。搜索流程（`_comb_best_period`）：σ=0.5
轻高斯平滑（容忍 AA 相位散布）→ pitch 2% 相对步长粗扫（每 pitch 分段
最优相位打分，对网格误差免疫）→ 前 3 个不同峰（8% 去重窗口）各自做
细 pitch 网格 × 全相位扫描 + 步长折半抛光 → 裁决：平局集（分数 ≥
best/1.15）中取最大 pitch（谐波家族最大竞争者即真值），整数吸附（round
后分数 ≥ 0.97×best）。

裁决规则：`confidence = quality × separation`（quality = 能量集中度
clamp(best_score, 0, 1)，separation = (best − 次优非同族分数)/best）。每轴
独立接受：`confidence ≥ COMB_ACCEPT_CONF`(0.35) 且 pitch > 0 时覆盖该轴
周期，低置信自动回退 G3/投票链结果（真实图全部低置信回退，零扰动；合成
硬边 7.5px 网格修复投票锁 30px 的对齐倍数误检：30.000/16×15 →
7.496/64×63）。梳相位仅作参考不透传 find_phase。两轴置信度均值记入
`Grid.comb_energy_conf` 供 A/B 分析。

**G5：JPEG 8×8 网格防护（`jpeg_grid_guard=True`）**

`detect_jpeg_grid`（参考 IPOL 2020 交叉差分法）：交叉差分
`C(x,y) = I(x+1,y+1) − I(x+1,y) − I(x,y+1) + I(x,y)` 抵消真实内容边缘
（一阶差分成对相减），仅保留量化阶跃在 JPEG 块角点的混合二阶差分响应。
对 64 个可能网格原点投票：只取强响应位置（|C| 超过 90 分位）投给
`(y mod 8, x mod 8)` bin，峰 bin 占比为 strength——JPEG 网格全部角点
mod 8 落入同一 bin，自然图像近似均匀铺满 64 bins。

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 显著性阈值 | `JPEG_GRID_STRENGTH_THRESHOLD` = 0.06 | 均匀期望 1/64≈0.0156 的约 3.8 倍；q=70 压缩实测 0.08-0.11（显著），PNG 对照 0.02-0.03（不显著） |
| 惩罚因子 | `JPEG_PENALTY_FACTOR` = 0.6 | 显著时对候选 c ∈ {7,8,9,15,16,17,23,24,25}（8/16/24 ±1）的 edge 分数乘 0.6（归一化前施加，实现候选间相对降权） |
| 作用范围 | 仅投票链 | G2 梳状终审/G3 峰值拟合为图像域独立判据，不受惩罚影响 |

检测结果 `(is_significant, phase, strength)` 记入 `Grid.jpeg_grid` 供
A/B 分析；真实图阈值 0.06 下均未触发（strength 0.017-0.020），零扰动。

**元信息（metadata）变化**：`Pipeline` 的 `metadata` 字典在原有
`w_logic/h_logic/px/py/grid_conf/unique_colors/extract_method` 基础上
新增 `low_confidence`（`conf < 0.4` 时为 True，供 GUI 提示人工确认）；
`comb_energy_conf` 与 `jpeg_grid` 通过 `Grid` 对象暴露（`result.grid`）。

**E1：提取均匀网格快速路径（自动触发）**

`extract.py::extract_blocks` 在均匀网格（`cell_ys/cell_xs` 为 None 的
phase+等距模型，或逐交点坐标等距）且 `method` 为 `median`/`mean` 时
走 `_extract_uniform_fast` 向量化快速路径：逐行带切片 + 按块宽分组的
`np.median/np.mean`（两算法结果仅由像素集合决定，与逐块循环逐位等价），
大图提速一个数量级；不均匀网格、`mode`/`kmeans`/`dominant` 方法或边界
退化时自动回退逐块路径（结果不变）。

**S1：长宽比守恒防护与两轴联合周期再搜索**

非整数块周期（如真实 9.5px）下两轴投票可能分别锁到不同的错误整数周期
（真实案例：两轴锁 10/8 → 输出 AR 畸变 20%，即转换"拉伸"）。防护链：
分开使用两轴周期后，若逻辑分辨率长宽比与原图相对差 >
`AR_GUARD_RATIO_DIFF`(0.12)，先做联合再搜索（`_ar_joint_research`）：
候选池含两轴投票值/FFT 主峰/ACF 峰/梳状 top-N 连续 pitch（`_comb_top_pitches`，
非整数可入池）及其 ×2/×0.5 扩展，组合评分以 AR 守恒（相对差 <
`AR_GUARD_ACCEPT_AR` 0.03）为主、两轴边界带强度（复用边缘积分图）为辅；
最优组合边界强度 ≥ 原组合 0.8 倍（`AR_GUARD_EDGE_FLOOR`）时采纳，否则
回退边界强度较高方向的正方形块保底。`detect()` 的 `aspect_guard_tol`
参数可覆盖触发阈值。实测 slice_06.jpg 畸变 20.2% → 0.4%（71×64），
其余 AR 守恒图零扰动。

**S2：检测尺寸上限与坐标映射（`detect_max_size`，默认 2048）**

大图放大路径的性能保护：`grid_detect` 输入图（resize 后）任一边超过
`detect_max_size` 时，检测改在放大前（denoise 后）原图上执行，完成后
`_map_grid_to_scale` 将 `px/py/phase_x/phase_y/cell_ys/cell_xs` 按缩放比
线性映射回放大坐标系（nearest 放大下精确 ×f，插值放大亚像素误差 <1px），
`w_logic/h_logic` 不变，extract 仍消费放大图与映射后网格。放大后 ≤上限
（小图）保持放大图上检测，保留"非整数周期 ×f 后更接近整数"的精度收益。
实测 image02.png（2048²）放大 2x：grid_detect 10.7s → 3.8s（检测在
2048² 原图执行），全流程 12.3s → 6.9s（≤10s 预算）。

---

## 4 块提取 extract

`extract.py::extract_blocks(img, grid, method="median", core_ratio=0.6)` 将每个
块压缩为 1 个代表色像素，直接输出逻辑分辨率的真像素图。`method` 支持
`median`/`mean`/`mode`/`kmeans`/`dominant`。

```python
def extract_blocks(img, grid, method="median", core_ratio=0.6):
    px, py = grid.px, grid.py
    use_cells = grid.cell_ys is not None and grid.cell_xs is not None

    # 核心区尺寸：块标称尺寸 × core_ratio（默认 0.6）
    cw = max(1, round(px * core_ratio));  ch = max(1, round(py * core_ratio))

    for j in range(h_logic):
        for i in range(w_logic):
            # 块边界：逐交点坐标（支持局部漂移）或 phase + 等距
            (by_start, by_end, bx_start, bx_end) = 块(j, i)

            # 边界去重（x 与 y 双向，防相邻块共享像素）：
            bx_start = max(bx_start, prev_bx_end)
            by_start = max(by_start, prev_by_end[i])

            # 核心区：块内居中，且不超出块边界
            core = img[cy_start:cy_end, cx_start:cx_end]

            # 代表色（method）：
            if method == "median":  rep = np.median(core)
            if method == "mean":    rep = np.mean(core)
            if method == "kmeans":  # 中位数估计主色，距离阈值分离渗透色
                center = np.median(core)
                dists  = |pixel - center|
                thr    = percentile(dists, 25)      # 25 分位
                rep    = mean(主色像素)               # 距离 <= thr 的像素均值
            if method == "mode":    # /32 量化后取众数 bin，再取 bin 内像素均值
                quantized = round(core / 32);  mode_q = argmax 众数
                rep = mean(quantized == mode_q 的像素)
            if method == "dominant":  # 感知空间分 bin 取众数，bin 内原始 RGB 均值
                space_px = rgb_to_oklab(core)       # 优先 OKLab（步长 0.02），
                                                    # 未就绪回退 Lab（步长 5）
                quantized = round(space_px / step);  mode_q = argmax 众数 bin
                rep = mean(space_px == mode_q 的原始 RGB 像素)

            out[j, i] = clip(rep, 0, 255)
```

要点：

- **核心区采样**规避块边缘杂色（抗锯齿、JPEG 伪影、相邻块渗透）；
- **x/y 双向去重叠**：`cell_ys` 逐交点局部漂移可能使相邻行/列重叠或反向，
  对称处理两个方向，保证相邻块不共享像素；
- **空块回填**：行方向用同行左邻值，否则用上一行同列值；
- **kmeans 方法**并非真正 K-means，而是"中位数主色 + 25 分位距离阈值"的轻量
  渗透色分离，速度远快于聚类；
- **dominant 方法**把核心区像素转到感知空间（优先 OKLab，回退 Lab），对
  小步长量化 bin 取众数，再取该 bin 内原始 RGB 像素均值作主色——比 mode 的
  RGB `/32` 量化更贴近人眼感知（Lab 通道步长 4-6 区间，OKLab 步长 0.02）；
- **均匀网格快速路径（E1，自动触发）**：均匀网格 + `median`/`mean` 时走
  `_extract_uniform_fast` 向量化路径（逐行带切片 + 分组聚合，与逐块循环
  逐位等价），详见 3.12「E1」段。

---

## 5 调色板精炼 color_quantize

`color_quantize.py::color_quantize(img, n_colors=16, space="rgb")` 用 K-means
将图像颜色量化为有限簇心，每个像素吸附到最近簇心。`space` 指定聚类颜色空间：
`"rgb"`（默认，与旧行为完全一致）/ `"lab"` / `"oklab"`；后两者将像素转到
对应感知空间做 KMeans（聚类更贴近人眼感知），簇心映射回 RGB 得到最终量化图。
渐变过渡区域自动变为硬边界（阶跃），且不会超出 [0,255]（无 unsharp mask 的
白边问题）。

```python
def color_quantize(img, n_colors=16, space="rgb"):
    pixels = img.reshape(-1, 3).astype(float64)

    if len(pixels) < n_colors: return img.copy()      # 像素太少

    # 聚合唯一色与计数（float 输入先取整——否则 0-255 色阶下浮点值几乎全唯一，
    # 聚合退化为全量 KMeans，B9 修复）
    unique_colors, inverse, counts = np.unique(round(pixels), return_inverse, return_counts)

    if len(unique_colors) <= n_colors: return img.copy()   # 无需聚类

    # 按 space 转聚类输入：rgb 直用唯一色；lab/oklab 转到感知空间（簇心需映射回 RGB）
    fit_colors = unique_colors if space == "rgb" else rgb_to_lab/oklab(unique_colors)

    # 对唯一色加权聚类（sample_weight = 频次，避免重复像素重复计算）
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3)
    kmeans.fit(fit_colors, sample_weight=counts)

    # 感知空间聚类的簇心映射回 RGB（lab_to_rgb / oklab_to_rgb）
    centers = lab_to_rgb/oklab_to_rgb(kmeans.cluster_centers_) if space != "rgb" else centers

    # 仅对唯一色 predict，再经 inverse 索引映射回所有像素
    all_labels = kmeans.predict(fit_colors)[inverse]
    quantized  = centers[all_labels].reshape(H, W, 3)

    # 还原 dtype：整型 clip+round 到 dtype 最大值
```

`src/core/color.py` 提供 `rgb_to_lab`/`lab_to_rgb`（基于 skimage）与
`rgb_to_oklab`/`oklab_to_rgb`（基于 Björn Ottosson 的 OKLab 公式，感知均匀性
优于 Lab），供 `space="lab"/"oklab"` 使用；`oklab_to_rgb` 对越出 sRGB 色域的
值裁剪到 [0,255]。

---

## 6 降噪与对比度增强 denoise

`denoise.py::denoise_ai_noise(img, method="nl_means", strength=0.5)` 收敛 AI
生成图像中的扩散色偏与 JPEG DCT 块效应。内部统一将输入归一化到 [0,1] 调用
skimage，再还原回 [0,255]。

| 方法 | skimage 函数 | 参数映射 | 特点 |
| --- | --- | --- | --- |
| `nl_means` | `denoise_nl_means` | `h=strength*0.03`，`patch_size=2`，`patch_distance=3` | 适合平滑区域的低频色偏与 DCT 伪影 |
| `tv_chambolle` | `denoise_tv_chambolle` | `weight=strength*0.1` | 去噪同时较好保留边缘 |
| `bilateral` | `denoise_bilateral` | `sigma_color=strength*0.1`，`sigma_spatial=2` | 平滑色偏的同时保留块边缘 |
| `none` / `strength<=0` | — | — | 直通返回副本 |

`apply_clahe(img, clip_limit=0.03)`：对去噪后图像做 CLAHE 局部对比度增强，
逐通道调用 `skimage.exposure.equalize_adapthist`（避免部分版本 `adapt_rgb`
转发 `channel_axis` 引发的 TypeError），提升网格检测对低对比度区域的识别。

**抗锯齿消除（`remove_anti_aliasing`）**

`aa_removal.py::remove_anti_aliasing(img_rgb, threshold=0.5, passes=2)` 在
`denoise_global` 阶段内（`enable_aa_removal=True` 时）清理 AI 伪像素图块
边界/网格线上的 1px 线性混合（抗锯齿）杂色，默认关闭。算法在 OKLAB 感知
空间用「三角形不等式」判定当前像素颜色是否位于邻域两主色之间，若是则吸附
到较近的主色，把混合色清理为纯块色：

```python
def remove_anti_aliasing(img_rgb, threshold=0.5, passes=2):
    oklab = _rgb_to_oklab(img_rgb)                  # 模块内自带 OKLAB 转换
    for _ in range(max(1, passes)):                 # 重复迭代收敛宽 AA 过渡带
        # 1) 8 邻域颜色按 RGB 每通道 32 级粗量化归为类别
        # 2) 取 8 邻域内频次最高的前两类作主色 a、b（含排除中心色的回退判定）
        # 3) 仅当 d(a,b) > threshold（两主色足够不同）才继续，避免破坏细节
        # 4) 三角形不等式：|d(p,a)+d(p,b)-d(a,b)| / d(a,b) 小于容差
        #    → p 位于 a-b 之间（AA 混合色）；主路径容差 _TRIANGLE_TOL=0.4，
        #      交叉 AA 回退路径用更严格的 _STRICT_TOL=0.05（防纯色角点误判）；
        #    且要求 p 与 a、b 均明显不同（真混合色而非纯块色）
        # 5) 吸附到更近的主色 a/b（平局取 a）
    return clip(arr, 0, 255)
```

参考 pixfix（lovelaced/pixfix）的 OKLAB 三角形不等式 AA 消除思路；模块内
自包含（自带约 15 行向量化 OKLAB 转换），不依赖 `color.py`。

---

## 7 调整大小 resize（放大）

调整大小逻辑内联于 `pipeline.py::_run_resize`（无独立模块），执行顺序固定为：

1. **放大**（若 `enable_upscale=True`）：提升网格检测分辨率；
2. **可选锐化**（若 `enable_sharpen=True`）：unsharp mask 提锐放大结果。

```python
def _run_resize(img):
    # 1) 先放大（默认关闭）
    if enable_upscale and upscale_factor > 1:
        if method == "nearest":
            # np.repeat 无插值：逐行逐列重复（像素艺术保持锐利）
            img = np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
        else:
            # skimage.transform.resize，order 映射：
            #   nearest=0, bilinear=1, bicubic=3, lanczos=5
            # anti_aliasing=False（保持锐边）、preserve_range=True
            img = resize(img, (H*factor, W*factor), order=order_map[method],
                         anti_aliasing=False, preserve_range=True)
    # 2) 可选锐化（unsharp mask），仍在放大后
    if enable_sharpen:
        img = _unsharp_mask(img, strength)
    return img
```

resize 阶段仅保留放大语义（缩小/块降采样功能已整体移除，含
`downsample_blocks` 及 `enable_downscale`/`downscale_factor`/`downscale_method`
参数）。`grid_detect`/`extract` 均在放大后图像的坐标系上工作，与网格坐标一致；
`min_p`/`max_p` 无需做任何折算。

**Unsharp Mask（`_unsharp_mask`）**：

```python
# 分通道：arr + strength * (arr - gaussian_filter(arr, sigma=radius, mode="reflect"))
# 3D 逐通道，结果 clip [0, 255]
```

默认关闭（`enable_sharpen=False`）：unsharp 放大高频，高强度时产生白边
（截断失真）——这是它与 K-means 量化的本质区别（量化是替换操作，无白边）。

---

## 8 输入输出与图像工具 io / utils

### 8.1 图像加载（load_image）

- `ImageOps.exif_transpose`：应用 EXIF 方向（手机/相机竖拍自动转正）；
- `convert("RGB")` → `normalize_image`：统一 `(H, W, 3)` float64 RGB 0-255；
- 文件不存在 / 无法解码：抛带路径的友好 `ValueError`（非 PIL 原始异常）。

### 8.2 图像保存（save_image）

- `clip → uint8` 后 `Image.fromarray(..., mode="RGB")`；
- `scale > 1` 时用 `Image.NEAREST` 最近邻放大（保持像素艺术锐利边缘）。

### 8.3 灰度转换（to_gray）

```python
# ITU-R BT.601 亮度系数
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### 8.4 图像归一化（utils.normalize_image）

- PIL Image → RGB float64；
- RGBA 输入丢弃 alpha 通道；
- float 输入且最大值 ≤ 1 时 ×255 放大到 0-255 色阶。

---

## 附录 A 全参数与阈值速查表

### A.1 流水线参数（PipelineParams）

| 参数 | 默认值 | 阶段 | 说明 |
| --- | --- | --- | --- |
| `enable_ai_denoise` | True | 降噪 | 启用图像级去噪 |
| `ai_denoise_method` | "nl_means" | 降噪 | none/nl_means/tv_chambolle/bilateral |
| `ai_denoise_strength` | 0.5 | 降噪 | 去噪强度 0-1 |
| `enable_clahe` | False | 降噪 | CLAHE 局部对比度增强 |
| `clahe_clip_limit` | 0.03 | 降噪 | CLAHE 裁剪限制 0.01-0.1 |
| `enable_aa_removal` | False | 降噪 | 抗锯齿消除预处理（OKLAB 三角形不等式吸附，默认关） |
| `aa_removal_passes` | 2 | 降噪 | AA 消除迭代次数 |
| `aa_removal_threshold` | 0.5 | 降噪 | AA 两主色距离阈值（低于不做吸附） |
| `denoise_grid_guard` | False | 降噪 | 去噪过强时自动减半强度重去噪一次（默认关） |
| `enable_upscale` | False | 调整大小 | 默认不放大（最近邻放大引入格点伪周期） |
| `upscale_factor` | 2 | 调整大小 | 放大倍数 |
| `upscale_method` | "nearest" | 调整大小 | nearest/bilinear/bicubic/lanczos |
| `enable_sharpen` | False | 调整大小 | unsharp mask 锐化（有白边风险，默认关） |
| `sharpen_strength` | 0.5 | 调整大小 | 锐化强度 0-1 |
| `min_p` | 3 | 网格检测 | 最小候选周期（像素） |
| `max_p` | 40 | 网格检测 | 最大候选周期（像素） |
| `user_hint` | None | 网格检测 | 用户逻辑分辨率提示 (w,h) |
| `phase_step` | 0.1 | 网格检测 | 相位扫描步长 |
| `snr_threshold` | 8.0 | 网格检测 | FFT 网格判定 SNR 阈值 |
| `edge_search_tolerance` | 3 | 网格检测 | 共享边界搜索半径（px） |
| `enable_subpixel_refine` | True | 网格检测 | 亚像素精炼 |
| `smooth_strength` | 0.5 | 网格检测 | 全局正则化混合强度（0=纯观测，1=纯模型） |
| `outlier_reject_ratio` | 0.5 | 网格检测 | 离群间距剔除阈值比例 |
| `detect_signal` | "gray" | 网格检测 | 检测信号模式 gray/oklab（G1；oklab 供等亮度异色块，默认 gray） |
| `detect_max_size` | 2048 | 网格检测 | 检测输入图最大边长：放大后超限则检测回退原图并映射坐标（S2） |
| `enable_pre_quantize` | False | 网格检测 | 检测前对检测信号做小调色板预量化（P2，默认关） |
| `enable_peak_lattice_fit` | True | 网格检测 | 峰值格点拟合周期精化（G3，含轴一致性防护，默认开） |
| `enable_comb_energy_score` | True | 网格检测 | 梳状能量集中度周期终审（G2，低置信自动回退，默认开） |
| `jpeg_grid_guard` | True | 网格检测 | JPEG 8×8 网格检测与候选降权（G5，默认开） |
| `extract_method` | "median" | 提取 | median/mean/mode/kmeans/dominant |
| `extract_core_ratio` | 0.6 | 提取 | 核心区采样比例 0.5-1.0 |
| `fix_square` | False | 提取 | 逻辑分辨率与正方形差 1 时自动修正 |
| `enable_palette_refine` | True | 精炼 | 启用 K-means 调色板精炼 |
| `palette_colors` | 16 | 精炼 | 调色板目标色数 |

### A.2 网格检测内部阈值/常量（grid_detect.py）

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `VOTE_BVR_LIMIT` | 10 | 2D BVR 只对廉价判据前 10 名候选计算 |
| 新路径投票权重 | ACF 0.3 / FFT 0.2 / BVR 0.1 / edge 0.4 | 周期总评分 |
| 旧路径投票权重 | ACF 0.4 / FFT 0.3 / BVR 0.3 | 无 edge_map 时保持旧行为 |
| 子谐波修正主判据 | e_ratio > 1.3 | 边界强度比 |
| comb 预筛 | comb(kP) > 1.2 × comb(P) | 修正前 comb 一致性 |
| 投票覆盖 FFT | conf > 0.3 或 e_vote > 1.2 × e_fft | 投票结果优先级 |
| 方向保护 | 小周期 edge ≥ 0.7 × 大周期 edge | ACF 纠正限制 |
| `_count_blocks` 越界容差 | 0.02 × period | round 后越界回退 |
| 候选分辨率权重 | 0.9 / 0.8 / 0.85 | 主选 + ±1 邻域 |
| 低置信拒绝 | conf < 0.05 且无 FFT | 抛 ValueError |
| 边缘峰阈值（方块） | 0.03 × 边长 | 有界边判定下限 |
| 方块有界边 | avg ≥ 0.06 且 coh ≥ 0.5 | 接受条件 |
| BFS 邻接容差 | adjacency_tol = 3.0 | 方块网格连接 |
| 全局正则单调间距 | 0.3 × period | 网格线最小间距 |
| 谐波梳 | k = 1..8，基准中位数×8 | 周期判别 |
| 2D BVR 相位步长 | max(1, period // 4) | 相位扫描 |
| `RUNS_STRONG_HIT_RATE` | 0.7 | runs 视为强证据的命中率下限（grid_detect 集成口径） |
| `RUNS_AGREE_REL` | 0.2 | runs 与 vote 一致的相对差阈值 |
| `RUNS_ADOPT_EDGE_RATIO` | 0.9 | runs > vote 采纳 runs 所需边界强度比 |
| `RUNS_SHRINK_EDGE_RATIO` | 1.3 | runs < vote 缩小周期所需显式边界强度比 |
| `PLAUSIBILITY_MIN_PERIOD` | 3 | 触发高分辨率门控的最小周期（1-2px 生成纹理） |
| `PLAUSIBILITY_EDGE_RATIO` | 1.3 | 采纳更大倍周期的边界带强度比 |
| `PLAUSIBILITY_MAX_STEPS` | 6 | 链式放大的最大步数（2→4→8→…） |
| runs 同色容差 | tol = 12.0 | `detect_integer_scale` 判定同色的像素差值（灰度级） |
| runs 采样间隔 | sample_stride = 4 | `detect_integer_scale` 每隔该行/列采样一条线 |
| `PEAK_LATTICE_MIN_PEAKS` / `MAX_FITS` | 4 / 20 | G3 拟合最少峰数 / k 扫描 minimize 调用数上限 |
| `PEAK_LATTICE_MAX_J` / `MAX_REL` | 0.15 / 0.3 | G3 接受精化的目标值上限 / 相对投票值偏移上限 |
| `PEAK_LATTICE_AXIS_GUARD_REL` | 0.02 | G3 轴一致性防护：精化后两轴相对分裂超过该值触发回退 |
| `PEAK_LATTICE_AXIS_ORIG_REL` | 0.05 | G3 轴一致性防护：精化前两轴视为一致的相对差上限 |
| `COMB_ACCEPT_CONF` | 0.35 | G2 终审采纳置信度阈值（低于回退投票链结果） |
| G2 粗扫 pitch 步长 | COMB_COARSE_REL_STEP = 0.02 | 相对步长 ~2%（[3,40] 约 131 个候选） |
| G2 精化候选 | COMB_REFINE_TOP_K = 3，去重窗口 0.08 | 粗扫分数前 3 个不同峰做细网格搜索 |
| `JPEG_GRID_STRENGTH_THRESHOLD` | 0.06 | G5 JPEG 网格显著性阈值（峰 bin 占比，均匀期望 ≈0.0156） |
| `JPEG_PENALTY_FACTOR` | 0.6 | G5 对 8/16/24 ±1 候选的 edge 分数降权因子 |
| `JPEG_PENALTY_PERIODS` | {7,8,9,15,16,17,23,24,25} | G5 惩罚的候选周期集合（8/16/24 ±1） |
| `JPEG_GRID_QUANTILE` | 90.0 | G5 交叉差分响应参与投票的分位（仅强响应投票） |

---

## 附录 B 术语表

| 术语 | 定义 |
| --- | --- |
| 像素图 / 像素画 | 由规则像素块构成的图像，每个块为单一颜色 |
| AI 生成像素图 | 扩散模型等生成的"伪像素图"：块内常有纹理、渐变、JPEG 伪影 |
| 规范像素图 | 每块纯色、边界锐利的标准像素艺术 |
| 逻辑分辨率 | 像素块数量（w_logic × h_logic），即输出图分辨率 |
| 块周期 px/py | 像素块在 x/y 方向的像素尺寸 |
| 子谐波 | 真实周期的整数分之一，导致误检的元凶 |
| 边缘强度图 | Sobel + 对比度归一化的 [0,1] 边缘响应 |
| 谐波梳（Spectral Comb） | 在谐波频率取谱幅值和的周期判别指标 |
| BVR | 块方差对比度（块间方差 / 块内方差） |
| 相位 | 网格在图像中的偏移起点 |
| 增量重跑 | 参数变更后从受影响阶段起重算 |
| 像素格点阵 | 最近邻放大产生的像素重复结构（2px 周期格点） |
| 完美像素 | 编辑器中消除 2px 宽对角伪影的绘制算法（Aseprite 同款） |




