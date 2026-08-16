# 网格检测精度与准确度升级方案

> 调研来源：GitHub 开源项目 + 学术论文 + OpenCV 官方文档（2026-08 检索）
> 关联：上一轮已修复「网格数量翻倍」（边界带边缘强度判据 + 子谐波 k=2..6 修正）；本方案在此基础上做**主动式**精度升级。

## 1. 目标

1. **周期准确**：主动选择基频，彻底取代「FFT 主峰 + 事后修正」的被动方案，根治子谐波误检（P→P/k，网格翻倍）。
2. **几何精度**：网格线/交点定位达到亚像素（<0.15px），支持轻微畸变与非等距网格。
3. **向后兼容**：Grid API 不变、现有 91 项测试不回归、无 OpenCV 时自动降级。

## 2. 调研结论（外部方法 → 本项目落地点）

| 外部方法 | 来源 | 核心思路 | 落地点 |
|---|---|---|---|
| Spectral Comb 谐波梳状搜索 | Signal Processing, 1999 | 基频 f0 使所有谐波 k·f0 能量之和最大；子谐波假峰缺对应谐波自动压分 | 周期估计升级（核心） |
| Autoperiod | Vlachos, 2005 | FFT 候选峰 + ACF 逐峰验证 + 梯度上升细化 | 融入 comb 流程 |
| GPA 几何相位分析 | J. Micromech. Microeng. 19(1):015012 | 谱带通 + IFFT 得亚像素相位场 | P vs P/k 判别 + 亚像素线定位 |
| OpenCV cornerSubPix | OpenCV 官方文档 | 角点响应 + 梯度加权最小二乘亚像素精化（~0.01px） | 网格交点精化（明暗交替图） |
| 全局网格外推 | arXiv:2104.14963 | Sobel 梯度沿网格线积分 + 全局尺度搜索 | 仿射外推填充低置信区 |
| 抛物线插值/矩法亚像素 | Tabatabai 灰矩 / Zernike 矩 | 抗锯齿过渡亚像素边缘（0.1px 级） | 与 GPA 混合的 hybrid 模式 |
| FFT 十字谱 + Hough | Marin, 2015 | 谱中十字亮线求两轴周期+旋转角 | 远期（需 OpenCV Hough） |
| U-Net 网格线分割 | arXiv:2012.08641 | 深度学习语义分割，最鲁棒 | 远期（需标注+GPU） |

## 3. 方案架构

### 3.1 Spectral Comb 主动选基频（Step 1）
- `_spectral_comb_score(profile, period, n_harmonics=6)`：对候选周期 p，累加 FFT 谱在 k·f0（k=1..min(6, max_p//p)）窄带峰值能量。真实周期谐波丰富得分高；子谐波 P/k 的谐波位置落在谱低能量区自动压分。
- `_comb_candidate_periods(profile, min_p, max_p)`：comb 得分 Top5。
- 融合：comb 作为第 5 判据并入 `_vote_period`，且**前置过滤**——comb Top1/Top2 ≥1.5 倍直接采纳；否则加权投票（ACF 0.25/FFT 0.15/BVR 0.1/边界带 0.3/comb 0.2）。
- 回滚开关：`use_comb_prefilter=False` 恢复旧投票。

### 3.2 GPA 相位场（Step 2）
- `_gpa_phase_field(sig, period, band_halfwidth=0.15)`：1D 投影谱在 f0=1/p 窄带通 + IFFT 得复场，相位 φ=atan2(Im,Re)。
- **P vs P/k 判别**：`_phase_field_smoothness(phase)`（解卷绕后相邻相位差的中位绝对偏差）——真实周期相位场平滑、子谐波剧烈跳变；comb 比值 <1.5 时以此定夺。
- **亚像素线定位**：`_gpa_line_positions(phase, period)` 由相位斜率得线位，与抛物线插值按 `subpixel_method="hybrid"` 混合。
- 回滚开关：`subpixel_method="parabola"`。

### 3.3 OpenCV cornerSubPix（Step 3）
- `_try_corner_subpix(gray, cell_ys, cell_xs)`：以边缘吸附交点为初值调 `cv2.cornerSubPix`（winSize=5, criteria=(EPS+COUNT,30,0.01)），角点响应均值门控，输出 `corner_conf`。
- 降级：`try: import cv2`；无 cv2 或非棋盘图返回 None。

### 3.4 仿射外推（Step 3）
- `_fit_affine_grid(cell_coords, counts, ...)`：加权仿射拟合 (i,j)→(x,y)，在全局正则化后对低置信/无观测区外推填充（交点覆盖率 >60% 门控）。
- 回滚开关：`enable_affine_extrapolation=False`。

### 3.5 Grid 扩展（向后兼容）
- 新增可选字段：`comb_score: float = 0.0`、`gpa_phase_x/gpa_phase_y: np.ndarray | None = None`、`corner_conf: float = 0.0`。
- `cell_ys/cell_xs` 语义不变；`extract.py` 零改动。

## 4. 预期收益

| 指标 | 现状 | 目标 |
|---|---|---|
| 块内纹理子谐波误检（网格翻倍） | 事后修正兜底（k=2..6） | 主动选基频，k 不限 |
| 网格线定位误差 | 整数级 + 抛物线插值 | <0.15px（GPA hybrid） |
| 明暗交替图交点精度 | 边缘吸附 | <0.1px（cornerSubPix） |
| 轻微畸变/非等距网格 | 逐交点边缘吸附 | + 仿射全局外推 |

## 5. 风险与回滚

| 方案 | 风险 | 验证/回滚 |
|---|---|---|
| Spectral Comb | 谐波阶数选择不当、大周期谐波数不足、短图谱分辨率低 | `use_comb_prefilter=False` |
| GPA | p<4px 混叠、相位卷绕噪声 | `subpixel_method="parabola"` |
| cornerSubPix | 非棋盘明暗交替图退化、低对比度 | 角点响应门控 + 无 cv2 自动跳过 |
| 仿射外推 | 桶形/极端畸变外推失真 | 覆盖率门控 + 开关可关 |
| OpenCV 依赖 | 新依赖体积 | try-import 降级，核心路径纯 numpy |

## 6. 工程实测结论（2026-08 落地后）

**本轮落地结果**（全部验证通过，默认行为零回归）：

| 方案 | 合成/规整网格 | 真实 AI 图 | 默认值 |
|---|---|---|---|
| Spectral Comb（第 5 判据 + 双一致门控） | ✅ 有效（k=2..6 场景全修复） | ⚠️ **有害**（块内纹理主导，真实图边界强度区分度仅 1.08，comb 将 px 从 16 拉到 4） | **关闭** |
| 迭代子谐波修正（edge >1.3 阈值） | ✅ 有效（4→12→24 多级修正） | ✅ 无回归 | 启用（上轮已有） |
| GPA 相位场（平滑度/线定位） | ✅ 分析原语可用（平滑度判别 P vs P/k 有效） | ⚠️ cell 定位与现有边缘引导+正则化同源，无增量；模糊区判据会误判 | 函数保留，不接入主链 |
| cornerSubPix 交点精化 | ✅ 有效（抗锯齿棋盘 <0.01px） | ⚠️ 理想二值图收敛到半像素（无亚像素信息）；真实图有漂移风险 | **关闭** |
| 仿射外推（RANSAC 离群替换） | ✅ 有效（离群点被模型替换） | ⚠️ 真实图无显著增益 | **关闭** |

**关键结论**：
1. **comb 从设计上偏好小周期**（谐波能量更集中），必须靠边界强度门控抵消；真实 AI 图边界强度区分度不足（1.08 vs 合成图 1.7-2.1），门控失效 → 默认关闭是正确决策。
2. **真实图放大路径存在隐藏翻倍**（块 30px 被检测 16px）：diag 达标标准（间距均匀）未检查 px 与真实值，掩盖了该问题；需要后续以「已知逻辑分辨率的标注图」做精度回归。
3. cornerSubPix 收敛目标 = 图像梯度最优（边缘过渡中心），对理想二值图收敛到半像素是**正确的**（坐标约定差异）；对低对比度真实图有漂移风险，故门控 + 默认关闭。
4. GPA 的实质价值在**相位平滑度判别**（0.005 vs 0.043 区分度明确），已作为分析原语保留。

## 7. 参考来源

- Spectral Comb: ScienceDirect S0165168499000900（Signal Processing 1999）
- Autoperiod: Vlachos 2005, rdrr.io/cran/fdars/man/autoperiod.html
- GPA: IOP J.Micromech.Microeng. 19(1):015012（2009）
- cornerSubPix: OpenCV 相机标定教程（docs.opencv.org）
- 全局网格外推: arXiv:2104.14963（国际象棋棋盘网格估计）
- FFT 十字谱 + Hough: Marin 2015（Malassez 计数板）
- 投影 Hough 双线束: Hansard 2014, arXiv:1401.6393
- 相位一致性边缘: Kovesi 1999
- U-Net 网格线分割: arXiv:2012.08641
- 反像素化: Kopf & Lischinski 2011, SIGGRAPH（Depixelizing Pixel Art）
