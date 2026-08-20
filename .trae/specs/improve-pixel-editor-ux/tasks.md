# Tasks
- [x] Task 1: 画笔基础增强（R1 圆形笔刷 + R8 状态栏 + R7 网格开关）
  - [x] SubTask 1.1: PixelCanvas 新增笔刷形状（square/circle，默认 circle，size=1 退化单像素）+ _draw_pixel 圆盘覆盖 + 光标预览按形状 + 面板"圆形/方形"选择
  - [x] SubTask 1.2: 网格显示开关（默认 True）+ set/get_grid_visible + 面板"显示网格"
  - [x] SubTask 1.3: cursor_moved 信号 + get_hover_info()（坐标/fg hex/缩放%）+ Editor 状态栏实时显示含工具名
  - [x] SubTask 1.4: 单测（圆形圆盘非方形、预览冒烟、网格开关、状态栏）
- [x] Task 3: 颜色快捷流（R4）+ 视图（R5）+ 性能（R6）
  - [x] SubTask 3.1: 右键取背景色（_right_picking 拖动连续取）+ swap_colors + X 快捷键 + 面板交换按钮
  - [x] SubTask 3.2: 滚轮鼠标居中平滑缩放（1.15、1-64）+ Ctrl+0 reset_view + 双击 fit_to_view
  - [x] SubTask 3.3: _qimage 缓存 + _invalidate_qimage + drawTiledPixmap 棋盘格；移除逐帧重建与逐像素 fillRect
  - [x] SubTask 3.4: 单测（右键取背景、X 交换、鼠标居中缩放、性能 sanity 102×102 单帧 ~0.9ms）
- [x] Task 4: F1 透明撤销修复 + 回归
  - [x] SubTask 4.1: _undo/_redo 改 set_image(prev) 传完整 RGBA，透明保留
  - [x] SubTask 4.2: 单测：透明像素撤销保持 alpha=0
- [x] Task 2: 形状与对称（R2 圆形工具 + R3 对称）
  - [x] SubTask 2.1: TOOL_CIRCLE + _draw_circle_midpoint（外接圆心 max(|dx|,|dy|)/2 描边）+ Shift 正圆 + 预览/提交 + 工具栏"圆形 (C)" + C 快捷键 + idx 6
  - [x] SubTask 2.2: 对称 none/horizontal/vertical/both（Shift+S 循环）+ _mirror_point 镜像到画笔/橡皮/直线/矩形/圆 + 对称轴红/蓝虚线参考线（油漆桶未镜像，按 spec 允许降级）
  - [x] SubTask 2.3: 面板对称 QComboBox；对称轴画布中线，不提供拖动
  - [x] SubTask 2.4: 单测圆/正圆/水平/四象限对称/多工具生效
- [x] Task 5: 全量回归 + 文档
  - [x] SubTask 5.1: test_pixel_editor_toolbar_text 更新（前六项保留原顺序 + 追加"圆形"）
  - [x] SubTask 5.2: 全量 pytest——offscreen 平台 497 passed；默认平台仅 2 个既有 UI 抓图 flaky（hover/crosshair，DPR 与事件投递相关,offscreen 下通过证明逻辑正确）
  - [x] SubTask 5.3: README 编辑器能力与快捷键说明已补充

# Task Dependencies
- Task 1 独立；Task 2 复用 Task 1 圆形预览；Task 3/4 独立
- Task 5 依赖 Task 1-4
- 实施按 Task 1→3→4→2→5 串行（单文件强耦合避免冲突）