# Tasks
- [x] Task 1: 核心颜色替换函数 `remap_color`
  - [x] 1.1 在 `src/core` 新增（或在 `color.py` 扩展）`remap_color(img, src_rgb, dst_rgb, tol)`：将匹配像素（欧氏/逐通道容差内）替换为目标色
  - [x] 1.2 单元测试：精确替换、容差替换、目标色与源色相同不变、退化输入
- [x] Task 2: 块提取收敛为 dominant 默认、移除 mode/mean
  - [x] 2.1 `src/pipeline.py`：`extract_method` 默认 `"kmeans"` → `"dominant"`
  - [x] 2.2 `src/core/extract.py`：允许集改为 `median/kmeans/dominant`；移除 `mean`/`mode` 分支（含 `_extract_blocks_loop` 与 `_extract_uniform_fast` 的 mean 路径）
  - [x] 2.3 `src/cli.py`：`--extract-method` choices 收敛、默认 `dominant`；help 更新
  - [x] 2.4 `src/gui/param_panel.py` `_ExtractPage`：下拉移除"均值/众数"，`dominant` 置默认
  - [x] 2.5 更新 `tests/test_extract.py`、`tests/test_pipeline.py`、`tests/test_cli.py` 中 mode/mean 用例与默认断言
- [x] Task 3: 调色板精炼页新增色卡与全局换色
  - [x] 3.1 读取精炼后结果唯一色，生成色卡（颜色样本 + 像素数），上限 `palette_colors`
  - [x] 3.2 勾选"启用统一全局色彩"时显示色卡，取消时隐藏
  - [x] 3.3 点击色卡 + 选择目标色（QColorDialog）→ `remap_color` 应用到结果图并刷新预览、emit 参数/结果变更
  - [x] 3.4 GUI 测试：勾选显示色卡、换色后像素被替换
- [x] Task 4: 像素编辑器新增"统一改变颜色"
  - [x] 4.1 编辑器增加一个动作/工具（菜单或工具栏"统一改色"），以当前前景色为目标色、以吸管/选取源色
  - [x] 4.2 调用统一 `remap_color` 作用于画布 `_image`，压一次撤销栈
  - [x] 4.3 GUI 测试：触发改色后画布中源色像素变为目标色并可撤销

# Task Dependencies
- Task 1 → 无（独立工具函数，先行）
- Task 2 → 依赖 1（无强依赖，可并行；但共享 extract 语义）
- Task 3 → 依赖 1（需 remap_color）
- Task 4 → 依赖 1（需 remap_color）
- Task 1/2 可并行；Task 3、Task 4 依赖 Task 1