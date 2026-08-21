# Checklist
- [x] `remap_color` 实现正确：精确/容差替换、同色不变、退化输入安全（tests/test_color_remap.py 6 用例通过）
- [x] `PipelineParams().extract_method == "dominant"`（实测 console 输出 dominant）
- [x] extract 允许集仅含 `median/kmeans/dominant`，`mode`/`mean` 分支已移除且传入时回退 median（grep 确认无 method=="mean"/"mode"）
- [x] CLI `--extract-method` 与 GUI 下拉仅暴露 `dominant/kmeans/median`，默认 `dominant`（help 与 combo 实测确认）
- [x] 调色板精炼页：勾选显示色卡、取消隐藏色卡（test_palette_page_swatches_show_on_enable 通过）
- [x] 调色板精炼页：点击色卡选择目标色后，结果图全局同色像素被替换并刷新预览（test_palette_page_remap_emits 通过；main_window 接入 _on_color_remapped）
- [x] 像素编辑器提供"统一改变颜色"入口（_build_color_panel 新增 replace_btn）
- [x] 编辑器统一改色后画布同色像素替换为目标色，且可撤销（test_pixel_editor_replace_color_undo / cancel 通过）
- [x] 全部相关测试通过：完整套件 492 通过（仅 2 个与本次无关的既有离屏 pixel_editor 失败，改动前即存在）