# Checklist

- [x] 圆形笔刷形状实现：点按/拖动为近似圆形填充；圆形预览与方形可选并可切换（默认圆形）
- [x] 网格显示开关：默认开启，关闭后不绘制像素网格线（对称轴除外）
- [x] 状态栏实时显示：坐标(x,y)、前景色 hex、缩放百分比、当前工具
- [x] 圆形工具（C 键）：中点画圆描边、Shift 锁正圆、预览/提交/并入撤销栈；工具栏追加"圆形"按钮且前六项顺序不变
- [x] 对称工具（Shift+S 循环 none/horizontal/vertical/both）：轴参考线显示；画笔/橡皮/直线/矩形/圆/填充绘制均镜像
- [x] 右键取背景色（含按住拖动连续取）；X 交换前景/背景色 + 面板交换按钮
- [x] 滚轮以鼠标居中平滑缩放（系数 1.15、1-64）、Ctrl+0 重置、双击 fit
- [x] 绘制性能：画布 QImage 缓存（去除每帧 tobytes+copy）、棋盘格 drawTiledPixmap；102×102 单帧绘制不卡顿（性能 sanity 记录）
- [x] 透明撤销：撤销/重做恢复完整 RGBA，透明像素保持 alpha=0
- [x] 既有 API 向后兼容：工具枚举、set_image/get_image、_draw_line_bresenham、_flood_fill、Alt 吸管、Shift 临时直线、完美像素行为、快捷键 B/E/I/G/L/U 均不变
- [x] tests/test_pixel_editor.py 新增用例全过；全量 pytest 除既有 UI 类 flaky（hover/crosshair）零新增失败
- [x] 文档（README）补充编辑器工具/快捷键/对称/交换色说明