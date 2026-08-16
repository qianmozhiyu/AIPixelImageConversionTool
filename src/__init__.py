"""AI 像素图转换工具包。

在包导入时设置 ``OMP_NUM_THREADS=1``，避免 sklearn KMeans 首次调用时
OpenMP 线程池初始化的 ~3s 开销。像素图数据集很小（通常 <10000 像素），
单线程 KMeans 反而更快。用户可通过显式设置环境变量覆盖此默认值。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
