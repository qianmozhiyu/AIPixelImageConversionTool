"""管线异步执行 worker。

在独立 QThread 中运行 :class:`~src.pipeline.Pipeline`，通过 Qt 信号将进度、
完成结果、错误与取消事件回传到主线程，避免阻塞 GUI 事件循环。

支持两种模式：
- 全量运行：传入 image + params，内部创建 Pipeline 并执行 ``run``。
- 增量重跑：传入已存在的 pipeline + reset_stage，调用 ``reset_from`` + ``build_result``。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..pipeline import Pipeline, PipelineParams, PipelineResult


class PipelineWorker(QThread):
    """异步执行管线的 QThread worker。

    Signals:
        progress(str, float): (stage_name, percent) 进度更新
        finished_result(PipelineResult): 处理完成，携带结果
        error(str): 处理出错，携带错误信息
        cancelled(): 处理被取消
    """

    progress = Signal(str, float)
    finished_result = Signal(object)  # PipelineResult
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        image: np.ndarray | None = None,
        params: PipelineParams | None = None,
        pipeline: Pipeline | None = None,
        reset_stage: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._image = image
        self._params = params
        self._pipeline = pipeline
        self._reset_stage = reset_stage
        self._cancelled = False

    def run(self):
        try:
            def progress_cb(stage, percent):
                if self._cancelled:
                    raise InterruptedError("用户取消")
                self.progress.emit(stage, percent)

            if self._pipeline is not None and self._reset_stage is not None:
                # 增量模式：在已有 pipeline 上 reset_from
                self._pipeline.params = self._params
                self._pipeline._progress = progress_cb
                self._pipeline.reset_from(self._reset_stage)
                result = self._pipeline.build_result()
            elif self._pipeline is not None:
                # 全量运行（使用已有 pipeline 对象）
                self._pipeline.params = self._params
                self._pipeline._progress = progress_cb
                result = self._pipeline.run(self._image)
            else:
                # 原始模式：内部创建 pipeline
                pipeline = Pipeline(self._params, progress_callback=progress_cb)
                result = pipeline.run(self._image)

            if self._cancelled:
                self.cancelled.emit()
            else:
                self.finished_result.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
