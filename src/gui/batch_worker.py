"""批量处理 worker。

在独立 QThread 中顺序处理文件夹内所有图片，每张完成后通过信号通知主线程。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..pipeline import Pipeline, PipelineParams
from ..core.io import load_image
from ..core.asset_manager import AssetManager
from ..core.config import load_preference

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BatchWorker(QThread):
    """批量处理 worker。

    Signals:
        progress(int, int, str): (current, total, filename) 进度更新
        image_done(str): 单张完成（asset_id，auto_add_asset 开启时入库）
        image_result(str, object, object): 单张结果（source_name, pixel_art,
            original，auto_add_asset 关闭时发出，供预览确认）
        finished_all(int): 全部完成（成功数）
        error(str, str): 单张出错（filename, error_msg）
        cancelled()
    """

    progress = Signal(int, int, str)   # current, total, filename
    image_done = Signal(str)            # asset_id
    image_result = Signal(str, object, object)  # source_name, pixel_art, original
    finished_all = Signal(int)          # success_count
    error = Signal(str, str)            # filename, error_msg
    cancelled = Signal()

    def __init__(
        self,
        folder: str,
        params: PipelineParams,
        asset_manager: AssetManager,
        parent=None,
    ):
        super().__init__(parent)
        self._folder = Path(folder)
        self._params = params
        self._asset_manager = asset_manager
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """顺序处理文件夹内所有图片。"""
        # 收集图片文件（目录不存在/无权限时捕获异常，避免线程静默终止）
        try:
            images = sorted([
                f for f in self._folder.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
            ])
        except Exception as e:  # FileNotFoundError / PermissionError / OSError 等
            self.error.emit(str(self._folder), f"无法读取目录: {e}")
            self.finished_all.emit(0)
            return
        total = len(images)
        if total == 0:
            self.finished_all.emit(0)
            return

        success = 0
        # 是否自动移入资产库（设置项，默认关闭）
        auto_add = load_preference("auto_add_asset", False)
        auto_add = auto_add in (True, "true", "True", 1, "1")
        for i, img_path in enumerate(images):
            if self._cancelled:
                self.cancelled.emit()
                return

            self.progress.emit(i + 1, total, img_path.name)

            try:
                img = load_image(str(img_path))
                pipeline = Pipeline(self._params)
                result = pipeline.run(img)
                source_name = img_path.stem
                if auto_add:
                    asset_id = self._asset_manager.add_asset(result.pixel_art, source_name)
                    self.image_done.emit(asset_id)
                # 始终发出结果供"预览"阶段收集（分页浏览、确认入库/重新生成）
                self.image_result.emit(source_name, result.pixel_art, img)
                success += 1
            except Exception as e:
                self.error.emit(img_path.name, str(e))

        self.finished_all.emit(success)
