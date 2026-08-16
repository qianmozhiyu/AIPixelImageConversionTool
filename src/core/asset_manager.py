"""资产存储管理。

管理用户转换生成的像素图资产，支持增删改查导出。
资产存储在本地文件系统中，每个资产包含图片文件 + 元数据 JSON。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .io import load_image, save_image


# 支持的图片扩展名
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AssetInfo:
    """资产元信息。"""
    id: str
    source_name: str        # 源伪像素图文件名（不含扩展名）
    created_at: float       # 生成时间戳
    thumb_path: str         # 缩略图路径
    width: int
    height: int
    # 额外元数据（参数摘要等）
    metadata: dict = None   # type: ignore

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AssetInfo":
        return cls(**d)


def _default_store_dir() -> Path:
    """默认资产存储目录 ~/.aipixel/assets/。"""
    return Path.home() / ".aipixel" / "assets"


def get_store_dir(custom_dir: Optional[str] = None) -> Path:
    """获取资产存储目录，优先使用自定义路径。"""
    if custom_dir:
        p = Path(custom_dir)
    else:
        p = _default_store_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


class AssetManager:
    """资产存储管理器。

    资产目录结构：
        store_dir/
            {asset_id}/
                image.png        # 原始像素图
                thumb.png        # 缩略图（最大 128x128）
                meta.json        # 元数据
    """

    THUMB_MAX_SIZE = 128

    def __init__(self, store_dir: Optional[str | Path] = None):
        """初始化资产管理器。

        Args:
            store_dir: 资产存储目录，None 时使用默认 ~/.aipixel/assets/
        """
        self.store_dir = Path(store_dir) if store_dir else _default_store_dir()
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _asset_dir(self, asset_id: str) -> Path:
        return self.store_dir / asset_id

    def _image_path(self, asset_id: str) -> Path:
        return self._asset_dir(asset_id) / "image.png"

    def _thumb_path(self, asset_id: str) -> Path:
        return self._asset_dir(asset_id) / "thumb.png"

    def _meta_path(self, asset_id: str) -> Path:
        return self._asset_dir(asset_id) / "meta.json"

    def _make_thumbnail(self, image: np.ndarray, path: Path) -> None:
        """生成缩略图（最近邻缩放到最大 128x128）。"""
        h, w = image.shape[:2]
        scale = min(self.THUMB_MAX_SIZE / h, self.THUMB_MAX_SIZE / w, 1.0)
        if scale < 1.0:
            th, tw = max(1, int(h * scale)), max(1, int(w * scale))
            # 最近邻缩放
            from skimage.transform import resize as _sk_resize
            thumb = _sk_resize(image, (th, tw), order=0, anti_aliasing=False, preserve_range=True)
            thumb = thumb.astype(np.uint8)
        else:
            thumb = image
        save_image(thumb, path)

    def add_asset(self, image: np.ndarray, source_name: str, metadata: Optional[dict] = None) -> str:
        """添加资产。

        Args:
            image: (H,W,3) uint8 像素图。
            source_name: 源伪像素图文件名（不含扩展名），用于命名。
            metadata: 额外元数据（参数摘要等）。

        Returns:
            asset_id: 资产唯一 ID。
        """
        asset_id = uuid.uuid4().hex[:12]
        asset_dir = self._asset_dir(asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)

        # 保存图片
        save_image(image, self._image_path(asset_id))
        # 保存缩略图
        self._make_thumbnail(image, self._thumb_path(asset_id))

        h, w = image.shape[:2]
        info = AssetInfo(
            id=asset_id,
            source_name=source_name,
            created_at=time.time(),
            thumb_path=str(self._thumb_path(asset_id)),
            width=w,
            height=h,
            metadata=metadata or {},
        )
        self._save_meta(info)
        return asset_id

    def _save_meta(self, info: AssetInfo) -> None:
        with open(self._meta_path(info.id), "w", encoding="utf-8") as f:
            json.dump(info.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_meta(self, asset_id: str) -> Optional[AssetInfo]:
        meta_path = self._meta_path(asset_id)
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return AssetInfo.from_dict(json.load(f))

    def list_assets(self) -> list[AssetInfo]:
        """列出所有资产，按创建时间降序（最新在前）。"""
        assets = []
        for item in self.store_dir.iterdir():
            if item.is_dir() and self._meta_path(item.name).exists():
                info = self._load_meta(item.name)
                if info:
                    assets.append(info)
        assets.sort(key=lambda a: a.created_at, reverse=True)
        return assets

    def load_asset(self, asset_id: str) -> Optional[np.ndarray]:
        """加载资产图片。"""
        path = self._image_path(asset_id)
        if not path.exists():
            return None
        return load_image(path)

    def load_thumbnail(self, asset_id: str) -> Optional[np.ndarray]:
        """加载缩略图。"""
        path = self._thumb_path(asset_id)
        if not path.exists():
            return None
        return load_image(path)

    def delete_asset(self, asset_id: str) -> bool:
        """删除资产。"""
        asset_dir = self._asset_dir(asset_id)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
            return True
        return False

    def export_asset(self, asset_id: str, dest_path: str | Path) -> bool:
        """导出资产到指定路径。"""
        src = self._image_path(asset_id)
        if not src.exists():
            return False
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True

    def update_asset(self, asset_id: str, image: np.ndarray) -> bool:
        """更新资产图片（编辑器保存回写）。"""
        if not self._image_path(asset_id).exists():
            return False
        save_image(image, self._image_path(asset_id))
        self._make_thumbnail(image, self._thumb_path(asset_id))
        # 更新元数据尺寸
        info = self._load_meta(asset_id)
        if info:
            h, w = image.shape[:2]
            info.width = w
            info.height = h
            self._save_meta(info)
        return True

    def get_info(self, asset_id: str) -> Optional[AssetInfo]:
        """获取资产信息。"""
        return self._load_meta(asset_id)
