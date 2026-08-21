# -*- mode: python ; coding: utf-8 -*-
# onedir（文件夹方式）打包配置。
# 裁剪策略（均已验证稳定）：
#   - PySide6 未用模块（纯 2D Widgets 应用，不需要 QML/网络/OpenGL/3D 等）
#   - PIL 未用图像插件（仅需 PNG/JPG）
#   - 二进制层过滤：软件 OpenGL 渲染器、QML/Quick/Network DLL、OpenSSL、AVIF/HEIF
#   - scipy/sklearn 不裁剪（依赖链过密，强行排除导致连环 ImportError）


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ---- PySide6 未用模块 ----
        'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets',
        'PySide6.QtNetwork', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
        'PySide6.QtSql', 'PySide6.QtMultimedia', 'PySide6.QtSvg',
        'PySide6.QtXml', 'PySide6.QtTest', 'PySide6.QtConcurrent',
        'PySide6.QtPrintSupport', 'PySide6.QtDBus', 'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets', 'PySide6.QtPositioning', 'PySide6.QtLocation',
        # ---- PIL 未用图像插件（仅需 PNG/JPG 解码与 _imaging 核心） ----
        'PIL._avif', 'PIL._heif', 'PIL._webp', 'PIL._tiff', 'PIL._imagingtiff',
        'PIL._mpeg', 'PIL._dds', 'PIL._gif', 'PIL._icns',
        # ---- scipy：仅排除 skimage.io 独用的 io 与已废弃 misc ----
        'scipy.io', 'scipy.misc',
    ],
    noarchive=False,
    optimize=0,
)

# 过滤无需的二进制（Qt QML/Quick/Network/OpenGL 等 DLL、软件 OpenGL 渲染器、
# OpenSSL、Pillow AVIF/HEIF 库）
a.binaries = [b for b in a.binaries if not any(x in b[0] for x in (
    'opengl32sw', 'Qt6Qml', 'Qt6Quick', 'Qt6Network', 'Qt6Sql', 'Qt6OpenGL',
    'Qt6Multimedia', 'Qt6Svg', 'Qt6Xml', 'Qt6Test', 'Qt6Concurrent',
    'Qt6PrintSupport', 'Qt6DBus', 'Qt6Positioning', 'Qt6Location',
    'libcrypto', 'libssl', 'avif', 'libavif', 'heif', 'libheif',
))]
# 过滤 Qt 的 QML 与翻译数据（纯 Widgets 应用不需要）
a.datas = [d for d in a.datas if not any(x in d[0] for x in ('qml', 'translations'))]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AIPixelTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='v1_2_0',
)
