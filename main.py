"""AIPixelTool 打包入口。

PyInstaller 以此文件为入口，src 作为正规包被自动收集。
运行 GUI：python main.py
"""
from src.gui.main_window import main
import sys

if __name__ == "__main__":
    sys.exit(main())
