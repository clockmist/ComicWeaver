#!/usr/bin/env python3
"""ComicWeaver 启动脚本 - 启动 Gradio 前端。"""
import io
import sys
from pathlib import Path

# Windows 控制台 UTF-8 兼容
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# 确保 src 在 sys.path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from comicweaver.ui import main

if __name__ == "__main__":
    print("ComicWeaver v0.1.0 - starting...")
    print("URL: http://127.0.0.1:7860")
    print("Note: This is a Mock implementation, no real LLM/diffusion calls.\n")
    main()
