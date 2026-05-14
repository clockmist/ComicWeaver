"""pytest 配置 - 把 src 加入 sys.path 以便导入 comicweaver。"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
