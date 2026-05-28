"""pytest 配置 - 把 src 加入 sys.path 以便导入 comicweaver。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_comicweaver_config(monkeypatch):
    """Keep tests independent from a developer's local API configuration."""
    keys = [
        "COMICWEAVER_CONFIG",
        "COMICWEAVER_LLM_PROVIDER",
        "COMICWEAVER_LLM_BASE_URL",
        "COMICWEAVER_LLM_API_KEY",
        "COMICWEAVER_LLM_MODEL",
        "COMICWEAVER_LLM_ENABLED",
        "COMICWEAVER_IMAGE_PROVIDER",
        "COMICWEAVER_IMAGE_SERVER_URL",
        "COMICWEAVER_IMAGE_MODEL",
        "COMICWEAVER_IMAGE_ENABLED",
        "COMICWEAVER_IMAGE_CHARACTER_WORKFLOW",
        "COMICWEAVER_IMAGE_PANEL_WORKFLOW",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COMICWEAVER_CONFIG", "__missing_test_config__.yaml")

    from comicweaver.config import reload_config

    reload_config()
    yield
    reload_config()
