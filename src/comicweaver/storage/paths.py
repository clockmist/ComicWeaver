"""路径管理 - 项目目录、缓存目录等的统一入口。"""
from __future__ import annotations

from pathlib import Path

from comicweaver.config import load_config


def root_dir() -> Path:
    """获取应用根目录(从入口文件向上找)。"""
    return Path.cwd()


def projects_root() -> Path:
    p = root_dir() / load_config().storage.projects_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_dir(project_id: str) -> Path:
    p = projects_root() / project_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root() -> Path:
    p = root_dir() / load_config().storage.cache_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_root() -> Path:
    p = root_dir() / load_config().storage.outputs_dir
    p.mkdir(parents=True, exist_ok=True)
    return p
