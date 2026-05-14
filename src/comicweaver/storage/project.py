"""轻量级项目持久化 - JSON文件,真实版本可换 SQLite。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from comicweaver.core import ComicProject, ComicState

from .paths import project_dir, projects_root


def save_project(project: ComicProject) -> Path:
    """保存项目到 projects/{project_id}/project.json。"""
    project.updated_at = time.time()
    path = project_dir(project.project_id) / "project.json"
    path.write_text(
        project.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_project(project_id: str) -> Optional[ComicProject]:
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        return None
    return ComicProject.model_validate_json(path.read_text(encoding="utf-8"))


def list_projects() -> list[ComicProject]:
    """列出所有已保存的项目。"""
    out: list[ComicProject] = []
    root = projects_root()
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        p = sub / "project.json"
        if p.exists():
            try:
                out.append(ComicProject.model_validate_json(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    out.sort(key=lambda x: x.updated_at, reverse=True)
    return out


def project_to_state(project: ComicProject) -> ComicState:
    return ComicState(**project.state)  # type: ignore[arg-type]


def state_to_project(state: ComicState, title: str = "") -> ComicProject:
    return ComicProject(
        project_id=state.get("project_id", ""),
        title=title or state.get("user_input", "Untitled")[:30],
        state=dict(state),
    )
