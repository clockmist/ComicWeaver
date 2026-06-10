"""轻量级项目持久化 - JSON文件,真实版本可换 SQLite。

v2.0: 新增 save_dev_log() 将开发者日志持久化为可读文本文件。
v3.0: 引入 checkpoint 机制 — project.json 精简为元信息，完整 state
       通过 checkpoints/ 下的 agent 级快照重建。
"""
from __future__ import annotations

import time
from pathlib import Path

from comicweaver.core import ComicProject, ComicState

from .paths import project_dir, projects_root


def save_project(project: ComicProject) -> Path:
    """保存项目到 projects/{project_id}/project.json。

    v2 (format_version >= 2): state 字段不写入完整数据，
    完整状态通过 checkpoints/ 目录管理。
    """
    project.updated_at = time.time()
    path = project_dir(project.project_id) / "project.json"
    path.write_text(
        project.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_project(project_id: str) -> ComicProject | None:
    """加载项目。

    v1 (format_version=1): 从 project.json 的 state 字段恢复。
    v2 (format_version>=2): 从 checkpoints/ 重建 state。
    """
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        return None

    project = ComicProject.model_validate_json(path.read_text(encoding="utf-8"))

    # v2+: 从 checkpoint 重建完整 state
    if project.format_version >= 2:
        from .checkpoint import load_latest_state

        checkpoint_state = load_latest_state(project_id)
        if checkpoint_state is not None:
            project.state = dict(checkpoint_state)
        elif not project.state.get("user_input"):
            # checkpoint 不存在且 state 缺少必要字段 → 从已有数据补全
            summary = project.state_summary or {}
            project.state.update({
                "project_id": project_id,
                "title": project.title,
                "user_input": summary.get("user_input", project.state.get("user_input", "")),
                "creation_mode": summary.get("creation_mode", project.state.get("creation_mode", "simple")),
                "style_preset": summary.get("style_preset", project.state.get("style_preset", "manga")),
                "interaction_mode": summary.get("interaction_mode", project.state.get("interaction_mode", "semi_auto")),
                "target_pages": summary.get("target_pages", project.state.get("target_pages", 4)),
                "current_phase": summary.get("current_phase", project.state.get("current_phase", "init")),
            })

    return project


def load_project_meta(project_id: str) -> ComicProject | None:
    """只加载项目元信息（不重建完整 state）。

    用于 UI 列表等只需要基本信息的场景，速度更快。
    """
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        return None

    project = ComicProject.model_validate_json(path.read_text(encoding="utf-8"))

    # v2: 从 state_summary 构造最小 state dict（兼容 UI 渲染）
    if project.format_version >= 2 and not project.state.get("user_input"):
        summary = project.state_summary or {}
        project.state.update({
            "project_id": project_id,
            "title": project.title,
            "user_input": summary.get("user_input", ""),
            "creation_mode": summary.get("creation_mode", "simple"),
            "style_preset": summary.get("style_preset", "manga"),
            "interaction_mode": summary.get("interaction_mode", "semi_auto"),
            "target_pages": summary.get("target_pages", 4),
            "current_phase": summary.get("current_phase", "init"),
        })

    return project


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
                project = ComicProject.model_validate_json(
                    p.read_text(encoding="utf-8")
                )
                # v2: 从 state_summary 填充最小 state（兼容 UI 渲染）
                if project.format_version >= 2 and not project.state.get("user_input"):
                    summary = project.state_summary or {}
                    project.state.update({
                        "project_id": project.project_id,
                        "title": project.title,
                        "user_input": summary.get("user_input", ""),
                        "creation_mode": summary.get("creation_mode", "simple"),
                        "style_preset": summary.get("style_preset", "manga"),
                        "interaction_mode": summary.get("interaction_mode", "semi_auto"),
                        "target_pages": summary.get("target_pages", 4),
                        "current_phase": summary.get("current_phase", "init"),
                    })
                out.append(project)
            except Exception:  # noqa: BLE001
                continue
    out.sort(key=lambda x: x.updated_at, reverse=True)
    return out


def project_to_state(project: ComicProject) -> ComicState:
    """将 ComicProject 转换为 ComicState。"""
    return ComicState(**project.state)  # type: ignore[arg-type]


def state_to_project(state: ComicState, title: str = "") -> ComicProject:
    """将 ComicState 转换为 ComicProject（v2 格式）。

    v2 格式中 state 字段只保留启动所需的最小字段，
    完整状态通过 checkpoints/ 目录管理。
    """
    # v2 最小状态：保留启动项目所需的必要字段
    essential_state = {
        "project_id": state.get("project_id", ""),
        "user_input": state.get("user_input", ""),
        "title": state.get("title", ""),
        "creation_mode": state.get("creation_mode", "simple"),
        "style_preset": state.get("style_preset", "manga"),
        "interaction_mode": state.get("interaction_mode", "semi_auto"),
        "target_pages": state.get("target_pages", 4),
        "current_phase": state.get("current_phase", "init"),
    }
    return ComicProject(
        project_id=state.get("project_id", ""),
        title=(
            title
            or state.get("title", "")
            or state.get("user_input", "Untitled")[:30]
        ),
        format_version=2,
        state=essential_state,
    )


def save_dev_log(dev_entries: list[dict], project_id: str) -> Path | None:
    """将开发者日志条目写入项目目录下的 dev_log.txt 文件。

    每条日志按 Agent 分段，带中文标注，便于快速定位各 Agent 的输出内容。

    Args:
        dev_entries: DevLogEntry.to_dict() 的列表
        project_id: 项目ID

    Returns:
        写入的文件路径，如果列表为空则返回 None
    """
    if not dev_entries:
        return None

    from comicweaver.utils.logging import write_dev_log_to_file

    proj_dir = project_dir(project_id)
    return write_dev_log_to_file(dev_entries, proj_dir)
