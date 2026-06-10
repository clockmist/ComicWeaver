"""项目相关回调：创建、打开、保存、列表。"""
from __future__ import annotations

import gradio as gr

from comicweaver.core import make_initial_state
from comicweaver.storage import save_project as storage_save_project
from comicweaver.storage import state_to_project

from ..renderers.project import render_project_cards, render_project_info
from ..renderers.workflow import render_workflow_left
from ..composables.live import render_live_content, render_phase_text_content
from ..session import SESSION


def create_project(
    user_input: str,
    project_name: str,
    creation_mode: str,
    style_preset: str,
    interaction_mode: str,
    target_pages: int,
) -> tuple[str, str]:
    if not user_input.strip():
        return "❌ 请输入剧本主题或完整剧本", ""

    SESSION.reset()
    SESSION.state = make_initial_state(
        user_input=user_input,
        title=project_name.strip(),
        creation_mode=creation_mode,
        style_preset=style_preset,
        interaction_mode=interaction_mode,
        target_pages=target_pages,
    )

    project_id = SESSION.state["project_id"]
    proj_title = SESSION.state.get("title", project_id)
    return (
        f"✅ 项目已创建: {proj_title} ({project_id})",
        f"项目名称: {proj_title}\n项目ID: {project_id}\n模式: {creation_mode} / {interaction_mode}\n"
        f"目标页数: {target_pages}\n风格: {style_preset}\n"
        f"剧本: {user_input[:60]}{'...' if len(user_input) > 60 else ''}",
    )


def list_projects_cards() -> str:
    """列出已保存项目，渲染为卡片网格 HTML。"""
    from comicweaver.storage import list_projects as storage_list_projects
    projects = storage_list_projects()
    card_data: list[dict] = []
    import datetime
    for p in projects:
        phase = p.state.get("current_phase", "init") if p.state else "init"
        dt = datetime.datetime.fromtimestamp(p.updated_at)
        card_data.append({
            "project_id": p.project_id,
            "title": p.title,
            "updated_at_str": dt.strftime("%Y-%m-%d %H:%M"),
            "phase": phase,
        })
    return render_project_cards(card_data)


def open_project_by_id(project_id: str) -> tuple[str, str, str, str, str, str, dict]:
    """通过项目 ID 字符串打开已保存项目。

    v3.0: 自动检测并迁移 v1 格式项目到 v2 checkpoint 格式。
    """
    from comicweaver.storage import load_project, project_to_state

    pid = (project_id or "").strip()
    if not pid:
        return (
            "❌ 请输入项目ID", list_projects_cards(), render_project_info(),
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            "init",
            gr.update(),
        )

    project = load_project(pid)
    if project is None:
        return (
            f"❌ 项目 {pid} 不存在", list_projects_cards(), render_project_info(),
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            "init",
            gr.update(),
        )

    # v3.0: 自动迁移 v1 项目
    if project.format_version < 2 and project.state:
        from comicweaver.storage import migrate_legacy_project
        migrate_legacy_project(pid)
        project = load_project(pid) or project

    state = project_to_state(project)
    SESSION.reset()
    SESSION.state = state

    character_db = state.get("character_db", {})
    char_preview: list[dict] = []
    for cid, cp in character_db.get("characters", {}).items():
        ref = cp.get("base_reference", {})
        img_path = ref.get("image_path", "")
        if img_path:
            char_preview.append({
                "kind": "character", "char_id": cid,
                "name": cp.get("name", cid), "image_path": img_path,
            })

    panel_images = state.get("panel_images", [])
    panel_preview: list[dict] = []
    for pi in panel_images:
        img_path = pi.get("image_path", "")
        if img_path:
            panel_preview.append({"panel_id": pi.get("panel_id", "?"), "image_path": img_path})

    right_html = render_live_content(
        panel_preview, char_preview,
        render_phase_text_content(state, SESSION.agent_outputs),
        state.get("bubble_placements", {}),
    )

    current_phase = state.get("current_phase", "init")
    left_html = render_workflow_left("", [], [], None, current_phase=current_phase)

    _next_phase: dict[str, str] = {
        "init": "story",
        "story": "character",
        "character": "script",
        "script": "storyboard",
        "storyboard": "image",
        "image": "bubble",
        "bubble": "layout",
        "layout": "layout",
    }
    next_phase = _next_phase.get(current_phase, "init")

    return (
        f"✅ 已加载: {project.title} — 已自动切换到「⚡ 创作」Tab",
        list_projects_cards(),
        render_project_info(state),
        left_html,
        right_html,
        next_phase,
        gr.update(selected="⚡ 创作"),
    )


def save_current_project() -> str:
    """手动保存当前项目。"""
    if SESSION.state is None:
        return "❌ 没有可保存的项目，请先创建项目"
    try:
        from comicweaver.storage import list_project_checkpoints

        proj = state_to_project(SESSION.state)
        project_id = SESSION.state.get("project_id", "")
        if project_id:
            proj.checkpoints_completed = list_project_checkpoints(project_id)
            current_phase = SESSION.state.get("current_phase", "init")
            # 将 current_phase 映射到文件夹名
            _phase_to_folder: dict[str, str] = {
                "story": "01_story", "character": "02_character",
                "script": "03_script", "storyboard": "04_storyboard",
                "image": "05_image", "bubble": "06_bubble", "layout": "07_layout",
            }
            proj.latest_checkpoint = _phase_to_folder.get(current_phase, "")
            proj.state_summary = {
                "title": SESSION.state.get("title", ""),
                "user_input": SESSION.state.get("user_input", ""),
                "current_phase": current_phase,
                "interaction_mode": SESSION.state.get("interaction_mode", "semi_auto"),
                "creation_mode": SESSION.state.get("creation_mode", "simple"),
                "style_preset": SESSION.state.get("style_preset", "manga"),
                "target_pages": SESSION.state.get("target_pages", 4),
            }
        storage_save_project(proj)
        return f"✅ 项目已保存: {proj.title} ({proj.project_id})"
    except Exception as exc:
        return f"❌ 保存失败: {exc}"
