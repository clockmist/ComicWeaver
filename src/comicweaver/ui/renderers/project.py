"""项目信息栏和项目卡片网格渲染。"""
from __future__ import annotations

import html


def render_project_info(state: dict | None = None) -> str:
    """渲染当前项目信息栏（创作流程 Tab 顶部）。"""
    if not state:
        return '<div style="color:#94a3b8;padding:8px;font-size:12px;">尚未创建项目</div>'

    title = state.get("title", "") or "未命名"
    pid = state.get("project_id", "?")
    phase = state.get("current_phase", "init")
    phase_labels = {
        "init": "等待开始", "story": "故事阶段", "script": "剧本阶段",
        "character": "角色阶段", "storyboard": "分镜阶段",
        "image": "图像阶段", "bubble": "台词阶段", "layout": "排版阶段",
    }
    phase_label = phase_labels.get(phase, phase)

    return f"""
    <div style="display:flex;align-items:center;gap:16px;padding:10px 16px;
                background:#f0f4ff;border:1px solid #cbd5e1;
                border-radius:10px;color:#1e293b;font-size:13px;">
        <div style="font-weight:700;font-size:15px;">📋 {html.escape(title)}</div>
        <div style="color:#64748b;font-size:11px;font-family:monospace;">ID: {html.escape(pid)}</div>
        <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
            <span style="background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:12px;
                         font-size:11px;">📌 {html.escape(phase_label)}</span>
        </div>
    </div>
    """


def render_project_cards(projects: list[dict]) -> str:
    """以卡片网格展示已保存的项目列表。"""
    if not projects:
        return (
            '<div class="cw-stream-empty">暂无已保存的项目。<br>'
            '在「项目」Tab 创建新项目后运行工作流，项目会自动保存。</div>'
        )

    phase_labels = {
        "init": "等待开始", "script": "剧本", "character": "角色",
        "storyboard": "分镜", "image": "图像", "layout": "排版",
    }

    cards = []
    for p in projects:
        pid = p.get("project_id", "?")
        title = p.get("title", "未命名")
        updated = p.get("updated_at_str", "?")
        phase = p.get("phase", "init")
        phase_label = phase_labels.get(phase, phase)

        cards.append(f"""
        <div class="cw-project-card" data-project-id="{html.escape(pid)}" role="button" tabindex="0">
            <div class="cw-project-card-title">{html.escape(title)}</div>
            <div class="cw-project-card-id">{html.escape(pid)}</div>
            <div class="cw-project-card-meta">
                <span class="cw-badge">{phase_label}</span>
                <span style="font-size:11px;color:#94a3b8;">{html.escape(updated)}</span>
            </div>
        </div>
        """)

    return f"""
    <div class="cw-project-cards">
        {"".join(cards)}
    </div>
    """
