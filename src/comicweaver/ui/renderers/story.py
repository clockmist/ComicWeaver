"""故事和剧本渲染：故事大纲、情感曲线SVG、剧本详情。"""
from __future__ import annotations

import html as html_mod

from .common import _resolve_char_name


def render_script_summary(script: dict, story: dict | None = None) -> str:
    """渲染剧本摘要卡片（v1.0：pages 格式 + StoryOutput 角色）。"""
    if not script:
        return '<div style="color:#64748b;padding:12px;">尚未生成剧本</div>'

    title = html_mod.escape(script.get("title", "Untitled"))
    summary = html_mod.escape(script.get("summary", ""))
    genres = script.get("genre", [])

    story_chars = story.get("characters", []) if story else []
    legacy_chars = script.get("characters", [])
    chars = story_chars if story_chars else legacy_chars

    pages = script.get("pages", [])
    legacy_scenes = script.get("scenes", [])
    total_panels = sum(len(p.get("panels", [])) for p in pages)
    scene_count = len(pages) if pages else len(legacy_scenes)

    char_html = "".join(
        f'<span class="cw-badge" style="margin-right:6px;">'
        f'{html_mod.escape(c.get("name", ""))} · {html_mod.escape(c.get("role", ""))}</span>'
        for c in chars
    )

    genre_html = " ".join(
        f'<span class="cw-badge">{html_mod.escape(g)}</span>' for g in genres
    )

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 8px 0;color:#1e293b;">📖 {title}</h3>
        <div style="color:#64748b;font-size:13px;margin-bottom:12px;">{summary}</div>
        <div style="margin-bottom:8px;">{genre_html}</div>
        <div style="font-size:13px;margin-bottom:6px;"><b>角色:</b></div>
        <div style="margin-bottom:12px;">{char_html}</div>
        <div style="display:flex;gap:24px;font-size:13px;color:#475569;">
            <div><b>{scene_count}</b> 页</div>
            <div><b>{total_panels}</b> 分格</div>
            <div><b>{len(chars)}</b> 角色</div>
        </div>
    </div>
    """


def render_emotion_curve(curve: list[float]) -> str:
    """简单SVG情感曲线图。"""
    if not curve:
        return '<div style="color:#64748b;padding:12px;">情感曲线生成中...</div>'

    w, h = 600, 120
    n = len(curve)
    if n < 2:
        return '<div style="color:#64748b;">数据点不足</div>'

    xs = [i * w / (n - 1) for i in range(n)]
    ys = [h - 10 - v * (h - 20) for v in curve]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))

    markers = []
    for i, v in enumerate(curve):
        if v >= 0.8:
            markers.append(
                f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="4" '
                f'fill="#ef4444" stroke="white" stroke-width="2"/>'
            )

    return f"""
    <div class="cw-card">
        <div style="font-size:13px;color:#475569;margin-bottom:8px;"><b>📈 情感曲线</b> (红点为高潮)</div>
        <svg width="{w}" height="{h}" style="display:block;">
            <defs>
                <linearGradient id="emo-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#ef4444" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.05"/>
                </linearGradient>
            </defs>
            <polyline points="{pts}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
            <polygon points="0,{h} {pts} {w},{h}" fill="url(#emo-grad)"/>
            {''.join(markers)}
        </svg>
    </div>
    """


# ============================================================================
# v0.4 详细阶段渲染
# ============================================================================


def _render_story_outline(story: dict) -> str:
    """渲染 StoryAgent 输出：完整叙事故事及元数据。"""
    title = html_mod.escape(story.get("title", "未命名"))
    author_note = html_mod.escape((story.get("author_note") or "")[:200])
    tone = html_mod.escape(story.get("tone", ""))
    genre = story.get("genre", [])
    story_text = (story.get("story_text") or "").strip()
    characters = story.get("characters", [])
    core_conflict = html_mod.escape((story.get("core_conflict") or ""))
    setting = html_mod.escape((story.get("setting") or ""))
    target_pages = story.get("target_pages", 4)

    genre_tags = " ".join(
        f'<span class="cw-badge">{html_mod.escape(str(g))}</span>' for g in genre
    )

    char_items = ""
    for c in characters:
        name = html_mod.escape(c.get("name", "?"))
        role = html_mod.escape(c.get("role", "?"))
        desc = html_mod.escape((c.get("brief_description") or "")[:120])
        char_items += (
            f'<div class="cw-phase-text-scene">'
            f'<b>{name}</b> <span class="cw-badge">{role}</span>'
            f'<br><span style="color:#64748b;">{desc}</span>'
            f'</div>'
        )

    paragraphs = story_text.split("\n\n") if story_text else ["等待生成..."]
    formatted_text = ""
    for para in paragraphs:
        if para.strip():
            formatted_text += (
                f'<p style="line-height:1.8;text-indent:2em;margin:0 0 8px 0;">'
                f'{html_mod.escape(para.strip())}</p>'
            )

    word_count = len(story_text) if story_text else 0

    return f"""
    <div class="cw-phase-text-panel">
        <h4>📖 故事: {title}</h4>
        <div class="cw-phase-text-scroll">
        <div class="cw-phase-text-item" style="margin-bottom:4px;">
            <span style="color:#64748b;font-style:italic;">{author_note}</span>
        </div>
        <div class="cw-phase-text-item" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
            {genre_tags}
            {f'<span class="cw-badge" style="background:#e0e7ff;color:#3730a3;">{tone}</span>' if tone else ''}
            <span style="font-size:11px;color:#94a3b8;margin-left:auto;">
                {target_pages}页 · {word_count}字
            </span>
        </div>
        {f'<div class="cw-phase-text-item"><span class="key">背景设定:</span> <span class="val">{setting}</span></div>' if setting else ''}
        {f'<div class="cw-phase-text-item"><span class="key">核心冲突:</span> <span class="val">{core_conflict}</span></div>' if core_conflict else ''}
        <div class="cw-phase-text-item" style="margin-top:12px;background:#fafbfc;
                    border-left:3px solid #3b82f6;padding:12px 16px;border-radius:0 8px 8px 0;
                    max-height:400px;overflow-y:auto;">
            {formatted_text}
        </div>
        {f'''<details style="margin-top:12px;">
            <summary style="cursor:pointer;color:#3b82f6;font-size:13px;font-weight:600;">
                👥 角色概览 ({len(characters)}人)
            </summary>
            {char_items}
        </details>''' if characters else ''}
        </div>
    </div>
    """


def _render_detailed_script(script: dict, char_map: dict[str, str], story: dict | None = None) -> str:
    """渲染完整剧本（v1.0：pages/panels 格式 + StoryOutput 角色）。"""
    title = html_mod.escape(script.get("title", "未命名"))
    summary = html_mod.escape((script.get("summary", "") or ""))
    genres = script.get("genre", [])

    story_chars = story.get("characters", []) if story else []
    legacy_chars = script.get("characters", [])
    characters = story_chars if story_chars else legacy_chars

    pages = script.get("pages", [])
    legacy_scenes = script.get("scenes", [])

    genre_tags = " ".join(
        f'<span class="cw-badge">{html_mod.escape(str(g))}</span>' for g in genres
    )

    char_list = ""
    for c in characters:
        name = html_mod.escape(c.get("name", "?"))
        role = html_mod.escape(c.get("role", "?"))
        desc = html_mod.escape((c.get("brief_description") or "")[:120])
        char_list += (
            f'<div class="cw-phase-text-scene">'
            f'<b>{name}</b> <span class="cw-badge">{role}</span>'
            + (f'<br><span style="color:#64748b;">{desc}</span>' if desc else "")
            + f'</div>'
        )

    # v1.0 路径：渲染 pages → panels
    panel_items = ""
    if pages:
        total_panels = sum(len(p.get("panels", [])) for p in pages)
        for page in pages:
            page_num = page.get("page_number", "?")
            page_note = html_mod.escape((page.get("page_note") or "")[:200])
            panels = page.get("panels", [])

            page_panels_html = ""
            for panel in panels:
                pid = html_mod.escape(panel.get("panel_id", "?"))
                order = panel.get("order_in_page", "?")
                purpose = html_mod.escape(panel.get("narrative_purpose", "")[:200])
                char_name = html_mod.escape(_resolve_char_name(panel.get("character_id", ""), char_map))
                action = html_mod.escape((panel.get("character_action") or "")[:120])
                dialogue = html_mod.escape((panel.get("dialogue_text") or "")[:150])
                tone = html_mod.escape(panel.get("dialogue_tone", ""))
                is_thought = panel.get("is_thought", False)
                narration = html_mod.escape((panel.get("narration") or "")[:150])
                emotion = panel.get("emotion", "")
                emotion_intensity = panel.get("emotion_intensity", 0.5)
                is_key = panel.get("is_key_panel", False)
                location = html_mod.escape(panel.get("location", ""))
                time_of_day = html_mod.escape(panel.get("time_of_day", ""))
                atmosphere = html_mod.escape(panel.get("atmosphere", ""))

                bubble_icon = "💭" if is_thought else "💬"
                tone_badge = f' <span class="cw-badge">{tone}</span>' if tone else ""
                key_badge = ' <span class="cw-badge" style="background:#fef3c7;color:#92400e;">关键格</span>' if is_key else ""

                page_panels_html += f"""
                <div class="cw-phase-text-scene" style="margin-bottom:6px;">
                    <div style="font-weight:600;margin-bottom:2px;">
                        格 {order} {key_badge}
                        {f'<span style="float:right;font-size:10px;color:#94a3b8;">{emotion} {emotion_intensity:.1f}</span>' if emotion else ''}
                    </div>
                    {f'<div style="color:#3b82f6;font-size:12px;margin-bottom:2px;">🎯 {purpose}</div>' if purpose else ''}
                    {f'<div style="font-size:11px;color:#64748b;">📍 {location}{" · " + time_of_day if time_of_day else ""}{" · " + atmosphere if atmosphere else ""}</div>' if (location or time_of_day or atmosphere) else ''}
                    {f'<div style="color:#64748b;font-style:italic;">📢 {narration}</div>' if narration else ''}
                    {f'<div>🎬 <b>{char_name}</b>: {action}</div>' if (char_name or action) else ''}
                    {f'<div>{bubble_icon} <b>{char_name}</b>{tone_badge}: {dialogue}</div>' if dialogue else ''}
                </div>
                """

            page_panels_html = page_panels_html or '<div style="color:#94a3b8;font-size:12px;">（无面板）</div>'

            panel_items += f"""
            <details style="margin-top:4px;" open>
                <summary style="cursor:pointer;color:#6366f1;font-size:13px;font-weight:600;">
                    📄 第 {page_num} 页 ({len(panels)} 格)
                    {f'<span style="color:#64748b;font-weight:400;font-size:12px;">— {page_note}</span>' if page_note else ''}
                </summary>
                <div style="margin-left:12px;border-left:2px solid #e0e7ff;padding-left:12px;margin-top:6px;">
                    {page_panels_html}
                </div>
            </details>
            """
    else:
        # 旧格式兼容路径
        for s in legacy_scenes:
            idx = s.get("order", s.get("scene_index", "?"))
            loc = html_mod.escape(s.get("location", "?"))
            time_of_day = html_mod.escape(s.get("time_of_day", ""))
            atmosphere = html_mod.escape(s.get("atmosphere", ""))
            narration = html_mod.escape((s.get("narration") or "")[:150])
            emotion = s.get("emotion_intensity", 0)

            chars_in_scene = s.get("characters_present", [])
            char_display = ", ".join(
                html_mod.escape(_resolve_char_name(cid, char_map))
                for cid in chars_in_scene
            ) if chars_in_scene else "无"

            actions = s.get("actions", [])
            action_items = ""
            for a in actions:
                actor = html_mod.escape(_resolve_char_name(a.get("actor", "?"), char_map))
                desc = html_mod.escape(a.get("description", "")[:80])
                action_items += f'<div style="margin-left:8px;">🎬 <b>{actor}</b>: {desc}</div>'

            dialogues = s.get("dialogues", [])
            dialogue_items = ""
            for d in dialogues:
                speaker = html_mod.escape(_resolve_char_name(d.get("speaker", "?"), char_map))
                text = html_mod.escape(d.get("text", "")[:120])
                tone = html_mod.escape(d.get("tone", ""))
                is_thought = d.get("is_thought", False)
                bubble_icon = "💭" if is_thought else "💬"
                tone_badge = f' <span class="cw-badge">{tone}</span>' if tone else ""
                dialogue_items += (
                    f'<div style="margin-left:8px;">{bubble_icon} <b>{speaker}</b>{tone_badge}: {text}</div>'
                )

            panel_items += f"""
            <div class="cw-phase-text-scene">
                <b>场景 {idx}</b> · {loc}
                {f' · {time_of_day}' if time_of_day else ''}
                {f' · {atmosphere}' if atmosphere else ''}
                <span style="float:right;font-size:10px;color:#94a3b8;">情绪 {emotion:.2f}</span>
                {f'<div style="color:#64748b;font-style:italic;">📢 {narration}</div>' if narration else ''}
                <div style="font-size:11px;color:#64748b;">角色: {char_display}</div>
                {action_items}
                {dialogue_items}
            </div>
            """

    page_count = len(pages) if pages else len(legacy_scenes)
    total_panels = sum(len(p.get("panels", [])) for p in pages) if pages else len(legacy_scenes)

    return f"""
    <div class="cw-phase-text-panel">
        <h4>📝 剧本: {title}</h4>
        <div class="cw-phase-text-scroll">
        <div class="cw-phase-text-item" style="margin-bottom:6px;">{summary}</div>
        <div class="cw-phase-text-item">{genre_tags}</div>
        <div class="cw-phase-text-item" style="margin-top:4px;">
            <span class="key">页数:</span> <span class="val">{page_count}</span>
            &nbsp;&nbsp;<span class="key">分格:</span> <span class="val">{total_panels}</span>
            &nbsp;&nbsp;<span class="key">角色数:</span> <span class="val">{len(characters)}</span>
        </div>
        <details style="margin-top:6px;">
            <summary style="cursor:pointer;color:#3b82f6;font-size:13px;font-weight:600;">
                👥 角色详情 ({len(characters)}人)
            </summary>
            {char_list}
        </details>
        <details style="margin-top:4px;" open>
            <summary style="cursor:pointer;color:#3b82f6;font-size:13px;font-weight:600;">
                🎬 分页面板详情 ({page_count}页 · {total_panels}格)
            </summary>
            {panel_items}
        </details>
        </div>
    </div>
    """
