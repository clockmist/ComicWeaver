"""分镜规划渲染：缩略图、详细表格、SVG页面预览。"""
from __future__ import annotations

import html as html_mod

from .common import _resolve_char_name


def render_storyboard_preview(plan: list[dict]) -> str:
    """渲染分镜规划缩略图。"""
    if not plan:
        return '<div style="color:#64748b;padding:12px;">尚未生成分镜</div>'

    pages_html = []
    for page in plan:
        page_num = page.get("page_number", "?")
        template = page.get("layout_template", "")
        is_climax = page.get("is_climax_page", False)
        panels = page.get("panels", [])

        frames_svg = []
        scale = 200
        for p in panels:
            bbox = p.get("bbox", {})
            x = bbox.get("x", 0) * scale
            y = bbox.get("y", 0) * scale * 1.4
            w = bbox.get("width", 0) * scale
            h = bbox.get("height", 0) * scale * 1.4
            shot = p.get("shot_size", "")
            frames_svg.append(
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" '
                f'fill="#dbeafe" stroke="#1e40af" stroke-width="2"/>'
                f'<text x="{x + 4:.0f}" y="{y + 14:.0f}" font-size="9" fill="#1e3a8a">'
                f'{p.get("order_in_page", "?")}·{shot[:6]}</text>'
            )

        climax_badge = (
            '<span class="cw-badge" style="background:#fee2e2;color:#991b1b;">高潮页</span>'
            if is_climax else ""
        )

        pages_html.append(f"""
        <div class="cw-card" style="display:inline-block;margin:6px;vertical-align:top;">
            <div style="margin-bottom:6px;font-size:12px;color:#1e293b;">
                <b>第 {page_num} 页</b> · {html_mod.escape(template)} {climax_badge}
            </div>
            <svg width="{scale}" height="{int(scale * 1.4)}" style="background:#f8fafc;border:1px solid #cbd5e1;">
                {''.join(frames_svg)}
            </svg>
        </div>
        """)

    return f'<div style="white-space:nowrap;overflow-x:auto;">{"".join(pages_html)}</div>'


def render_storyboard_detail(plan: list[dict]) -> str:
    """渲染分镜详细规划（每面板包含 shot/angle/action/prompt）。"""
    if not plan:
        return (
            '<div style="color:#64748b;padding:12px;">分镜数据尚未生成。<br>'
            '请在「创作流程」Tab 运行工作流后查看。</div>'
        )

    panels_html = []
    for page in plan:
        page_num = page.get("page_number", "?")
        template = page.get("layout_template", "?")
        is_climax = page.get("is_climax_page", False)
        panels = page.get("panels", [])

        panel_rows = []
        for pn in panels:
            prompt_pack = pn.get("prompt_pack", {})
            chars = ", ".join(pn.get("characters_in_panel", [])) or "无"
            bubbles = pn.get("speech_bubble_hints", [])
            bubble_texts = "; ".join(
                f'{b.get("dialogue_index", "?")}:{b.get("bubble_type", "speech")}'
                for b in bubbles[:3]
            ) or "无对话"

            panel_rows.append(f"""
            <tr>
                <td style="padding:4px 8px;font-size:12px;font-family:monospace;">
                    {html_mod.escape(pn.get('panel_id', '?'))}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    {pn.get('order_in_page', '?')}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    <span class="cw-badge">{html_mod.escape(pn.get('shot_size', '?'))}</span></td>
                <td style="padding:4px 8px;font-size:12px;">
                    {html_mod.escape(pn.get('camera_angle', '?'))}</td>
                <td style="padding:4px 8px;font-size:12px;max-width:140px;
                           overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                    title="{html_mod.escape(pn.get('primary_action', ''))}">
                    {html_mod.escape((pn.get('primary_action', '') or '')[:40])}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    {pn.get('emotion_intensity', 0):.2f}</td>
                <td style="padding:4px 8px;font-size:11px;color:#64748b;">{html_mod.escape(chars)}</td>
                <td style="padding:4px 8px;font-size:11px;max-width:200px;
                           overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                    title="{html_mod.escape(prompt_pack.get('positive_prompt', ''))}">
                    {html_mod.escape((prompt_pack.get('positive_prompt', '') or '')[:60])}</td>
            </tr>
            """)

        climax_badge = (
            '<span class="cw-badge" style="background:#fee2e2;color:#991b1b;">高潮页</span>'
            if is_climax else ""
        )

        panels_html.append(f"""
        <div class="cw-card" style="margin-bottom:12px;">
            <h4 style="margin:0 0 8px 0;color:#1e293b;">
                第 {page_num} 页 · {html_mod.escape(template)} {climax_badge}
                <span style="font-size:12px;color:#64748b;">
                    ({len(panels)} 面板 · 平均情绪 {page.get('page_emotion_avg', 0):.2f})
                </span>
            </h4>
            <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:12px;">
                    <thead>
                        <tr style="background:#f8fafc;color:#475569;">
                            <th style="padding:6px 8px;text-align:left;">面板ID</th>
                            <th style="padding:6px 8px;text-align:left;">序号</th>
                            <th style="padding:6px 8px;text-align:left;">景别</th>
                            <th style="padding:6px 8px;text-align:left;">角度</th>
                            <th style="padding:6px 8px;text-align:left;">动作</th>
                            <th style="padding:6px 8px;text-align:left;">情绪</th>
                            <th style="padding:6px 8px;text-align:left;">角色</th>
                            <th style="padding:6px 8px;text-align:left;">提示词</th>
                        </tr>
                    </thead>
                    <tbody>{''.join(panel_rows)}</tbody>
                </table>
            </div>
        </div>
        """)

    return f'<div>{"".join(panels_html)}</div>'


def _render_page_svg(page_data: dict) -> str:
    """渲染单个页面的故事板 SVG 缩略图。"""
    panels = page_data.get("panels", []) if isinstance(page_data, dict) else getattr(page_data, "panels", [])
    if not panels:
        return '<div style="color:#94a3b8;padding:20px;">无面板数据</div>'

    svg_w = 300
    svg_h = 420
    rects = []
    for i, pnl in enumerate(panels):
        bbox = pnl.get("bbox", {}) if isinstance(pnl, dict) else getattr(pnl, "bbox", None)
        if bbox:
            x = (bbox.get("x", 0) if isinstance(bbox, dict) else getattr(bbox, "x", 0)) * svg_w
            y = (bbox.get("y", 0) if isinstance(bbox, dict) else getattr(bbox, "y", 0)) * svg_h
            w = (bbox.get("width", 0.25) if isinstance(bbox, dict) else getattr(bbox, "width", 0.25)) * svg_w
            h = (bbox.get("height", 0.25) if isinstance(bbox, dict) else getattr(bbox, "height", 0.25)) * svg_h
        else:
            cols = 2
            rows = (len(panels) + 1) // 2
            col = i % cols
            row = i // cols
            cell_w = svg_w / cols
            cell_h = svg_h / rows
            x = col * cell_w + 4
            y = row * cell_h + 4
            w = cell_w - 8
            h = cell_h - 8

        order = pnl.get("order_in_panel", i + 1) if isinstance(pnl, dict) else getattr(pnl, "order_in_panel", i + 1)
        shot = pnl.get("shot_size", "") if isinstance(pnl, dict) else getattr(pnl, "shot_size", "")
        shot_label = str(shot)[:2] if shot else "?"

        rects.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="rgba(59,130,246,0.15)" stroke="#3b82f6" stroke-width="1.5" rx="2"/>'
            f'<text x="{x + w/2}" y="{y + h/2 - 6}" text-anchor="middle" '
            f'font-size="10" fill="#1e40af">{order}</text>'
            f'<text x="{x + w/2}" y="{y + h/2 + 8}" text-anchor="middle" '
            f'font-size="8" fill="#64748b">{shot_label}</text>'
        )

    return f"""
    <svg width="{svg_w}" height="{svg_h}" style="border:1px solid #e2e8f0;border-radius:4px;
         background:#fafbfc;">
        <rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#fafbfc" rx="4"/>
        {"".join(rects)}
    </svg>
    """


def _render_detailed_storyboard(storyboard: list[dict], char_map: dict[str, str]) -> str:
    """渲染完整分镜规划：每页每格的对话、动作、角色、镜头。"""
    if not storyboard:
        return ""

    total_panels = sum(len(p.get("panels", [])) for p in storyboard)
    pages_html = ""

    for p in storyboard:
        page_num = p.get("page_number", "?")
        template = html_mod.escape(p.get("layout_template", ""))
        layout_hint = html_mod.escape(p.get("layout_hint", ""))
        panels = p.get("panels", [])
        is_climax = p.get("is_climax_page", False)
        climax_mark = ' 🔥 高潮页' if is_climax else ''
        emotion_avg = p.get("page_emotion_avg", 0)

        panel_items = ""
        for pn in panels:
            pid = html_mod.escape(pn.get("panel_id", "?"))
            order = pn.get("order_in_page", "?")
            shot = html_mod.escape(pn.get("shot_size", "?"))
            angle = html_mod.escape(pn.get("camera_angle", "?"))
            shape = html_mod.escape(pn.get("shape", "rectangle"))
            mood = html_mod.escape(pn.get("mood", ""))
            emotion = pn.get("emotion_intensity", 0)
            prompt_pack = pn.get("prompt_pack", {})
            prompt_text = html_mod.escape(
                (prompt_pack.get("positive_prompt") or pn.get("primary_action") or "")[:300]
            )

            chars_in_panel = pn.get("characters_in_panel", [])
            char_display = ", ".join(
                html_mod.escape(_resolve_char_name(cid, char_map))
                for cid in chars_in_panel
            ) if chars_in_panel else "无"

            dialogues = pn.get("dialogues_in_panel", [])
            dialogue_items = ""
            for d in dialogues:
                speaker = html_mod.escape(_resolve_char_name(d.get("speaker", "?"), char_map))
                text = html_mod.escape(d.get("text", "")[:100])
                tone = html_mod.escape(d.get("tone", ""))
                is_thought = d.get("is_thought", False)
                icon = "💭" if is_thought else "💬"
                tone_tag = f' [{tone}]' if tone else ""
                dialogue_items += (
                    f'<div style="margin-left:4px;font-size:11px;">'
                    f'{icon} <b>{speaker}</b>{tone_tag}: {text}</div>'
                )

            bubble_hints = pn.get("speech_bubble_hints", [])
            bubble_items = ""
            if bubble_hints:
                for bh in bubble_hints[:3]:
                    btype = html_mod.escape(bh.get("bubble_type", "speech"))
                    pos = html_mod.escape(bh.get("suggested_position", "auto"))
                    bubble_items += f'<span class="cw-badge" style="font-size:9px;">{btype}@{pos}</span> '

            panel_items += f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                        padding:6px 8px;margin:4px 0;font-size:11px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <b>{pid}</b> (第{order}格)
                    <span style="font-size:10px;color:#94a3b8;">情绪 {emotion:.2f}</span>
                </div>
                <div style="color:#475569;margin:2px 0;">
                    <span class="cw-badge">{shot}</span>
                    <span class="cw-badge">{angle}</span>
                    <span class="cw-badge" style="background:#e0e7ff;color:#3730a3;">{shape}</span>
                    {f'<span class="cw-badge" style="background:#fef3c7;color:#92400e;">{mood}</span>' if mood else ''}
                </div>
                <div><span class="key">角色:</span> {char_display}</div>
                {f'<div style="margin-top:2px;"><span class="key">Prompt</span> <span style="font-size:10px;color:#64748b;word-break:break-all;">{prompt_text}</span></div>' if prompt_text else ''}
                {f'<div style="margin-top:2px;">{dialogue_items}</div>' if dialogue_items else ''}
                {f'<div style="margin-top:2px;">💭气泡: {bubble_items}</div>' if bubble_items else ''}
            </div>
            """

        pages_html += f"""
        <div style="margin-bottom:8px;">
            <div style="font-weight:600;color:#1e293b;font-size:13px;margin-bottom:4px;
                        padding:4px 8px;background:#e2e8f0;border-radius:4px;">
                第 {page_num} 页 · {template}
                {f' · {layout_hint}' if layout_hint else ''}
                {climax_mark}
                <span style="font-weight:normal;font-size:11px;color:#64748b;">
                    ({len(panels)}格 · 平均情绪 {emotion_avg:.2f})
                </span>
            </div>
            {panel_items}
        </div>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>🎬 分镜规划 ({len(storyboard)}页 / {total_panels}格)</h4>
        <div class="cw-phase-text-scroll">
        {pages_html}
        </div>
    </div>
    """
