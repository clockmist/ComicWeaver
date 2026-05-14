"""HTML 渲染辅助函数 - 生成富前端展示。"""
from __future__ import annotations

import html
import time


_PHASE_AGENTS = [
    ("script_agent", "📝 剧本"),
    ("character_agent", "👤 角色"),
    ("storyboard_agent", "🎬 分镜"),
    ("image_agent", "🎨 图像"),
    ("layout_agent", "📐 排版"),
    ("reviewer_agent", "✅ 审查"),
]


def render_agent_grid(active: str = "", done: list[str] | None = None) -> str:
    """渲染6个Agent的状态卡片网格。"""
    done = done or []
    cells = []
    for agent_id, label in _PHASE_AGENTS:
        cls = "cw-agent-pill"
        if agent_id == "reviewer_agent":
            cls += " review" if active == agent_id else ""
        if agent_id == active:
            cls += " active"
        elif agent_id in done:
            cls += " done"
        cells.append(f'<div class="{cls}">{label}</div>')
    return f'<div class="cw-agent-grid">{"".join(cells)}</div>'


def render_score_bars(dim_scores: dict[str, float]) -> str:
    """渲染审查评分条。"""
    if not dim_scores:
        return '<div style="color:#64748b;font-size:13px;">尚无评分数据</div>'

    bars = []
    for dim, score in dim_scores.items():
        cls = "fill"
        if score < 4:
            cls += " low"
        elif score < 7:
            cls += " mid"
        pct = max(0, min(100, score * 10))
        bars.append(f"""
        <div class="cw-score-bar">
            <div class="label">{html.escape(dim)}</div>
            <div class="bar"><div class="{cls}" style="width:{pct}%"></div></div>
            <div class="value">{score:.1f}</div>
        </div>
        """)
    return f'<div class="cw-card">{"".join(bars)}</div>'


def render_log(events: list[dict], max_lines: int = 50) -> str:
    """渲染日志(类似终端输出)。"""
    lines = []
    for evt in events[-max_lines:]:
        ts = time.strftime("%H:%M:%S", time.localtime(evt.get("timestamp", time.time())))
        agent = evt.get("agent", "")
        evt_type = evt.get("type", "log")
        content = evt.get("content")

        css_cls = f"log-{evt_type}"
        prefix = f"[{ts}] [{agent}]"

        if evt_type == "progress":
            try:
                pct = float(content) * 100
                msg = f"进度 {pct:.0f}%"
            except (TypeError, ValueError):
                msg = str(content)
        elif evt_type == "thinking":
            msg = f"💭 {content}"
        elif evt_type == "done":
            msg = "✓ 完成"
        elif evt_type == "error":
            msg = f"✗ {content}"
        elif evt_type == "checkpoint":
            msg = f"⏸  {content}"
            css_cls = "log-checkpoint"
        else:
            msg = str(content) if content is not None else ""
        lines.append(f'<div class="{css_cls}">{html.escape(prefix)} {html.escape(msg)}</div>')

    body = "\n".join(lines) if lines else '<div style="color:#64748b;">等待开始...</div>'
    return f'<div class="cw-log">{body}</div>'


def render_checkpoint(pending: dict | None) -> str:
    """渲染当前确认点提示。"""
    if not pending:
        return '<div style="color:#64748b;font-size:13px;padding:12px;">当前没有等待确认的节点</div>'

    label = pending.get("label", "确认")
    payload = pending.get("payload", {})
    preview_lines = []
    for k, v in payload.items():
        if isinstance(v, list):
            v_str = f"{len(v)} 项" if len(v) > 5 else ", ".join(str(x) for x in v)
        else:
            v_str = str(v)
        preview_lines.append(f"<div><b>{html.escape(str(k))}:</b> {html.escape(v_str)}</div>")
    preview_html = "".join(preview_lines)

    return f"""
    <div class="cw-checkpoint">
        <h3>⏸ {html.escape(label)}</h3>
        <div style="font-size:13px;color:#78350f;">
            生产Agent已完成本阶段输出,审查Agent已通过质量检查。
            请确认是否继续下一阶段。
        </div>
        <div class="preview">{preview_html}</div>
    </div>
    """


def render_script_summary(script: dict) -> str:
    """渲染剧本摘要卡片。"""
    if not script:
        return '<div style="color:#64748b;padding:12px;">尚未生成剧本</div>'

    title = html.escape(script.get("title", "Untitled"))
    summary = html.escape(script.get("summary", ""))
    chars = script.get("characters", [])
    scenes = script.get("scenes", [])
    genres = script.get("genre", [])

    char_html = "".join(
        f'<span class="cw-badge" style="margin-right:6px;">'
        f'{html.escape(c.get("name", ""))} · {html.escape(c.get("role", ""))}</span>'
        for c in chars
    )

    genre_html = " ".join(
        f'<span class="cw-badge">{html.escape(g)}</span>' for g in genres
    )

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 8px 0;color:#1e293b;">📖 {title}</h3>
        <div style="color:#64748b;font-size:13px;margin-bottom:12px;">{summary}</div>
        <div style="margin-bottom:8px;">{genre_html}</div>
        <div style="font-size:13px;margin-bottom:6px;"><b>角色:</b></div>
        <div style="margin-bottom:12px;">{char_html}</div>
        <div style="display:flex;gap:24px;font-size:13px;color:#475569;">
            <div><b>{len(scenes)}</b> 场景</div>
            <div><b>{len(chars)}</b> 角色</div>
            <div><b>{len(script.get('emotion_curve', []))}</b> 情感节点</div>
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
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # 高潮点标记
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

        # 渲染框架预览
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
                <b>第 {page_num} 页</b> · {html.escape(template)} {climax_badge}
            </div>
            <svg width="{scale}" height="{int(scale * 1.4)}" style="background:#f8fafc;border:1px solid #cbd5e1;">
                {''.join(frames_svg)}
            </svg>
        </div>
        """)

    return f'<div style="white-space:nowrap;overflow-x:auto;">{"".join(pages_html)}</div>'


def render_review_history(reviews: list[dict]) -> str:
    """渲染审查历史时间线。"""
    if not reviews:
        return '<div style="color:#64748b;padding:12px;">审查记录为空</div>'

    rows = []
    for r in reviews:
        decision = r.get("decision", "?")
        score = r.get("score", 0)
        agent = r.get("agent", "?")
        color = {
            "pass": "#10b981",
            "revise": "#f59e0b",
            "escalate": "#ef4444",
        }.get(decision, "#94a3b8")
        icon = {"pass": "✓", "revise": "↻", "escalate": "⚠"}.get(decision, "•")
        issues = r.get("issues", [])
        issue_text = "; ".join(issues[:2]) if issues else "无明显问题"
        rows.append(f"""
        <tr>
            <td style="padding:6px 12px;color:{color};font-weight:600;">{icon} {decision.upper()}</td>
            <td style="padding:6px 12px;">{html.escape(agent)}</td>
            <td style="padding:6px 12px;font-weight:600;">{score:.1f}</td>
            <td style="padding:6px 12px;font-size:12px;color:#64748b;">{html.escape(issue_text)}</td>
        </tr>
        """)
    return f"""
    <div class="cw-card">
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
                <tr style="background:#f8fafc;color:#475569;">
                    <th style="padding:8px 12px;text-align:left;">决策</th>
                    <th style="padding:8px 12px;text-align:left;">Agent</th>
                    <th style="padding:8px 12px;text-align:left;">评分</th>
                    <th style="padding:8px 12px;text-align:left;">问题摘要</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """
