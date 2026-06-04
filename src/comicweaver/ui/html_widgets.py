"""HTML 渲染辅助函数 - 生成富前端展示。"""
from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any


def _gradio_img_src(file_path: str) -> str:
    """将绝对/相对文件路径转为 Gradio 可用的 img src URL。

    Gradio 6.x 的 /gradio_api/file= 需要相对于工作目录的路径。
    """
    try:
        rel = os.path.relpath(str(file_path), os.getcwd())
    except (ValueError, OSError):
        rel = str(file_path)
    return rel.replace("\\", "/")


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
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))

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


# ============================================================================
# 增强：Agent 输出面板
# ============================================================================


def _dict_preview(d: dict, max_depth: int = 2, current_depth: int = 0) -> str:
    """递归渲染字典为缩进 HTML，键带颜色、值截断。"""
    if current_depth >= max_depth or not isinstance(d, dict):
        if isinstance(d, str) and len(d) > 200:
            return html.escape(d[:200] + "...")
        return html.escape(str(d)[:200])
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> '
                f'<span style="color:#64748b;">{{... {len(v)} keys}}</span>'
                f'{_dict_preview(v, max_depth, current_depth + 1)}'
                f'</div>'
            )
        elif isinstance(v, list):
            if len(v) == 0:
                lines.append(f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> []')
            elif isinstance(v[0], dict):
                items = "".join(
                    f'<details style="margin-left:8px;"><summary style="cursor:pointer;color:#3b82f6;">'
                    f'[{i}]</summary>{_dict_preview(item, max_depth, current_depth + 1)}</details>'
                    for i, item in enumerate(v[:15])
                )
                more = f' ... +{len(v) - 15} more' if len(v) > 15 else ""
                lines.append(
                    f'<div style="margin-left:{current_depth*16}px;">'
                    f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> [{len(v)} items]{more}'
                    f'{items}</div>'
                )
            else:
                preview = ", ".join(html.escape(str(x)) for x in v[:10])
                more = f" ... +{len(v) - 10}" if len(v) > 10 else ""
                lines.append(
                    f'<div style="margin-left:{current_depth*16}px;">'
                    f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> '
                    f'[{preview}{more}]</div>'
                )
        elif isinstance(v, str):
            display = v[:100] + "..." if len(v) > 100 else v
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> '
                f'<span style="color:#334155;">{html.escape(display)}</span></div>'
            )
        else:
            lines.append(
                f'<div style="margin-left:{current_depth*16}px;">'
                f'<b style="color:#1e40af;">{html.escape(str(k))}:</b> '
                f'<span style="color:#0f172a;">{html.escape(str(v))}</span></div>'
            )
    return "\n".join(lines)


def render_agent_outputs(agent_outputs: dict[str, dict]) -> str:
    """渲染所有 Agent 的输出面板（可折叠卡片）。"""
    if not agent_outputs:
        return '<div style="color:#64748b;padding:12px;">工作流尚未运行，无 Agent 输出数据。<br>请在「创作流程」Tab 启动工作流后再查看。</div>'

    agents_order = [
        ("script_agent", "📝 剧本 Agent", "剧本理解"),
        ("character_agent", "👤 角色 Agent", "角色管理"),
        ("storyboard_agent", "🎬 分镜 Agent", "分镜设计"),
        ("image_agent", "🎨 图像 Agent", "面板图像生成"),
        ("layout_agent", "📐 排版 Agent", "页面排版合成"),
        ("reviewer_agent", "✅ 审查 Agent", "质量审查"),
    ]

    cards = []
    for agent_id, label, subtitle in agents_order:
        outputs = agent_outputs.get(agent_id, [])
        card_content = ""
        if not outputs:
            card_content = '<div style="color:#94a3b8;font-size:13px;padding:8px;">暂无输出数据</div>'
        else:
            for i, out in enumerate(outputs):
                out_id = out.get("panel_id") or out.get("title") or f"#{i + 1}"
                output_data = out.get("output", {})
                card_content += (
                    f'<details style="margin-bottom:4px;" open>'
                    f'<summary style="cursor:pointer;font-size:13px;font-weight:600;'
                    f'color:#1e293b;padding:6px 0;">'
                    f'{html.escape(str(out_id))}</summary>'
                    f'<div style="font-size:12px;font-family:monospace;background:#f8fafc;'
                    f'padding:8px 12px;border-radius:6px;max-height:400px;overflow-y:auto;">'
                    f'{_dict_preview(output_data, max_depth=3)}'
                    f'</div></details>'
                )

        cards.append(f"""
        <div class="cw-agent-output-card">
            <div class="cw-agent-output-header">
                <span style="font-size:18px;">{label}</span>
                <span style="font-size:12px;color:#64748b;">{subtitle}</span>
            </div>
            <div style="padding:8px 12px;">{card_content}</div>
        </div>
        """)

    return f'<div style="display:flex;flex-direction:column;gap:12px;">{"".join(cards)}</div>'


def render_dev_log(dev_entries: list[dict], filter_category: str = "all",
                   filter_level: str = "all", max_entries: int = 100) -> str:
    """渲染开发者日志查看器（带筛选）。"""
    if not dev_entries:
        return '<div style="color:#64748b;padding:12px;">暂无开发者日志。<br>请在「创作流程」Tab 启动工作流后自动生成。</div>'

    # 筛选
    filtered = []
    for e in dev_entries:
        if filter_category != "all" and e.get("category", "") != filter_category:
            continue
        if filter_level != "all" and e.get("level", "") != filter_level:
            continue
        filtered.append(e)

    entries_to_show = filtered[-max_entries:]
    if not entries_to_show:
        return '<div style="color:#64748b;padding:12px;">当前筛选条件下无匹配日志。</div>'

    # 分类映射颜色
    cat_colors: dict[str, str] = {
        "agent_input": "#3b82f6",
        "agent_output": "#10b981",
        "state_change": "#8b5cf6",
        "review_decision": "#f59e0b",
        "error": "#ef4444",
        "performance": "#06b6d4",
        "checkpoint": "#f97316",
        "workflow": "#64748b",
    }
    level_icons: dict[str, str] = {
        "trace": "·",
        "debug": "🔍",
        "info": "ℹ️",
        "warn": "⚠️",
        "error": "❌",
        "perf": "⏱",
    }

    lines = []
    for e in entries_to_show:
        ts = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", time.time())))
        cat = e.get("category", "?")
        lvl = e.get("level", "info")
        agent = e.get("agent", "?")
        msg = html.escape(str(e.get("message", ""))[:200])
        dur = e.get("duration_ms", 0)
        dur_str = f" [{dur:.0f}ms]" if dur > 0 else ""
        color = cat_colors.get(cat, "#94a3b8")
        icon = level_icons.get(lvl, "·")

        # 可点击展开详细数据
        data_dict = e.get("data", {})
        data_json = json.dumps(data_dict, ensure_ascii=False, default=str) if data_dict else ""
        data_preview = ""
        if data_json:
            data_preview = (
                f'<details style="margin-left:16px;font-size:11px;">'
                f'<summary style="cursor:pointer;color:{color};">展开数据</summary>'
                f'<pre style="background:#0f172a;color:#e2e8f0;padding:8px;border-radius:4px;'
                f'max-height:200px;overflow:auto;font-size:11px;">'
                f'{html.escape(data_json[:2000])}'
                f'</pre></details>'
            )

        lines.append(
            f'<div style="padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:12px;">'
            f'<span style="color:#94a3b8;">{ts}</span> '
            f'<span style="color:{color};font-weight:600;">{icon} [{cat}]</span> '
            f'<span style="color:#1e293b;">[{html.escape(agent)}]</span> '
            f'{msg}{dur_str}'
            f'{data_preview}'
            f'</div>'
        )

    stats = (
        f'<div style="font-size:12px;color:#64748b;margin-bottom:8px;">'
        f'显示 {len(entries_to_show)}/{len(dev_entries)} 条日志'
        f'</div>'
    )
    body = "\n".join(lines)
    return f'<div class="cw-dev-log">{stats}<div style="max-height:500px;overflow-y:auto;">{body}</div></div>'


def render_performance_summary(dev_entries: list[dict]) -> str:
    """渲染性能统计表格。"""
    perf_entries = [
        e for e in dev_entries
        if e.get("category") == "performance"
    ]
    if not perf_entries:
        return '<div style="color:#64748b;font-size:13px;padding:12px;">暂无性能数据（需运行工作流后生成）</div>'

    rows = []
    total_ms = 0.0
    for e in perf_entries:
        agent = e.get("agent", "?")
        dur = e.get("duration_ms", 0)
        total_ms += dur
        dur_str = f"{dur:.0f}ms" if dur < 1000 else f"{dur / 1000:.1f}s"
        data = e.get("data", {})
        retries = data.get("retry_count", 0)
        status = data.get("status", "ok")
        status_icon = {"ok": "✅", "partial": "⚠️"}.get(status, "✅")

        rows.append(f"""
        <tr>
            <td style="padding:6px 12px;text-align:center;">{status_icon}</td>
            <td style="padding:6px 12px;font-weight:500;">{html.escape(agent)}</td>
            <td style="padding:6px 12px;text-align:right;font-family:monospace;">{dur_str}</td>
            <td style="padding:6px 12px;text-align:center;">{retries}</td>
        </tr>
        """)

    total_str = f"{total_ms:.0f}ms" if total_ms < 1000 else f"{total_ms / 1000:.1f}s"
    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">⏱ 性能统计</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
                <tr style="background:#f8fafc;color:#475569;">
                    <th style="padding:8px 12px;text-align:center;">状态</th>
                    <th style="padding:8px 12px;text-align:left;">Agent</th>
                    <th style="padding:8px 12px;text-align:right;">耗时</th>
                    <th style="padding:8px 12px;text-align:center;">重试</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
            <tfoot>
                <tr style="font-weight:700;border-top:2px solid #e2e8f0;">
                    <td colspan="2" style="padding:8px 12px;text-align:right;">总计</td>
                    <td style="padding:8px 12px;text-align:right;font-family:monospace;">{total_str}</td>
                    <td></td>
                </tr>
            </tfoot>
        </table>
    </div>
    """


def render_character_profiles(character_db: dict | None,
                               agent_outputs: dict[str, list[dict]] | None = None) -> str:
    """渲染角色档案卡片（含参考图）。"""
    if not character_db:
        return '<div style="color:#64748b;padding:12px;">角色数据尚未生成。<br>请在「创作流程」Tab 运行工作流后查看。</div>'

    characters = character_db.get("characters", {})
    if not characters:
        return '<div style="color:#64748b;padding:12px;">角色列表为空</div>'

    cards = []
    for char_id, profile in characters.items():
        name = html.escape(profile.get("name", char_id))
        traits = profile.get("visual_traits", {})
        ref = profile.get("base_reference", {})

        # 视觉特征
        trait_items = []
        for tk, tv in traits.items():
            if tv:
                trait_items.append(f'<span class="cw-badge" style="margin:2px;">{html.escape(tk)}: {html.escape(str(tv))}</span>')

        # 参考图
        img_path = ref.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<div style="text-align:center;margin:8px 0;">'
                f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                f'style="max-width:150px;max-height:200px;border:1px solid #e2e8f0;'
                f'border-radius:8px;" alt="{name} 参考图"/>'
                f'</div>'
            )

        # 元数据
        meta_info = (
            f'<div style="font-size:11px;color:#94a3b8;">'
            f'来源: {ref.get("source", "?")} · '
            f'置信度: {ref.get("confidence", 0):.2f} · '
            f'锁定: {"是" if profile.get("locked") else "否"}'
            f'</div>'
        )

        cards.append(f"""
        <div class="cw-character-card" style="display:inline-block;width:220px;
                    margin:6px;vertical-align:top;padding:12px;">
            <h4 style="margin:0 0 4px 0;color:#1e293b;">{name}</h4>
            <div style="font-size:11px;color:#64748b;margin-bottom:8px;">ID: {html.escape(char_id)}</div>
            {img_html}
            <div style="margin:6px 0;">{''.join(trait_items)}</div>
            {meta_info}
        </div>
        """)

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">👤 角色档案 ({len(characters)})</h3>
        <div style="display:flex;flex-wrap:wrap;">{''.join(cards)}</div>
    </div>
    """


def render_panel_images_gallery(panel_images: list[dict]) -> str:
    """渲染单张面板图像画廊（非最终合成页）。"""
    if not panel_images:
        return '<div style="color:#64748b;padding:12px;">面板图像尚未生成。<br>请在「创作流程」Tab 运行工作流后查看。</div>'

    items = []
    for pi in panel_images:
        pid = pi.get("panel_id", "?")
        path = pi.get("image_path", "")
        backend = pi.get("backend", "?")
        gen_time = pi.get("generation_time_ms", 0)
        prompt = pi.get("prompt_used", "")[:80] or ""

        img_html = ""
        if path and os.path.exists(str(path)):
            safe_path = _gradio_img_src(str(path))
            img_html = (
                f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                f'style="width:100%;height:180px;object-fit:cover;border-radius:6px;" '
                f'alt="{html.escape(pid)}" loading="lazy"/>'
            )
        else:
            img_html = (
                f'<div style="width:100%;height:180px;background:#f1f5f9;border-radius:6px;'
                f'display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:12px;">'
                f'无图像</div>'
            )

        items.append(f"""
        <div style="display:inline-block;width:200px;margin:6px;vertical-align:top;
                    background:white;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">
            {img_html}
            <div style="padding:8px;">
                <div style="font-size:12px;font-weight:600;color:#1e293b;"
                     title="{html.escape(pid)}">{html.escape(pid)}</div>
                <div style="font-size:11px;color:#64748b;">后端: {html.escape(backend)} · {gen_time}ms</div>
                <div style="font-size:10px;color:#94a3b8;margin-top:2px;overflow:hidden;
                            text-overflow:ellipsis;white-space:nowrap;"
                     title="{html.escape(prompt)}">{html.escape(prompt)}</div>
            </div>
        </div>
        """)

    return f"""
    <div class="cw-card">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">🎨 面板图像 ({len(panel_images)})</h3>
        <div style="white-space:nowrap;overflow-x:auto;padding:4px;">{''.join(items)}</div>
    </div>
    """


def render_project_info(state: dict | None = None) -> str:
    """渲染当前项目信息栏（创作流程 Tab 顶部）。"""
    if not state:
        return '<div style="color:#94a3b8;padding:8px;font-size:12px;">尚未创建项目</div>'

    title = state.get("title", "") or "未命名"
    pid = state.get("project_id", "?")
    phase = state.get("current_phase", "init")
    phase_labels = {
        "init": "等待开始", "script": "剧本阶段", "character": "角色阶段",
        "storyboard": "分镜阶段", "image": "图像阶段", "layout": "排版阶段",
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


def render_live_panel_preview(
    panel_images: list[dict],
    character_images: list[dict] | None = None,
) -> str:
    """渲染实时预览区：角色人设图 + 面板图像。"""
    if character_images is None:
        character_images = []

    has_chars = bool(character_images)
    has_panels = bool(panel_images)

    if not has_chars and not has_panels:
        return '<div style="color:#64748b;padding:8px;font-size:12px;">⏳ 等待图像生成...</div>'

    parts: list[str] = []

    # 角色人设图
    if has_chars:
        char_items = []
        for ci in character_images:
            name = ci.get("name", ci.get("char_id", "?"))
            path = ci.get("image_path", "")
            img_html = _preview_thumbnail(path, str(name))
            char_items.append(f"""
            <div style="display:inline-block;width:140px;margin:4px;vertical-align:top;
                        background:white;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">
                {img_html}
                <div style="padding:4px 6px;">
                    <div style="font-size:11px;font-weight:600;color:#1e293b;">👤 {html.escape(str(name))}</div>
                </div>
            </div>
            """)
        parts.append(
            f'<div style="margin-bottom:6px;font-size:11px;color:#64748b;">'
            f'👤 角色人设 ({len(character_images)})</div>'
            f'<div style="white-space:nowrap;overflow-x:auto;padding:4px;">{"".join(char_items)}</div>'
        )

    # 面板图像
    if has_panels:
        panel_items = []
        for pi in panel_images:
            pid = pi.get("panel_id", "?") if isinstance(pi, dict) else getattr(pi, "panel_id", "?")
            path = pi.get("image_path", "") if isinstance(pi, dict) else getattr(pi, "image_path", "")
            img_html = _preview_thumbnail(str(path), str(pid))
            panel_items.append(f"""
            <div style="display:inline-block;width:140px;margin:4px;vertical-align:top;
                        background:white;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">
                {img_html}
                <div style="padding:4px 6px;">
                    <div style="font-size:11px;font-weight:600;color:#1e293b;">🎨 {html.escape(str(pid))}</div>
                </div>
            </div>
            """)
        parts.append(
            f'<div style="margin-bottom:6px;font-size:11px;color:#64748b;">'
            f'🎨 面板图像 ({len(panel_images)})</div>'
            f'<div style="white-space:nowrap;overflow-x:auto;padding:4px;">{"".join(panel_items)}</div>'
        )

    return f"""
    <div class="cw-live-preview">
        {"".join(parts)}
    </div>
    """


def _preview_thumbnail(path: str, alt: str) -> str:
    """单个预览缩略图（120px 高，cover 裁剪）。"""
    if path and os.path.exists(str(path)):
        safe_path = _gradio_img_src(str(path))
        return (
            f'<img src="/gradio_api/file={html.escape(safe_path)}" '
            f'style="width:100%;height:120px;object-fit:cover;border-radius:4px;" '
            f'alt="{html.escape(alt)}" loading="lazy"/>'
        )
    return (
        f'<div style="width:100%;height:120px;background:#f1f5f9;border-radius:4px;'
        f'display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:10px;">'
        f'无图像</div>'
    )


def render_storyboard_detail(plan: list[dict]) -> str:
    """渲染分镜详细规划（每面板包含 shot/angle/action/prompt）。"""
    if not plan:
        return '<div style="color:#64748b;padding:12px;">分镜数据尚未生成。<br>请在「创作流程」Tab 运行工作流后查看。</div>'

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
                    {html.escape(pn.get('panel_id', '?'))}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    {pn.get('order_in_page', '?')}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    <span class="cw-badge">{html.escape(pn.get('shot_size', '?'))}</span></td>
                <td style="padding:4px 8px;font-size:12px;">
                    {html.escape(pn.get('camera_angle', '?'))}</td>
                <td style="padding:4px 8px;font-size:12px;max-width:140px;
                           overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                    title="{html.escape(pn.get('primary_action', ''))}">
                    {html.escape((pn.get('primary_action', '') or '')[:40])}</td>
                <td style="padding:4px 8px;font-size:12px;">
                    {pn.get('emotion_intensity', 0):.2f}</td>
                <td style="padding:4px 8px;font-size:11px;color:#64748b;">{html.escape(chars)}</td>
                <td style="padding:4px 8px;font-size:11px;max-width:200px;
                           overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                    title="{html.escape(prompt_pack.get('positive_prompt', ''))}">
                    {html.escape((prompt_pack.get('positive_prompt', '') or '')[:60])}</td>
            </tr>
            """)

        climax_badge = (
            '<span class="cw-badge" style="background:#fee2e2;color:#991b1b;">高潮页</span>'
            if is_climax else ""
        )

        panels_html.append(f"""
        <div class="cw-card" style="margin-bottom:12px;">
            <h4 style="margin:0 0 8px 0;color:#1e293b;">
                第 {page_num} 页 · {html.escape(template)} {climax_badge}
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


def render_review_full_detail(review_results: list[dict]) -> str:
    """渲染完整审查详情（含 Schema 校验/质量评分/建议）。"""
    if not review_results:
        return '<div style="color:#64748b;padding:12px;">审查数据尚未生成。<br>请在「创作流程」Tab 运行工作流后查看。</div>'

    cards = []
    for r in review_results:
        agent = r.get("agent", "?")
        decision = r.get("decision", "?")
        score = r.get("score", 0)
        issues = r.get("issues", [])
        suggestions = r.get("suggestions", [])
        strengths = r.get("strengths", [])
        dim_scores = r.get("dimension_scores", {})
        schema_check = r.get("schema_check", {})
        escalate = r.get("escalation", {}) or {}

        color = {"pass": "#10b981", "revise": "#f59e0b", "escalate": "#ef4444"}.get(decision, "#94a3b8")
        icon = {"pass": "✅", "revise": "🔄", "escalate": "⚠️"}.get(decision, "❓")

        # Schema 校验
        sc_html = ""
        if schema_check:
            passed = schema_check.get("passed", True)
            missing = schema_check.get("missing_fields", [])
            invalid = schema_check.get("invalid_fields", [])
            errors = schema_check.get("errors", [])
            sc_status = "✅ 通过" if passed else "❌ 未通过"
            sc_html = (
                f'<div style="font-size:12px;margin-top:6px;padding:6px;background:#f8fafc;'
                f'border-radius:4px;">'
                f'<b>Schema 校验: {sc_status}</b>'
                + (f'<div style="color:#ef4444;">缺失: {", ".join(missing)}</div>' if missing else "")
                + (f'<div style="color:#f59e0b;">无效: {", ".join(invalid)}</div>' if invalid else "")
                + (f'<div style="color:#ef4444;">错误: {", ".join(errors)}</div>' if errors else "")
                + f'</div>'
            )

        # 维度评分
        dim_html = ""
        if dim_scores:
            bars = "".join(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;margin:2px 0;">'
                f'<span style="width:80px;text-align:right;color:#475569;">{html.escape(dim)}</span>'
                f'<div style="flex:1;height:4px;background:#e2e8f0;border-radius:2px;overflow:hidden;">'
                f'<div style="height:100%;width:{max(0, min(100, val * 10))}%;'
                f'background:{_score_color(val)};"></div></div>'
                f'<span style="width:32px;text-align:right;font-weight:600;">{val:.1f}</span>'
                f'</div>'
                for dim, val in dim_scores.items()
            )
            dim_html = f'<div style="margin:6px 0;padding:4px 8px;">{bars}</div>'

        # 建议/优势/问题
        issues_html = ""
        if issues:
            issues_html = (
                f'<div style="font-size:11px;margin:4px 0;">'
                f'<b style="color:#ef4444;">问题:</b> {"; ".join(html.escape(i) for i in issues[:5])}'
                f'</div>'
            )
        strength_html = ""
        if strengths:
            strength_html = (
                f'<div style="font-size:11px;margin:4px 0;">'
                f'<b style="color:#10b981;">优势:</b> {"; ".join(html.escape(s) for s in strengths[:3])}'
                f'</div>'
            )
        suggestion_html = ""
        if suggestions:
            suggestion_html = (
                f'<div style="font-size:11px;margin:4px 0;">'
                f'<b style="color:#3b82f6;">建议:</b> {"; ".join(html.escape(s) for s in suggestions[:3])}'
                f'</div>'
            )

        # 升级详情
        escalate_html = ""
        if escalate.get("headline"):
            escalate_html = (
                f'<div style="font-size:12px;margin:6px 0;padding:6px;background:#fef2f2;'
                f'border-radius:4px;border-left:3px solid #ef4444;">'
                f'<b>⚠️ 升级: {html.escape(escalate.get("headline", ""))}</b>'
                f'<div style="color:#991b1b;">{html.escape(escalate.get("suggested_action", ""))}</div>'
                f'</div>'
            )

        cards.append(f"""
        <div class="cw-card" style="margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <span style="font-size:20px;">{icon}</span>
                <span style="font-weight:600;color:#1e293b;">{html.escape(agent)}</span>
                <span style="color:{color};font-weight:700;font-size:18px;">{score:.1f}</span>
                <span class="cw-badge" style="background:{color}20;color:{color};font-weight:600;">
                    {decision.upper()}</span>
            </div>
            {dim_html}
            {sc_html}
            {strength_html}
            {issues_html}
            {suggestion_html}
            {escalate_html}
        </div>
        """)

    return f'<div>{"".join(cards)}</div>'


def _score_color(val: float) -> str:
    if val >= 7:
        return "linear-gradient(90deg, #10b981, #34d399)"
    if val >= 4:
        return "linear-gradient(90deg, #fbbf24, #f59e0b)"
    return "linear-gradient(90deg, #f87171, #ef4444)"


# ============================================================================
# Phase 3: 逐页漫画阅读器 + 气泡叠加 + 分镜对比
# ============================================================================


def render_page_reader(
    final_pages: list[dict],
    current_idx: int = 0,
    show_bubbles: bool = False,
) -> str:
    """渲染逐页漫画阅读器。

    Parameters
    ----------
    final_pages:
        最终合成页面列表（来自 LayoutOutput.final_pages）。
    current_idx:
        当前显示的页面索引（0-based）。
    show_bubbles:
        是否叠加对话气泡边界标注。
    """
    if not final_pages:
        return '<div style="color:#64748b;padding:24px;text-align:center;">暂无漫画页面。<br>请先在「创作流程」Tab 运行工作流。</div>'

    total = len(final_pages)
    idx = max(0, min(current_idx, total - 1))
    page = final_pages[idx]

    page_id = page.get("page_id", "?")
    page_num = page.get("page_number", idx + 1)
    image_path = page.get("image_path", "")
    width = page.get("width_px", 1240)
    height = page.get("height_px", 1754)
    bubbles = page.get("bubbles", [])

    # 主图像
    img_html = ""
    if image_path and os.path.exists(str(image_path)):
        safe_path = _gradio_img_src(str(image_path))
        img_html = (
            f'<a href="/gradio_api/file={html.escape(safe_path)}" target="_blank">'
            f'<img src="/gradio_api/file={html.escape(safe_path)}" '
            f'style="max-width:100%;max-height:70vh;cursor:zoom-in;border-radius:8px;'
            f'box-shadow:0 4px 16px rgba(0,0,0,0.15);" '
            f'alt="第{page_num}页" />'
            f'</a>'
        )
    else:
        img_html = (
            f'<div style="width:{min(width, 600)}px;height:{min(height, 800)}px;'
            f'background:#f1f5f9;border-radius:8px;display:flex;align-items:center;'
            f'justify-content:center;color:#94a3b8;">'
            f'页面图像未生成</div>'
        )

    # 气泡叠加层
    bubble_overlay = ""
    if show_bubbles and bubbles:
        bubble_divs = []
        for b in bubbles:
            bbox = b.get("bbox", {})
            bx = bbox.get("x", 0) if isinstance(bbox, dict) else getattr(bbox, "x", 0)
            by = bbox.get("y", 0) if isinstance(bbox, dict) else getattr(bbox, "y", 0)
            bw = bbox.get("width", 0) if isinstance(bbox, dict) else getattr(bbox, "width", 0)
            bh = bbox.get("height", 0) if isinstance(bbox, dict) else getattr(bbox, "height", 0)
            text = b.get("text", "")[:30] if isinstance(b, dict) else getattr(b, "text", "")[:30]
            btype = b.get("bubble_type", "speech") if isinstance(b, dict) else getattr(b, "bubble_type", "speech")

            type_colors = {
                "speech": "rgba(59,130,246,0.2)", "thought": "rgba(139,92,246,0.2)",
                "shout": "rgba(239,68,68,0.25)", "whisper": "rgba(168,85,247,0.2)",
                "narration": "rgba(245,158,11,0.2)",
            }
            bg = type_colors.get(btype, "rgba(59,130,246,0.15)")

            bubble_divs.append(
                f'<div class="cw-bubble-overlay" style="left:{bx*100}%;top:{by*100}%;'
                f'width:{bw*100}%;height:{bh*100}%;background:{bg};">'
                f'<span class="cw-bubble-text">{html.escape(text)}</span>'
                f'</div>'
            )
        bubble_overlay = (
            f'<div style="position:absolute;top:0;left:0;width:100%;height:100%;'
            f'pointer-events:none;">{"".join(bubble_divs)}</div>'
        )

    return f"""
    <div class="cw-page-reader" style="text-align:center;">
        <div style="margin-bottom:12px;font-size:14px;color:#64748b;">
            第 <strong style="color:#1e293b;">{page_num}</strong> / {total} 页
            <span style="margin-left:8px;font-size:12px;color:#94a3b8;">({html.escape(str(page_id))})</span>
        </div>
        <div style="position:relative;display:inline-block;max-width:100%;">
            {img_html}
            {bubble_overlay}
        </div>
        <div class="cw-page-nav" style="margin-top:16px;display:flex;justify-content:center;gap:16px;">
            <button class="cw-page-nav-btn" {"disabled" if idx == 0 else ""}
                    style="padding:8px 20px;border:1px solid #d1d5db;border-radius:6px;
                           background:white;cursor:pointer;font-size:14px;color:#374151;"
                    onclick="document.dispatchEvent(new CustomEvent('cw-page-nav', {{detail: 'prev'}}))">
                ◀ 上一页
            </button>
            <span style="padding:8px 12px;font-size:14px;color:#64748b;align-self:center;">
                {idx + 1} / {total}
            </span>
            <button class="cw-page-nav-btn" {"disabled" if idx >= total - 1 else ""}
                    style="padding:8px 20px;border:1px solid #d1d5db;border-radius:6px;
                           background:white;cursor:pointer;font-size:14px;color:#374151;"
                    onclick="document.dispatchEvent(new CustomEvent('cw-page-nav', {{detail: 'next'}}))">
                下一页 ▶
            </button>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#94a3b8;">
            💡 点击图片可在新标签页中查看原图
        </div>
    </div>
    """


def render_comparison_view(plan: list[dict], final_pages: list[dict]) -> str:
    """渲染分镜规划 vs 最终成品的并排对比。

    将故事板布局（SVG 预览）与最终渲染页面进行并排比较。
    """
    if not plan and not final_pages:
        return '<div style="color:#64748b;padding:12px;">暂无数据。<br>请先在「创作流程」Tab 运行工作流后查看。</div>'

    # 为每个页面匹配 storyboard 和 final page
    final_by_id: dict[str, dict] = {}
    for fp in final_pages:
        pid = fp.get("page_id", "") if isinstance(fp, dict) else getattr(fp, "page_id", "")
        if pid:
            final_by_id[pid] = fp

    rows = []
    for page_data in plan:
        page_id = page_data.get("page_id", "") if isinstance(page_data, dict) else getattr(page_data, "page_id", "")
        page_num = page_data.get("page_number", "?") if isinstance(page_data, dict) else getattr(page_data, "page_number", "?")

        # 故事板 SVG
        storyboard_svg = _render_page_svg(page_data)

        # 最终成品图像
        final_img = ""
        matched = final_by_id.get(page_id)
        if matched:
            path = matched.get("image_path", "") if isinstance(matched, dict) else getattr(matched, "image_path", "")
            if path and os.path.exists(str(path)):
                safe_path = _gradio_img_src(str(path))
                final_img = (
                    f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                    f'style="max-width:100%;max-height:400px;border-radius:6px;'
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.1);" '
                    f'alt="第{page_num}页成品" />'
                )
            else:
                final_img = '<div style="color:#94a3b8;padding:40px;">尚未渲染</div>'
        else:
            final_img = '<div style="color:#94a3b8;padding:40px;">尚未渲染</div>'

        rows.append(f"""
        <div class="cw-comparison-row" style="display:flex;gap:16px;margin-bottom:16px;
                    background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;">
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">📐 分镜规划 — 第{page_num}页</div>
                {storyboard_svg}
            </div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">🎨 最终成品 — 第{page_num}页</div>
                {final_img}
            </div>
        </div>
        """)

    return f"""
    <div class="cw-comparison-grid">
        <h3 style="margin:0 0 12px 0;color:#1e293b;">🔄 分镜 vs 成品对比</h3>
        {"".join(rows) if rows else '<div style="color:#64748b;padding:12px;">没有可对比的页面。</div>'}
    </div>
    """


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
            # 均匀分布
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


# ============================================================================
# v0.3 前端重设计：Claude Code 风格渲染函数
# ============================================================================

_PHASE_AGENTS_V3 = [
    ("script_agent", "📝 剧本"),
    ("character_agent", "👤 角色"),
    ("storyboard_agent", "🎬 分镜"),
    ("image_agent", "🎨 图像"),
    ("layout_agent", "📐 排版"),
]


def render_agent_status_bar(active: str, done: list[str], elapsed_s: float = 0.0) -> str:
    """紧凑型 Agent 状态条 — 水平排列的圆点 + 标签。

    elapsed_s: 当前活跃 agent 的已运行秒数（仅 active 时显示）。
    """
    pills = []
    for agent_id, label in _PHASE_AGENTS_V3:
        if agent_id == active:
            cls = "active"
            dot = "●"
        elif agent_id in done:
            cls = "done"
            dot = "✓"
        else:
            cls = "pending"
            dot = "○"
        pills.append(
            f'<span class="cw-as-pill cw-as-{cls}">'
            f'{dot} {label}'
            f'</span>'
        )

    elapsed_html = ""
    if active and elapsed_s > 0:
        if elapsed_s < 60:
            elapsed_str = f"{elapsed_s:.0f}s"
        else:
            elapsed_str = f"{elapsed_s/60:.1f}min"
        elapsed_html = f'<span class="cw-elapsed active">⏱ {elapsed_str}</span>'

    return f"""
    <div class="cw-agent-status">
        {''.join(pills)}
        {elapsed_html}
    </div>
    """


def render_stream_log(events: list[dict]) -> str:
    """Claude Code 风格流式日志 — 暗色终端，时间线事件。

    事件类型与样式映射：
    - thinking → 紫色，左侧缩进
    - progress → 蓝色进度条
    - log / dev_* → 灰色信息
    - done → 绿色 ✓
    - error → 红色 ✗
    - checkpoint → 黄色高亮
    """
    if not events:
        return '<div class="cw-stream-log"><div class="cw-stream-empty">等待工作流启动...</div></div>'

    lines = []
    for evt in events[-80:]:  # 最近 80 条
        ts = evt.get("timestamp", 0)
        time_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "--:--:--"
        evt_type = str(evt.get("type", "log"))
        content = str(evt.get("content", ""))
        agent = str(evt.get("agent", ""))

        # 类型 → CSS class + 图标
        type_css, icon = _stream_entry_style(evt_type, content)

        # 构造条目
        lines.append(
            f'<div class="cw-stream-entry {type_css}">'
            f'<span class="cw-stream-time">{time_str}</span> '
            f'<span class="cw-stream-icon">{icon}</span> '
            f'<span class="cw-stream-content">{html.escape(content)}</span>'
            f'</div>'
        )

    return f"""
    <div class="cw-stream-log" id="cw-stream-log">
        {"".join(lines)}
    </div>
    """


def _stream_entry_style(evt_type: str, content: str) -> tuple[str, str]:
    """Map event type to CSS class and icon."""
    if evt_type in ("thinking", "progress"):
        return "cw-stream-thinking", "💭"
    if evt_type == "done":
        return "cw-stream-done", "✓"
    if evt_type == "error":
        return "cw-stream-error", "✗"
    if evt_type == "checkpoint":
        return "cw-stream-checkpoint", "⏸"
    if evt_type.startswith("dev_"):
        return "cw-stream-debug", "🔧"
    if evt_type == "partial":
        return "cw-stream-partial", "📤"
    return "cw-stream-info", "▸"


def render_project_cards(projects: list[dict]) -> str:
    """以卡片网格展示已保存的项目列表。

    每个 dict 应有: project_id, title, updated_at_str, phase
    """
    if not projects:
        return '<div class="cw-stream-empty">暂无已保存的项目。<br>在「项目」Tab 创建新项目后运行工作流，项目会自动保存。</div>'

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


def render_workflow_left(
    active: str,
    done: list[str],
    events: list[dict],
    checkpoint: dict | None,
    elapsed_s: float = 0.0,
    current_phase: str | None = None,
) -> str:
    """组合渲染工作流左栏：阶段跳转 + 状态条 + 流式日志 + checkpoint。"""
    if current_phase is None:
        current_phase = _agent_to_phase(active, done)
    phase_jump = render_phase_jump_buttons(current_phase, done)
    status = render_agent_status_bar(active, done, elapsed_s)
    log = render_stream_log(events)
    cp = render_inline_checkpoint(checkpoint)
    return f"""
    <div class="cw-workflow-left">
        {phase_jump}
        {status}
        {log}
        {cp}
    </div>
    """


def _agent_to_phase(active: str, done: list[str]) -> str:
    """根据活跃/已完成 agent 推断当前阶段。"""
    agent_to_phase = {
        "script_agent": "script",
        "character_agent": "character",
        "storyboard_agent": "storyboard",
        "image_agent": "image",
        "layout_agent": "layout",
    }
    if active and active in agent_to_phase:
        return agent_to_phase[active]
    # 返回最后完成的阶段
    phase_order = ["script", "character", "storyboard", "image", "layout"]
    reverse_agents = {v: k for k, v in agent_to_phase.items()}
    last_phase = "init"
    for phase in phase_order:
        agent = reverse_agents.get(phase, "")
        if agent in done:
            last_phase = phase
    return last_phase


def render_inline_checkpoint(checkpoint: dict | None) -> str:
    """内联 Checkpoint 面板 — 嵌入在流式日志底部。"""
    if not checkpoint:
        return ""

    label = checkpoint.get("label", "确认")
    payload = checkpoint.get("payload", {})
    phase = payload.get("phase", "?")
    pid = payload.get("project_id", "?")

    items = "".join(
        f'<span style="margin-right:12px;font-size:12px;"><strong>{html.escape(str(k))}</strong>: {html.escape(str(v)[:40])}</span>'
        for k, v in payload.items()
    )

    return f"""
    <div class="cw-inline-checkpoint">
        <div class="cw-checkpoint-header">
            ⏸ <strong>{html.escape(label)}</strong>
        </div>
        <div class="cw-checkpoint-meta">{items}</div>
        <div class="cw-checkpoint-hint">
            请选择「接受并继续」或「重新生成」。你可以在下方输入框中提供具体的修改指导。
        </div>
    </div>
    """


# ============================================================================
# v0.4 新增：阶段跳转、文字内容展示、指导信息
# ============================================================================

_PHASE_ORDER = ["init", "script", "character", "storyboard", "image", "layout"]
_PHASE_LABELS: dict[str, str] = {
    "init": "初始",
    "script": "剧本",
    "character": "角色",
    "storyboard": "分镜",
    "image": "图像",
    "layout": "排版",
}


def render_phase_jump_buttons(current_phase: str, done_agents: list[str] | None = None) -> str:
    """渲染阶段跳转按钮组 — 允许用户查看/跳到特定阶段。

    当前阶段高亮，已完成阶段可点击，未完成阶段禁用。
    """
    if done_agents is None:
        done_agents = []

    # 将 done_agents 映射到 phase
    agent_to_phase = {
        "script_agent": "script",
        "character_agent": "character",
        "storyboard_agent": "storyboard",
        "image_agent": "image",
        "layout_agent": "layout",
    }
    done_phases = {"init"}
    for a in done_agents:
        if a in agent_to_phase:
            done_phases.add(agent_to_phase[a])

    buttons = []
    for phase in _PHASE_ORDER:
        label = _PHASE_LABELS.get(phase, phase)
        if phase == current_phase:
            cls = "current"
        elif phase in done_phases:
            cls = ""
        else:
            cls = "disabled"
        data_phase = phase
        buttons.append(
            f'<span class="cw-phase-jump-btn {cls}" '
            f'data-phase="{html.escape(data_phase)}">{html.escape(label)}</span>'
        )

    return f"""
    <div class="cw-phase-jump-bar">
        <span class="cw-phase-jump-label">阶段:</span>
        {"".join(buttons)}
    </div>
    """


def render_phase_text_content(state: dict | None, agent_outputs: dict[str, list[dict]]) -> str:
    """根据当前 state 和 agent_outputs 渲染所有已完成阶段的详细文字内容。

    用户需要足够信息来判断是否重新生成该阶段。
    """
    if not state:
        return '<div class="cw-phase-text-empty">尚未创建项目</div>'

    parts: list[str] = []

    # 解析角色名映射（供分镜/排版阶段引用）
    character_db = state.get("character_db", {})
    char_id_to_name: dict[str, str] = {}
    if character_db:
        for cid, cp in character_db.get("characters", {}).items():
            char_id_to_name[cid] = cp.get("name", cid)

    # 剧本阶段完成 → 显示完整剧本
    script = state.get("structured_script", {})
    if script:
        parts.append(_render_detailed_script(script, char_id_to_name))

    # 角色阶段完成 → 显示完整角色特征
    if character_db:
        parts.append(_render_detailed_characters(character_db))

    # 分镜阶段完成 → 显示每页每格的详细信息
    storyboard = state.get("storyboard_plan", [])
    if storyboard:
        parts.append(_render_detailed_storyboard(storyboard, char_id_to_name))

    # 图像阶段完成 → 显示每张图的生成参数
    panel_images = state.get("panel_images", [])
    if panel_images:
        parts.append(_render_detailed_images(panel_images))

    # 排版阶段完成 → 显示布局和最终图片
    final_pages = state.get("final_pages", [])
    if final_pages:
        parts.append(_render_detailed_final_pages(final_pages, storyboard))

    if not parts:
        return '<div class="cw-phase-text-empty">等待工作流启动...</div>'

    return "".join(parts)


# ============================================================================
# 详细渲染辅助函数
# ============================================================================

def _resolve_char_name(char_id: str, char_map: dict[str, str]) -> str:
    """将角色ID解析为显示名称。"""
    return char_map.get(char_id, char_id)


def _render_detailed_script(script: dict, char_map: dict[str, str]) -> str:
    """渲染完整剧本：每个场景的叙述、对话、动作。"""
    title = html.escape(script.get("title", "未命名"))
    summary = html.escape((script.get("summary", "") or ""))
    genres = script.get("genre", [])
    scenes = script.get("scenes", [])
    characters = script.get("characters", [])

    genre_tags = " ".join(
        f'<span class="cw-badge">{html.escape(str(g))}</span>' for g in genres
    )

    # 角色列表
    char_list = ""
    for c in characters:
        name = html.escape(c.get("name", "?"))
        role = html.escape(c.get("role", "?"))
        appearance = html.escape((c.get("appearance") or "")[:60])
        personality = html.escape((c.get("personality") or "")[:60])
        char_list += (
            f'<div class="cw-phase-text-scene">'
            f'<b>{name}</b> <span class="cw-badge">{role}</span>'
            + (f'<br><span style="color:#64748b;">外貌: {appearance}</span>' if appearance else "")
            + (f'<br><span style="color:#64748b;">性格: {personality}</span>' if personality else "")
            + f'</div>'
        )

    # 场景详情
    scene_items = ""
    for s in scenes:
        idx = s.get("order", s.get("scene_index", "?"))
        loc = html.escape(s.get("location", "?"))
        time_of_day = html.escape(s.get("time_of_day", ""))
        atmosphere = html.escape(s.get("atmosphere", ""))
        narration = html.escape((s.get("narration") or "")[:150])
        emotion = s.get("emotion_intensity", 0)

        # 角色
        chars_in_scene = s.get("characters_present", [])
        char_display = ", ".join(
            html.escape(_resolve_char_name(cid, char_map))
            for cid in chars_in_scene
        ) if chars_in_scene else "无"

        # 动作
        actions = s.get("actions", [])
        action_items = ""
        for a in actions:
            actor = html.escape(_resolve_char_name(a.get("actor", "?"), char_map))
            desc = html.escape(a.get("description", "")[:80])
            action_items += f'<div style="margin-left:8px;">🎬 <b>{actor}</b>: {desc}</div>'

        # 对话
        dialogues = s.get("dialogues", [])
        dialogue_items = ""
        for d in dialogues:
            speaker = html.escape(_resolve_char_name(d.get("speaker", "?"), char_map))
            text = html.escape(d.get("text", "")[:120])
            tone = html.escape(d.get("tone", ""))
            is_thought = d.get("is_thought", False)
            bubble_icon = "💭" if is_thought else "💬"
            tone_badge = f' <span class="cw-badge">{tone}</span>' if tone else ""
            dialogue_items += (
                f'<div style="margin-left:8px;">{bubble_icon} <b>{speaker}</b>{tone_badge}: {text}</div>'
            )

        scene_items += f"""
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

    return f"""
    <div class="cw-phase-text-panel">
        <h4>📝 剧本: {title}</h4>
        <div class="cw-phase-text-scroll">
        <div class="cw-phase-text-item" style="margin-bottom:6px;">{summary}</div>
        <div class="cw-phase-text-item">{genre_tags}</div>
        <div class="cw-phase-text-item" style="margin-top:4px;">
            <span class="key">场景数:</span> <span class="val">{len(scenes)}</span>
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
                🎬 场景详情 ({len(scenes)}场)
            </summary>
            {scene_items}
        </details>
        </div>
    </div>
    """


def _render_detailed_characters(character_db: dict) -> str:
    """渲染完整角色设计：外貌特征、标签、参考图。"""
    characters = character_db.get("characters", {})
    if not characters:
        return ""

    items = ""
    for cid, cp in characters.items():
        name = html.escape(cp.get("name", cid))
        gender = html.escape(cp.get("gender_tag", ""))
        appearance_prompt = html.escape((cp.get("appearance_prompt") or "")[:200])
        core_tags = html.escape((cp.get("core_tags") or "")[:200])
        seed = cp.get("seed", 0)

        # 结构化外貌特征
        traits = cp.get("visual_traits", {})
        hair = html.escape(str(traits.get("hair", ""))[:40]) if traits.get("hair") else ""
        eyes = html.escape(str(traits.get("eyes", ""))[:40]) if traits.get("eyes") else ""
        body = html.escape(str(traits.get("body", ""))[:40]) if traits.get("body") else ""
        clothing = html.escape(str(traits.get("clothing", ""))[:40]) if traits.get("clothing") else ""
        distinctive = traits.get("distinctive", [])
        distinctive_str = ", ".join(html.escape(str(d)[:30]) for d in distinctive) if distinctive else ""

        trait_rows = ""
        if hair:
            trait_rows += f'<div><span class="key">发型:</span> <span class="val">{hair}</span></div>'
        if eyes:
            trait_rows += f'<div><span class="key">眼睛:</span> <span class="val">{eyes}</span></div>'
        if body:
            trait_rows += f'<div><span class="key">身材:</span> <span class="val">{body}</span></div>'
        if clothing:
            trait_rows += f'<div><span class="key">服装:</span> <span class="val">{clothing}</span></div>'
        if distinctive_str:
            trait_rows += f'<div><span class="key">特征:</span> <span class="val">{distinctive_str}</span></div>'

        # 参考图
        ref = cp.get("base_reference", {})
        img_path = ref.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                f'style="max-width:120px;max-height:160px;border:1px solid #e2e8f0;'
                f'border-radius:6px;margin:4px 0;" alt="{name} 参考图"/>'
            )

        items += f"""
        <div class="cw-phase-text-scene">
            <b>{name}</b>
            {f' <span class="cw-badge">{gender}</span>' if gender else ''}
            {f' <span class="cw-badge" style="background:#f1f5f9;color:#64748b;">种子: {seed}</span>' if seed else ''}
            {img_html}
            {f'<div style="font-size:11px;color:#475569;margin-top:4px;">{trait_rows}</div>' if trait_rows else ''}
            {f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">标签: {core_tags}</div>' if core_tags else ''}
            {f'<div style="font-size:10px;color:#94a3b8;">描述: {appearance_prompt}</div>' if appearance_prompt else ''}
        </div>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>👤 角色设计 ({len(characters)}人)</h4>
        <div class="cw-phase-text-scroll">
        {items}
        </div>
    </div>
    """


def _render_detailed_storyboard(storyboard: list[dict], char_map: dict[str, str]) -> str:
    """渲染完整分镜规划：每页每格的对话、动作、角色、镜头。"""
    if not storyboard:
        return ""

    total_panels = sum(len(p.get("panels", [])) for p in storyboard)
    pages_html = ""

    for p in storyboard:
        page_num = p.get("page_number", "?")
        template = html.escape(p.get("layout_template", ""))
        layout_hint = html.escape(p.get("layout_hint", ""))
        panels = p.get("panels", [])
        is_climax = p.get("is_climax_page", False)
        climax_mark = ' 🔥 高潮页' if is_climax else ''
        emotion_avg = p.get("page_emotion_avg", 0)

        # 每个面板的详细信息
        panel_items = ""
        for pn in panels:
            pid = html.escape(pn.get("panel_id", "?"))
            order = pn.get("order_in_page", "?")
            shot = html.escape(pn.get("shot_size", "?"))
            angle = html.escape(pn.get("camera_angle", "?"))
            shape = html.escape(pn.get("shape", "rectangle"))
            action = html.escape((pn.get("primary_action") or "")[:100])
            pose = html.escape((pn.get("pose_hint") or "")[:80])
            expression = html.escape((pn.get("expression") or "")[:80])
            setting = html.escape((pn.get("setting") or "")[:60])
            mood = html.escape(pn.get("mood", ""))
            lighting = html.escape((pn.get("scene_lighting") or "")[:60])
            weather = html.escape((pn.get("weather") or "")[:30])
            time_of_day = html.escape((pn.get("time_of_day") or "")[:30])
            emotion = pn.get("emotion_intensity", 0)

            # 此面板中的角色
            chars_in_panel = pn.get("characters_in_panel", [])
            char_display = ", ".join(
                html.escape(_resolve_char_name(cid, char_map))
                for cid in chars_in_panel
            ) if chars_in_panel else "无"

            # 此面板中的对话
            dialogues = pn.get("dialogues_in_panel", [])
            dialogue_items = ""
            for d in dialogues:
                speaker = html.escape(_resolve_char_name(d.get("speaker", "?"), char_map))
                text = html.escape(d.get("text", "")[:100])
                tone = html.escape(d.get("tone", ""))
                is_thought = d.get("is_thought", False)
                icon = "💭" if is_thought else "💬"
                tone_tag = f' [{tone}]' if tone else ""
                dialogue_items += (
                    f'<div style="margin-left:4px;font-size:11px;">'
                    f'{icon} <b>{speaker}</b>{tone_tag}: {text}</div>'
                )

            # 气泡提示
            bubble_hints = pn.get("speech_bubble_hints", [])
            bubble_items = ""
            if bubble_hints:
                for bh in bubble_hints[:3]:
                    btype = html.escape(bh.get("bubble_type", "speech"))
                    pos = html.escape(bh.get("suggested_position", "auto"))
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
                {f'<div><span class="key">场景:</span> {setting}</div>' if setting else ''}
                {f'<div><span class="key">光照:</span> {lighting}' + (f' · {weather}' if weather else '') + (f' · {time_of_day}' if time_of_day else '') + '</div>' if (lighting or weather or time_of_day) else ''}
                <div><span class="key">角色:</span> {char_display}</div>
                {f'<div><span class="key">动作:</span> {action}</div>' if action else ''}
                {f'<div><span class="key">姿势:</span> {pose}</div>' if pose else ''}
                {f'<div><span class="key">表情:</span> {expression}</div>' if expression else ''}
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


def _render_detailed_images(panel_images: list[dict]) -> str:
    """渲染每张面板图像的生成详情：提示词、种子、参数。"""
    if not panel_images:
        return ""

    backends: dict[str, int] = {}
    total_time = 0
    for pi in panel_images:
        backend = pi.get("backend", "?")
        backends[backend] = backends.get(backend, 0) + 1
        total_time += pi.get("generation_time_ms", 0)

    backend_str = " · ".join(f"{k}: {v}张" for k, v in backends.items())
    avg_time = total_time / len(panel_images) if panel_images else 0
    time_str = f"{avg_time/1000:.1f}s" if avg_time >= 1000 else f"{avg_time:.0f}ms"

    # 每张面板的详情（可折叠）
    panel_items = ""
    for pi in panel_images:
        pid = html.escape(pi.get("panel_id", "?"))
        backend = html.escape(pi.get("backend", "?"))
        seed = pi.get("seed", 0)
        steps = pi.get("steps", 0)
        cfg = pi.get("cfg_scale", 0)
        gen_time = pi.get("generation_time_ms", 0)
        gen_str = f"{gen_time/1000:.1f}s" if gen_time >= 1000 else f"{gen_time:.0f}ms"
        prompt = html.escape((pi.get("prompt_used") or "")[:300])
        neg_prompt = html.escape((pi.get("negative_prompt_used") or "")[:200])
        chars = pi.get("characters_present", [])
        char_str = ", ".join(html.escape(str(c)) for c in chars) if chars else "无"

        # 缩略图
        img_path = pi.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                f'style="max-width:100%;max-height:160px;border:1px solid #e2e8f0;'
                f'border-radius:4px;margin:4px 0;" alt="{pid}" loading="lazy"/>'
            )

        panel_items += f"""
        <details style="margin:2px 0;font-size:11px;">
            <summary style="cursor:pointer;color:#3b82f6;font-weight:500;">
                {pid} · {backend} · ⏱{gen_str} · 🌱{seed}
            </summary>
            <div style="padding:4px 8px;background:#f8fafc;border-radius:4px;">
                {img_html}
                <div><span class="key">角色:</span> {char_str}</div>
                <div><span class="key">参数:</span> steps={steps} cfg={cfg}</div>
                <div style="font-size:10px;color:#64748b;max-height:60px;overflow-y:auto;">
                    <span class="key">正向提示词:</span> {prompt}
                </div>
                {f'<div style="font-size:10px;color:#94a3b8;max-height:40px;overflow-y:auto;"><span class="key">负向:</span> {neg_prompt}</div>' if neg_prompt else ''}
            </div>
        </details>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>🎨 图像生成 ({len(panel_images)}张面板)</h4>
        <div class="cw-phase-text-scroll">
        <div class="cw-phase-text-item">
            <span class="key">后端分布:</span> <span class="val">{backend_str}</span>
            &nbsp;&nbsp;<span class="key">平均耗时:</span> <span class="val">{time_str}/张</span>
        </div>
        <div style="margin-top:4px;">
            {panel_items}
        </div>
        </div>
    </div>
    """


def _render_detailed_final_pages(final_pages: list[dict], storyboard: list[dict]) -> str:
    """渲染完整排版结果：布局网格、对话气泡、最终图片。"""
    if not final_pages:
        return ""

    # 从分镜中提取布局信息
    storyboard_by_page: dict[str, dict] = {}
    for p in storyboard:
        pid = p.get("page_id", "")
        if pid:
            storyboard_by_page[pid] = p

    items = ""
    for fp in final_pages:
        page_num = fp.get("page_number", "?")
        page_id = html.escape(fp.get("page_id", "?"))
        w = fp.get("width_px", 0)
        h = fp.get("height_px", 0)

        # 从分镜获取布局信息
        sb = storyboard_by_page.get(fp.get("page_id", ""), {})
        panels_in_page = sb.get("panels", [])
        panel_count = len(panels_in_page)
        # 推断网格布局
        grid_desc = _infer_grid(panel_count)

        # 布局警告
        warnings = fp.get("layout_warnings", [])
        warn_html = ""
        if warnings:
            warn_items = "".join(
                f'<div style="color:#f59e0b;font-size:10px;">⚠ {html.escape(str(w)[:100])}</div>'
                for w in warnings[:3]
            )
            warn_html = f'<div style="margin:4px 0;">{warn_items}</div>'

        # 气泡详情
        bubbles = fp.get("bubbles", [])
        bubble_html = ""
        if bubbles:
            bubble_items = ""
            for b in bubbles[:8]:
                btype = html.escape(b.get("bubble_type", "speech"))
                text = html.escape((b.get("text") or "")[:60])
                font_size = b.get("font_size_pt", 12)
                occlusion = b.get("occlusion_score", 0)
                occ_color = "#10b981" if occlusion < 0.3 else ("#f59e0b" if occlusion < 0.7 else "#ef4444")
                bubble_items += (
                    f'<div style="font-size:10px;margin:1px 0;">'
                    f'<span class="cw-badge" style="font-size:9px;">{btype}</span>'
                    f' {text}'
                    f' <span style="color:{occ_color};">(遮挡:{occlusion:.2f})</span>'
                    f' <span style="color:#94a3b8;">字号:{font_size}pt</span>'
                    f'</div>'
                )
            bubble_html = f"""
            <details style="margin-top:4px;font-size:11px;">
                <summary style="cursor:pointer;color:#3b82f6;">💬 对话气泡 ({len(bubbles)}个)</summary>
                {bubble_items}
            </details>
            """

        # 最终图片
        img_path = fp.get("image_path", "")
        img_html = ""
        if img_path and os.path.exists(str(img_path)):
            safe_path = _gradio_img_src(str(img_path))
            img_html = (
                f'<a href="/gradio_api/file={html.escape(safe_path)}" target="_blank">'
                f'<img src="/gradio_api/file={html.escape(safe_path)}" '
                f'style="max-width:100%;max-height:200px;border:1px solid #e2e8f0;'
                f'border-radius:6px;margin:4px 0;cursor:zoom-in;" '
                f'alt="第{page_num}页" loading="lazy"/>'
                f'</a>'
            )

        items += f"""
        <div class="cw-phase-text-scene">
            <b>第 {page_num} 页</b> · {w}×{h}px · {grid_desc} ({panel_count}格)
            {img_html}
            {warn_html}
            {bubble_html}
        </div>
        """

    return f"""
    <div class="cw-phase-text-panel">
        <h4>📐 排版完成 ({len(final_pages)}页)</h4>
        <div class="cw-phase-text-scroll">
        {items}
        </div>
    </div>
    """


def _infer_grid(panel_count: int) -> str:
    """根据面板数量推断网格布局名称。"""
    mapping = {1: "1×1", 2: "1×2 或 2×1", 3: "1×3 或 L形",
               4: "2×2", 5: "2×3(少1)", 6: "2×3", 7: "2×4(少1)", 8: "2×4"}
    return mapping.get(panel_count, f"{panel_count}格")


def render_live_content(
    panel_images: list[dict],
    character_images: list[dict],
    text_html: str,
) -> str:
    """渲染创作Tab右侧面板：文字内容 + 图像预览。"""
    image_preview = render_live_panel_preview(panel_images, character_images)

    return f"""
    <div class="cw-live-content">
        {text_html}
        {image_preview}
    </div>
    """
