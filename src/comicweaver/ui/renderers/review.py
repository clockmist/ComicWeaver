"""审查评分渲染：评分条、审查历史、审查详情。"""
from __future__ import annotations

import html


def _score_color(val: float) -> str:
    if val >= 7:
        return "linear-gradient(90deg, #10b981, #34d399)"
    if val >= 4:
        return "linear-gradient(90deg, #fbbf24, #f59e0b)"
    return "linear-gradient(90deg, #f87171, #ef4444)"


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


def render_review_full_detail(review_results: list[dict]) -> str:
    """渲染完整审查详情（含 Schema 校验/质量评分/建议）。"""
    if not review_results:
        return (
            '<div style="color:#64748b;padding:12px;">审查数据尚未生成。<br>'
            '请在「创作流程」Tab 运行工作流后查看。</div>'
        )

    cards = []
    for r in review_results:
        agent_id = r.get("agent", "?")
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
                <span style="font-weight:600;color:#1e293b;">{html.escape(agent_id)}</span>
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
