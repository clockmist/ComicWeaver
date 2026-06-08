"""开发者日志渲染：日志查看器、性能统计。"""
from __future__ import annotations

import html as html_mod
import json
import time


def render_dev_log(dev_entries: list[dict], filter_category: str = "all",
                   filter_level: str = "all", max_entries: int = 100) -> str:
    """渲染开发者日志查看器（带筛选）。"""
    if not dev_entries:
        return (
            '<div style="color:#64748b;padding:12px;">暂无开发者日志。<br>'
            '请在「创作流程」Tab 启动工作流后自动生成。</div>'
        )

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
    cat_labels: dict[str, str] = {
        "agent_input": "📥 输入",
        "agent_output": "📤 输出",
        "state_change": "🔄 状态",
        "review_decision": "🔍 审查",
        "error": "❌ 错误",
        "performance": "⏱ 性能",
        "checkpoint": "⏸ 确认",
        "workflow": "📋 流程",
    }
    level_icons: dict[str, str] = {
        "trace": "·",
        "debug": "🔍",
        "info": "ℹ️",
        "warn": "⚠️",
        "error": "❌",
        "perf": "⏱",
    }
    level_labels: dict[str, str] = {
        "trace": "追踪",
        "debug": "调试",
        "info": "信息",
        "warn": "警告",
        "error": "错误",
        "perf": "性能",
    }
    agent_labels: dict[str, str] = {
        "story_agent": "故事创作",
        "character_agent": "角色设计",
        "script_agent": "剧本生成",
        "storyboard_agent": "分镜规划",
        "image_agent": "图像生成",
        "bubble_agent": "台词气泡",
        "layout_agent": "排版合成",
        "workflow": "工作流",
    }

    lines = []
    for e in entries_to_show:
        ts = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", time.time())))
        cat = e.get("category", "?")
        lvl = e.get("level", "info")
        agent = e.get("agent", "?")
        msg = html_mod.escape(str(e.get("message", ""))[:200])
        dur = e.get("duration_ms", 0)
        dur_str = f" [{dur:.0f}ms]" if dur > 0 else ""
        color = cat_colors.get(cat, "#94a3b8")
        icon = level_icons.get(lvl, "·")

        # 中文标签
        cat_label = cat_labels.get(cat, cat)
        lvl_label = level_labels.get(lvl, lvl)
        agent_label = agent_labels.get(agent, agent)

        data_dict = e.get("data", {})
        data_json = json.dumps(data_dict, ensure_ascii=False, default=str) if data_dict else ""
        data_preview = ""
        if data_json:
            data_preview = (
                f'<details style="margin-left:16px;font-size:11px;">'
                f'<summary style="cursor:pointer;color:{color};">展开数据</summary>'
                f'<pre style="background:#0f172a;color:#e2e8f0;padding:8px;border-radius:4px;'
                f'max-height:200px;overflow:auto;font-size:11px;">'
                f'{html_mod.escape(data_json[:2000])}'
                f'</pre></details>'
            )

        lines.append(
            f'<div style="padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:12px;">'
            f'<span style="color:#94a3b8;">{ts}</span> '
            f'<span style="color:{color};font-weight:600;" title="{html_mod.escape(lvl_label)}">{icon} {html_mod.escape(cat_label)}</span> '
            f'<span style="color:#1e293b;" title="{html_mod.escape(agent)}">[{html_mod.escape(agent_label)}]</span> '
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
            <td style="padding:6px 12px;font-weight:500;">{html_mod.escape(agent)}</td>
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
