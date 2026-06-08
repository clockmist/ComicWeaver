"""工作流UI渲染：Agent状态条、流式日志、确认面板、阶段跳转。"""
from __future__ import annotations

import html as html_mod
import time


# ============================================================================
# Agent 列表
# ============================================================================

_PHASE_AGENTS = [
    ("script_agent", "📝 剧本"),
    ("character_agent", "👤 角色"),
    ("storyboard_agent", "🎬 分镜"),
    ("image_agent", "🎨 图像"),
    ("layout_agent", "📐 排版"),
]

_PHASE_AGENTS_V3 = [
    ("story_agent", "📖 故事"),
    ("character_agent", "👤 角色"),
    ("script_agent", "📝 剧本"),
    ("storyboard_agent", "🎬 分镜"),
    ("image_agent", "🎨 图像"),
    ("bubble_agent", "💬 台词"),
    ("layout_agent", "📐 排版"),
]

_PHASE_ORDER = ["init", "story", "character", "script", "storyboard", "image", "bubble", "layout"]
_PHASE_LABELS: dict[str, str] = {
    "init": "初始",
    "story": "故事",
    "script": "剧本",
    "character": "角色",
    "storyboard": "分镜",
    "image": "图像",
    "bubble": "台词",
    "layout": "排版",
}


def _agent_to_phase(active: str, done: list[str]) -> str:
    """根据活跃/已完成 agent 推断当前阶段。"""
    agent_to_phase = {
        "story_agent": "story",
        "script_agent": "script",
        "character_agent": "character",
        "storyboard_agent": "storyboard",
        "image_agent": "image",
        "bubble_agent": "bubble",
        "layout_agent": "layout",
    }
    if active and active in agent_to_phase:
        return agent_to_phase[active]
    phase_order = ["story", "script", "character", "storyboard", "image", "bubble", "layout"]
    reverse_agents = {v: k for k, v in agent_to_phase.items()}
    last_phase = "init"
    for phase in phase_order:
        agent = reverse_agents.get(phase, "")
        if agent in done:
            last_phase = phase
    return last_phase


# ============================================================================
# Agent 状态网格
# ============================================================================


def render_agent_grid(active: str = "", done: list[str] | None = None) -> str:
    """渲染6个Agent的状态卡片网格。"""
    done = done or []
    cells = []
    for agent_id, label in _PHASE_AGENTS:
        cls = "cw-agent-pill"
        if agent_id == active:
            cls += " active"
        elif agent_id in done:
            cls += " done"
        cells.append(f'<div class="{cls}">{label}</div>')
    return f'<div class="cw-agent-grid">{"".join(cells)}</div>'


def render_agent_status_bar(active: str, done: list[str], elapsed_s: float = 0.0) -> str:
    """紧凑型 Agent 状态条 — 水平排列的圆点 + 标签。"""
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


# ============================================================================
# 日志渲染
# ============================================================================


def render_log(events: list[dict], max_lines: int = 50) -> str:
    """渲染日志（类似终端输出）。"""
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
        lines.append(f'<div class="{css_cls}">{html_mod.escape(prefix)} {html_mod.escape(msg)}</div>')

    body = "\n".join(lines) if lines else '<div style="color:#64748b;">等待开始...</div>'
    return f'<div class="cw-log">{body}</div>'


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


def render_stream_log(events: list[dict]) -> str:
    """Claude Code 风格流式日志 — 暗色终端，时间线事件。"""
    if not events:
        return '<div class="cw-stream-log"><div class="cw-stream-empty">等待工作流启动...</div></div>'

    lines = []
    for evt in events[-80:]:
        ts = evt.get("timestamp", 0)
        time_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "--:--:--"
        evt_type = str(evt.get("type", "log"))
        content = str(evt.get("content", ""))

        type_css, icon = _stream_entry_style(evt_type, content)
        lines.append(
            f'<div class="cw-stream-entry {type_css}">'
            f'<span class="cw-stream-time">{time_str}</span> '
            f'<span class="cw-stream-icon">{icon}</span> '
            f'<span class="cw-stream-content">{html_mod.escape(content)}</span>'
            f'</div>'
        )

    return f"""
    <div class="cw-stream-log" id="cw-stream-log">
        {"".join(lines)}
    </div>
    """


# ============================================================================
# Checkpoint 面板
# ============================================================================


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
        preview_lines.append(f"<div><b>{html_mod.escape(str(k))}:</b> {html_mod.escape(v_str)}</div>")
    preview_html = "".join(preview_lines)

    return f"""
    <div class="cw-checkpoint">
        <h3>⏸ {html_mod.escape(label)}</h3>
        <div style="font-size:13px;color:#78350f;">
            生产Agent已完成本阶段输出，审查Agent已通过质量检查。
            请确认是否继续下一阶段。
        </div>
        <div class="preview">{preview_html}</div>
    </div>
    """


def render_inline_checkpoint(checkpoint: dict | None) -> str:
    """内联 Checkpoint 面板 — 嵌入在流式日志底部。"""
    if not checkpoint:
        return ""

    label = checkpoint.get("label", "确认")
    payload = checkpoint.get("payload", {})

    items = "".join(
        f'<span style="margin-right:12px;font-size:12px;">'
        f'<strong>{html_mod.escape(str(k))}</strong>: {html_mod.escape(str(v)[:40])}</span>'
        for k, v in payload.items()
    )

    return f"""
    <div class="cw-inline-checkpoint">
        <div class="cw-checkpoint-header">
            ⏸ <strong>{html_mod.escape(label)}</strong>
        </div>
        <div class="cw-checkpoint-meta">{items}</div>
        <div class="cw-checkpoint-hint">
            请选择「接受并继续」或「重新生成」。你可以在下方输入框中提供具体的修改指导。
        </div>
    </div>
    """


# ============================================================================
# 阶段跳转
# ============================================================================


def render_phase_jump_buttons(current_phase: str, done_agents: list[str] | None = None) -> str:
    """渲染阶段跳转按钮组。"""
    if done_agents is None:
        done_agents = []

    agent_to_phase = {
        "story_agent": "story",
        "script_agent": "script",
        "character_agent": "character",
        "storyboard_agent": "storyboard",
        "image_agent": "image",
        "bubble_agent": "bubble",
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
        buttons.append(
            f'<span class="cw-phase-jump-btn {cls}" '
            f'data-phase="{html_mod.escape(phase)}">{html_mod.escape(label)}</span>'
        )

    return f"""
    <div class="cw-phase-jump-bar">
        <span class="cw-phase-jump-label">阶段:</span>
        {"".join(buttons)}
    </div>
    """


# ============================================================================
# 组合渲染
# ============================================================================


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
