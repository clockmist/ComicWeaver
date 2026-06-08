"""结果加载回调：各 Tab 的数据获取与渲染。"""
from __future__ import annotations

import os

from ..renderers.characters import render_character_profiles
from ..renderers.dev import render_dev_log, render_performance_summary
from ..renderers.images import render_panel_images_gallery
from ..renderers.layout import render_comparison_view, render_page_reader
from ..renderers.story import render_emotion_curve, render_script_summary
from ..renderers.storyboard import render_storyboard_detail, render_storyboard_preview
from ..renderers.agents import render_agent_outputs
from ..session import SESSION


def load_results(page_idx: int = 0, show_bubbles: bool = False) -> tuple[str, str, str, str, str, list]:
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return empty, empty, empty, empty, empty, []

    state = SESSION.state
    script = state.get("structured_script", {}) or {}
    story = state.get("developed_story", {}) or {}
    curve = state.get("emotion_curve", []) or []
    plan = state.get("storyboard_plan", []) or []
    final_pages = state.get("final_pages", []) or []

    page_reader_html = render_page_reader(final_pages, page_idx, show_bubbles)
    comparison_html = render_comparison_view(plan, final_pages)

    page_gallery = []
    for p in final_pages:
        path = p.get("image_path", "")
        if os.path.exists(path):
            page_gallery.append(path)

    return (
        render_script_summary(script, story),
        render_emotion_curve(curve),
        render_storyboard_preview(plan),
        page_reader_html,
        comparison_html,
        page_gallery,
    )


def load_all_results(page_idx: int = 0, show_bubbles: bool = False) -> tuple:
    """Tab 3 预览：加载所有结果数据。"""
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return (empty, empty, empty, empty, empty, empty)

    state = SESSION.state
    script = state.get("structured_script", {}) or {}
    story = state.get("developed_story", {}) or {}
    curve = state.get("emotion_curve", []) or []
    plan = state.get("storyboard_plan", []) or []
    final_pages = state.get("final_pages", []) or []
    character_db = state.get("character_db", {}) or {}

    return (
        render_script_summary(script, story),
        render_emotion_curve(curve),
        render_storyboard_preview(plan),
        render_page_reader(final_pages, page_idx, show_bubbles),
        render_comparison_view(plan, final_pages),
        render_character_profiles(character_db, SESSION.agent_outputs),
    )


def navigate_page(direction: str, page_idx: int, show_bubbles: bool) -> tuple[str, int]:
    """页面导航回调。"""
    final_pages = SESSION.state.get("final_pages", []) if SESSION.state else []
    total = len(final_pages)
    if direction == "prev":
        page_idx = max(0, page_idx - 1)
    elif direction == "next":
        page_idx = min(total - 1, page_idx + 1)
    return render_page_reader(final_pages, page_idx, show_bubbles), page_idx


def get_download_files() -> list[str]:
    """获取所有最终页面的文件路径用于下载。"""
    if SESSION.state is None:
        return []
    pages = SESSION.state.get("final_pages", [])
    paths = []
    for p in pages:
        path = p.get("image_path", "")
        if path and os.path.exists(path):
            paths.append(path)
    return paths


def load_agent_outputs_tab() -> tuple[str, str, str]:
    """加载 Agent 输出 Tab 的所有内容。"""
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return empty, empty, empty

    agent_outputs = SESSION.agent_outputs
    character_db = SESSION.state.get("character_db") or {}
    plan = SESSION.state.get("storyboard_plan", []) or []

    return (
        render_agent_outputs(agent_outputs),
        render_character_profiles(character_db, agent_outputs),
        render_storyboard_detail(plan),
    )


def load_dev_log_tab(filter_category: str = "all",
                     filter_level: str = "all") -> tuple[str, str, str]:
    """加载开发者日志 Tab 的所有内容。"""
    dev_entries = SESSION.dev_log
    if not dev_entries:
        empty = (
            '<div style="color:#64748b;padding:12px;">暂无开发者日志。<br>'
            '请在「创作流程」Tab 启动工作流后自动生成。</div>'
        )
        return empty, empty, empty

    error_count = sum(1 for e in dev_entries if e.get("level") == "error")
    warn_count = sum(1 for e in dev_entries if e.get("level") == "warn")
    perf_count = sum(1 for e in dev_entries if e.get("category") == "performance")
    input_count = sum(1 for e in dev_entries if e.get("category") == "agent_input")

    stats_html = f"""
    <div class="cw-stat-row">
        <div class="cw-stat-card">
            <div class="number">{len(dev_entries)}</div>
            <div class="label">总日志条数</div>
        </div>
        <div class="cw-stat-card" style="border-color:#ef4444;">
            <div class="number" style="color:#ef4444;">{error_count}</div>
            <div class="label">错误</div>
        </div>
        <div class="cw-stat-card" style="border-color:#f59e0b;">
            <div class="number" style="color:#f59e0b;">{warn_count}</div>
            <div class="label">警告</div>
        </div>
        <div class="cw-stat-card" style="border-color:#06b6d4;">
            <div class="number" style="color:#06b6d4;">{perf_count}</div>
            <div class="label">性能记录</div>
        </div>
        <div class="cw-stat-card" style="border-color:#3b82f6;">
            <div class="number" style="color:#3b82f6;">{input_count}</div>
            <div class="label">输入追踪</div>
        </div>
    </div>
    """

    return (
        stats_html,
        render_dev_log(dev_entries, filter_category, filter_level),
        render_performance_summary(dev_entries),
    )


def load_dev_log_compact(filter_category: str = "all",
                         filter_level: str = "all") -> str:
    """紧凑版开发者日志（用于调试折叠面板）。"""
    dev_entries = SESSION.dev_log
    if not dev_entries:
        return '<div style="color:#64748b;padding:12px;">暂无日志。启动工作流后自动生成。</div>'
    return render_dev_log(dev_entries, filter_category, filter_level)
