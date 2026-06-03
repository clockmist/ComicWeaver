"""Gradio 前端主入口 - 多Tab界面。

设计:使用 gr.update 流式更新,通过 generator 函数产出多次输出。
每次工作流暂停在 checkpoint 时,函数 yield 当前状态后返回,
用户响应后通过另一个按钮重新启动 generator 继续。
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator

import gradio as gr

from comicweaver.core import ComicState, make_initial_state
from comicweaver.orchestrator import CheckpointSignal, ComicWorkflow, WorkflowEvent
from comicweaver.storage import save_project, state_to_project

from .html_widgets import (
    render_agent_grid,
    render_agent_outputs,
    render_character_profiles,
    render_checkpoint,
    render_comparison_view,
    render_dev_log,
    render_emotion_curve,
    render_live_panel_preview,
    render_log,
    render_page_reader,
    render_panel_images_gallery,
    render_performance_summary,
    render_project_info,
    render_review_full_detail,
    render_review_history,
    render_score_bars,
    render_script_summary,
    render_storyboard_detail,
    render_storyboard_preview,
)
from .styles import CUSTOM_CSS

from comicweaver.utils.logging import DevLogEntry

# ============================================================================
# 应用会话状态(单进程单用户简化方案)
# ============================================================================

class Session:
    """单个用户会话的状态容器。"""

    def __init__(self) -> None:
        self.state: ComicState | None = None
        self.workflow: ComicWorkflow | None = None
        self.event_log: list[dict] = []
        self.done_agents: list[str] = []
        self.active_agent: str = ""
        self.review_scores: dict[str, float] = {}
        self.last_checkpoint: dict | None = None
        self.workflow_done: bool = False
        self.error: str | None = None
        # 新增：Agent 输出和开发者日志收集
        self.agent_outputs: dict[str, list[dict]] = {}  # agent_id -> [output dicts]
        self.dev_log: list[dict] = []                    # DevLogEntry dicts
        self.panel_images_preview: list[dict] = []       # 实时面板预览
        # asyncio 资源
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def reset(self) -> None:
        self.event_log = []
        self.done_agents = []
        self.active_agent = ""
        self.review_scores = {}
        self.last_checkpoint = None
        self.workflow_done = False
        self.error = None
        self.workflow = None
        self.agent_outputs = {}
        self.dev_log = []
        self.panel_images_preview = []
        self._task = None
        self._loop = None


SESSION = Session()


# ============================================================================
# 工作流后台执行
# ============================================================================

async def _consume_workflow(session: Session) -> None:
    """消费工作流事件,持续更新session状态。直到完成或暂停在checkpoint。"""
    if session.workflow is None or session.state is None:
        return

    try:
        async for msg in session.workflow.arun(session.state):
            # CheckpointSignal -> 暂停
            if isinstance(msg, CheckpointSignal):
                session.last_checkpoint = {
                    "id": msg.checkpoint_id,
                    "label": msg.label,
                    "payload": msg.payload,
                }
                session.event_log.append({
                    "agent": "workflow",
                    "type": "checkpoint",
                    "content": f"等待用户确认: {msg.label}",
                    "timestamp": msg.timestamp,
                })
                # 等待外部通过 session.workflow.respond() 设置 future
                # 这里 _consume_workflow 任务会被 yield-await 阻塞,
                # 因为 _handle_checkpoint 内部会 await self._pending_response
                # 注意:CheckpointSignal yield之后控制权回到这里,然后继续
                # async for 会再次进入生成器,生成器内部 await 会阻塞
                continue

            # WorkflowMessage
            if hasattr(msg, "event"):
                if msg.event == WorkflowEvent.NODE_START:
                    session.active_agent = msg.node
                    session.event_log.append({
                        "agent": msg.node,
                        "type": "log",
                        "content": f"启动 {msg.node}",
                        "timestamp": msg.timestamp,
                    })
                elif msg.event == WorkflowEvent.NODE_END:
                    if msg.node not in session.done_agents:
                        session.done_agents.append(msg.node)
                    session.active_agent = ""
                    session.event_log.append({
                        "agent": msg.node,
                        "type": "done",
                        "content": f"{msg.node} 完成",
                        "timestamp": msg.timestamp,
                    })
                elif msg.event in (
                    WorkflowEvent.REVIEW_PASS,
                    WorkflowEvent.REVIEW_REVISE,
                    WorkflowEvent.REVIEW_ESCALATE,
                ):
                    payload = msg.payload or {}
                    score = payload.get("score", 0)
                    if "dimension_scores" in payload:
                        session.review_scores = payload["dimension_scores"]
                    session.event_log.append({
                        "agent": "reviewer",
                        "type": "log",
                        "content": f"{msg.event.value.upper()} · 评分 {score:.1f}",
                        "timestamp": msg.timestamp,
                    })
                elif msg.event == WorkflowEvent.WORKFLOW_DONE:
                    session.workflow_done = True
                    session.event_log.append({
                        "agent": "workflow",
                        "type": "done",
                        "content": "工作流完成 ✓",
                        "timestamp": msg.timestamp,
                    })
                    if session.state:
                        proj = state_to_project(session.state)
                        save_project(proj)
                continue

            # DevLogEntry
            if isinstance(msg, DevLogEntry):
                session.dev_log.append(msg.to_dict())
                # 同时在事件日志中显示简化版本
                session.event_log.append({
                    "agent": msg.agent,
                    "type": f"dev_{msg.category.value}",
                    "content": f"[{msg.level.value}] {msg.message[:120]}",
                    "timestamp": msg.timestamp,
                })
                continue

            # StreamEvent
            if hasattr(msg, "agent") and hasattr(msg, "type"):
                evt_type = (
                    msg.type.value if hasattr(msg.type, "value")
                    else str(msg.type)
                )
                # 改进：dict 内容显示实际摘要而非仅 key 列表
                content = msg.content
                if isinstance(content, dict):
                    if evt_type == "done":
                        # DONE 事件：保存完整输出按 agent 归类
                        agent_name = msg.agent
                        out_list = session.agent_outputs.setdefault(agent_name, [])
                        out_list.append({
                            "panel_id": content.get("panel_id", ""),
                            "title": content.get("title", ""),
                            "output": content,
                            "timestamp": msg.timestamp,
                        })
                        # 实时面板预览：image_agent 完成时收集图像
                        if agent_name == "image_agent" and "panel_image" in content:
                            session.panel_images_preview.append(content["panel_image"])
                        # 日志中显示摘要
                        keys = list(content.keys())
                        summary = f"✓ 完成 · 输出 {len(content)} 个字段: {', '.join(keys[:8])}"
                        if len(keys) > 8:
                            summary += f" ...+{len(keys) - 8}"
                        content = summary
                    else:
                        # PARTIAL_OUTPUT 等：显示 top 级别 key + 值摘要
                        parts = []
                        for k, v in list(content.items())[:5]:
                            if isinstance(v, list):
                                parts.append(f"{k}=[{len(v)} items]")
                            elif isinstance(v, dict):
                                parts.append(f"{k}={{...{len(v)} keys}}")
                            elif isinstance(v, str) and len(v) > 40:
                                parts.append(f"{k}='{v[:40]}...'")
                            else:
                                parts.append(f"{k}={v}")
                        content = "; ".join(parts)

                session.event_log.append({
                    "agent": msg.agent,
                    "type": evt_type,
                    "content": content,
                    "timestamp": msg.timestamp,
                })
    except Exception as exc:  # noqa: BLE001
        session.error = str(exc)
        session.event_log.append({
            "agent": "workflow",
            "type": "error",
            "content": f"{exc.__class__.__name__}: {exc}",
            "timestamp": time.time(),
        })


def _ensure_loop(session: Session) -> asyncio.AbstractEventLoop:
    """获取或创建session专属事件循环(运行在后台线程)。"""
    if session._loop and session._loop.is_running():
        return session._loop
    import threading
    loop = asyncio.new_event_loop()
    session._loop = loop

    def runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=runner, daemon=True).start()
    # 等待 loop 启动
    while not loop.is_running():
        time.sleep(0.01)
    return loop


# ============================================================================
# UI 回调
# ============================================================================

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


def start_workflow() -> Iterator[tuple]:
    """启动工作流并周期性产出UI更新。"""
    if SESSION.state is None:
        yield (
            "❌ 请先在「项目创建」Tab创建项目",
            render_agent_grid(),
            render_log([]),
            render_checkpoint(None),
            render_score_bars({}),
            render_live_panel_preview([]),
            render_project_info(),
            gr.update(interactive=False),  # accept_btn
            gr.update(interactive=False),  # regen_btn
        )
        return

    # 检测是否为恢复：若已有 workflow 且来自已保存项目，则复用
    current_phase = SESSION.state.get("current_phase", "init")
    is_resume = current_phase not in ("init",)
    if SESSION.workflow is None:
        if is_resume:
            SESSION.workflow = ComicWorkflow(resume_phase=current_phase)
        else:
            SESSION.workflow = ComicWorkflow()
    SESSION.event_log = []
    SESSION.done_agents = []
    SESSION.workflow_done = False
    SESSION.last_checkpoint = None
    SESSION.panel_images_preview = []

    loop = _ensure_loop(SESSION)
    SESSION._task = asyncio.run_coroutine_threadsafe(
        _consume_workflow(SESSION), loop
    )

    resume_note = " (从断点恢复)" if is_resume else ""
    yield (
        f"🚀 工作流已启动{resume_note}...",
        render_agent_grid(SESSION.active_agent, SESSION.done_agents),
        render_log(SESSION.event_log),
        render_checkpoint(SESSION.last_checkpoint),
        render_score_bars(SESSION.review_scores),
        render_live_panel_preview(SESSION.panel_images_preview),
        render_project_info(SESSION.state),
        gr.update(interactive=False),  # accept_btn
        gr.update(interactive=False),  # regen_btn
    )

    while True:
        time.sleep(0.5)
        at_checkpoint = SESSION.last_checkpoint is not None
        status = f"🚀 运行中...{resume_note}" if not resume_note else "🚀 运行中..."
        if SESSION.workflow_done:
            status = "✅ 工作流已完成"
        elif SESSION.error:
            status = f"❌ {SESSION.error}"
        elif at_checkpoint:
            status = f"⏸ 等待确认: {SESSION.last_checkpoint['label']}"

        yield (
            status,
            render_agent_grid(SESSION.active_agent, SESSION.done_agents),
            render_log(SESSION.event_log),
            render_checkpoint(SESSION.last_checkpoint),
            render_score_bars(SESSION.review_scores),
            render_live_panel_preview(SESSION.panel_images_preview),
            render_project_info(SESSION.state),
            gr.update(interactive=at_checkpoint),  # accept_btn
            gr.update(interactive=at_checkpoint),  # regen_btn
        )

        if SESSION.workflow_done or SESSION.error:
            break
        if at_checkpoint:
            break


def respond_checkpoint(decision: str) -> Iterator[tuple]:
    """响应当前确认点,继续执行工作流。"""
    if SESSION.workflow is None or SESSION.last_checkpoint is None:
        yield (
            "❌ 没有等待响应的确认点",
            render_agent_grid(SESSION.active_agent, SESSION.done_agents),
            render_log(SESSION.event_log),
            render_checkpoint(None),
            render_score_bars(SESSION.review_scores),
            render_live_panel_preview(SESSION.panel_images_preview),
            render_project_info(SESSION.state),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
        return

    SESSION.workflow.respond(decision)
    decision_label = {"accept": "接受并继续", "regenerate": "重新生成"}.get(decision, decision)
    SESSION.event_log.append({
        "agent": "user",
        "type": "log",
        "content": f"用户决策: {decision_label}",
        "timestamp": time.time(),
    })
    SESSION.last_checkpoint = None

    yield (
        f"✅ 已响应: {decision_label},工作流继续...",
        render_agent_grid(SESSION.active_agent, SESSION.done_agents),
        render_log(SESSION.event_log),
        render_checkpoint(None),
        render_score_bars(SESSION.review_scores),
        render_live_panel_preview(SESSION.panel_images_preview),
        render_project_info(SESSION.state),
        gr.update(interactive=False),  # 响应后立刻禁用，等待下一轮
        gr.update(interactive=False),
    )

    while True:
        time.sleep(0.5)
        at_checkpoint = SESSION.last_checkpoint is not None
        status = "🚀 运行中..."
        if SESSION.workflow_done:
            status = "✅ 工作流已完成"
        elif SESSION.error:
            status = f"❌ {SESSION.error}"
        elif at_checkpoint:
            status = f"⏸ 等待确认: {SESSION.last_checkpoint['label']}"

        yield (
            status,
            render_agent_grid(SESSION.active_agent, SESSION.done_agents),
            render_log(SESSION.event_log),
            render_checkpoint(SESSION.last_checkpoint),
            render_score_bars(SESSION.review_scores),
            render_live_panel_preview(SESSION.panel_images_preview),
            render_project_info(SESSION.state),
            gr.update(interactive=at_checkpoint),
            gr.update(interactive=at_checkpoint),
        )

        if SESSION.workflow_done or SESSION.error or at_checkpoint:
            break


def load_results(page_idx: int = 0, show_bubbles: bool = False) -> tuple[str, str, str, str, str, str, list]:
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return empty, empty, empty, empty, empty, empty, []

    script = SESSION.state.get("structured_script", {}) or {}
    curve = SESSION.state.get("emotion_curve", []) or []
    plan = SESSION.state.get("storyboard_plan", []) or []
    reviews = SESSION.state.get("review_results", []) or []
    final_pages = SESSION.state.get("final_pages", []) or []

    # 页面阅读器
    page_reader_html = render_page_reader(final_pages, page_idx, show_bubbles)
    # 分镜对比
    comparison_html = render_comparison_view(plan, final_pages)

    page_gallery = []
    for p in final_pages:
        path = p.get("image_path", "")
        if os.path.exists(path):
            page_gallery.append(path)

    return (
        render_script_summary(script),
        render_emotion_curve(curve),
        render_storyboard_preview(plan),
        render_review_history(reviews),
        page_reader_html,
        comparison_html,
        page_gallery,
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


def load_agent_outputs_tab() -> tuple[str, str, str, str, str]:
    """加载 Agent 输出 Tab 的所有内容。"""
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return empty, empty, empty, empty, empty

    agent_outputs = SESSION.agent_outputs
    character_db = SESSION.state.get("character_db") or {}
    plan = SESSION.state.get("storyboard_plan", []) or []
    panel_images = SESSION.state.get("panel_images", []) or []
    reviews = SESSION.state.get("review_results", []) or []

    return (
        render_agent_outputs(agent_outputs),
        render_character_profiles(character_db, agent_outputs),
        render_panel_images_gallery(panel_images),
        render_storyboard_detail(plan),
        render_review_full_detail(reviews),
    )


def load_dev_log_tab(filter_category: str = "all",
                     filter_level: str = "all") -> tuple[str, str, str]:
    """加载开发者日志 Tab 的所有内容。"""
    dev_entries = SESSION.dev_log
    if not dev_entries:
        empty = '<div style="color:#64748b;padding:12px;">暂无开发者日志。<br>请在「创作流程」Tab 启动工作流后自动生成。</div>'
        return empty, empty, empty

    # 统计
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


# ============================================================================
# 项目管理回调
# ============================================================================


def list_projects_ui() -> list[list]:
    """列出所有已保存的项目，返回 Dataframe 行数据。"""
    from comicweaver.storage import list_projects as storage_list_projects
    projects = storage_list_projects()
    rows: list[list] = []
    for p in projects:
        phase = p.state.get("current_phase", "init") if p.state else "init"
        rows.append([
            p.project_id,
            p.title,
            _format_timestamp(p.updated_at),
            phase,
        ])
    return rows


def _format_timestamp(ts: float) -> str:
    """格式化 Unix 时间戳为可读字符串。"""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M")


def open_project_callback(selected_row: list) -> tuple:
    """打开选中的已保存项目。"""
    from comicweaver.storage import load_project, project_to_state

    if not selected_row or not selected_row[0]:
        empty = '<div style="color:#64748b;padding:12px;">请先在项目列表中选择一个项目</div>'
        return ("❌ 未选择项目", empty, empty, empty, empty, empty, empty,
                gr.update(interactive=False), gr.update(interactive=False))

    project_id = str(selected_row[0])
    project = load_project(project_id)
    if project is None:
        empty = '<div style="color:#64748b;padding:12px;">项目不存在</div>'
        return (f"❌ 项目 {project_id} 不存在", empty, empty, empty, empty, empty, empty,
                gr.update(interactive=False), gr.update(interactive=False))

    state = project_to_state(project)
    SESSION.reset()
    SESSION.state = state
    SESSION.workflow = ComicWorkflow.from_saved_project(project_id)

    return (
        f"✅ 已加载项目: {project.title} ({project_id})",
        render_agent_grid(),
        render_log([]),
        render_checkpoint(None),
        render_score_bars({}),
        render_live_panel_preview([]),
        render_project_info(state),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def save_current_project() -> str:
    """手动保存当前项目。"""
    if SESSION.state is None:
        return "❌ 没有可保存的项目，请先创建项目"
    from comicweaver.storage import save_project, state_to_project
    try:
        proj = state_to_project(SESSION.state)
        save_project(proj)
        return f"✅ 项目已保存: {proj.title} ({proj.project_id})"
    except Exception as exc:
        return f"❌ 保存失败: {exc}"


# ============================================================================
# UI 构建
# ============================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ComicWeaver") as demo:
        # 头部
        gr.HTML("""
        <div class="cw-header">
            <h1>🎨 ComicWeaver</h1>
            <div class="subtitle">
                多Agent协作的自动化漫画创作系统 · v0.2.0 API-ready
            </div>
        </div>
        """)

        with gr.Tabs():
            # ---- Tab 1 ----
            with gr.Tab("📝 项目创建"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. 输入剧本主题或完整剧本")
                        user_input = gr.Textbox(
                            label="剧本内容",
                            placeholder="例如:\n"
                                        "• 简单模式: \"一个少年在雨夜的城市中寻找失踪的朋友\"\n"
                                        "• 详细模式: 提供分场景的详细描述\n"
                                        "• 改编模式: 粘贴小说原文",
                            lines=6,
                        )
                        project_name = gr.Textbox(
                            label="项目名称（可选）",
                            placeholder="留空则自动从剧本截取...",
                            value="",
                        )
                        gr.Markdown("### 2. 配置创作参数")
                        with gr.Row():
                            creation_mode = gr.Dropdown(
                                label="创作模式",
                                choices=[
                                    ("简单模式 - 一句话主题自动扩展", "simple"),
                                    ("专业模式 - 详细剧本", "detailed"),
                                    ("改编模式 - 小说→漫画", "adaptation"),
                                ],
                                value="simple",
                            )
                            style_preset = gr.Dropdown(
                                label="风格预设",
                                choices=[
                                    ("日漫风格", "manga"),
                                    ("美漫风格", "comic"),
                                    ("条漫风格", "webtoon"),
                                    ("写实风格", "realistic"),
                                ],
                                value="manga",
                            )
                        with gr.Row():
                            interaction_mode = gr.Dropdown(
                                label="交互模式",
                                choices=[
                                    ("半自动 - 关键节点确认 (推荐)", "semi_auto"),
                                    ("全自动 - 全程不打断", "full_auto"),
                                    ("全交互 - 每步确认", "full_interactive"),
                                ],
                                value="semi_auto",
                            )
                            target_pages = gr.Slider(
                                label="目标页数",
                                minimum=1, maximum=20, value=4, step=1,
                            )
                        create_btn = gr.Button(
                            "创建项目",
                            variant="primary",
                            elem_classes="cw-btn-primary",
                            size="lg",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 项目信息")
                        create_status = gr.Textbox(
                            label="状态",
                            interactive=False,
                            elem_classes="cw-card",
                        )
                        project_info = gr.Textbox(
                            label="详细信息",
                            lines=8,
                            interactive=False,
                            elem_classes="cw-card",
                        )

                create_btn.click(
                    create_project,
                    inputs=[user_input, project_name, creation_mode, style_preset,
                            interaction_mode, target_pages],
                    outputs=[create_status, project_info],
                )

            # ---- Tab 2 ----
            with gr.Tab("🚀 创作流程"):
                gr.Markdown("### 实时监控 6 个 Agent 的协作")

                gr.Markdown("#### 当前项目")
                project_info_html = gr.HTML(render_project_info())

                with gr.Row():
                    start_btn = gr.Button(
                        "▶ 启动工作流",
                        variant="primary",
                        elem_classes="cw-btn-primary",
                        size="lg",
                    )
                    workflow_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        scale=3,
                    )

                gr.Markdown("#### Agent 流水线")
                agent_grid_html = gr.HTML(render_agent_grid())

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 实时日志")
                        log_html = gr.HTML(render_log([]))
                    with gr.Column(scale=1):
                        gr.Markdown("#### 最近一次审查评分")
                        score_html = gr.HTML(render_score_bars({}))

                gr.Markdown("#### 实时面板预览")
                panel_preview_html = gr.HTML(render_live_panel_preview([]))

                gr.Markdown("#### 用户确认 (半自动模式)")
                checkpoint_html = gr.HTML(render_checkpoint(None))

                with gr.Row():
                    accept_btn = gr.Button("✅ 接受并继续",
                                            variant="primary",
                                            elem_classes="cw-btn-primary")
                    regen_btn = gr.Button("↻ 重新生成本阶段",
                                            elem_classes="cw-btn-secondary")

                with gr.Row():
                    save_btn = gr.Button("💾 保存当前项目",
                                         elem_classes="cw-btn-secondary")
                    save_status = gr.Textbox(
                        label="", interactive=False, scale=3,
                    )

                wf_outputs = [workflow_status, agent_grid_html, log_html,
                              checkpoint_html, score_html, panel_preview_html,
                              project_info_html, accept_btn, regen_btn]

                start_btn.click(start_workflow, outputs=wf_outputs)
                accept_btn.click(
                    lambda: (yield from respond_checkpoint("accept")),
                    outputs=wf_outputs,
                )
                regen_btn.click(
                    lambda: (yield from respond_checkpoint("regenerate")),
                    outputs=wf_outputs,
                )
                save_btn.click(
                    save_current_project,
                    outputs=[save_status],
                )

            # ---- Tab 3: 结果浏览（增强）----
            with gr.Tab("📖 结果浏览"):
                gr.Markdown("### 查看各阶段产出")
                with gr.Row():
                    load_btn = gr.Button("🔄 刷新结果", variant="primary")
                    download_btn = gr.DownloadButton(
                        "📥 下载所有页面",
                        variant="secondary",
                    )

                with gr.Accordion("📝 剧本结构", open=True):
                    script_html = gr.HTML()
                with gr.Accordion("📈 情感曲线", open=True):
                    curve_html = gr.HTML()
                with gr.Accordion("🎬 分镜规划", open=False):
                    storyboard_html = gr.HTML()

                # -- 逐页阅读器 --
                with gr.Accordion("🎨 漫画页面阅读器", open=True):
                    page_state = gr.State(0)
                    bubble_toggle = gr.State(False)

                    with gr.Row():
                        prev_btn = gr.Button("◀ 上一页", scale=1)
                        next_btn = gr.Button("下一页 ▶", scale=1)
                        bubble_checkbox = gr.Checkbox(
                            label="显示对话气泡边界",
                            value=False,
                            scale=2,
                        )

                    page_reader_html = gr.HTML(render_page_reader([], 0))

                with gr.Accordion("🔄 分镜 vs 成品对比", open=False):
                    comparison_html = gr.HTML()

                with gr.Accordion("✅ 审查历史", open=False):
                    review_html = gr.HTML()

                with gr.Accordion("🎨 页面缩略图", open=False):
                    page_gallery = gr.Gallery(
                        label="漫画页面",
                        columns=2,
                        height="auto",
                        elem_classes="cw-page-viewer",
                    )

                load_btn.click(
                    load_results,
                    inputs=[page_state, bubble_toggle],
                    outputs=[script_html, curve_html, storyboard_html,
                              review_html, page_reader_html, comparison_html,
                              page_gallery],
                )
                prev_btn.click(
                    lambda s, b: navigate_page("prev", s, b),
                    inputs=[page_state, bubble_toggle],
                    outputs=[page_reader_html, page_state],
                )
                next_btn.click(
                    lambda s, b: navigate_page("next", s, b),
                    inputs=[page_state, bubble_toggle],
                    outputs=[page_reader_html, page_state],
                )
                bubble_checkbox.change(
                    lambda s, b: (render_page_reader(
                        SESSION.state.get("final_pages", []) if SESSION.state else [],
                        s, b), b),
                    inputs=[page_state, bubble_checkbox],
                    outputs=[page_reader_html, bubble_toggle],
                )
                download_btn.click(
                    get_download_files,
                    outputs=[download_btn],
                )

            # ---- Tab 4: Agent 输出详情 ----
            with gr.Tab("📊 Agent输出"):
                gr.Markdown("### 各 Agent 的完整输入/输出详情")
                load_agent_btn = gr.Button("🔄 刷新 Agent 输出", variant="primary")

                with gr.Accordion("📦 原始输出结构", open=True):
                    agent_outputs_html = gr.HTML()
                with gr.Accordion("👤 角色档案", open=False):
                    character_profiles_html = gr.HTML()
                with gr.Accordion("🎨 面板图像画廊", open=False):
                    panel_gallery_html = gr.HTML()
                with gr.Accordion("🎬 分镜详细规划", open=False):
                    storyboard_detail_html = gr.HTML()
                with gr.Accordion("✅ 审查完整详情", open=False):
                    review_full_html = gr.HTML()

                load_agent_btn.click(
                    load_agent_outputs_tab,
                    outputs=[agent_outputs_html, character_profiles_html,
                              panel_gallery_html, storyboard_detail_html,
                              review_full_html],
                )

            # ---- Tab 5: 开发者日志 ----
            with gr.Tab("🔧 开发者日志"):
                gr.Markdown("### 结构化开发者日志（Agent 输入/输出/性能/错误）")

                with gr.Row():
                    with gr.Column(scale=1):
                        filter_category = gr.Dropdown(
                            label="筛选类别",
                            choices=[
                                ("全部", "all"),
                                ("Agent 输入", "agent_input"),
                                ("Agent 输出", "agent_output"),
                                ("状态变更", "state_change"),
                                ("审查决策", "review_decision"),
                                ("错误", "error"),
                                ("性能", "performance"),
                                ("确认点", "checkpoint"),
                                ("工作流", "workflow"),
                            ],
                            value="all",
                        )
                    with gr.Column(scale=1):
                        filter_level = gr.Dropdown(
                            label="筛选级别",
                            choices=[
                                ("全部", "all"),
                                ("TRACE", "trace"),
                                ("DEBUG", "debug"),
                                ("INFO", "info"),
                                ("WARN", "warn"),
                                ("ERROR", "error"),
                                ("PERF", "perf"),
                            ],
                            value="all",
                        )
                    with gr.Column(scale=1):
                        refresh_dev_btn = gr.Button("🔄 刷新日志", variant="primary")

                dev_stats_html = gr.HTML()
                dev_log_html = gr.HTML()
                with gr.Accordion("⏱ 性能统计", open=False):
                    perf_summary_html = gr.HTML()

                refresh_dev_btn.click(
                    load_dev_log_tab,
                    inputs=[filter_category, filter_level],
                    outputs=[dev_stats_html, dev_log_html, perf_summary_html],
                )

            # ---- Tab 7: 项目管理 ----
            with gr.Tab("📂 项目管理"):
                gr.Markdown("### 已保存的项目")

                with gr.Row():
                    refresh_projects_btn = gr.Button("🔄 刷新项目列表", variant="primary")
                    open_selected_btn = gr.Button("📂 打开选中项目", variant="primary")

                project_table = gr.Dataframe(
                    headers=["项目ID", "名称", "更新时间", "当前阶段"],
                    datatype=["str", "str", "str", "str"],
                    label="项目列表",
                    interactive=False,
                    row_count=(5, "dynamic"),
                )
                project_select = gr.State([])

                open_project_status = gr.Textbox(
                    label="操作状态", interactive=False,
                    elem_classes="cw-card",
                )

                # 当打开项目后，同步刷新结果视图的占位组件
                open_script_html = gr.HTML(visible=False)
                open_curve_html = gr.HTML(visible=False)
                open_storyboard_html = gr.HTML(visible=False)
                open_review_html = gr.HTML(visible=False)

                refresh_projects_btn.click(
                    list_projects_ui,
                    outputs=[project_table],
                )
                open_selected_btn.click(
                    open_project_callback,
                    inputs=[project_table],
                    outputs=[open_project_status, agent_grid_html, log_html,
                             checkpoint_html, score_html, panel_preview_html,
                             project_info_html, accept_btn, regen_btn],
                )

            # ---- Tab 6: 关于 ----
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                ## ComicWeaver v0.2.0 — LangGraph 重构版

                **当前状态:** 基于 LangGraph 工作流引擎，默认使用本地 fallback，
                可通过配置接入 LLM API 与 ComfyUI。未配置真实后端时运行无需 GPU。

                ### 系统组成
                - **5 个生产 Agent:** 剧本 / 角色 / 分镜 / 图像 / 排版
                - **1 个审查 Agent:** 横向贯穿，质量守门
                - **编排层:** 基于 LangGraph StateGraph 的工作流引擎
                - **3 种交互模式:** 半自动 / 全自动 / 全交互
                - **开发者日志:** 结构化 Agent 输入/输出/性能/错误追踪

                ### 新增功能 (v0.2.0)
                1. LangGraph StateGraph 重构编排层
                2. 完整的 Agent 输入/输出可视化
                3. 结构化开发者日志系统（5 个级别 × 8 个类别）
                4. 面板图像独立画廊
                5. 分镜详细规划展示
                6. 角色档案卡片 + 参考图预览
                7. 完整审查详情（Schema 校验/质量评分/升级处理）
                8. 性能计时统计

                ### 文档
                - [LangGraph 重构说明](LangGraph重构说明.md)
                - [UI输出与日志改进方案](UI输出与日志改进方案.md)
                """)

        gr.HTML("""
        <div style="text-align:center;padding:16px;color:#94a3b8;font-size:12px;">
            ComicWeaver © 2026 · 复旦大学计算机图形学课程项目
        </div>
        """)

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
