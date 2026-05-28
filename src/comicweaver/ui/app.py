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
    render_checkpoint,
    render_emotion_curve,
    render_log,
    render_review_history,
    render_score_bars,
    render_script_summary,
    render_storyboard_preview,
)
from .styles import CUSTOM_CSS

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

            # StreamEvent
            if hasattr(msg, "agent") and hasattr(msg, "type"):
                session.event_log.append({
                    "agent": msg.agent,
                    "type": msg.type.value if hasattr(msg.type, "value") else str(msg.type),
                    "content": msg.content if not isinstance(msg.content, dict)
                                else f"[partial] {list(msg.content.keys())}",
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
        creation_mode=creation_mode,
        style_preset=style_preset,
        interaction_mode=interaction_mode,
        target_pages=target_pages,
    )

    project_id = SESSION.state["project_id"]
    return (
        f"✅ 项目已创建: {project_id}",
        f"项目ID: {project_id}\n模式: {creation_mode} / {interaction_mode}\n"
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
        )
        return

    SESSION.workflow = ComicWorkflow()
    SESSION.event_log = []
    SESSION.done_agents = []
    SESSION.workflow_done = False
    SESSION.last_checkpoint = None

    loop = _ensure_loop(SESSION)
    SESSION._task = asyncio.run_coroutine_threadsafe(
        _consume_workflow(SESSION), loop
    )

    # 周期性轮询session状态
    yield (
        "🚀 工作流已启动...",
        render_agent_grid(SESSION.active_agent, SESSION.done_agents),
        render_log(SESSION.event_log),
        render_checkpoint(SESSION.last_checkpoint),
        render_score_bars(SESSION.review_scores),
    )

    while True:
        time.sleep(0.5)
        status = "🚀 运行中..."
        if SESSION.workflow_done:
            status = "✅ 工作流已完成"
        elif SESSION.error:
            status = f"❌ {SESSION.error}"
        elif SESSION.last_checkpoint:
            status = f"⏸ 等待确认: {SESSION.last_checkpoint['label']}"

        yield (
            status,
            render_agent_grid(SESSION.active_agent, SESSION.done_agents),
            render_log(SESSION.event_log),
            render_checkpoint(SESSION.last_checkpoint),
            render_score_bars(SESSION.review_scores),
        )

        if SESSION.workflow_done or SESSION.error:
            break
        if SESSION.last_checkpoint:
            # 暂停在checkpoint,等待用户响应
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
        )
        return

    SESSION.workflow.respond(decision)
    SESSION.event_log.append({
        "agent": "user",
        "type": "log",
        "content": f"用户决策: {decision}",
        "timestamp": time.time(),
    })
    SESSION.last_checkpoint = None

    yield (
        f"✅ 已响应: {decision},工作流继续...",
        render_agent_grid(SESSION.active_agent, SESSION.done_agents),
        render_log(SESSION.event_log),
        render_checkpoint(None),
        render_score_bars(SESSION.review_scores),
    )

    while True:
        time.sleep(0.5)
        status = "🚀 运行中..."
        if SESSION.workflow_done:
            status = "✅ 工作流已完成"
        elif SESSION.error:
            status = f"❌ {SESSION.error}"
        elif SESSION.last_checkpoint:
            status = f"⏸ 等待确认: {SESSION.last_checkpoint['label']}"

        yield (
            status,
            render_agent_grid(SESSION.active_agent, SESSION.done_agents),
            render_log(SESSION.event_log),
            render_checkpoint(SESSION.last_checkpoint),
            render_score_bars(SESSION.review_scores),
        )

        if SESSION.workflow_done or SESSION.error or SESSION.last_checkpoint:
            break


def load_results() -> tuple[str, str, str, str, list]:
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return empty, empty, empty, empty, []

    script = SESSION.state.get("structured_script", {}) or {}
    curve = SESSION.state.get("emotion_curve", []) or []
    plan = SESSION.state.get("storyboard_plan", []) or []
    reviews = SESSION.state.get("review_results", []) or []
    final_pages = SESSION.state.get("final_pages", []) or []

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
        page_gallery,
    )


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
                    inputs=[user_input, creation_mode, style_preset,
                            interaction_mode, target_pages],
                    outputs=[create_status, project_info],
                )

            # ---- Tab 2 ----
            with gr.Tab("🚀 创作流程"):
                gr.Markdown("### 实时监控 6 个 Agent 的协作")

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

                gr.Markdown("#### 用户确认 (半自动模式)")
                checkpoint_html = gr.HTML(render_checkpoint(None))

                with gr.Row():
                    accept_btn = gr.Button("✅ 接受并继续",
                                            variant="primary",
                                            elem_classes="cw-btn-primary")
                    regen_btn = gr.Button("↻ 重新生成本阶段",
                                            elem_classes="cw-btn-secondary")
                    skip_btn = gr.Button("⏭ 跳过此阶段",
                                            elem_classes="cw-btn-secondary")

                start_btn.click(
                    start_workflow,
                    outputs=[workflow_status, agent_grid_html, log_html,
                              checkpoint_html, score_html],
                )
                accept_btn.click(
                    lambda: (yield from respond_checkpoint("accept")),
                    outputs=[workflow_status, agent_grid_html, log_html,
                              checkpoint_html, score_html],
                )
                regen_btn.click(
                    lambda: (yield from respond_checkpoint("regenerate")),
                    outputs=[workflow_status, agent_grid_html, log_html,
                              checkpoint_html, score_html],
                )
                skip_btn.click(
                    lambda: (yield from respond_checkpoint("skip")),
                    outputs=[workflow_status, agent_grid_html, log_html,
                              checkpoint_html, score_html],
                )

            # ---- Tab 3 ----
            with gr.Tab("📖 结果浏览"):
                gr.Markdown("### 查看各阶段产出")
                load_btn = gr.Button("🔄 刷新结果", variant="primary")

                with gr.Accordion("📝 剧本结构", open=True):
                    script_html = gr.HTML()
                with gr.Accordion("📈 情感曲线", open=True):
                    curve_html = gr.HTML()
                with gr.Accordion("🎬 分镜规划", open=False):
                    storyboard_html = gr.HTML()
                with gr.Accordion("✅ 审查历史", open=False):
                    review_html = gr.HTML()
                with gr.Accordion("🎨 最终漫画页面", open=True):
                    page_gallery = gr.Gallery(
                        label="漫画页面",
                        columns=2,
                        height="auto",
                        elem_classes="cw-page-viewer",
                    )

                load_btn.click(
                    load_results,
                    outputs=[script_html, curve_html, storyboard_html,
                              review_html, page_gallery],
                )

            # ---- Tab 4: 关于 ----
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                ## ComicWeaver v0.1.0 - 初始框架

                **当前状态:** 默认使用本地 fallback,可通过配置接入 LLM API 与 ComfyUI。
                未配置真实后端时运行无需 GPU。

                ### 系统组成
                - **5 个生产 Agent:** 剧本 / 角色 / 分镜 / 图像 / 排版
                - **1 个审查 Agent:** 横向贯穿,质量守门
                - **编排层:** 支持 LangGraph 风格的状态机工作流
                - **3 种交互模式:** 半自动 / 全自动 / 全交互

                ### 后续迭代
                1. 接入真实 LLM (Qwen2.5-7B / ChatGLM4)
                2. 接入 ComfyUI / SDXL 图像生成
                3. 实现 IP-Adapter 链式参考机制
                4. 完善对话气泡的智能避让算法
                5. 多格式导出 (PDF / PSD)

                ### 文档
                - [设计文档](docs/开发思路框架.md)
                - [Agent 接口规范](docs/agents/README.md)
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
