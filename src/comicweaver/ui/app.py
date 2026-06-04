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
from pathlib import Path

import gradio as gr

from comicweaver.core import ComicState, make_initial_state
from comicweaver.orchestrator import CheckpointSignal, ComicWorkflow, WorkflowEvent
from comicweaver.storage import save_project, state_to_project

from .html_widgets import (
    render_agent_outputs,
    render_agent_status_bar,
    render_character_profiles,
    render_comparison_view,
    render_dev_log,
    render_emotion_curve,
    render_inline_checkpoint,
    render_live_content,
    render_live_panel_preview,
    render_page_reader,
    render_panel_images_gallery,
    render_performance_summary,
    render_phase_text_content,
    render_project_cards,
    render_project_info,
    render_review_full_detail,
    render_review_history,
    render_score_bars,
    render_script_summary,
    render_storyboard_detail,
    render_storyboard_preview,
    render_stream_log,
    render_workflow_left,
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
        self.character_preview: list[dict] = []          # 角色人设图预览
        # v0.4: 时间追踪和阶段信息
        self.agent_start_time: float = 0.0              # 当前 agent 开始时间
        self.current_phase: str = "init"                # 工作流当前阶段
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
        self.character_preview = []
        self.agent_start_time = 0.0
        self.current_phase = "init"
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
                    session.agent_start_time = time.time()
                    # 更新 current_phase
                    agent_to_phase = {
                        "story_agent": "story",
                        "script_agent": "script", "character_agent": "character",
                        "storyboard_agent": "storyboard", "image_agent": "image",
                        "layout_agent": "layout",
                    }
                    session.current_phase = agent_to_phase.get(msg.node, "init")
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
                        # 实时预览：角色人设图
                        if agent_name == "character_agent" and "character_db" in content:
                            cdb = content.get("character_db") or {}
                            chars = cdb.get("characters", {})
                            for cid, cp in chars.items():
                                ref = cp.get("base_reference", {})
                                img_path = ref.get("image_path", "")
                                if img_path:
                                    session.character_preview.append({
                                        "kind": "character",
                                        "char_id": cid,
                                        "name": cp.get("name", cid),
                                        "image_path": img_path,
                                    })
                        # 实时面板预览：image_agent 完成时收集图像
                        if agent_name == "image_agent" and "panel_image" in content:
                            session.panel_images_preview.append(content["panel_image"])
                            # 同步更新 state.panel_images（逐面板累积）
                            pis = session.state.setdefault("panel_images", [])
                            pis.append(content["panel_image"])
                        # 同步更新 session.state：LangGraph 仅在节点返回时同步 state，
                        # 但 checkpoint 在节点返回前触发，故需在此提前写入
                        if session.state is not None:
                            if agent_name == "story_agent":
                                session.state["developed_story"] = content
                                session.state["narrative_structure"] = content.get("narrative_structure", {})
                            elif agent_name == "script_agent":
                                session.state["structured_script"] = content
                                session.state["emotion_curve"] = content.get("emotion_curve", [])
                            elif agent_name == "character_agent":
                                session.state["character_db"] = content.get("character_db") or {}
                            elif agent_name == "storyboard_agent":
                                pages = content.get("pages", [])
                                if pages:
                                    session.state["storyboard_plan"] = pages
                            elif agent_name == "layout_agent":
                                session.state["final_pages"] = content.get("final_pages", [])
                                session.state["exports"] = content.get("exports", [])
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


# 阶段定义：顺序、前置依赖、需要清除的字段
_PHASE_PREREQUISITES: dict[str, list[str]] = {
    "init": [],
    "story": [],
    "script": ["developed_story"],
    "character": ["structured_script"],
    "storyboard": ["structured_script", "character_db"],
    "image": ["structured_script", "character_db", "storyboard_plan"],
    "layout": ["structured_script", "character_db", "storyboard_plan", "panel_images"],
}
# 每个阶段需要清除的输出字段（从该阶段开始，清除自身及之后所有阶段的输出）
_PHASE_CLEAR_FIELDS: dict[str, list[str]] = {
    "init": [
        "developed_story", "narrative_structure",
        "structured_script", "emotion_curve", "character_db", "reference_chain",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "story": [
        "developed_story", "narrative_structure",
        "structured_script", "emotion_curve", "character_db", "reference_chain",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "script": [
        "structured_script", "emotion_curve", "character_db", "reference_chain",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "character": [
        "character_db", "reference_chain",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "storyboard": [
        "storyboard_plan", "layout_grids",
        "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "image": [
        "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "layout": [
        "final_pages", "exports",
    ],
}


def _validate_and_reset_state(state: ComicState, start_phase: str) -> str | None:
    """验证前置条件并清除对应阶段的输出数据。

    返回 None 表示成功，否则返回错误消息字符串。
    """
    # 1. 检查前置依赖
    prerequisites = _PHASE_PREREQUISITES.get(start_phase, [])
    for field in prerequisites:
        value = state.get(field)
        if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
            phase_label = _PHASE_LABELS_CN.get(start_phase, start_phase)
            field_label = {
                "structured_script": "剧本",
                "character_db": "角色设计",
                "storyboard_plan": "分镜规划",
                "panel_images": "面板图像",
            }.get(field, field)
            return f"❌ 无法从「{phase_label}」阶段开始：缺少前置数据「{field_label}」。\n请先从头开始或从已完成的更早阶段开始。"

    # 2. 清除该阶段及之后阶段的输出
    clear_fields = _PHASE_CLEAR_FIELDS.get(start_phase, [])
    for field in clear_fields:
        if field in state:
            if field in ("emotion_curve", "reference_chain", "storyboard_plan",
                         "layout_grids", "panel_images", "generation_metadata",
                         "final_pages", "exports"):
                state[field] = []  # type: ignore[literal-required]
            else:
                state[field] = {}  # type: ignore[literal-required]

    # 3. 始终清除的通用字段
    state["review_results"] = []
    state["retry_counts"] = {}
    state["pending_checkpoint"] = None
    state["user_decisions"] = []
    state["stream_messages"] = []
    state["errors"] = []

    # 4. 更新 current_phase
    state["current_phase"] = start_phase

    return None


_PHASE_LABELS_CN = {
    "init": "从头开始", "story": "故事阶段",
    "script": "剧本阶段", "character": "角色阶段",
    "storyboard": "分镜阶段", "image": "图像阶段", "layout": "排版阶段",
}


def start_workflow(start_phase: str = "init") -> Iterator[tuple]:
    """启动工作流并周期性产出UI更新（v0.4 7元组）。

    start_phase: "init"(默认从头开始) 或 "script"/"character"/"storyboard"/"image"/"layout"
    """
    if SESSION.state is None:
        yield (
            "❌ 请先在「📋 项目」Tab 创建或打开项目",
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            render_project_info(),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, value=""),
        )
        return

    # 确定实际启动阶段
    if start_phase and start_phase != "init":
        resume_phase = start_phase
    else:
        resume_phase = None  # None = 从头开始 (init → script)

    # 验证前置条件，清除该阶段及之后的输出数据
    effective_phase = resume_phase or "init"
    error_msg = _validate_and_reset_state(SESSION.state, effective_phase)
    if error_msg:
        yield (
            error_msg,
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            render_project_info(SESSION.state),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, value=""),
        )
        return

    is_resume = resume_phase is not None

    # 始终创建新的 workflow 实例（使用用户选择的阶段）
    SESSION.workflow = ComicWorkflow(resume_phase=resume_phase)
    SESSION.event_log = []
    SESSION.done_agents = []
    SESSION.workflow_done = False
    SESSION.last_checkpoint = None
    SESSION.panel_images_preview = []
    SESSION.character_preview = []
    SESSION.agent_outputs = {}
    SESSION.agent_start_time = time.time()

    loop = _ensure_loop(SESSION)
    SESSION._task = asyncio.run_coroutine_threadsafe(
        _consume_workflow(SESSION), loop
    )

    resume_note = f" (从 {resume_phase} 恢复)" if is_resume else ""
    yield (
        f"🚀 工作流已启动{resume_note}...",
        render_workflow_left(
            SESSION.active_agent, SESSION.done_agents,
            SESSION.event_log, SESSION.last_checkpoint,
        ),
        render_live_content(
            SESSION.panel_images_preview, SESSION.character_preview,
            render_phase_text_content(SESSION.state, SESSION.agent_outputs),
        ),
        render_project_info(SESSION.state),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False, value=""),
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

        # 计算当前 agent 耗时
        elapsed = 0.0
        if SESSION.active_agent and SESSION.agent_start_time > 0:
            elapsed = time.time() - SESSION.agent_start_time

        yield (
            status,
            render_workflow_left(
                SESSION.active_agent, SESSION.done_agents,
                SESSION.event_log, SESSION.last_checkpoint,
                elapsed,
            ),
            render_live_content(
                SESSION.panel_images_preview, SESSION.character_preview,
                render_phase_text_content(SESSION.state, SESSION.agent_outputs),
            ),
            render_project_info(SESSION.state),
            gr.update(interactive=at_checkpoint),
            gr.update(interactive=at_checkpoint),
            gr.update(interactive=at_checkpoint, value=""),
        )

        if SESSION.workflow_done or SESSION.error:
            break
        if at_checkpoint:
            break


def respond_checkpoint(decision: str, guidance: str = "") -> Iterator[tuple]:
    """响应当前确认点,继续执行工作流（v0.4 7元组）。

    decision: "accept" 或 "regenerate"
    guidance: 用户可选的指导信息
    """
    if SESSION.workflow is None or SESSION.last_checkpoint is None:
        yield (
            "❌ 没有等待响应的确认点",
            render_workflow_left(
                SESSION.active_agent, SESSION.done_agents,
                SESSION.event_log, SESSION.last_checkpoint,
            ),
            render_live_content([], [], ""),
            render_project_info(SESSION.state),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, value=""),
        )
        return

    # 发送响应（带指导信息）
    SESSION.workflow.respond(decision, guidance)
    decision_label = {"accept": "接受并继续", "regenerate": "重新生成"}.get(decision, decision)
    guidance_note = f" (指导: {guidance[:50]})" if guidance else ""
    SESSION.event_log.append({
        "agent": "user",
        "type": "log",
        "content": f"用户决策: {decision_label}{guidance_note}",
        "timestamp": time.time(),
    })

    # 如果是重新生成，清除对应阶段的预览数据和 state 字段
    if decision == "regenerate":
        cp_id = SESSION.last_checkpoint.get("id", "")
        if cp_id == "after_story":
            SESSION.state["developed_story"] = {}
            SESSION.state["narrative_structure"] = {}
        elif cp_id == "after_script":
            SESSION.state["structured_script"] = {}
            SESSION.state["emotion_curve"] = []
        elif cp_id == "after_character":
            SESSION.state["character_db"] = {}
            SESSION.character_preview = []
        elif cp_id == "after_storyboard":
            SESSION.state["storyboard_plan"] = []
        elif cp_id == "after_image":
            SESSION.state["panel_images"] = []
            SESSION.panel_images_preview = []
        elif cp_id == "after_layout":
            SESSION.state["final_pages"] = []
            SESSION.state["exports"] = []
            SESSION.panel_images_preview = []

    SESSION.last_checkpoint = None

    yield (
        f"✅ 已响应: {decision_label},工作流继续...",
        render_workflow_left(
            SESSION.active_agent, SESSION.done_agents,
            SESSION.event_log, SESSION.last_checkpoint,
        ),
        render_live_content(
            SESSION.panel_images_preview, SESSION.character_preview,
            render_phase_text_content(SESSION.state, SESSION.agent_outputs),
        ),
        render_project_info(SESSION.state),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False, value=""),
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

        elapsed = 0.0
        if SESSION.active_agent and SESSION.agent_start_time > 0:
            elapsed = time.time() - SESSION.agent_start_time

        yield (
            status,
            render_workflow_left(
                SESSION.active_agent, SESSION.done_agents,
                SESSION.event_log, SESSION.last_checkpoint,
                elapsed,
            ),
            render_live_content(
                SESSION.panel_images_preview, SESSION.character_preview,
                render_phase_text_content(SESSION.state, SESSION.agent_outputs),
            ),
            render_project_info(SESSION.state),
            gr.update(interactive=at_checkpoint),
            gr.update(interactive=at_checkpoint),
            gr.update(interactive=at_checkpoint, value="" if not at_checkpoint else None),
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


def list_projects_cards() -> str:
    """列出已保存项目，渲染为卡片网格 HTML。"""
    from comicweaver.storage import list_projects as storage_list_projects
    projects = storage_list_projects()
    card_data: list[dict] = []
    import datetime
    for p in projects:
        phase = p.state.get("current_phase", "init") if p.state else "init"
        dt = datetime.datetime.fromtimestamp(p.updated_at)
        card_data.append({
            "project_id": p.project_id,
            "title": p.title,
            "updated_at_str": dt.strftime("%Y-%m-%d %H:%M"),
            "phase": phase,
        })
    return render_project_cards(card_data)


def open_project_by_id(project_id: str) -> tuple[str, str, str, str, str, str, dict]:
    """通过项目 ID 字符串打开已保存项目。
    返回 (status, cards, proj_info, left_panel, right_panel, start_phase, tabs_update)。
    """
    from comicweaver.storage import load_project, project_to_state

    pid = (project_id or "").strip()
    if not pid:
        return (
            "❌ 请输入项目ID", list_projects_cards(), render_project_info(),
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            "init",
            gr.update(),
        )

    project = load_project(pid)
    if project is None:
        return (
            f"❌ 项目 {pid} 不存在", list_projects_cards(), render_project_info(),
            render_workflow_left("", [], [], None),
            render_live_content([], [], ""),
            "init",
            gr.update(),
        )

    state = project_to_state(project)
    SESSION.reset()
    SESSION.state = state

    # 从已保存状态中提取角色预览图
    character_db = state.get("character_db", {})
    char_preview: list[dict] = []
    for cid, cp in character_db.get("characters", {}).items():
        ref = cp.get("base_reference", {})
        img_path = ref.get("image_path", "")
        if img_path:
            char_preview.append({
                "kind": "character", "char_id": cid,
                "name": cp.get("name", cid), "image_path": img_path,
            })

    # 从已保存状态中提取面板图像预览
    panel_images = state.get("panel_images", [])
    panel_preview: list[dict] = []
    for pi in panel_images:
        img_path = pi.get("image_path", "")
        if img_path:
            panel_preview.append({"panel_id": pi.get("panel_id", "?"), "image_path": img_path})

    # 渲染右侧面板：已保存阶段的文字内容 + 图像预览
    right_html = render_live_content(
        panel_preview, char_preview,
        render_phase_text_content(state, SESSION.agent_outputs),
    )

    # 渲染左侧面板：显示当前阶段,无活跃/已完成 agent
    current_phase = state.get("current_phase", "init")
    left_html = render_workflow_left("", [], [], None, current_phase=current_phase)

    # 自动设置 start_phase 为下一阶段（半自动模式下继续工作流）
    _next_phase: dict[str, str] = {
        "init": "story",
        "story": "script",
        "script": "character",
        "character": "storyboard",
        "storyboard": "image",
        "image": "layout",
        "layout": "layout",
    }
    next_phase = _next_phase.get(current_phase, "init")

    return (
        f"✅ 已加载: {project.title} — 已自动切换到「⚡ 创作」Tab",
        list_projects_cards(),
        render_project_info(state),
        left_html,
        right_html,
        next_phase,
        gr.update(selected="⚡ 创作"),
    )


def load_all_results(page_idx: int = 0, show_bubbles: bool = False) -> tuple:
    """Tab 3 预览：加载所有结果数据。"""
    if SESSION.state is None:
        empty = '<div style="color:#64748b;padding:12px;">请先创建项目并运行工作流</div>'
        return (empty, empty, empty, empty, empty, empty, empty)

    state = SESSION.state
    script = state.get("structured_script", {}) or {}
    curve = state.get("emotion_curve", []) or []
    plan = state.get("storyboard_plan", []) or []
    reviews = state.get("review_results", []) or []
    final_pages = state.get("final_pages", []) or []
    character_db = state.get("character_db", {}) or {}

    return (
        render_script_summary(script),
        render_emotion_curve(curve),
        render_storyboard_preview(plan),
        render_review_history(reviews),
        render_page_reader(final_pages, page_idx, show_bubbles),
        render_comparison_view(plan, final_pages),
        render_character_profiles(character_db, SESSION.agent_outputs),
    )


def load_dev_log_compact(filter_category: str = "all",
                         filter_level: str = "all") -> str:
    """紧凑版开发者日志（用于调试折叠面板）。"""
    dev_entries = SESSION.dev_log
    if not dev_entries:
        return '<div style="color:#64748b;padding:12px;">暂无日志。启动工作流后自动生成。</div>'
    return render_dev_log(dev_entries, filter_category, filter_level)


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

HEAD_HTML = """<script>
(function() {
    function _findInput(wrapId) {
        var wrap = document.getElementById(wrapId);
        if (!wrap) return null;
        return wrap.querySelector('textarea, input');
    }
    function _setNativeValue(field, value) {
        var proto = field instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        var desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) desc.set.call(field, value);
        else field.value = value;
        field.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: value}));
        field.dispatchEvent(new Event('change', {bubbles: true}));
        field.focus();
    }
    document.addEventListener('click', function(e) {
        var card = e.target.closest('.cw-project-card');
        if (!card) return;
        var pid = card.getAttribute('data-project-id');
        if (!pid) return;
        e.preventDefault();
        var field = _findInput('open-project-id-input');
        if (field) {
            _setNativeValue(field, pid);
        }
    });
})();
</script>"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ComicWeaver") as demo:
        # 头部
        gr.HTML("""
        <div class="cw-header">
            <h1>🎨 ComicWeaver</h1>
            <div class="subtitle">
                多Agent协作的自动化漫画创作系统 · v0.4.0
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ================================================================
            # Tab 1: 📋 项目
            # ================================================================
            with gr.Tab("📋 项目"):
                with gr.Row():
                    # 左侧：创建新项目
                    with gr.Column(scale=2):
                        gr.Markdown("### ✨ 创建新项目")
                        user_input = gr.Textbox(
                            label="剧本内容",
                            placeholder="输入一句话主题或完整剧本...",
                            lines=4,
                        )
                        with gr.Row():
                            project_name = gr.Textbox(
                                label="项目名称（可选）",
                                placeholder="留空则自动截取...",
                                scale=2,
                            )
                            target_pages = gr.Slider(
                                label="目标页数", minimum=1, maximum=20,
                                value=4, step=1, scale=1,
                            )
                        with gr.Row():
                            creation_mode = gr.Dropdown(
                                label="创作模式",
                                choices=[("简单", "simple"), ("专业", "detailed"), ("改编", "adaptation")],
                                value="simple", scale=1,
                            )
                            style_preset = gr.Dropdown(
                                label="风格",
                                choices=[("日漫", "manga"), ("美漫", "comic"), ("条漫", "webtoon"), ("写实", "realistic")],
                                value="manga", scale=1,
                            )
                            interaction_mode = gr.Dropdown(
                                label="交互",
                                choices=[("半自动", "semi_auto"), ("全自动", "full_auto"), ("全交互", "full_interactive")],
                                value="semi_auto", scale=1,
                            )
                        create_btn = gr.Button(
                            "🚀 创建项目", variant="primary",
                            elem_classes="cw-btn-primary", size="lg",
                        )
                        create_status = gr.Textbox(label="", interactive=False, elem_id="create-project-status")

                    # 右侧：已保存项目
                    with gr.Column(scale=3):
                        with gr.Row():
                            gr.Markdown("### 📂 已保存的项目")
                            refresh_projects_btn = gr.Button(
                                "🔄 刷新", scale=0, elem_classes="cw-btn-secondary",
                            )
                        project_cards_html = gr.HTML(render_project_cards([]))
                        with gr.Row():
                            open_project_id = gr.Textbox(
                                label="输入项目ID打开", placeholder="粘贴项目ID...",
                                scale=2, elem_id="open-project-id-input",
                            )
                            open_btn = gr.Button(
                                "📂 打开", scale=0, elem_classes="cw-btn-primary",
                                elem_id="open-project-btn",
                            )
                            save_current_btn = gr.Button(
                                "💾 保存当前", scale=0, elem_classes="cw-btn-secondary",
                            )

                create_btn.click(
                    create_project,
                    inputs=[user_input, project_name, creation_mode, style_preset,
                            interaction_mode, target_pages],
                    outputs=[create_status],
                ).then(
                    list_projects_cards, outputs=[project_cards_html],
                )
                refresh_projects_btn.click(
                    list_projects_cards, outputs=[project_cards_html],
                )
                # open_btn.click 在下方所有 Tab 定义完成后注册
                save_current_btn.click(
                    save_current_project,
                    outputs=[create_status],
                ).then(
                    list_projects_cards, outputs=[project_cards_html],
                )

            # ================================================================
            # Tab 2: ⚡ 创作  (Claude Code 风格)
            # ================================================================
            with gr.Tab("⚡ 创作"):
                # 项目信息栏
                project_info_html = gr.HTML(render_project_info())

                # 控制栏（含阶段选择）
                with gr.Row():
                    start_phase = gr.Dropdown(
                        label="起始阶段",
                        choices=[
                            ("从头开始", "init"),
                            ("故事阶段", "story"),
                            ("剧本阶段", "script"),
                            ("角色阶段", "character"),
                            ("分镜阶段", "storyboard"),
                            ("图像阶段", "image"),
                            ("排版阶段", "layout"),
                        ],
                        value="init", scale=1,
                    )
                    start_btn = gr.Button(
                        "▶ 启动工作流", variant="primary",
                        elem_classes="cw-btn-primary", size="lg", scale=2,
                    )
                    workflow_status = gr.Textbox(
                        label="", interactive=False, scale=4,
                        placeholder="就绪 — 请先创建或打开项目",
                    )
                    save_wf_btn = gr.Button(
                        "💾 保存", elem_classes="cw-btn-secondary", scale=0,
                    )

                # 左右分栏：流式日志 | 实时预览（文字+图像）
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        left_panel_html = gr.HTML(
                            render_workflow_left("", [], [], None)
                        )
                        # 指导输入 + 按钮组
                        guidance_input = gr.Textbox(
                            label="🎯 用户指导（可选）",
                            placeholder="例如：'让角色看起来更年轻一些'、'给第二页增加一个特写镜头'...",
                            lines=2,
                            interactive=False,
                            elem_classes="cw-guidance-input",
                        )
                        with gr.Row():
                            accept_btn = gr.Button(
                                "✅ 接受并继续", variant="primary",
                                elem_classes="cw-btn-primary", interactive=False,
                            )
                            regen_btn = gr.Button(
                                "↻ 重新生成", elem_classes="cw-btn-secondary",
                                interactive=False,
                            )
                    with gr.Column(scale=1):
                        panel_preview_html = gr.HTML(
                            render_live_content([], [], "")
                        )

                # 调试面板（折叠）
                with gr.Accordion("🔧 开发者调试", open=False):
                    with gr.Row():
                        dev_filter_category = gr.Dropdown(
                            label="类别", value="all", scale=1,
                            choices=[("全部","all"),("输入","agent_input"),("输出","agent_output"),
                                     ("状态","state_change"),("审查","review_decision"),
                                     ("错误","error"),("性能","performance"),("工作流","workflow")],
                        )
                        dev_filter_level = gr.Dropdown(
                            label="级别", value="all", scale=1,
                            choices=[("全部","all"),("TRACE","trace"),("DEBUG","debug"),
                                     ("INFO","info"),("WARN","warn"),("ERROR","error")],
                        )
                        dev_refresh_btn = gr.Button("🔄 刷新", scale=0)
                    dev_html = gr.HTML()

                # 输出列表 (v0.4: 7元组)
                wf_outputs = [
                    workflow_status, left_panel_html, panel_preview_html,
                    project_info_html, accept_btn, regen_btn, guidance_input,
                ]

                start_btn.click(
                    start_workflow,
                    inputs=[start_phase],
                    outputs=wf_outputs,
                )
                accept_btn.click(
                    lambda g: (yield from respond_checkpoint("accept", g)),
                    inputs=[guidance_input],
                    outputs=wf_outputs,
                )
                regen_btn.click(
                    lambda g: (yield from respond_checkpoint("regenerate", g)),
                    inputs=[guidance_input],
                    outputs=wf_outputs,
                )
                save_wf_btn.click(
                    save_current_project,
                    outputs=[workflow_status],
                )
                dev_refresh_btn.click(
                    load_dev_log_compact,
                    inputs=[dev_filter_category, dev_filter_level],
                    outputs=[dev_html],
                )

            # ================================================================
            # Tab 3: 📖 预览
            # ================================================================
            with gr.Tab("📖 预览"):
                with gr.Row():
                    load_results_btn = gr.Button("🔄 刷新", variant="primary", scale=0)
                    download_btn = gr.DownloadButton("📥 下载所有页面", variant="secondary", scale=0)

                # 逐页阅读器
                gr.Markdown("### 🎨 漫画页面")
                page_state = gr.State(0)
                bubble_toggle_state = gr.State(False)
                with gr.Row():
                    prev_btn = gr.Button("◀ 上一页", scale=1)
                    next_btn = gr.Button("下一页 ▶", scale=1)
                    bubble_checkbox = gr.Checkbox(
                        label="显示对话气泡边界", value=False, scale=2,
                    )
                page_reader_html = gr.HTML(render_page_reader([], 0))

                # 详情 Accordion
                with gr.Accordion("📝 剧本 & 分镜", open=False):
                    with gr.Row():
                        script_html = gr.HTML(scale=1)
                        curve_html = gr.HTML(scale=1)
                    storyboard_html = gr.HTML()
                    comparison_html = gr.HTML()
                with gr.Accordion("👤 角色 & 审查", open=False):
                    character_profiles_html = gr.HTML()
                    review_html = gr.HTML()

                load_results_btn.click(
                    load_all_results,
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[script_html, curve_html, storyboard_html,
                              review_html, page_reader_html, comparison_html,
                              character_profiles_html],
                )
                prev_btn.click(
                    lambda s, b: navigate_page("prev", s, b),
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[page_reader_html, page_state],
                )
                next_btn.click(
                    lambda s, b: navigate_page("next", s, b),
                    inputs=[page_state, bubble_toggle_state],
                    outputs=[page_reader_html, page_state],
                )
                bubble_checkbox.change(
                    lambda s, b: (render_page_reader(
                        SESSION.state.get("final_pages", []) if SESSION.state else [],
                        s, b), b),
                    inputs=[page_state, bubble_checkbox],
                    outputs=[page_reader_html, bubble_toggle_state],
                )
                download_btn.click(
                    get_download_files,
                    outputs=[download_btn],
                )

        # ==== 跨 Tab 事件注册（需要 Tab 2 组件已定义）====
        _open_outputs = [create_status, project_cards_html, project_info_html,
                         left_panel_html, panel_preview_html, start_phase, tabs]
        open_btn.click(
            open_project_by_id,
            inputs=[open_project_id],
            outputs=_open_outputs,
        )
        # 输入框按 Enter 也能打开项目
        open_project_id.submit(
            open_project_by_id,
            inputs=[open_project_id],
            outputs=_open_outputs,
        )

        gr.HTML("""
        <div style="text-align:center;padding:16px;color:#94a3b8;font-size:12px;">
            ComicWeaver © 2026 · v0.4.0 Claude Code Edition
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
        head=HEAD_HTML,
        css=CUSTOM_CSS,
        allowed_paths=[str(Path.cwd())],
    )


if __name__ == "__main__":
    main()
