"""工作流回调：启动、轮询、确认点响应。"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import gradio as gr

from comicweaver.core import ComicState
from comicweaver.orchestrator import CheckpointSignal, ComicWorkflow, WorkflowEvent
from comicweaver.storage import save_dev_log, save_project, state_to_project
from comicweaver.utils.logging import DevLogEntry

from ..composables.live import render_live_content, render_phase_text_content
from ..renderers.project import render_project_info
from ..renderers.workflow import render_workflow_left
from ..session import Session, SESSION

# ============================================================================
# 阶段管理
# ============================================================================

_PHASE_PREREQUISITES: dict[str, list[str]] = {
    "init": [],
    "story": [],
    "character": ["developed_story"],
    "script": ["developed_story", "character_db"],
    "storyboard": ["structured_script", "character_db"],
    "image": ["structured_script", "character_db", "storyboard_plan"],
    "bubble": ["storyboard_plan", "panel_images"],
    "layout": ["storyboard_plan", "panel_images", "bubble_placements"],
}

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
    "character": [
        "character_db", "reference_chain",
        "structured_script", "emotion_curve",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "script": [
        "structured_script", "emotion_curve",
        "storyboard_plan", "layout_grids", "panel_images", "generation_metadata",
        "final_pages", "exports",
    ],
    "storyboard": [
        "storyboard_plan", "layout_grids",
        "panel_images", "generation_metadata",
        "bubble_placements", "bubbled_panel_images", "final_pages", "exports",
    ],
    "image": [
        "panel_images", "generation_metadata",
        "bubble_placements", "bubbled_panel_images", "final_pages", "exports",
    ],
    "bubble": [
        "bubble_placements", "bubbled_panel_images",
        "final_pages", "exports",
    ],
    "layout": [
        "final_pages", "exports",
    ],
}

_PHASE_LABELS_CN = {
    "init": "从头开始", "story": "故事阶段",
    "script": "剧本阶段", "character": "角色阶段",
    "storyboard": "分镜阶段", "image": "图像阶段",
    "bubble": "台词阶段", "layout": "排版阶段",
}


def _validate_and_reset_state(state: ComicState, start_phase: str) -> str | None:
    """验证前置条件并清除对应阶段的输出数据。返回 None 表示成功。"""
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
                "bubble_placements": "台词气泡",
            }.get(field, field)
            return (
                f"❌ 无法从「{phase_label}」阶段开始：缺少前置数据「{field_label}」。\n"
                f"请先从头开始或从已完成的更早阶段开始。"
            )

    clear_fields = _PHASE_CLEAR_FIELDS.get(start_phase, [])
    for field in clear_fields:
        if field in state:
            if field in ("emotion_curve", "reference_chain", "storyboard_plan",
                         "layout_grids", "panel_images", "generation_metadata",
                         "final_pages", "exports"):
                state[field] = []  # type: ignore[literal-required]
            else:
                state[field] = {}  # type: ignore[literal-required]

    state["review_results"] = []
    state["retry_counts"] = {}
    state["pending_checkpoint"] = None
    state["user_decisions"] = []
    state["stream_messages"] = []
    state["errors"] = []
    state["current_phase"] = start_phase

    return None


# ============================================================================
# 工作流执行
# ============================================================================


async def _consume_workflow(session: Session) -> None:
    """消费工作流事件，持续更新 session 状态。"""
    if session.workflow is None or session.state is None:
        return

    try:
        async for msg in session.workflow.arun(session.state):
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
                continue

            if hasattr(msg, "event"):
                if msg.event == WorkflowEvent.NODE_START:
                    session.active_agent = msg.node
                    session.agent_start_time = time.time()
                    agent_to_phase = {
                        "story_agent": "story",
                        "character_agent": "character",
                        "script_agent": "script",
                        "storyboard_agent": "storyboard",
                        "image_agent": "image",
                        "bubble_agent": "bubble",
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
                        # 同时持久化开发者日志为可读文本文件
                        if session.dev_log:
                            save_dev_log(session.dev_log, session.state.get("project_id", "unknown"))
                continue

            if isinstance(msg, DevLogEntry):
                session.dev_log.append(msg.to_dict())
                session.event_log.append({
                    "agent": msg.agent,
                    "type": f"dev_{msg.category.value}",
                    "content": f"[{msg.level.value}] {msg.message[:120]}",
                    "timestamp": msg.timestamp,
                })
                continue

            if hasattr(msg, "agent") and hasattr(msg, "type"):
                evt_type = (
                    msg.type.value if hasattr(msg.type, "value")
                    else str(msg.type)
                )
                content = msg.content
                if isinstance(content, dict):
                    if evt_type == "done":
                        agent_name = msg.agent
                        out_list = session.agent_outputs.setdefault(agent_name, [])
                        out_list.append({
                            "panel_id": content.get("panel_id", ""),
                            "title": content.get("title", ""),
                            "output": content,
                            "timestamp": msg.timestamp,
                        })
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
                        if agent_name == "image_agent" and "panel_image" in content:
                            session.panel_images_preview.append(content["panel_image"])
                            pis = session.state.setdefault("panel_images", [])
                            pis.append(content["panel_image"])
                        if session.state is not None:
                            if agent_name == "story_agent":
                                session.state["developed_story"] = content
                                session.state["narrative_structure"] = content.get("narrative_structure", {})
                            elif agent_name == "script_agent":
                                session.state["structured_script"] = content
                                curve: list[float] = []
                                for p in content.get("pages", []):
                                    for panel in p.get("panels", []):
                                        if panel.get("emotion_intensity") is not None:
                                            curve.append(float(panel["emotion_intensity"]))
                                session.state["emotion_curve"] = curve or content.get("emotion_curve", [])
                            elif agent_name == "character_agent":
                                session.state["character_db"] = content.get("character_db") or {}
                            elif agent_name == "storyboard_agent":
                                pages = content.get("pages", [])
                                if pages:
                                    session.state["storyboard_plan"] = pages
                            elif agent_name == "bubble_agent":
                                session.state["bubble_placements"] = content.get("bubble_placements", {})
                                session.state["bubbled_panel_images"] = content.get("bubbled_panel_images", {})
                            elif agent_name == "layout_agent":
                                session.state["final_pages"] = content.get("final_pages", [])
                                session.state["exports"] = content.get("exports", [])
                        keys = list(content.keys())
                        summary = f"✓ 完成 · 输出 {len(content)} 个字段: {', '.join(keys[:8])}"
                        if len(keys) > 8:
                            summary += f" ...+{len(keys) - 8}"
                        content = summary
                    else:
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
    except (asyncio.CancelledError, GeneratorExit, RuntimeError):
        # 工作流被取消或异步生成器被关闭，静默清理
        pass
    except Exception as exc:  # noqa: BLE001
        session.error = str(exc)
        session.event_log.append({
            "agent": "workflow",
            "type": "error",
            "content": f"{exc.__class__.__name__}: {exc}",
            "timestamp": time.time(),
        })


def _ensure_loop(session: Session) -> asyncio.AbstractEventLoop:
    """获取或创建 session 专属事件循环（运行在后台线程）。

    如果已有任务在运行，先取消并清理旧的循环。
    """
    # 取消旧任务
    if session._task is not None and not session._task.done():
        session._task.cancel()
    session._task = None

    # 停止旧事件循环
    if session._loop is not None:
        try:
            if session._loop.is_running():
                session._loop.call_soon_threadsafe(session._loop.stop)
        except Exception:
            pass
        session._loop = None

    import threading
    loop = asyncio.new_event_loop()
    session._loop = loop

    def runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=runner, daemon=True).start()
    while not loop.is_running():
        time.sleep(0.01)
    return loop


# ============================================================================
# UI 回调
# ============================================================================


def start_workflow(start_phase: str = "init") -> Iterator[tuple]:
    """启动工作流并周期性产出UI更新。"""
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

    if start_phase and start_phase != "init":
        resume_phase = start_phase
    else:
        resume_phase = None

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
            SESSION.state.get("bubble_placements", {}),
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
            gr.update(interactive=at_checkpoint, value=""),
        )

        if SESSION.workflow_done or SESSION.error:
            break
        if at_checkpoint:
            break


def respond_checkpoint(decision: str, guidance: str = "") -> Iterator[tuple]:
    """响应当前确认点，继续执行工作流。"""
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

    SESSION.workflow.respond(decision, guidance)
    decision_label = {"accept": "接受并继续", "regenerate": "重新生成"}.get(decision, decision)
    guidance_note = f" (指导: {guidance[:50]})" if guidance else ""
    SESSION.event_log.append({
        "agent": "user",
        "type": "log",
        "content": f"用户决策: {decision_label}{guidance_note}",
        "timestamp": time.time(),
    })

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
            SESSION.state["bubble_placements"] = {}  # 图像重生成后气泡也需重新计算
        elif cp_id == "after_bubble":
            SESSION.state["bubble_placements"] = {}
            SESSION.state["bubbled_panel_images"] = {}
        elif cp_id == "after_layout":
            SESSION.state["final_pages"] = []
            SESSION.state["exports"] = []
            SESSION.panel_images_preview = []

    SESSION.last_checkpoint = None

    yield (
        f"✅ 已响应: {decision_label}，工作流继续...",
        render_workflow_left(
            SESSION.active_agent, SESSION.done_agents,
            SESSION.event_log, SESSION.last_checkpoint,
        ),
        render_live_content(
            SESSION.panel_images_preview, SESSION.character_preview,
            render_phase_text_content(SESSION.state, SESSION.agent_outputs),
            SESSION.state.get("bubble_placements", {}),
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
