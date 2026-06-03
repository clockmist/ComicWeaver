"""工作流图执行引擎 - 基于 LangGraph StateGraph 实现。

使用 LangGraph 管理 agent 执行流水线的状态转换、事件流和人工确认点。
ComicWorkflow 类保持向后兼容的 arun()/respond() 接口。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from comicweaver.agents import (
    CharacterAgent,
    ImageAgent,
    LayoutAgent,
    ReviewerAgent,
    ScriptAgent,
    StoryboardAgent,
)
from comicweaver.core import (
    AgentContext,
    CharacterDB,
    CharacterDraft,
    CharacterInput,
    ComicState,
    CreationMode,
    ImageInput,
    InteractionMode,
    LayoutInput,
    PageLayout,
    PanelImage,
    ReviewDecision,
    ReviewFeedback,
    ReviewInput,
    Scene,
    ScriptInput,
    StoryboardInput,
    StreamEventType,
)
from comicweaver.utils.logging import (
    log_agent_error,
    log_agent_input,
    log_agent_output,
    log_checkpoint,
    log_fallback,
    log_performance,
    log_review,
    log_state_change,
    log_workflow_event,
    summarize_character_output,
    summarize_image_output,
    summarize_layout_output,
    summarize_review_output,
    summarize_script_output,
    summarize_storyboard_output,
)

from .types import CheckpointSignal, WorkflowEvent, WorkflowMessage, should_pause

# ============================================================================
# 条件路由辅助函数
# ============================================================================


def _make_route_checkpoint(checkpoint_id: str):
    """创建条件边路由函数 —— 判断是否需要进入确认节点。"""

    def route(state: ComicState) -> str:
        if should_pause(state, checkpoint_id):
            return "checkpoint"
        return "continue"

    return route


# ============================================================================
# Checkpoint 信息构建
# ============================================================================


def _build_checkpoint_payload(state: ComicState, checkpoint_id: str) -> dict:
    """根据当前 state 构建确认节点的预览信息。"""
    payloads: dict[str, dict] = {
        "after_script": {
            "title": state.get("structured_script", {}).get("title"),
            "scene_count": len(state.get("structured_script", {}).get("scenes", [])),
            "characters": [
                c.get("name")
                for c in state.get("structured_script", {}).get("characters", [])
            ],
        },
        "after_character": {
            "characters": list(
                (state.get("character_db") or {}).get("characters", {}).keys()
            ),
        },
        "after_storyboard": {
            "pages": len(state.get("storyboard_plan", [])),
            "panels": sum(
                len(p.get("panels", []))
                for p in state.get("storyboard_plan", [])
            ),
        },
        "after_layout": {
            "pages": [p.get("image_path") for p in state.get("final_pages", [])],
        },
    }
    return payloads.get(checkpoint_id, {})


_CHECKPOINT_LABELS: dict[str, str] = {
    "after_script": "剧本结构确认",
    "after_character": "角色设计确认",
    "after_storyboard": "分镜规划确认",
    "after_layout": "最终漫画确认",
}


def _make_checkpoint_node(checkpoint_id: str):
    """创建 LangGraph 确认节点 —— 调用 interrupt() 暂停工作流。"""

    def node(state: ComicState) -> ComicState:
        label = _CHECKPOINT_LABELS.get(checkpoint_id, checkpoint_id)
        payload = _build_checkpoint_payload(state, checkpoint_id)
        signal = CheckpointSignal(
            checkpoint_id=checkpoint_id,
            label=label,
            payload=payload,
        )

        # 使用 LangGraph interrupt 暂停工作流
        decision = interrupt(signal)

        # 开发者日志：确认点交互
        writer = get_stream_writer()
        writer(log_checkpoint(checkpoint_id, decision))

        # 记录用户决策（resume 后由 LangGraph 状态管理持久化）
        state.setdefault("user_decisions", []).append({
            "checkpoint": checkpoint_id,
            "decision": decision,
        })
        state["pending_checkpoint"] = None

        return state

    return node


# ============================================================================
# ComicWorkflow —— 基于 LangGraph 的工作流执行器
# ============================================================================


class ComicWorkflow:
    """漫画创作工作流执行器（基于 LangGraph StateGraph）。

    用法:
        wf = ComicWorkflow()
        async for msg in wf.arun(state):
            ...  # msg 可能是 StreamEvent / WorkflowMessage / CheckpointSignal

        # 用户响应确认点
        wf.respond("accept")
    """

    def __init__(self) -> None:
        self.script_agent = ScriptAgent()
        self.character_agent = CharacterAgent()
        self.storyboard_agent = StoryboardAgent()
        self.image_agent = ImageAgent()
        self.layout_agent = LayoutAgent()
        self.reviewer = ReviewerAgent()

        # 构建并编译 LangGraph 图
        self._graph = self._build_graph()
        self._compiled = self._graph.compile(checkpointer=MemorySaver())

        # thread-safe 响应通道（由 Gradio 线程调用 respond，工作流线程消费）
        self._response_queue: asyncio.Queue | None = None
        self._loop_for_response: asyncio.AbstractEventLoop | None = None
        self._current_config: dict | None = None

    def _ensure_queue(self) -> asyncio.Queue:
        if self._response_queue is None:
            self._response_queue = asyncio.Queue()
        return self._response_queue

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ComicState)

        # -- 生产节点 --
        builder.add_node("script", self._node_script)
        builder.add_node("character", self._node_character)
        builder.add_node("storyboard", self._node_storyboard)
        builder.add_node("image", self._node_image)
        builder.add_node("layout", self._node_layout)

        # -- 审查节点 --
        builder.add_node("review_script", self._node_review_script)
        builder.add_node("review_character", self._node_review_character)
        builder.add_node("review_storyboard", self._node_review_storyboard)
        builder.add_node("review_layout", self._node_review_layout)

        # -- 确认节点 --
        builder.add_node("checkpoint_script", _make_checkpoint_node("after_script"))
        builder.add_node("checkpoint_character", _make_checkpoint_node("after_character"))
        builder.add_node("checkpoint_storyboard", _make_checkpoint_node("after_storyboard"))
        builder.add_node("checkpoint_layout", _make_checkpoint_node("after_layout"))

        # -- 边：script → review → [checkpoint?] → character --
        builder.add_edge(START, "script")
        builder.add_edge("script", "review_script")
        builder.add_conditional_edges(
            "review_script",
            _make_route_checkpoint("after_script"),
            {"checkpoint": "checkpoint_script", "continue": "character"},
        )
        builder.add_edge("checkpoint_script", "character")

        # -- 边：character → review → [checkpoint?] → storyboard --
        builder.add_edge("character", "review_character")
        builder.add_conditional_edges(
            "review_character",
            _make_route_checkpoint("after_character"),
            {"checkpoint": "checkpoint_character", "continue": "storyboard"},
        )
        builder.add_edge("checkpoint_character", "storyboard")

        # -- 边：storyboard → review → [checkpoint?] → image --
        builder.add_edge("storyboard", "review_storyboard")
        builder.add_conditional_edges(
            "review_storyboard",
            _make_route_checkpoint("after_storyboard"),
            {"checkpoint": "checkpoint_storyboard", "continue": "image"},
        )
        builder.add_edge("checkpoint_storyboard", "image")

        # -- 边：image → layout → review → [checkpoint?] → END --
        builder.add_edge("image", "layout")
        builder.add_edge("layout", "review_layout")
        builder.add_conditional_edges(
            "review_layout",
            _make_route_checkpoint("after_layout"),
            {"checkpoint": "checkpoint_layout", "continue": END},
        )
        builder.add_edge("checkpoint_layout", END)

        return builder

    # ------------------------------------------------------------------
    # 生产节点实现
    # ------------------------------------------------------------------

    async def _node_script(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "script_agent"))
        writer(log_workflow_event("进入阶段", "script"))

        state["current_phase"] = "script"
        ctx = self._make_context(state)
        inputs = ScriptInput(
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
            raw_text=state.get("user_input", ""),
            target_pages=state.get("target_pages", 4),
            style_hint=state.get("style_preset", "manga"),
        )

        # 开发者日志：Agent 输入
        writer(log_agent_input("script_agent", {
            "creation_mode": inputs.creation_mode.value,
            "raw_text": inputs.raw_text[:200],
            "target_pages": inputs.target_pages,
            "style_hint": inputs.style_hint or "none",
        }))

        try:
            async for evt in self.script_agent.astream(inputs, ctx):
                writer(evt)
                if evt.type == StreamEventType.DONE and evt.content:
                    state["structured_script"] = evt.content
                    state["emotion_curve"] = evt.content.get("emotion_curve", [])
                    # 开发者日志：Agent 输出摘要
                    writer(log_agent_output("script_agent",
                        summarize_script_output(evt.content)))
        except Exception:
            writer(log_agent_error("script_agent", "执行失败",
                {"raw_text": inputs.raw_text[:100]}))
            raise

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("script_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "script_agent"))
        return state

    async def _node_character(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "character_agent"))
        writer(log_workflow_event("进入阶段", "character"))

        state["current_phase"] = "character"
        ctx = self._make_context(state)
        script_dict = state.get("structured_script", {})
        drafts = [
            CharacterDraft.model_validate(c)
            for c in script_dict.get("characters", [])
        ]
        inputs = CharacterInput(
            operation="init",
            character_drafts=drafts,
            style_preset=state.get("style_preset", "manga"),
        )

        # 开发者日志：Agent 输入
        writer(log_agent_input("character_agent", {
            "operation": "init",
            "draft_count": len(drafts),
            "draft_names": [d.name for d in drafts],
            "style_preset": state.get("style_preset", "manga"),
        }))

        try:
            async for evt in self.character_agent.astream(inputs, ctx):
                writer(evt)
                if evt.type == StreamEventType.DONE and evt.content:
                    state["character_db"] = evt.content.get("character_db") or {}
                    writer(log_agent_output("character_agent",
                        summarize_character_output(evt.content)))
        except Exception:
            writer(log_agent_error("character_agent", "执行失败",
                {"drafts": [d.name for d in drafts]}))
            raise

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("character_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "character_agent"))
        return state

    async def _node_storyboard(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "storyboard_agent"))
        writer(log_workflow_event("进入阶段", "storyboard"))

        state["current_phase"] = "storyboard"
        ctx = self._make_context(state)
        script_dict = state.get("structured_script", {})
        scenes = [Scene.model_validate(s) for s in script_dict.get("scenes", [])]
        cdb_dict = state.get("character_db") or {}
        cdb = CharacterDB.model_validate(cdb_dict) if cdb_dict else None

        inputs = StoryboardInput(
            scenes=scenes,
            emotion_curve=state.get("emotion_curve", []),
            character_db=cdb,
            style_preset=state.get("style_preset", "manga"),
            target_pages=state.get("target_pages", 4),
        )

        # 开发者日志：Agent 输入
        writer(log_agent_input("storyboard_agent", {
            "scene_count": len(scenes),
            "emotion_curve_peaks": (
                max(state.get("emotion_curve", [0])) if state.get("emotion_curve") else 0
            ),
            "target_pages": inputs.target_pages,
            "has_character_db": cdb is not None,
        }))

        try:
            async for evt in self.storyboard_agent.astream(inputs, ctx):
                writer(evt)
                if evt.type == StreamEventType.DONE and evt.content:
                    state["storyboard_plan"] = evt.content.get("pages", [])
                    writer(log_agent_output("storyboard_agent",
                        summarize_storyboard_output(evt.content)))
        except Exception:
            writer(log_agent_error("storyboard_agent", "执行失败",
                {"scene_count": len(scenes), "target_pages": inputs.target_pages}))
            raise

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("storyboard_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "storyboard_agent"))
        return state

    async def _node_image(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "image_agent"))
        writer(log_workflow_event("进入阶段", "image"))

        state["current_phase"] = "image"
        ctx = self._make_context(state)
        pages = [PageLayout.model_validate(p) for p in state.get("storyboard_plan", [])]
        all_panel_images: list[dict] = []
        total_panels = sum(len(page.panels) for page in pages)

        writer(log_agent_input("image_agent", {
            "total_panels": total_panels,
            "style_preset": state.get("style_preset", "manga"),
            "backend_preference": "auto",
        }))

        panel_idx = 0
        for page in pages:
            for panel in page.panels:
                panel_idx += 1
                inputs = ImageInput(
                    panel_plan=panel,
                    style_preset=state.get("style_preset", "manga"),
                )
                try:
                    async for evt in self.image_agent.astream(inputs, ctx):
                        writer(evt)
                        if evt.type == StreamEventType.DONE and evt.content:
                            pi = evt.content.get("panel_image")
                            if pi:
                                all_panel_images.append(pi)
                            # 检测 fallback
                            backend_used = evt.content.get("backend_used", "?")
                            fallback_chain = evt.content.get("fallback_chain", [])
                            if fallback_chain and len(fallback_chain) > 1:
                                writer(log_fallback(
                                    "image_agent",
                                    fallback_chain[0],
                                    fallback_chain[-1],
                                ))
                            writer(log_agent_output("image_agent",
                                summarize_image_output(evt.content)))
                except Exception:
                    writer(log_agent_error("image_agent",
                        f"面板 {panel.panel_id} 生成失败",
                        {"panel_id": panel.panel_id, "panel_idx": panel_idx}))
                    raise

        state["panel_images"] = all_panel_images
        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("image_agent", elapsed, status=f"{len(all_panel_images)}/{total_panels} panels"))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "image_agent"))
        return state

    async def _node_layout(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "layout_agent"))
        writer(log_workflow_event("进入阶段", "layout"))

        state["current_phase"] = "layout"
        ctx = self._make_context(state)
        pages = [PageLayout.model_validate(p) for p in state.get("storyboard_plan", [])]
        panel_imgs = {
            p["panel_id"]: PanelImage.model_validate(p)
            for p in state.get("panel_images", [])
        }

        inputs = LayoutInput(
            pages=pages,
            panel_images=panel_imgs,
            style_preset=state.get("style_preset", "manga"),
            page_width_px=1240,
            page_height_px=1754,
            margin_px=40,
            gutter_px=10,
            reading_direction="ltr",
        )

        # 开发者日志：Agent 输入
        writer(log_agent_input("layout_agent", {
            "page_count": len(pages),
            "panel_image_count": len(panel_imgs),
            "panel_ids": list(panel_imgs.keys()),
            "export_formats": inputs.export_formats,
        }))

        try:
            async for evt in self.layout_agent.astream(inputs, ctx):
                writer(evt)
                if evt.type == StreamEventType.DONE and evt.content:
                    state["final_pages"] = evt.content.get("final_pages", [])
                    state["exports"] = evt.content.get("exports", [])
                    writer(log_agent_output("layout_agent",
                        summarize_layout_output(evt.content)))
        except Exception:
            writer(log_agent_error("layout_agent", "执行失败",
                {"page_count": len(pages), "panel_count": len(panel_imgs)}))
            raise

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("layout_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "layout_agent"))
        return state

    # ------------------------------------------------------------------
    # 审查节点实现
    # ------------------------------------------------------------------

    async def _node_review_script(self, state: ComicState) -> ComicState:
        return await self._run_review(
            state, "script_agent", "rubric_script_v1",
            target_output=state.get("structured_script", {}),
        )

    async def _node_review_character(self, state: ComicState) -> ComicState:
        return await self._run_review(
            state, "character_agent", "rubric_character_v1",
            target_output=state.get("character_db", {}),
        )

    async def _node_review_storyboard(self, state: ComicState) -> ComicState:
        return await self._run_review(
            state, "storyboard_agent", "rubric_storyboard_v1",
            target_output={
                "pages": state.get("storyboard_plan", []),
                "total_panels": sum(
                    len(p.get("panels", []))
                    for p in state.get("storyboard_plan", [])
                ),
            },
        )

    async def _node_review_layout(self, state: ComicState) -> ComicState:
        return await self._run_review(
            state, "layout_agent", "rubric_layout_v1",
            target_output={
                "final_pages": state.get("final_pages", []),
                "overall_metrics": {"bubbles_overflow_count": 0},
            },
        )

    async def _run_review(
        self,
        state: ComicState,
        target_agent: str,
        target_rubric_id: str,
        target_output: dict,
    ) -> ComicState:
        """对生产 Agent 的输出执行审查并更新 state。"""
        t0 = time.perf_counter()
        writer = get_stream_writer()
        ctx = self._make_context(state)
        retry_counts = state.get("retry_counts", {})
        retry = int(retry_counts.get(target_agent, 0))

        review_input = ReviewInput(
            target_agent=target_agent,
            target_rubric_id=target_rubric_id,
            target_output=target_output,
            retry_count=retry,
            max_retries=3,
        )

        # 开发者日志：审查输入
        writer(log_agent_input(f"reviewer({target_agent})", {
            "target_agent": target_agent,
            "rubric_id": target_rubric_id,
            "retry_count": retry,
            "max_retries": 3,
            "target_keys": list(target_output.keys())[:10],
        }))

        result: dict[str, Any] = {}
        async for evt in self.reviewer.astream(review_input, ctx):
            writer(evt)
            if evt.type == StreamEventType.DONE and evt.content:
                result = evt.content
                writer(log_agent_output(f"reviewer({target_agent})",
                    summarize_review_output(evt.content)))

        if not result:
            return state

        feedback = ReviewFeedback.model_validate(result.get("feedback", {}))
        review_log = state.setdefault("review_results", [])
        review_log.append({
            "agent": target_agent,
            "decision": feedback.decision.value,
            "score": feedback.overall_score,
            "issues": feedback.issues,
        })

        # 开发者日志：审查决策
        writer(log_review(
            target_agent,
            feedback.decision.value,
            feedback.overall_score,
            dimension_scores=feedback.dimension_scores,
            issues=feedback.issues,
        ))

        if feedback.decision == ReviewDecision.PASS:
            writer(WorkflowMessage(
                WorkflowEvent.REVIEW_PASS, target_agent,
                {"score": feedback.overall_score},
            ))
        elif feedback.decision == ReviewDecision.REVISE:
            retry_counts[target_agent] = retry + 1
            state["retry_counts"] = retry_counts
            writer(WorkflowMessage(
                WorkflowEvent.REVIEW_REVISE, target_agent,
                {"score": feedback.overall_score, "issues": feedback.issues},
            ))
        else:
            writer(WorkflowMessage(
                WorkflowEvent.REVIEW_ESCALATE, target_agent,
                {
                    "score": feedback.overall_score,
                    "summary": result.get("escalation_summary"),
                },
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance(f"reviewer({target_agent})", elapsed))
        return state

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def arun(self, state: ComicState) -> AsyncIterator[object]:
        """异步执行工作流，流式吐出消息（向后兼容的接口）。

        内部使用 LangGraph astream 驱动节点执行。
        当遇到 interrupt() 确认点时暂停，等待 respond() 后继续。
        """
        workflow_t0 = time.perf_counter()
        yield WorkflowMessage(WorkflowEvent.NODE_START, "workflow")
        yield log_workflow_event("工作流启动", f"project={state.get('project_id', '?')}, mode={state.get('interaction_mode', '?')}")

        config: dict[str, Any] = {
            "configurable": {"thread_id": state.get("project_id", "default")}
        }
        self._current_config = config
        input_data: Any = state  # 初始输入为完整 state
        prev_phase = state.get("current_phase", "init")

        try:
            while True:
                # 执行图直到下一个 interrupt 或 END
                async for mode, data in self._compiled.astream(
                    input_data, config, stream_mode=["custom", "updates"]
                ):
                    if mode == "custom":
                        yield data
                    elif mode == "updates":
                        # langgraph 内部的 __interrupt__ 标记跳过
                        if "__interrupt__" in data:
                            continue
                        # 将节点返回的更新合并到本地 state
                        for _node_name, node_update in data.items():
                            if isinstance(node_update, dict):
                                for key, value in node_update.items():
                                    state[key] = value  # type: ignore[literal-required]
                                # 检测阶段变更
                                new_phase = node_update.get("current_phase")
                                if new_phase and new_phase != prev_phase:
                                    yield log_state_change(prev_phase, new_phase)
                                    prev_phase = new_phase

                # 检查是否需要处理 interrupt
                snapshot = self._compiled.get_state(config)
                if snapshot is None or snapshot.next == ():
                    break  # 工作流完成

                if snapshot.interrupts:
                    interrupt_value = snapshot.interrupts[0].value
                    if isinstance(interrupt_value, CheckpointSignal):
                        # 将确认信息写入 state（供 UI 读取）
                        state["pending_checkpoint"] = {
                            "id": interrupt_value.checkpoint_id,
                            "label": interrupt_value.label,
                            "payload": interrupt_value.payload,
                        }
                        # 向 UI 发送确认信号
                        yield interrupt_value

                        # 等待用户响应（线程安全）
                        # 注意：user_decisions 由 checkpoint 节点在 resume 后记录
                        self._loop_for_response = asyncio.get_running_loop()
                        queue = self._ensure_queue()
                        decision = await queue.get()

                        if decision == "skip":
                            # 中止工作流
                            yield log_workflow_event("工作流中止", f"用户在 {interrupt_value.checkpoint_id} 选择了 skip")
                            return

                        # 用 Command(resume=...) 继续执行
                        input_data = Command(resume=decision)
                    else:
                        break
                else:
                    break

        except Exception as exc:  # noqa: BLE001
            yield log_agent_error("workflow", str(exc),
                {"type": exc.__class__.__name__, "phase": state.get("current_phase", "?")})
            yield WorkflowMessage(
                WorkflowEvent.ERROR, "workflow",
                {"error": str(exc), "type": exc.__class__.__name__},
            )
            raise

        total_elapsed = (time.perf_counter() - workflow_t0) * 1000
        yield log_performance("workflow(total)", total_elapsed)
        yield log_workflow_event("工作流完成",
            f"耗时 {total_elapsed:.0f}ms, "
            f"reviews={len(state.get('review_results', []))}, "
            f"pages={len(state.get('final_pages', []))}")
        yield WorkflowMessage(WorkflowEvent.WORKFLOW_DONE, "workflow", state)

    def respond(self, decision: str) -> None:
        """用户响应当前确认点。decision: accept | regenerate | skip

        线程安全 —— 可从任意线程调用。
        """
        queue = self._ensure_queue()
        if self._loop_for_response and self._loop_for_response.is_running():
            self._loop_for_response.call_soon_threadsafe(
                queue.put_nowait, decision
            )
        else:
            try:
                queue.put_nowait(decision)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _make_context(self, state: ComicState) -> AgentContext:
        return AgentContext(
            project_id=state.get("project_id", "proj"),
            interaction_mode=InteractionMode(state.get("interaction_mode", "semi_auto")),
            style_preset=state.get("style_preset", "manga"),
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
        )
