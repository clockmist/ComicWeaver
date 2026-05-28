"""工作流图执行引擎 - 编排 5 个生产Agent + 审查Agent + 确认节点。

实现选型说明: 在初始框架中先用纯 Python 实现,
未来可平滑替换为 langgraph.StateGraph(已在 docs 中规划)。
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

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

from .types import CheckpointSignal, WorkflowEvent, WorkflowMessage, should_pause


class ComicWorkflow:
    """漫画创作工作流执行器。

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

        # thread-safe 响应通道(由Gradio线程调用 respond,工作流线程消费)
        self._response_queue: asyncio.Queue | None = None
        self._loop_for_response: asyncio.AbstractEventLoop | None = None

    def _ensure_queue(self) -> asyncio.Queue:
        if self._response_queue is None:
            self._response_queue = asyncio.Queue()
        return self._response_queue

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def arun(self, state: ComicState) -> AsyncIterator[object]:
        """异步执行工作流,流式吐出消息。"""
        yield WorkflowMessage(WorkflowEvent.NODE_START, "workflow")

        try:
            async for msg in self._run_phase_script(state):
                yield msg
            async for msg in self._handle_checkpoint(state, "after_script"):
                yield msg

            async for msg in self._run_phase_character(state):
                yield msg
            async for msg in self._handle_checkpoint(state, "after_character"):
                yield msg

            async for msg in self._run_phase_storyboard(state):
                yield msg
            async for msg in self._handle_checkpoint(state, "after_storyboard"):
                yield msg

            async for msg in self._run_phase_images(state):
                yield msg

            async for msg in self._run_phase_layout(state):
                yield msg
            async for msg in self._handle_checkpoint(state, "after_layout"):
                yield msg

            yield WorkflowMessage(WorkflowEvent.WORKFLOW_DONE, "workflow", state)
        except Exception as exc:  # noqa: BLE001
            yield WorkflowMessage(
                WorkflowEvent.ERROR, "workflow",
                {"error": str(exc), "type": exc.__class__.__name__},
            )
            raise

    def respond(self, decision: str) -> None:
        """用户响应当前确认点。decision: accept | regenerate | skip
        线程安全 - 可从任意线程调用。
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
    # 各阶段实现
    # ------------------------------------------------------------------

    async def _run_phase_script(self, state: ComicState) -> AsyncIterator[object]:
        state["current_phase"] = "script"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "script_agent")

        ctx = self._make_context(state)
        inputs = ScriptInput(
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
            raw_text=state.get("user_input", ""),
            target_pages=state.get("target_pages", 4),
            style_hint=state.get("style_preset", "manga"),
        )

        # 流式执行
        async for evt in self.script_agent.astream(inputs, ctx):
            yield evt
            if evt.type == StreamEventType.DONE:
                state["structured_script"] = evt.content
                state["emotion_curve"] = evt.content.get("emotion_curve", [])

        # 审查
        async for evt in self._review(
            state,
            target_agent="script_agent",
            target_rubric_id="rubric_script_v1",
            target_output=state["structured_script"],
        ):
            yield evt

        yield WorkflowMessage(WorkflowEvent.NODE_END, "script_agent")

    async def _run_phase_character(self, state: ComicState) -> AsyncIterator[object]:
        state["current_phase"] = "character"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "character_agent")

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

        async for evt in self.character_agent.astream(inputs, ctx):
            yield evt
            if evt.type == StreamEventType.DONE:
                state["character_db"] = evt.content.get("character_db") or {}

        async for evt in self._review(
            state,
            target_agent="character_agent",
            target_rubric_id="rubric_character_v1",
            target_output=state["character_db"],
        ):
            yield evt

        yield WorkflowMessage(WorkflowEvent.NODE_END, "character_agent")

    async def _run_phase_storyboard(self, state: ComicState) -> AsyncIterator[object]:
        state["current_phase"] = "storyboard"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "storyboard_agent")

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

        async for evt in self.storyboard_agent.astream(inputs, ctx):
            yield evt
            if evt.type == StreamEventType.DONE:
                state["storyboard_plan"] = evt.content.get("pages", [])

        async for evt in self._review(
            state,
            target_agent="storyboard_agent",
            target_rubric_id="rubric_storyboard_v1",
            target_output={"pages": state["storyboard_plan"],
                           "total_panels": sum(len(p.get("panels", []))
                                                for p in state["storyboard_plan"])},
        ):
            yield evt

        yield WorkflowMessage(WorkflowEvent.NODE_END, "storyboard_agent")

    async def _run_phase_images(self, state: ComicState) -> AsyncIterator[object]:
        state["current_phase"] = "image"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "image_agent")

        ctx = self._make_context(state)
        pages = [PageLayout.model_validate(p) for p in state.get("storyboard_plan", [])]
        all_panel_images: list[dict] = []

        for page in pages:
            for panel in page.panels:
                inputs = ImageInput(
                    panel_plan=panel,
                    style_preset=state.get("style_preset", "manga"),
                )
                async for evt in self.image_agent.astream(inputs, ctx):
                    yield evt
                    if evt.type == StreamEventType.DONE:
                        pi = evt.content.get("panel_image")
                        if pi:
                            all_panel_images.append(pi)

        state["panel_images"] = all_panel_images
        yield WorkflowMessage(WorkflowEvent.NODE_END, "image_agent")

    async def _run_phase_layout(self, state: ComicState) -> AsyncIterator[object]:
        state["current_phase"] = "layout"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "layout_agent")

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
        )

        async for evt in self.layout_agent.astream(inputs, ctx):
            yield evt
            if evt.type == StreamEventType.DONE:
                state["final_pages"] = evt.content.get("final_pages", [])
                state["exports"] = evt.content.get("exports", [])

        async for evt in self._review(
            state,
            target_agent="layout_agent",
            target_rubric_id="rubric_layout_v1",
            target_output={
                "final_pages": state["final_pages"],
                "overall_metrics": {"bubbles_overflow_count": 0},
            },
        ):
            yield evt

        yield WorkflowMessage(WorkflowEvent.NODE_END, "layout_agent")

    # ------------------------------------------------------------------
    # 审查与暂停
    # ------------------------------------------------------------------

    async def _review(
        self,
        state: ComicState,
        target_agent: str,
        target_rubric_id: str,
        target_output: dict,
    ) -> AsyncIterator[object]:
        """对生产Agent的输出执行审查。"""
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

        result = None
        async for evt in self.reviewer.astream(review_input, ctx):
            yield evt
            if evt.type == StreamEventType.DONE:
                result = evt.content

        if result is None:
            return

        feedback = ReviewFeedback.model_validate(result.get("feedback", {}))
        review_log = state.setdefault("review_results", [])
        review_log.append({
            "agent": target_agent,
            "decision": feedback.decision.value,
            "score": feedback.overall_score,
            "issues": feedback.issues,
        })

        if feedback.decision == ReviewDecision.PASS:
            yield WorkflowMessage(
                WorkflowEvent.REVIEW_PASS, target_agent,
                {"score": feedback.overall_score},
            )
        elif feedback.decision == ReviewDecision.REVISE:
            retry_counts[target_agent] = retry + 1
            state["retry_counts"] = retry_counts
            yield WorkflowMessage(
                WorkflowEvent.REVIEW_REVISE, target_agent,
                {"score": feedback.overall_score, "issues": feedback.issues},
            )
            # 在初始框架中,REVISE后只记录,不实际重跑(避免增加复杂度)
            # 真实实现应在此触发 agent.revise() 并循环
        else:
            yield WorkflowMessage(
                WorkflowEvent.REVIEW_ESCALATE, target_agent,
                {
                    "score": feedback.overall_score,
                    "summary": result.get("escalation_summary"),
                },
            )

    async def _handle_checkpoint(
        self, state: ComicState, checkpoint_id: str
    ) -> AsyncIterator[object]:
        """处理确认点 - 如需暂停则yield CheckpointSignal。"""
        if not should_pause(state, checkpoint_id):
            return

        payload = {
            "after_script": {
                "title": state.get("structured_script", {}).get("title"),
                "scene_count": len(state.get("structured_script", {}).get("scenes", [])),
                "characters": [
                    c.get("name") for c in state.get("structured_script", {}).get("characters", [])
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
        }.get(checkpoint_id, {})

        signal = CheckpointSignal(
            checkpoint_id=checkpoint_id,
            label=({
                "after_script": "剧本结构确认",
                "after_character": "角色设计确认",
                "after_storyboard": "分镜规划确认",
                "after_layout": "最终漫画确认",
            }).get(checkpoint_id, checkpoint_id),
            payload=payload,
        )
        state["pending_checkpoint"] = {
            "id": signal.checkpoint_id,
            "label": signal.label,
            "payload": signal.payload,
        }

        yield signal

        # 记录当前事件循环,以便外部线程通过 call_soon_threadsafe 投递响应
        self._loop_for_response = asyncio.get_running_loop()
        queue = self._ensure_queue()

        # 等待用户响应
        decision = await queue.get()

        state.setdefault("user_decisions", []).append({
            "checkpoint": checkpoint_id,
            "decision": decision,
        })
        # 清除pending标记
        state["pending_checkpoint"] = None

        if decision == "skip":
            # 跳过后续阶段(简化实现:抛异常终止)
            raise StopAsyncIteration

    async def _maybe_pause(self, state: ComicState, checkpoint_id: str) -> bool:
        """如需暂停则触发CheckpointSignal,等待用户响应。

        返回 True 表示继续,False 表示中止工作流。
        """
        if not should_pause(state, checkpoint_id):
            return True

        # 收集本阶段产出供前端预览
        payload = {
            "after_script": {
                "title": state.get("structured_script", {}).get("title"),
                "scene_count": len(state.get("structured_script", {}).get("scenes", [])),
                "characters": [
                    c.get("name") for c in state.get("structured_script", {}).get("characters", [])
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
        }.get(checkpoint_id, {})

        signal = CheckpointSignal(
            checkpoint_id=checkpoint_id,
            label=({
                "after_script": "剧本结构确认",
                "after_character": "角色设计确认",
                "after_storyboard": "分镜规划确认",
                "after_layout": "最终漫画确认",
            }).get(checkpoint_id, checkpoint_id),
            payload=payload,
        )
        # 记录到state,前端可读取
        state["pending_checkpoint"] = {
            "id": signal.checkpoint_id,
            "label": signal.label,
            "payload": signal.payload,
        }

        # 触发等待
        loop = asyncio.get_running_loop()
        self._pending_response = loop.create_future()

        # 直接抛出 CheckpointSignal 给上层(由 arun 的yield承担)
        # 这里我们不能直接yield,所以由调用方约定:
        # arun 在 _maybe_pause 内不能yield CheckpointSignal,
        # 改为通过 state["pending_checkpoint"] + 单独通知机制。
        # 简化方案: 使用回调式pause,不暂停内部await,前端通过其他通道响应。
        # 实际实现见 ComicWorkflow.arun 中的写法 - 这里做最简化的不暂停,
        # 前端通过 should_pause 的预判提前停止事件流。
        return True

    def _make_context(self, state: ComicState) -> AgentContext:
        return AgentContext(
            project_id=state.get("project_id", "proj"),
            interaction_mode=InteractionMode(state.get("interaction_mode", "semi_auto")),
            style_preset=state.get("style_preset", "manga"),
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
        )
