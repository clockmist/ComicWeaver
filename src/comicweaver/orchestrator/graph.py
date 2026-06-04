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
    ScriptAgent,
    StoryboardAgent,
)
from comicweaver.api import ApiBackendError
from comicweaver.config import load_config
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
    Scene,
    ScriptInput,
    StoryboardInput,
    StreamEventType,
)
from comicweaver.storage import save_project, state_to_project
from comicweaver.utils.logging import (
    log_agent_error,
    log_agent_input,
    log_agent_output,
    log_comfyui_request,
    log_comfyui_response,
    log_fallback,
    log_performance,
    log_state_change,
    log_workflow_event,
    summarize_character_output,
    summarize_image_output,
    summarize_layout_output,
    summarize_script_output,
    summarize_storyboard_output,
)

from .types import (
    CheckpointSignal,
    KEY_CHECKPOINTS,
    WorkflowEvent,
    WorkflowMessage,
    should_pause,
)

class ComicWorkflow:
    """漫画创作工作流执行器（基于 LangGraph StateGraph）。

    用法:
        wf = ComicWorkflow()
        async for msg in wf.arun(state):
            ...  # msg 可能是 StreamEvent / WorkflowMessage / CheckpointSignal

        # 用户响应确认点
        wf.respond("accept")

    恢复已保存项目:
        wf = ComicWorkflow.from_saved_project(project_id)
        async for msg in wf.arun(wf._saved_state):
            ...
    """

    def __init__(self, resume_phase: str | None = None) -> None:
        self.config = load_config()
        self.script_agent = ScriptAgent()
        self.character_agent = CharacterAgent()
        self.storyboard_agent = StoryboardAgent()
        self.image_agent = ImageAgent()
        self.layout_agent = LayoutAgent()

        self._resume_phase = resume_phase

        # 构建并编译 LangGraph 图
        self._graph = self._build_graph()
        self._compiled = self._graph.compile(checkpointer=MemorySaver())

        # thread-safe 响应通道（由 Gradio 线程调用 respond，工作流线程消费）
        self._response_queue: asyncio.Queue | None = None
        self._loop_for_response: asyncio.AbstractEventLoop | None = None
        self._current_config: dict | None = None

        # Checkpoint pause/resume
        self._response_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_decision: str | None = None

        # 跨会话恢复：存储已加载的 state
        self._saved_state: ComicState | None = None

    def _ensure_queue(self) -> asyncio.Queue:
        if self._response_queue is None:
            self._response_queue = asyncio.Queue()
        return self._response_queue

    # ------------------------------------------------------------------
    # 图构建（简化版 — 无审查 Agent，线性流水线）
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ComicState)

        # -- 5 个生产节点，线性串联 --
        builder.add_node("script", self._node_script)
        builder.add_node("character", self._node_character)
        builder.add_node("storyboard", self._node_storyboard)
        builder.add_node("image", self._node_image)
        builder.add_node("layout", self._node_layout)

        # -- 边：script → character → storyboard → image → layout → END --
        builder.add_edge("script", "character")
        builder.add_edge("character", "storyboard")
        builder.add_edge("storyboard", "image")
        builder.add_edge("image", "layout")
        builder.add_edge("layout", END)

        # -- 入口：根据 resume_phase 决定从哪个节点开始 --
        phase_to_node: dict[str, str] = {
            "init": "script",
            "script": "character",
            "character": "storyboard",
            "storyboard": "image",
            "image": "layout",
            "layout": "layout",
        }
        start_node = "script"
        if self._resume_phase and self._resume_phase in phase_to_node:
            start_node = phase_to_node[self._resume_phase]
        builder.add_edge(START, start_node)

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

        # --- Regenerate loop: 用户选择"重新生成"时重新执行 agent ---
        while True:
            try:
                async for evt in self.script_agent.astream(inputs, ctx):
                    writer(evt)
                    if evt.type == StreamEventType.DONE and evt.content:
                        state["structured_script"] = evt.content
                        state["emotion_curve"] = evt.content.get("emotion_curve", [])
                        writer(log_agent_output("script_agent",
                            summarize_script_output(evt.content)))
            except Exception as exc:
                writer(log_agent_error("script_agent", f"执行失败: {exc}",
                    {"raw_text": inputs.raw_text[:100], "error_type": type(exc).__name__}))
                raise

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_script"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_script",
                label=KEY_CHECKPOINTS["after_script"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break
            # 用户选择重新生成 → 循环回到 agent 执行

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

        # --- Regenerate loop ---
        while True:
            try:
                async for evt in self.character_agent.astream(inputs, ctx):
                    writer(evt)
                    if evt.type == StreamEventType.DONE and evt.content:
                        state["character_db"] = evt.content.get("character_db") or {}
                        writer(log_agent_output("character_agent",
                            summarize_character_output(evt.content)))
                        # 记录每个角色的 ComfyUI 生成参数（种子 + 外观 prompt）
                        cdb = evt.content.get("character_db") or {}
                        chars = cdb.get("characters", {})
                        for cid, cp in chars.items():
                            writer(log_comfyui_request(
                                "character_agent",
                                kind="character_reference",
                                prompt=cp.get("base_reference", {}).get("generation_prompt", ""),
                                negative_prompt="(see agent hardcoded negative)",
                                seed=cp.get("seed", 0),
                                width=1024,
                                height=1024,
                                workflow_path=self.config.image.workflow_character_path,
                                metadata={
                                    "char_id": cid,
                                    "name": cp.get("name", ""),
                                    "appearance_prompt": cp.get("appearance_prompt", ""),
                                    "gender_tag": cp.get("gender_tag", "1girl"),
                                },
                            ))
            except Exception as exc:
                writer(log_agent_error("character_agent", f"执行失败: {exc}",
                    {"drafts": [d.name for d in drafts], "error_type": type(exc).__name__}))
                raise

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_character"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_character",
                label=KEY_CHECKPOINTS["after_character"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break

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

        # --- Regenerate loop ---
        while True:
            try:
                async for evt in self.storyboard_agent.astream(inputs, ctx):
                    writer(evt)
                    if evt.type == StreamEventType.DONE and evt.content:
                        pages = evt.content.get("pages", [])
                        if not pages:
                            raise ApiBackendError(
                                "StoryboardAgent returned empty pages — "
                                "LLM did not generate any panel designs"
                            )
                        state["storyboard_plan"] = pages
                        writer(log_agent_output("storyboard_agent",
                            summarize_storyboard_output(evt.content)))
            except Exception as exc:
                writer(log_agent_error("storyboard_agent", f"执行失败: {exc}",
                    {"scene_count": len(scenes), "target_pages": inputs.target_pages, "error_type": type(exc).__name__}))
                raise

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_storyboard"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_storyboard",
                label=KEY_CHECKPOINTS["after_storyboard"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break

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
        raw_storyboard = state.get("storyboard_plan", [])
        if not raw_storyboard:
            raise ApiBackendError(
                "storyboard_plan is empty — no panels to generate images for"
            )
        pages = [PageLayout.model_validate(p) for p in raw_storyboard]
        all_panel_images: list[dict] = []
        total_panels = sum(len(page.panels) for page in pages)

        # 加载 CharacterDB 供 ImageAgent 读取角色种子和外观 prompt
        cdb_dict = state.get("character_db") or {}
        character_db = CharacterDB.model_validate(cdb_dict) if cdb_dict else None

        writer(log_agent_input("image_agent", {
            "total_panels": total_panels,
            "style_preset": "black and white manga (hardcoded)",
            "backend_preference": "auto",
            "character_db_available": character_db is not None,
            "character_count": len(character_db.characters) if character_db else 0,
        }))

        panel_idx = 0
        for page in pages:
            for panel in page.panels:
                panel_idx += 1
                # 传入 character_db 以启用固定种子 + 角色外观 prompt
                inputs = ImageInput(
                    panel_plan=panel,
                    character_db=character_db,
                    style_preset="black and white manga",
                )
                try:
                    async for evt in self.image_agent.astream(inputs, ctx):
                        writer(evt)
                        if evt.type == StreamEventType.DONE and evt.content:
                            pi = evt.content.get("panel_image")
                            if pi:
                                all_panel_images.append(pi)
                                # 记录完整的 ComfyUI 生图参数
                                writer(log_comfyui_request(
                                    "image_agent",
                                    kind="panel",
                                    prompt=pi.get("prompt_used", ""),
                                    negative_prompt=pi.get("negative_prompt_used", ""),
                                    seed=pi.get("seed", 0),
                                    width=pi.get("width", 0),
                                    height=pi.get("height", 0),
                                    workflow_path=self.config.image.workflow_panel_path,
                                    metadata={
                                        "panel_id": pi.get("panel_id", "?"),
                                        "characters": pi.get("characters_present", []),
                                        "backend": pi.get("backend", "?"),
                                    },
                                ))
                                writer(log_comfyui_response(
                                    "image_agent",
                                    kind="panel",
                                    image_path=pi.get("image_path", ""),
                                    backend=pi.get("backend", "?"),
                                    seed=pi.get("seed", 0),
                                ))
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
                except Exception as exc:
                    writer(log_agent_error("image_agent",
                        f"面板 {panel.panel_id} 生成失败: {exc}",
                        {"panel_id": panel.panel_id, "panel_idx": panel_idx, "error_type": type(exc).__name__}))
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
        raw_storyboard = state.get("storyboard_plan", [])
        if not raw_storyboard:
            raise ApiBackendError(
                "storyboard_plan is empty — cannot compose layout without panels"
            )
        pages = [PageLayout.model_validate(p) for p in raw_storyboard]
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

        # --- Regenerate loop ---
        while True:
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

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_layout"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_layout",
                label=KEY_CHECKPOINTS["after_layout"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("layout_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "layout_agent"))
        return state

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def arun(self, state: ComicState) -> AsyncIterator[object]:
        """异步执行工作流，流式吐出消息。

        线性流水线，5 个 Agent 顺序执行。
        支持 CheckpointSignal 暂停与 resume。
        """
        # 捕获事件循环引用（供 respond() 线程安全调用）
        self._loop = asyncio.get_running_loop()

        # 跨会话恢复：合并已保存的 state
        if self._saved_state is not None:
            for key, value in self._saved_state.items():
                if key not in state or not state[key]:
                    state[key] = value  # type: ignore[literal-required]

        workflow_t0 = time.perf_counter()
        resume_note = ""
        if self._resume_phase and self._resume_phase not in ("init",):
            resume_note = f" (从 {self._resume_phase} 恢复)"
        yield WorkflowMessage(WorkflowEvent.NODE_START, "workflow")
        yield log_workflow_event(
            f"工作流启动{resume_note}",
            f"project={state.get('project_id', '?')}",
        )

        config: dict[str, Any] = {
            "configurable": {"thread_id": state.get("project_id", "default")}
        }
        self._current_config = config
        prev_phase = state.get("current_phase", "init")

        try:
            async for mode, data in self._compiled.astream(
                state, config, stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    yield data
                elif mode == "updates":
                    if "__interrupt__" in data:
                        continue
                    for _node_name, node_update in data.items():
                        if isinstance(node_update, dict):
                            for key, value in node_update.items():
                                state[key] = value  # type: ignore[literal-required]
                            new_phase = node_update.get("current_phase")
                            if new_phase and new_phase != prev_phase:
                                yield log_state_change(prev_phase, new_phase)
                                prev_phase = new_phase

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
            f"pages={len(state.get('final_pages', []))}")
        yield WorkflowMessage(WorkflowEvent.WORKFLOW_DONE, "workflow", state)

    # ------------------------------------------------------------------
    # Checkpoint / Resume 机制
    # ------------------------------------------------------------------

    def _save_checkpoint(self, state: ComicState) -> None:
        """在每个 Agent 完成后自动保存项目状态。"""
        try:
            proj = state_to_project(state)
            save_project(proj)
        except Exception:
            pass  # 保存失败不应中断工作流

    async def _await_response(self) -> None:
        """阻塞当前节点，直到 UI 线程调用 respond() 恢复。"""
        self._response_event = asyncio.Event()
        await self._response_event.wait()

    def respond(self, decision: str) -> None:
        """由 UI 线程调用，恢复暂停的工作流。

        使用 call_soon_threadsafe 安全地通知后台 asyncio 事件循环。
        """
        self._last_decision = decision
        if self._response_event and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._response_event.set)

    @classmethod
    def from_saved_project(cls, project_id: str) -> "ComicWorkflow | None":
        """从已保存的项目创建可恢复的 ComicWorkflow。

        加载 project.json，读取 current_phase，构造从对应节点开始的工作流。
        返回 None 表示项目不存在或无法加载。
        """
        from comicweaver.storage import load_project, project_to_state
        project = load_project(project_id)
        if project is None:
            return None
        state = project_to_state(project)
        phase = state.get("current_phase", "init")
        wf = cls(resume_phase=phase)
        wf._saved_state = state
        return wf

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
