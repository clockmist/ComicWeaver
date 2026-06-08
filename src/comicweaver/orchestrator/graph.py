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

from comicweaver.agents import (
    BubbleAgent,
    CharacterAgent,
    ImageAgent,
    LayoutAgent,
    ScriptAgent,
    StoryAgent,
    StoryboardAgent,
)
from comicweaver.agents.layout_agent import solve_page_bboxes
from comicweaver.api import ApiBackendError
from comicweaver.config import load_config
from comicweaver.core import (
    Action,
    AgentContext,
    BoundingBox,
    BubbleInput,
    CharacterDB,
    CharacterDraft,
    CharacterInput,
    ComicState,
    CreationMode,
    Dialogue,
    ImageInput,
    InteractionMode,
    LayoutInput,
    NarrativeStructure,
    PageLayout,
    PanelImage,
    Scene,
    ScriptInput,
    ScriptPage,
    StoryboardInput,
    StoryInput,
    StoryOutput,
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
    summarize_bubble_output,
    summarize_character_output,
    summarize_image_output,
    summarize_layout_output,
    summarize_script_output,
    summarize_storyboard_output,
)

from .types import (
    KEY_CHECKPOINTS,
    CheckpointSignal,
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
        self.story_agent = StoryAgent()
        self.script_agent = ScriptAgent()
        self.character_agent = CharacterAgent()
        self.storyboard_agent = StoryboardAgent()
        self.image_agent = ImageAgent()
        self.bubble_agent = BubbleAgent()
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
        self._last_guidance: str = ""

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

        # -- 7 个生产节点，线性串联 (v1.0: character 移到 script 之前) --
        builder.add_node("story", self._node_story)
        builder.add_node("character", self._node_character)
        builder.add_node("script", self._node_script)
        builder.add_node("storyboard", self._node_storyboard)
        builder.add_node("image", self._node_image)
        builder.add_node("bubble", self._node_bubble)
        builder.add_node("layout", self._node_layout)

        # -- 边：story → character → script → storyboard → image → bubble → layout → END --
        builder.add_edge("story", "character")
        builder.add_edge("character", "script")
        builder.add_edge("script", "storyboard")
        builder.add_edge("storyboard", "image")
        builder.add_edge("image", "bubble")
        builder.add_edge("bubble", "layout")
        builder.add_edge("layout", END)

        # -- 入口：根据 resume_phase 决定从哪个节点开始 --
        # identity 映射：resume_phase 指定直接从该节点开始（重新执行该阶段）
        phase_to_node: dict[str, str] = {
            "init": "story",
            "story": "story",
            "script": "script",
            "character": "character",
            "storyboard": "storyboard",
            "image": "image",
            "bubble": "bubble",
            "layout": "layout",
        }
        start_node = "story"
        if self._resume_phase and self._resume_phase in phase_to_node:
            start_node = phase_to_node[self._resume_phase]
        builder.add_edge(START, start_node)

        return builder

    # ------------------------------------------------------------------
    # 生产节点实现
    # ------------------------------------------------------------------

    async def _node_story(self, state: ComicState) -> ComicState:
        """v0.4: StoryAgent — expand raw user input into a complete narrative story."""
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "story_agent"))
        writer(log_workflow_event("进入阶段", "story"))

        state["current_phase"] = "story"
        ctx = self._make_context(state)
        inputs = StoryInput(
            raw_text=state.get("user_input", ""),
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
            target_pages=state.get("target_pages", 4),
            style_hint=state.get("style_preset", "manga"),
        )

        writer(log_agent_input("story_agent", {
            "raw_text": inputs.raw_text[:200],
            "target_pages": inputs.target_pages,
        }))

        # --- Regenerate loop: 用户选择"重新生成"时重新执行 agent ---
        while True:
            try:
                async for evt in self.story_agent.astream(inputs, ctx):
                    writer(evt)
                    if evt.type == StreamEventType.DONE and evt.content:
                        state["developed_story"] = evt.content
                        word_count = len(evt.content.get("story_text", ""))
                        writer(log_agent_output("story_agent", {
                            "title": evt.content.get("title", ""),
                            "genre": evt.content.get("genre", []),
                            "tone": evt.content.get("tone", ""),
                            "word_count": word_count,
                            "character_count": len(evt.content.get("characters", [])),
                        }))
            except Exception as exc:
                writer(log_agent_error("story_agent", f"执行失败: {exc}",
                    {"raw_text": inputs.raw_text[:100], "error_type": type(exc).__name__}))
                raise

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_story"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_story",
                label=KEY_CHECKPOINTS["after_story"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break
            state["developed_story"] = {}
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("story_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "story_agent"))
        return state

    async def _node_script(self, state: ComicState) -> ComicState:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "script_agent"))
        writer(log_workflow_event("进入阶段", "script"))

        state["current_phase"] = "script"
        ctx = self._make_context(state)

        # v1.0: Build story + character_db for context-aware script generation
        story_dict = state.get("developed_story") or {}
        story = StoryOutput.model_validate(story_dict) if story_dict else None

        cdb_dict = state.get("character_db") or {}
        character_db = CharacterDB.model_validate(cdb_dict) if cdb_dict else None

        inputs = ScriptInput(
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
            raw_text=state.get("user_input", ""),
            story=story,
            character_db=character_db,
            target_pages=state.get("target_pages", 4),
            style_hint=state.get("style_preset", "manga"),
        )

        # 开发者日志：Agent 输入
        writer(log_agent_input("script_agent", {
            "creation_mode": inputs.creation_mode.value,
            "has_story": story is not None,
            "has_character_db": character_db is not None,
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
                        # v1.0: 从 pages 提取 emotion_curve（替代旧 script 的 emotion_curve）
                        state["emotion_curve"] = _extract_emotion_curve(evt.content)
                        writer(log_agent_output("script_agent",
                            summarize_script_output(evt.content)))
            except Exception as exc:
                writer(log_agent_error("script_agent", f"执行失败: {exc}",
                    {"raw_text": (story.story_text if story else inputs.raw_text)[:100],
                     "error_type": type(exc).__name__}))
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
            state["structured_script"] = {}
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

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

        # v1.0: 从 StoryOutput.characters 构建 CharacterDraft（而非 ScriptOutput）
        story_dict = state.get("developed_story") or {}
        story_chars = story_dict.get("characters", []) if story_dict else []
        drafts = _story_chars_to_drafts(story_chars)
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
            # 用户选择重新生成 → 清除旧数据
            state["character_db"] = {}
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

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

        # v0.6: 优先传 pages（PanelTask），StoryboardAgent 直接消费
        raw_pages = script_dict.get("pages") or []
        if raw_pages:
            pages = [ScriptPage.model_validate(p) for p in raw_pages]
            scenes = []
            source_format = "pages(v1.0-direct)"
        else:
            pages = []
            scenes = [Scene.model_validate(s) for s in script_dict.get("scenes", [])]
            source_format = "scenes(legacy)"

        cdb_dict = state.get("character_db") or {}
        cdb = CharacterDB.model_validate(cdb_dict) if cdb_dict else None

        # Build story context
        story_dict = state.get("developed_story") or {}
        story_summary = story_dict.get("author_note", "") or story_dict.get("story_text", "")[:200] if story_dict else ""
        ns_dict = state.get("narrative_structure") or {}
        narrative_structure = NarrativeStructure.model_validate(ns_dict) if ns_dict else None

        inputs = StoryboardInput(
            pages=pages,
            scenes=scenes,
            emotion_curve=state.get("emotion_curve", []),
            character_db=cdb,
            style_preset=state.get("style_preset", "manga"),
            target_pages=state.get("target_pages", 4),
            story_summary=story_summary,
            narrative_structure=narrative_structure,
        )

        # 开发者日志：Agent 输入
        total_panels = sum(len(p.panels) for p in pages) if pages else len(scenes)
        writer(log_agent_input("storyboard_agent", {
            "panel_count": total_panels,
            "page_count": len(pages) if pages else len(scenes),
            "source_format": source_format,
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
                    {"panel_count": total_panels, "target_pages": inputs.target_pages, "error_type": type(exc).__name__, "source_format": source_format}))
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
            # 用户选择重新生成 → 清除旧数据
            state["storyboard_plan"] = []
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

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
        layout_grids, image_sizes = _precompute_layout_targets(pages)
        state["storyboard_plan"] = [page.model_dump() for page in pages]
        state["layout_grids"] = layout_grids
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
            "layout_precomputed": True,
        }))

        panel_idx = 0
        for page in pages:
            for panel in page.panels:
                panel_idx += 1
                # v0.4: Build reference_windows from character_db
                ref_windows = self._build_ref_windows(panel, character_db)
                # 传入 character_db 以启用固定种子 + 角色外观 prompt
                inputs = ImageInput(
                    panel_plan=panel,
                    reference_windows=ref_windows,
                    character_db=character_db,
                    style_preset="black and white manga",
                    width=image_sizes.get(panel.panel_id, (1024, 1024))[0],
                    height=image_sizes.get(panel.panel_id, (1024, 1024))[1],
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

        # --- Checkpoint & auto-save (介于图像生成与台词气泡之间) ---
        self._save_checkpoint(state)
        if should_pause(state, "after_image"):
            writer(CheckpointSignal(
                checkpoint_id="after_image",
                label=KEY_CHECKPOINTS["after_image"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                    "panel_count": len(all_panel_images),
                },
            ))
            await self._await_response()
            if self._last_decision == "regenerate":
                state["panel_images"] = []
                state["bubble_placements"] = {}
                if self._last_guidance:
                    writer(log_workflow_event("用户指导", self._last_guidance[:200]))
                writer(WorkflowMessage(WorkflowEvent.NODE_END, "image_agent"))
                return state

        writer(WorkflowMessage(WorkflowEvent.NODE_END, "image_agent"))
        return state

    async def _node_bubble(self, state: ComicState) -> ComicState:
        """台词气泡放置阶段 — BubbleAgent 计算所有面板的气泡坐标。"""
        t0 = time.perf_counter()
        writer = get_stream_writer()
        writer(WorkflowMessage(WorkflowEvent.NODE_START, "bubble_agent"))
        writer(log_workflow_event("进入阶段", "bubble"))

        state["current_phase"] = "bubble"
        ctx = self._make_context(state)
        raw_storyboard = state.get("storyboard_plan", [])
        if not raw_storyboard:
            raise ApiBackendError(
                "storyboard_plan is empty — cannot place bubbles without panels"
            )
        pages = [PageLayout.model_validate(p) for p in raw_storyboard]
        panel_imgs = {
            p["panel_id"]: PanelImage.model_validate(p)
            for p in state.get("panel_images", [])
        }

        # 统计对话信息
        total_dialogues = sum(
            len(panel.dialogues_in_panel) for page in pages for panel in page.panels
        )
        writer(log_agent_input("bubble_agent", {
            "page_count": len(pages),
            "panel_image_count": len(panel_imgs),
            "total_dialogues": total_dialogues,
        }))

        inputs = BubbleInput(
            pages=pages,
            panel_images=panel_imgs,
            page_width_px=1240,
            page_height_px=1754,
            margin_px=40,
            gutter_px=10,
            reading_direction="ltr",
        )

        # --- Regenerate loop ---
        while True:
            try:
                async for evt in self.bubble_agent.astream(inputs, ctx):
                    writer(evt)
                    if evt.type == StreamEventType.DONE and evt.content:
                        state["bubble_placements"] = evt.content.get(
                            "bubble_placements", {}
                        )
                        writer(log_agent_output("bubble_agent",
                            summarize_bubble_output(evt.content)))
            except Exception as exc:
                writer(log_agent_error("bubble_agent", f"执行失败: {exc}",
                    {"page_count": len(pages), "error_type": type(exc).__name__}))
                raise

            # --- Checkpoint & auto-save ---
            self._save_checkpoint(state)
            if not should_pause(state, "after_bubble"):
                break
            writer(CheckpointSignal(
                checkpoint_id="after_bubble",
                label=KEY_CHECKPOINTS["after_bubble"],
                payload={
                    "phase": state.get("current_phase", "?"),
                    "project_id": state.get("project_id", "?"),
                },
            ))
            await self._await_response()
            if self._last_decision != "regenerate":
                break
            state["bubble_placements"] = {}
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

        elapsed = (time.perf_counter() - t0) * 1000
        writer(log_performance("bubble_agent", elapsed))
        writer(WorkflowMessage(WorkflowEvent.NODE_END, "bubble_agent"))
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

        # 从 bubble_placements state 读取气泡数据
        bubble_placements = state.get("bubble_placements", {})

        inputs = LayoutInput(
            pages=pages,
            panel_images=panel_imgs,
            bubble_placements=bubble_placements,
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
            # 用户选择重新生成 → 清除旧数据
            state["final_pages"] = []
            state["exports"] = []
            if self._last_guidance:
                writer(log_workflow_event("用户指导", self._last_guidance[:200]))

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

        # 每次运行使用唯一的 thread_id，避免 LangGraph MemorySaver 回放旧状态
        run_id = f"{state.get('project_id', 'default')}_{int(time.time() * 1000)}"
        config: dict[str, Any] = {
            "configurable": {"thread_id": run_id}
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

    def respond(self, decision: str, guidance: str = "") -> None:
        """由 UI 线程调用，恢复暂停的工作流。

        使用 call_soon_threadsafe 安全地通知后台 asyncio 事件循环。
        guidance: 用户可选的指导信息，用于 regenerate 时传递给 agent。
        """
        self._last_decision = decision
        self._last_guidance = guidance
        if self._response_event and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._response_event.set)

    @classmethod
    def from_saved_project(cls, project_id: str) -> ComicWorkflow | None:
        """从已保存的项目创建可恢复的 ComicWorkflow。

        加载 project.json，读取 current_phase，从下一个阶段继续执行
        （如 current_phase="character" → 从 storyboard 继续）。
        返回 None 表示项目不存在或无法加载。
        """
        from comicweaver.storage import load_project, project_to_state
        project = load_project(project_id)
        if project is None:
            return None
        state = project_to_state(project)
        phase = state.get("current_phase", "init")
        # 计算下一个阶段：已完成的 phase → 从下一个 phase 开始
        _next_phase: dict[str, str] = {
            "init": "script",
            "script": "character",
            "character": "storyboard",
            "storyboard": "image",
            "image": "bubble",
            "bubble": "layout",
            "layout": "layout",
        }
        resume_phase = _next_phase.get(phase, "script")
        wf = cls(resume_phase=resume_phase)
        wf._saved_state = state
        return wf

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_ref_windows(
        self, panel, character_db: CharacterDB | None
    ) -> dict:
        """v0.4: Build reference windows for a panel from character_db."""
        from comicweaver.core import ReferenceWindow, WeightedReference
        if not character_db:
            return {}
        windows: dict[str, ReferenceWindow] = {}
        for char_id in panel.characters_in_panel:
            if char_id in character_db.characters:
                cp = character_db.characters[char_id]
                ref_path = cp.base_reference.image_path
                windows[char_id] = ReferenceWindow(
                    panel_id=panel.panel_id,
                    char_id=char_id,
                    references=[
                        WeightedReference(
                            image_path=ref_path,
                            weight=1.0,
                            role="base",
                        )
                    ],
                    window_strategy="base_only",
                )
        return windows

    def _make_context(self, state: ComicState) -> AgentContext:
        return AgentContext(
            project_id=state.get("project_id", "proj"),
            interaction_mode=InteractionMode(state.get("interaction_mode", "semi_auto")),
            style_preset=state.get("style_preset", "manga"),
            creation_mode=CreationMode(state.get("creation_mode", "simple")),
        )


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

_VALID_ROLES = {"protagonist", "antagonist", "supporting", "extra"}

_ROLE_NORMALIZE: dict[str, str] = {
    "deuteragonist": "supporting",
    "tritagonist": "supporting",
    "villain": "antagonist",
    "hero": "protagonist",
    "main": "protagonist",
    " lead": "protagonist",
    "side": "supporting",
    "minor": "extra",
    "background": "extra",
    "rival": "antagonist",
    "mentor": "supporting",
}


def _normalize_role(raw: str) -> str:
    """将 LLM 自由输出的角色类型映射到合法的 CharacterDraft.role 枚举值。"""
    role = str(raw).strip().lower()
    if role in _VALID_ROLES:
        return role
    return _ROLE_NORMALIZE.get(role, "supporting")


def _story_chars_to_drafts(story_chars: list[dict]) -> list[CharacterDraft]:
    """v1.0: 将 StoryOutput.characters 转换为 CharacterDraft 列表。"""
    drafts = []
    for i, c in enumerate(story_chars):
        drafts.append(CharacterDraft(
            char_id=f"char_{i:03d}",
            name=c.get("name", f"角色{i}"),
            role=_normalize_role(c.get("role", "supporting")),
            appearance=c.get("brief_description", ""),
            first_appearance_scene=0,
        ))
    return drafts


def _panel_tasks_to_scenes(raw_pages: list[dict]) -> list[Scene]:
    """v1.0 桥接：将新格式 pages(PanelTask) 转换为旧格式 scenes(Scene)。

    StoryboardAgent 暂时仍消费 Scene，后续改造后移除。
    每个 PanelTask 映射为一个 panel_hint=1 的 Scene。
    """
    scenes = []
    for page in raw_pages:
        for panel in page.get("panels", []):
            panel_id = panel.get("panel_id", "")
            char_id = panel.get("character_id", "")
            scenes.append(Scene(
                scene_id=panel_id,
                order=len(scenes),
                location=panel.get("location", ""),
                time_of_day=panel.get("time_of_day", ""),
                atmosphere=panel.get("atmosphere", ""),
                characters_present=[char_id] if char_id else [],
                actions=[Action(actor=char_id, description=panel.get("character_action", ""))]
                    if panel.get("character_action") else [],
                dialogues=[Dialogue(
                    speaker=char_id,
                    text=panel.get("dialogue_text", ""),
                    tone=panel.get("dialogue_tone", "neutral"),
                    is_thought=panel.get("is_thought", False),
                )] if panel.get("dialogue_text") else [],
                narration=panel.get("narration") or None,
                emotion_intensity=panel.get("emotion_intensity", 0.5),
                panel_hint=1,
            ))
    return scenes


def _extract_emotion_curve(script_content: dict) -> list[float]:
    """v1.0: 从 pages 的面板中提取 emotion_intensity 序列。

    替代旧 script 输出中的 emotion_curve 字段。
    """
    pages = script_content.get("pages", [])
    curve: list[float] = []
    for page in pages:
        for panel in page.get("panels", []):
            intensity = panel.get("emotion_intensity")
            if intensity is not None:
                curve.append(float(intensity))
    return curve


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _generation_size_from_bbox(
    bbox: BoundingBox,
    *,
    page_width_px: int = 1240,
    page_height_px: int = 1754,
    margin_px: int = 40,
    multiple: int = 64,
    min_short_side: int = 384,
    max_long_side: int = 1536,
) -> tuple[int, int]:
    """Convert a solved page bbox into a model-friendly generation size.

    The final compositor works in page pixels, but diffusion backends are more
    stable on dimensions aligned to latent-friendly multiples. This preserves
    the solved panel aspect ratio while keeping resolution in a practical range.
    """
    inner_w = max(1, page_width_px - margin_px * 2)
    inner_h = max(1, page_height_px - margin_px * 2)
    slot_w = max(1.0, inner_w * bbox.width)
    slot_h = max(1.0, inner_h * bbox.height)

    short_side = min(slot_w, slot_h)
    long_side = max(slot_w, slot_h)
    scale = 1.0
    if short_side < min_short_side:
        scale = max(scale, min_short_side / short_side)
    if long_side * scale > max_long_side:
        scale = max_long_side / long_side

    width = _round_to_multiple(slot_w * scale, multiple)
    height = _round_to_multiple(slot_h * scale, multiple)

    if max(width, height) > max_long_side:
        cap_scale = max_long_side / max(width, height)
        width = _round_to_multiple(width * cap_scale, multiple)
        height = _round_to_multiple(height * cap_scale, multiple)

    return width, height


def _precompute_layout_targets(
    pages: list[PageLayout],
    *,
    page_width_px: int = 1240,
    page_height_px: int = 1754,
    margin_px: int = 40,
    gutter_px: int = 10,
) -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """Solve final panel bboxes before image generation and derive image sizes."""
    layout_grids: list[dict] = []
    image_sizes: dict[str, tuple[int, int]] = {}

    for page in pages:
        bboxes = solve_page_bboxes(
            page,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
            margin_px=margin_px,
            gutter_px=gutter_px,
        )
        panel_entries = []
        for panel, bbox in zip(page.panels, bboxes, strict=False):
            width, height = _generation_size_from_bbox(
                bbox,
                page_width_px=page_width_px,
                page_height_px=page_height_px,
                margin_px=margin_px,
            )
            panel.bbox = bbox
            panel.size_ratio = bbox.width * bbox.height
            image_sizes[panel.panel_id] = (width, height)
            panel_entries.append({
                "panel_id": panel.panel_id,
                "bbox": bbox.model_dump(),
                "target_width": width,
                "target_height": height,
            })
        layout_grids.append({
            "page_id": page.page_id,
            "page_number": page.page_number,
            "width_px": page_width_px,
            "height_px": page_height_px,
            "panels": panel_entries,
        })

    return layout_grids, image_sizes
