"""Storyboard planning agent with API and local backends."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    BubbleHint,
    CameraAngle,
    PageLayout,
    PanelPlan,
    PanelShape,
    PromptPack,
    Scene,
    ShotSize,
    StoryboardInput,
    StoryboardOutput,
    StreamEvent,
    StreamEventType,
)

# Layout hints — intent passed to LayoutAgent (NO coordinates).
# LayoutAgent uses these to guide its recursive binary-partition solver.


def _select_layout_hint(panel_count: int, climax: bool, action_heavy: bool) -> str:
    """Return a layout *hint* for the page (intent, not coordinates).

    The LayoutAgent uses this to bias its split-direction heuristics.
    """
    if climax:
        return "climax"
    if action_heavy:
        return "action"
    if panel_count >= 5:
        return "dialogue"  # many panels → likely dialogue-heavy
    if panel_count <= 1:
        return "climax"     # single panel → splash / establishing
    return "standard"


def _select_shot(scene: Scene, panel_idx: int) -> ShotSize:
    """根据场景内容选取景别。"""
    if scene.emotion_intensity >= 0.85:
        return ShotSize.EXTREME_CLOSE if panel_idx % 2 == 0 else ShotSize.CLOSE
    if scene.dialogues:
        return ShotSize.MEDIUM if panel_idx == 0 else ShotSize.CLOSE
    if not scene.characters_present:
        return ShotSize.LONG
    return ShotSize.MEDIUM


def _select_angle(scene: Scene) -> CameraAngle:
    if scene.emotion_intensity >= 0.8:
        return CameraAngle.LOW
    if scene.atmosphere and "压抑" in scene.atmosphere:
        return CameraAngle.HIGH
    return CameraAngle.EYE_LEVEL


def _build_prompt(scene: Scene, shot: ShotSize, angle: CameraAngle, style: str) -> PromptPack:
    chars = "、".join(scene.characters_present) or "no character"
    pos = (
        f"{style} style, "
        f"{shot.value.replace('_', ' ')} shot, "
        f"{angle.value.replace('_', ' ')}, "
        f"characters: {chars}, "
        f"location: {scene.location}, "
        f"atmosphere: {scene.atmosphere}, "
        f"emotion intensity {scene.emotion_intensity:.2f}, "
        f"masterpiece, best quality, detailed"
    )
    neg = "low quality, blurry, distorted, multiple panels, comic page, watermark, text"
    return PromptPack(
        positive_prompt=pos,
        negative_prompt=neg,
        style_tags=[style, "manga"],
        composition_tags=[shot.value, angle.value],
        quality_tags=["masterpiece", "best quality"],
    )


class StoryboardAgent(BaseAgent[StoryboardInput, StoryboardOutput]):
    """Storyboard planning agent."""

    name = "storyboard_agent"
    version = "0.2.0-api"
    rubric_id = "rubric_storyboard_v1"

    async def run(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        if self.config.llm.is_available:
            try:
                return await self._run_api(inputs, context)
            except ApiBackendError:
                if not self.config.runtime.fallback_to_local:
                    raise
        return await self._run_local(inputs, context)

    async def _run_api(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        client = OpenAICompatibleLLMClient(self.config.llm)
        payload = {
            "task": "generate_comic_storyboard",
            "input_schema": StoryboardInput.model_json_schema(),
            "output_schema": StoryboardOutput.model_json_schema(),
            "input": inputs.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "layout_coordinate_system": "Panel bbox values use normalized x/y/width/height in [0, 1].",
            "requirements": [
                "Return JSON only.",
                "Each page must contain panels with valid bbox values.",
                "Each panel must include prompt_pack for image generation.",
                "Preserve scene and character ids from the input.",
            ],
        }
        data = await asyncio.to_thread(
            client.complete_json,
            "You convert comic scripts into page-level storyboards.",
            payload,
        )
        output = StoryboardOutput.model_validate(data)
        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[f"LLM API provider: {self.config.llm.provider}"],
            )
        })

    async def _run_local(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> StoryboardOutput:
        await self._sleep_for_demo(0.3)

        # 把场景按目标panel数展开
        panel_specs: list[tuple[Scene, int]] = []
        for scene in inputs.scenes:
            for i in range(max(1, scene.panel_hint)):
                panel_specs.append((scene, i))

        # 按目标panels_per_page分页
        per_page = max(1, inputs.target_panels_per_page)
        pages: list[PageLayout] = []
        page_idx = 0
        for start in range(0, len(panel_specs), per_page):
            chunk = panel_specs[start:start + per_page]
            climax = any(s.emotion_intensity >= 0.85 for s, _ in chunk)
            action_heavy = any(
                s.actions and len(s.actions) >= 2 for s, _ in chunk
            )

            # Layout hint only — LayoutAgent computes actual bboxes
            layout_hint = _select_layout_hint(len(chunk), climax, action_heavy)

            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []
            for i, (scene, panel_idx) in enumerate(chunk):
                shot = _select_shot(scene, panel_idx)
                angle = _select_angle(scene)
                panel_id = f"{page_id}_p{i + 1:02d}"
                bubble_hints = [
                    BubbleHint(
                        dialogue_index=di,
                        suggested_position="top_right" if di % 2 == 0 else "bottom_left",
                        bubble_type="speech" if not d.is_thought else "thought",
                    )
                    for di, d in enumerate(scene.dialogues)
                ]
                panels.append(
                    PanelPlan(
                        panel_id=panel_id,
                        page_id=page_id,
                        order_in_page=i + 1,
                        source_scene_id=scene.scene_id,
                        source_dialogue_indices=list(range(len(scene.dialogues))),
                        # bbox uses default (full page) — LayoutAgent overrides
                        shape=PanelShape.RECTANGLE,
                        shot_size=shot,
                        camera_angle=angle,
                        characters_in_panel=list(scene.characters_present),
                        primary_action=scene.actions[0].description if scene.actions else "",
                        setting=scene.location,
                        mood=scene.atmosphere,
                        emotion_intensity=scene.emotion_intensity,
                        prompt_pack=_build_prompt(scene, shot, angle, inputs.style_preset),
                        dialogues_in_panel=list(scene.dialogues) if panel_idx == 0 else [],
                        speech_bubble_hints=bubble_hints if panel_idx == 0 else [],
                    )
                )

            avg_emotion = sum(s.emotion_intensity for s, _ in chunk) / len(chunk)
            pages.append(
                PageLayout(
                    page_id=page_id,
                    page_number=page_idx + 1,
                    layout_template=layout_hint,  # keep for backward compat; real intent in layout_hint
                    layout_hint=layout_hint,
                    panels=panels,
                    page_emotion_avg=avg_emotion,
                    is_climax_page=climax,
                )
            )
            page_idx += 1

        return StoryboardOutput(
            pages=pages,
            total_panels=sum(len(p.panels) for p in pages),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"生成{len(pages)}页, {sum(len(p.panels) for p in pages)}格",
                ],
            ),
        )

    async def astream(
        self, inputs: StoryboardInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "开始分镜规划...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "为每场景分配分镜数...")
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "选择布局模板与景别...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)

        await self._sleep_for_demo(0.15)
        yield self._make_event(StreamEventType.THINKING, "拼装提示词...")

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"pages": len(output.pages), "panels": output.total_panels},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())
