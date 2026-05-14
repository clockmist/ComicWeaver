"""分镜设计 Agent - Mock 实现。"""
from __future__ import annotations

from typing import AsyncIterator

from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    BoundingBox,
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


# 简单布局模板:每页固定bbox列表(x, y, w, h),坐标0-1
_LAYOUT_TEMPLATES = {
    "grid_2x2": [
        (0.0, 0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5, 0.5),
        (0.0, 0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5, 0.5),
    ],
    "grid_3x2": [
        (0.0, 0.0, 0.5, 0.33),
        (0.5, 0.0, 0.5, 0.33),
        (0.0, 0.33, 0.5, 0.33),
        (0.5, 0.33, 0.5, 0.33),
        (0.0, 0.66, 0.5, 0.34),
        (0.5, 0.66, 0.5, 0.34),
    ],
    "splash_top": [
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5, 0.5),
    ],
    "diagonal": [
        (0.0, 0.0, 0.6, 0.45),
        (0.4, 0.0, 0.6, 0.45),
        (0.0, 0.55, 1.0, 0.45),
    ],
    "full_bleed": [
        (0.0, 0.0, 1.0, 1.0),
    ],
}


def _select_layout(panel_count: int, climax: bool) -> str:
    if climax:
        return "splash_top" if panel_count <= 3 else "diagonal"
    if panel_count <= 1:
        return "full_bleed"
    if panel_count <= 4:
        return "grid_2x2"
    return "grid_3x2"


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
    """分镜设计 Agent (Mock)."""

    name = "storyboard_agent"
    version = "0.1.0-mock"
    rubric_id = "rubric_storyboard_v1"

    async def run(
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
            template = _select_layout(len(chunk), climax)
            slots = _LAYOUT_TEMPLATES[template]

            page_id = f"page_{page_idx + 1:03d}"
            panels: list[PanelPlan] = []
            for i, (scene, panel_idx) in enumerate(chunk):
                slot = slots[i % len(slots)]
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
                        bbox=BoundingBox(x=slot[0], y=slot[1], width=slot[2], height=slot[3]),
                        shape=PanelShape.SPLASH if template == "full_bleed" else PanelShape.RECTANGLE,
                        size_ratio=slot[2] * slot[3],
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
                    layout_template=template,
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
