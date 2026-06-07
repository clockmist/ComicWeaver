"""台词气泡放置 Agent — 独立于排版，负责计算面板中对话气泡的位置和尺寸。

从 LayoutAgent 中拆分出来，职责:
1. 预计算面板 bbox（为气泡放置提供空间约束）
2. 对每个有对话的面板放置气泡
3. 输出归一化坐标和样式参数，供 LayoutAgent 合成使用

当前版本 (v0.1.0) 完全保留原有的 place_bubbles() 逻辑，
后续版本将集成 YOLO 人脸检测和自适应样式。
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    BubbleInput,
    BubbleOutput,
    BubblePlacementResult,
    PageLayout,
    PanelImage,
    StreamEvent,
    StreamEventType,
)

from .layout.bubbles import place_bubbles
from .layout_agent import solve_page_bboxes


class BubbleAgent(BaseAgent[BubbleInput, BubbleOutput]):
    """台词气泡放置 Agent — 计算气泡坐标和样式参数。"""

    name = "bubble_agent"
    version = "0.1.0"
    rubric_id = "rubric_bubble_v1"

    async def run(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        all_placements: dict[str, list[BubblePlacementResult]] = {}
        total_bubbles = 0
        face_stats: dict[str, int] = {}

        for page in inputs.pages:
            page_id = page.page_id

            # Step 1: 预计算面板 bbox（复用 solve_page_bboxes）
            bboxes = solve_page_bboxes(
                page,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                gutter_px=inputs.gutter_px,
            )

            # Step 2: 放置气泡（复用现有的 place_bubbles）
            # 当前不传 faces_by_panel（第二步再加入 YOLO）
            bubbles = place_bubbles(
                page.panels,
                bboxes,
                inputs.panel_images,
                font_size_pt=inputs.font_config.base_size_pt,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                faces_by_panel=None,  # 第二步接入 YOLO
            )

            # Step 3: 转为 BubblePlacementResult（供输出和 LayoutAgent 消费）
            results: list[BubblePlacementResult] = []
            for bp in bubbles:
                results.append(BubblePlacementResult(
                    panel_id=bp.panel_id,
                    dialogue_index=bp.dialogue_index,
                    bubble_type=bp.bubble_type,
                    speaker=bp.speaker,
                    text=bp.text,
                    x=bp.x,
                    y=bp.y,
                    w=bp.w,
                    h=bp.h,
                    font_size_pt=inputs.font_config.base_size_pt,
                    tail_direction=bp.tail_direction,
                    occlusion_score=bp.occlusion_score,
                ))
                total_bubbles += 1

            all_placements[page_id] = results
            face_stats[page_id] = 0  # 第二步将填充实际人脸数

        return BubbleOutput(
            bubble_placements=all_placements,
            total_bubbles=total_bubbles,
            face_detection_stats=face_stats,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"已放置{total_bubbles}个气泡",
                    f"页尺寸: {inputs.page_width_px}x{inputs.page_height_px}",
                ],
            ),
        )

    async def astream(
        self, inputs: BubbleInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """流式输出气泡放置进度。"""
        yield self._make_event(StreamEventType.LOG, "开始台词气泡放置...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        n = len(inputs.pages)
        for i in range(n):
            page = inputs.pages[i]
            dialogue_count = sum(
                len(p.dialogues_in_panel) for p in page.panels
            )
            yield self._make_event(
                StreamEventType.THINKING,
                f"第{i + 1}页: {len(page.panels)}个面板, {dialogue_count}条对话...",
            )
            yield self._make_event(
                StreamEventType.PROGRESS, 0.1 + (i + 1) / max(n, 1) * 0.7
            )

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"total_bubbles": output.total_bubbles},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())
