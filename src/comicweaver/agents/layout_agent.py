"""Page layout and composition agent.

The LayoutAgent is responsible for the final stage of comic creation:

1. **Weight calculation** — determine each panel's visual importance from metadata
2. **Layout solving** — recursive binary partition to assign bboxes
3. **Bubble placement** — position dialogue bubbles based on text & hints
4. **Page compositing** — render the final page image

When an LLM is configured it is used for layout-quality assessment only
(the solving itself is algorithmic and deterministic).
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    BoundingBox,
    ExportArtifact,
    FinalPage,
    LayoutInput,
    LayoutMetrics,
    LayoutOutput,
    StreamEvent,
    StreamEventType,
)

from .layout import (
    PageContext,
    PageGeometry,
    PanelSlot,
    compose_page,
    normalize_weights,
    place_bubbles,
    solve_layout,
)
from .layout.solver import apply_gutters
from .layout.templates import LayoutHint


class LayoutAgent(BaseAgent[LayoutInput, LayoutOutput]):
    """Page layout and composition agent.

    Uses algorithmic layout solving (no LLM required).  An optional LLM path
    (``_run_api``) may be added later for quality assessment.
    """

    name = "layout_agent"
    version = "0.3.0-layout"
    rubric_id = "rubric_layout_v1"

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def run(self, inputs: LayoutInput, context: AgentContext) -> LayoutOutput:
        await self._sleep_for_demo(0.2)

        if self.config.llm.is_available:
            # Future: LLM quality-assessment path
            # For now, fall through to algorithmic
            pass

        return self._run_algorithmic(inputs, context)

    def _run_algorithmic(
        self, inputs: LayoutInput, context: AgentContext,
    ) -> LayoutOutput:
        """Pure-algorithm layout pipeline."""
        geometry = PageGeometry(
            width_px=inputs.page_width_px,
            height_px=inputs.page_height_px,
            margin_px=inputs.margin_px,
            gutter_px=inputs.gutter_px,
        )

        final_pages: list[FinalPage] = []
        total_bubbles = 0

        for page in inputs.pages:
            # ---- Step 1: calculate panel weights ----
            page_ctx = PageContext(
                is_climax_page=page.is_climax_page,
                page_emotion_avg=page.page_emotion_avg,
            )
            weights = normalize_weights(page.panels, page_ctx)

            # ---- Step 2: recursive-partition layout solve ----
            hint: LayoutHint = _coerce_hint(page.layout_hint)
            bboxes = solve_layout(page.panels, weights, hint=hint)

            # ---- Step 3: apply gutters ----
            bboxes = apply_gutters(bboxes, geometry)

            # ---- Step 4: place dialogue bubbles ----
            bubbles = place_bubbles(
                page.panels, bboxes, inputs.panel_images,
                font_size_pt=inputs.font_config.base_size_pt,
            )

            # ---- Step 5: composite the page image ----
            panel_slots = [
                PanelSlot(
                    panel_id=panel.panel_id,
                    bbox=bbox,
                    shot_size=panel.shot_size.value
                              if hasattr(panel.shot_size, "value")
                              else str(panel.shot_size),
                    emotion_intensity=panel.emotion_intensity,
                )
                for panel, bbox in zip(page.panels, bboxes, strict=False)
            ]
            page_path = compose_page(
                project_id=context.project_id,
                page_id=page.page_id,
                page_number=page.page_number,
                panel_slots=panel_slots,
                panel_images=inputs.panel_images,
                bubbles=bubbles,
                width_px=inputs.page_width_px,
                height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
            )

            # ---- Step 6: build FinalPage ----
            placed = [
                # Convert BubblePlacement → PlacedBubble for the output model
                _bubble_to_placed(b, bboxes, page.panels)
                for b in bubbles
            ]
            final_pages.append(
                FinalPage(
                    page_id=page.page_id,
                    page_number=page.page_number,
                    image_path=page_path,
                    width_px=inputs.page_width_px,
                    height_px=inputs.page_height_px,
                    bubbles=placed,
                )
            )
            total_bubbles += len(placed)

        # ---- Exports ----
        exports: list[ExportArtifact] = []
        for fp in final_pages:
            fsize = os.path.getsize(fp.image_path) if os.path.exists(fp.image_path) else 0
            exports.append(
                ExportArtifact(
                    format="png",
                    file_path=fp.image_path,
                    file_size_bytes=fsize,
                    page_count=1,
                )
            )

        # ---- Metrics ----
        overflow = sum(
            1 for fp in final_pages
            for b in fp.bubbles
            if b.occlusion_score >= 0.7
        )
        metrics = LayoutMetrics(
            total_pages=len(final_pages),
            total_bubbles=total_bubbles,
            avg_occlusion_score=0.05,  # placeholder; image analysis will refine
            bubbles_overflow_count=overflow,
        )

        return LayoutOutput(
            final_pages=final_pages,
            exports=exports,
            overall_metrics=metrics,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"已合成{len(final_pages)}页, {total_bubbles}个气泡",
                    f"页尺寸: {inputs.page_width_px}×{inputs.page_height_px}",
                ],
            ),
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def astream(
        self, inputs: LayoutInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "开始排版合成...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        n = len(inputs.pages)
        for i in range(n):
            await self._sleep_for_demo(0.1)
            yield self._make_event(
                StreamEventType.THINKING,
                f"计算第{i + 1}页面板比重与布局...",
            )
            yield self._make_event(StreamEventType.PROGRESS, 0.1 + (i + 1) / max(n, 1) * 0.6)

        await self._sleep_for_demo(0.1)
        yield self._make_event(StreamEventType.THINKING, "定位对话气泡...")
        yield self._make_event(StreamEventType.PROGRESS, 0.8)

        await self._sleep_for_demo(0.1)
        yield self._make_event(StreamEventType.THINKING, "合成最终页面...")

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"page_count": len(output.final_pages), "bubbles": output.overall_metrics.total_bubbles},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_VALID_HINTS = frozenset({"standard", "climax", "action", "dialogue", "establishing"})


def _coerce_hint(raw: str) -> LayoutHint:
    """Coerce a raw hint string to a valid LayoutHint."""
    if raw in _VALID_HINTS:
        return raw  # type: ignore[return-value]
    return "standard"


def _bubble_to_placed(
    bp,  # BubblePlacement
    bboxes: list[BoundingBox],
    panels,  # list[PanelPlan]
):
    """Convert internal BubblePlacement to the schema PlacedBubble."""
    from comicweaver.core.schema import PlacedBubble

    return PlacedBubble(
        bubble_id=f"{bp.panel_id}_b{bp.dialogue_index}",
        panel_id=bp.panel_id,
        dialogue_index=bp.dialogue_index,
        bubble_type=bp.bubble_type,
        bbox=BoundingBox(x=bp.x, y=bp.y, width=bp.w, height=bp.h),
        text=bp.text,
        font_size_pt=12,
        occlusion_score=bp.occlusion_score,
    )
