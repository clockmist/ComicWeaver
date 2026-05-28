"""Page layout and composition agent."""
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
    PlacedBubble,
    StreamEvent,
    StreamEventType,
)
from comicweaver.storage.placeholder import generate_page_placeholder


class LayoutAgent(BaseAgent[LayoutInput, LayoutOutput]):
    """Page layout and composition agent."""

    name = "layout_agent"
    version = "0.2.0-local"
    rubric_id = "rubric_layout_v1"

    async def run(self, inputs: LayoutInput, context: AgentContext) -> LayoutOutput:
        await self._sleep_for_demo(0.2)

        final_pages: list[FinalPage] = []
        total_bubbles = 0

        for page in inputs.pages:
            # 收集每个panel的图像路径与坐标
            panel_paths: list[tuple[str, tuple[float, float, float, float]]] = []
            dialogues: list[tuple[str, str, tuple[float, float]]] = []
            placed_bubbles: list[PlacedBubble] = []

            for panel in page.panels:
                pi = inputs.panel_images.get(panel.panel_id)
                if pi is None:
                    continue
                panel_paths.append(
                    (
                        pi.image_path,
                        (panel.bbox.x, panel.bbox.y, panel.bbox.width, panel.bbox.height),
                    )
                )
                # 在每个panel右上角放置对话气泡
                for di, dialogue in enumerate(panel.dialogues_in_panel[:2]):
                    bx = panel.bbox.x + panel.bbox.width * 0.55
                    by = panel.bbox.y + 0.02 + di * 0.06
                    dialogues.append((dialogue.speaker, dialogue.text, (bx, by)))
                    placed_bubbles.append(
                        PlacedBubble(
                            bubble_id=f"{panel.panel_id}_b{di}",
                            panel_id=panel.panel_id,
                            dialogue_index=di,
                            bubble_type="thought" if dialogue.is_thought else "speech",
                            bbox=BoundingBox(
                                x=bx, y=by, width=0.18, height=0.05,
                            ),
                            text=dialogue.text,
                            font_size_pt=12,
                            occlusion_score=0.05,
                        )
                    )

            page_path = generate_page_placeholder(
                project_id=context.project_id,
                page_id=page.page_id,
                page_number=page.page_number,
                panel_paths=panel_paths,
                dialogues=dialogues,
            )

            final_pages.append(
                FinalPage(
                    page_id=page.page_id,
                    page_number=page.page_number,
                    image_path=page_path,
                    bubbles=placed_bubbles,
                )
            )
            total_bubbles += len(placed_bubbles)

        # 导出
        exports: list[ExportArtifact] = [
            ExportArtifact(
                format="png",
                file_path=p.image_path,
                file_size_bytes=os.path.getsize(p.image_path)
                                  if os.path.exists(p.image_path) else 0,
                page_count=1,
            )
            for p in final_pages
        ]

        return LayoutOutput(
            final_pages=final_pages,
            exports=exports,
            overall_metrics=LayoutMetrics(
                total_pages=len(final_pages),
                total_bubbles=total_bubbles,
                avg_occlusion_score=0.05,
                bubbles_overflow_count=0,
            ),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[f"已合成{len(final_pages)}页"],
            ),
        )

    async def astream(
        self, inputs: LayoutInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(StreamEventType.LOG, "开始排版合成...")
        n = len(inputs.pages)
        for i in range(n):
            await self._sleep_for_demo(0.1)
            yield self._make_event(
                StreamEventType.LOG, f"合成第{i + 1}/{n}页..."
            )
            yield self._make_event(StreamEventType.PROGRESS, (i + 1) / max(n, 1))

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"page_count": len(output.final_pages)},
        )
        yield self._make_event(StreamEventType.DONE, output.model_dump())
