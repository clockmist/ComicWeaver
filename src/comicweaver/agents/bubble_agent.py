"""台词气泡放置 Agent — 独立于排版，负责计算面板中对话气泡的位置和尺寸。

从 LayoutAgent 中拆分出来，职责:
1. 预计算面板 bbox（为气泡放置提供空间约束）
2. YOLO 人脸检测 — 识别面板中的人物面部区域
3. 对每个有对话的面板放置气泡，自动避开人脸
4. 输出归一化坐标和样式参数，供 LayoutAgent 合成使用

当前版本 (v0.2.0) 已集成 YOLO 人脸检测，
后续版本将加入自适应气泡尺寸和多样化样式。
"""
from __future__ import annotations

import asyncio
import logging
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
from .layout.face_detector import FaceRegion
from .layout.face_detector_yolo import detect_faces_yolo
from .layout_agent import solve_page_bboxes

logger = logging.getLogger(__name__)


class BubbleAgent(BaseAgent[BubbleInput, BubbleOutput]):
    """台词气泡放置 Agent — 计算气泡坐标和样式参数。"""

    name = "bubble_agent"
    version = "0.2.0"
    rubric_id = "rubric_bubble_v1"

    async def run(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        all_placements: dict[str, list[BubblePlacementResult]] = {}
        total_bubbles = 0
        face_stats: dict[str, int] = {}

        # Step 1: 预计算所有页面的 bbox + 收集面板bbox映射
        page_bboxes_map: dict[str, list] = {}
        for page in inputs.pages:
            bboxes = solve_page_bboxes(
                page,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                gutter_px=inputs.gutter_px,
            )
            page_bboxes_map[page.page_id] = bboxes

        # Step 2: YOLO 人脸检测（并行处理所有有对话的面板）
        faces_by_panel = await self._detect_faces(inputs)
        total_faces = sum(len(f) for f in faces_by_panel.values())
        logger.info(
            "YOLO 人脸检测完成: %d 个面板中检测到 %d 张人脸",
            len(faces_by_panel), total_faces,
        )

        # Step 3: 按页放置气泡
        for page in inputs.pages:
            page_id = page.page_id
            bboxes = page_bboxes_map[page_id]

            # 放置气泡（传入 YOLO 人脸检测结果以优化位置）
            # font_size_pt=0 启用自适应尺寸（由 calculate_bubble_size 根据面板大小计算）
            bubbles = place_bubbles(
                page.panels,
                bboxes,
                inputs.panel_images,
                font_size_pt=0,  # 自适应尺寸
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                faces_by_panel=faces_by_panel,
            )

            # 转为 BubblePlacementResult（供输出和 LayoutAgent 消费）
            results: list[BubblePlacementResult] = []
            for bp in bubbles:
                panel_faces = faces_by_panel.get(bp.panel_id, [])
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
                    font_size_pt=bp.font_size_pt,
                    tail_direction=bp.tail_direction,
                    occlusion_score=bp.occlusion_score,
                    face_count=len(panel_faces),
                    style=bp.style,
                ))
                total_bubbles += 1

            all_placements[page_id] = results

        # 填充 face_detection_stats（panel_id → 人脸数）
        face_stats = {
            panel_id: len(faces)
            for panel_id, faces in faces_by_panel.items()
        }

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
                    f"YOLO 检测到 {total_faces} 张人脸 (分布在 {len(faces_by_panel)} 个面板)",
                    f"页尺寸: {inputs.page_width_px}x{inputs.page_height_px}",
                ],
            ),
        )

    async def _detect_faces(
        self, inputs: BubbleInput
    ) -> dict[str, list[FaceRegion]]:
        """YOLO 人脸检测 — 并行处理所有有对话的面板。

        只检测有对话的面板（无对话的面板不需要气泡放置），
        使用 asyncio.gather 并行检测以最大化吞吐量。

        返回
        -------
        dict[str, list[FaceRegion]]
            panel_id → 检测到的人脸列表（无人脸则为空列表）。
        """
        # 收集所有需要检测的面板及其图像路径
        tasks: dict[str, str] = {}  # panel_id → image_path
        for page in inputs.pages:
            for panel in page.panels:
                if not panel.dialogues_in_panel:
                    continue
                panel_img = inputs.panel_images.get(panel.panel_id)
                if panel_img and panel_img.image_path:
                    tasks[panel.panel_id] = panel_img.image_path

        if not tasks:
            return {}

        # 并行 YOLO 检测（从 config 读取模型路径和置信度）
        yolo_cfg = getattr(self.config, "yolo", None)
        model_path = yolo_cfg.model_path if yolo_cfg else "yolo/face_yolov8n.pt"
        confidence = yolo_cfg.confidence_threshold if yolo_cfg else 0.3
        logger.info("开始 YOLO 人脸检测: %d 个面板 (model=%s)", len(tasks), model_path)
        results = await asyncio.gather(
            *[
                detect_faces_yolo(path, model_path=model_path, confidence_threshold=confidence)
                for path in tasks.values()
            ],
            return_exceptions=True,
        )

        # 组装结果
        faces_by_panel: dict[str, list[FaceRegion]] = {}
        for (panel_id, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.warning("面板 %s YOLO 检测异常: %s", panel_id, result)
                faces_by_panel[panel_id] = []
            else:
                faces_by_panel[panel_id] = result
                if result:
                    logger.info(
                        "面板 %s: 检测到 %d 张人脸", panel_id, len(result)
                    )

        return faces_by_panel

    async def astream(
        self, inputs: BubbleInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """流式输出气泡放置进度。"""
        yield self._make_event(StreamEventType.LOG, "开始台词气泡放置...")
        yield self._make_event(StreamEventType.PROGRESS, 0.05)

        # 统计面板和对话数
        n = len(inputs.pages)
        dialogue_panels = sum(
            1 for page in inputs.pages
            for panel in page.panels
            if panel.dialogues_in_panel
        )
        yield self._make_event(
            StreamEventType.THINKING,
            f"YOLO 人脸检测: {dialogue_panels} 个有对话的面板...",
        )
        yield self._make_event(StreamEventType.PROGRESS, 0.15)

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
                StreamEventType.PROGRESS, 0.15 + (i + 1) / max(n, 1) * 0.65
            )

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"total_bubbles": output.total_bubbles},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())
