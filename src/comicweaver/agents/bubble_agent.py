"""台词气泡放置 Agent — 独立于排版，负责计算面板中对话气泡的位置和尺寸。

从 LayoutAgent 中拆分出来，职责:
1. 预计算面板 bbox（为气泡放置提供空间约束）
2. YOLO 人脸检测 — 识别面板中的人物面部区域
3. 对每个有对话的面板放置气泡，自动避开人脸
4. 输出归一化坐标和样式参数，供 LayoutAgent 合成使用
5. ★v0.3.0: 直接在面板图像上绘制气泡，输出带台词的预览图

当前版本 (v0.3.0) 已集成 YOLO 人脸检测 + 面板级气泡渲染。
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from PIL import Image

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

from .layout.bubbles import BubblePlacement, place_bubbles
from .layout.compositor import render_bubbles_on_panel_image
from .layout.face_detector import FaceRegion
from .layout.face_detector_yolo import detect_faces_yolo
from .layout_agent import solve_page_bboxes

logger = logging.getLogger(__name__)


class BubbleAgent(BaseAgent[BubbleInput, BubbleOutput]):
    """台词气泡放置 Agent — 计算气泡坐标和样式参数，并生成面板预览图。"""

    name = "bubble_agent"
    version = "0.3.0"
    rubric_id = "rubric_bubble_v1"

    async def run(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        all_placements: dict[str, list[BubblePlacementResult]] = {}
        total_bubbles = 0
        face_stats: dict[str, int] = {}

        # Step 1: 预计算所有页面的 bbox + 收集面板bbox映射
        page_bboxes_map: dict[str, list] = {}
        panel_bbox_map: dict[str, object] = {}  # panel_id → BoundingBox
        for page in inputs.pages:
            bboxes = solve_page_bboxes(
                page,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                gutter_px=inputs.gutter_px,
            )
            page_bboxes_map[page.page_id] = bboxes
            for panel, bbox in zip(page.panels, bboxes):
                panel_bbox_map[panel.panel_id] = bbox

        # Step 2: YOLO 人脸检测（并行处理所有有对话的面板）
        faces_by_panel = await self._detect_faces(inputs)
        total_faces = sum(len(f) for f in faces_by_panel.values())
        logger.info(
            "YOLO 人脸检测完成: %d 个面板中检测到 %d 张人脸",
            len(faces_by_panel), total_faces,
        )

        # Step 3: 按页放置气泡
        page_bubbles: dict[str, list[BubblePlacement]] = {}  # page_id → bubbles
        for page in inputs.pages:
            page_id = page.page_id
            bboxes = page_bboxes_map[page_id]

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

            page_bubbles[page_id] = bubbles

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

        # Step 4: ★ 在面板图像上实际绘制气泡，生成预览图
        bubbled_images: dict[str, str] = {}
        for page in inputs.pages:
            for panel in page.panels:
                pid = panel.panel_id
                panel_img_obj = inputs.panel_images.get(pid)
                if not panel_img_obj or not panel_img_obj.image_path:
                    continue

                img_path = panel_img_obj.image_path
                if not os.path.exists(img_path):
                    continue

                # 收集该面板的气泡
                panel_bubbles_list = [
                    bp for pg_bubbles in page_bubbles.values()
                    for bp in pg_bubbles
                    if bp.panel_id == pid
                ]
                if not panel_bubbles_list:
                    continue

                bbox = panel_bbox_map.get(pid)
                if bbox is None:
                    continue

                try:
                    img = Image.open(img_path)
                    bubbled = render_bubbles_on_panel_image(
                        img, bbox, panel_bubbles_list,
                    )

                    # 保存到同目录，加 _bubbled 后缀
                    out_path = Path(img_path).parent / f"{pid}_bubbled.png"
                    bubbled.save(str(out_path))
                    bubbled_images[pid] = str(out_path)
                    logger.debug("面板 %s 气泡预览图已保存: %s", pid, out_path)
                except Exception as exc:
                    logger.warning("面板 %s 气泡渲染失败: %s", pid, exc)

        # 填充 face_detection_stats
        face_stats = {
            panel_id: len(faces)
            for panel_id, faces in faces_by_panel.items()
        }

        return BubbleOutput(
            bubble_placements=all_placements,
            total_bubbles=total_bubbles,
            face_detection_stats=face_stats,
            bubbled_panel_images=bubbled_images,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[
                    f"已放置{total_bubbles}个气泡",
                    f"YOLO 检测到 {total_faces} 张人脸 (分布在 {len(faces_by_panel)} 个面板)",
                    f"页尺寸: {inputs.page_width_px}x{inputs.page_height_px}",
                    f"已生成 {len(bubbled_images)} 张气泡预览图",
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

        # 顺序 YOLO 检测 — YOLO 模型不支持多线程并发，必须逐个处理
        yolo_cfg = getattr(self.config, "yolo", None)
        model_path = yolo_cfg.model_path if yolo_cfg else "yolo/face_yolov8n.pt"
        # 漫画/动漫人脸较难检测，默认使用较低置信度阈值
        confidence = yolo_cfg.confidence_threshold if yolo_cfg else 0.3
        logger.info(
            "开始 YOLO 人脸检测: %d 个面板 (model=%s, confidence=%.2f, 顺序处理)",
            len(tasks), model_path, confidence,
        )

        # 逐个面板检测（线程安全）
        faces_by_panel: dict[str, list[FaceRegion]] = {}
        detected_count = 0
        failed_count = 0
        for panel_id, path in tasks.items():
            try:
                result = await detect_faces_yolo(
                    path, model_path=model_path, confidence_threshold=confidence,
                )
                faces_by_panel[panel_id] = result
                if result:
                    detected_count += 1
                    logger.info(
                        "面板 %s: 检测到 %d 张人脸 (置信度: %s)",
                        panel_id, len(result),
                        ", ".join(f"{f.x:.2f},{f.y:.2f}" for f in result[:3]),
                    )
            except Exception as exc:
                logger.warning("面板 %s YOLO 检测异常: %s", panel_id, exc)
                faces_by_panel[panel_id] = []
                failed_count += 1

        if failed_count > 0:
            logger.warning(
                "⚠️ %d/%d 个面板的人脸检测失败，这些面板将使用纯启发式气泡放置",
                failed_count, len(tasks),
            )
        if detected_count == 0 and failed_count == 0:
            logger.info(
                "未检测到任何人脸。可能原因: 1) 漫画风格与训练数据不匹配 "
                "2) 面板中人脸过小 3) 置信度阈值 %.2f 偏高。"
                "气泡将使用纯启发式放置。",
                confidence,
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
            {
                "total_bubbles": output.total_bubbles,
                "bubbled_images": len(output.bubbled_panel_images),
            },
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())
