"""台词气泡放置 Agent — 独立于排版，负责计算面板中对话气泡的位置和尺寸。

从 LayoutAgent 中拆分出来，职责:
1. 预计算面板 bbox（为气泡放置提供空间约束）
2. YOLO 人脸检测 — 识别面板中的人物面部区域
3. 对每个有对话的面板放置气泡，自动避开人脸
   - 有人脸 → 5×5 网格 + 多维度评分
   - 有人物但无人脸 → 仅四角放置
   - 无人物（纯场景）→ 8 个分散候选位
4. 输出归一化坐标和样式参数，供 LayoutAgent 合成使用
5. 在面板图像上绘制气泡，输出带台词的预览图
6. LLM 驱动的用户反馈解释 + 选择性气泡重放置（v0.4.0）

当前版本 (v0.4.0) 已集成 YOLO 人脸检测 + 5×5 网格评分 + LLM 反馈修正。
"""
from __future__ import annotations

import asyncio
import json
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


# ---------------------------------------------------------------------------
# LLM feedback interpreter — system prompt
# ---------------------------------------------------------------------------

_BUBBLE_FEEDBACK_SYSTEM_PROMPT = """\
You are a comic dialogue bubble placement expert. Your job is to interpret user \
feedback about bubble positions and translate it into precise placement adjustments.

## Background
- This is a black-and-white (manga) comic generation system.
- Each panel has already had bubbles placed on it algorithmically.
- The user is unhappy with some bubble positions and wants adjustments.

## Bubble coordinate system
- All coordinates are NORMALIZED [0.0, 1.0], relative to the FULL PAGE.
- x=0 is left edge, x=1 is right edge. y=0 is top edge, y=1 is bottom edge.
- Each bubble has: (x, y, w, h) defining its bounding box (top-left corner + size).
- tail_direction: where the bubble's tail points ("up", "down", "left", "right",
  "up_left", "up_right", "down_left", "down_right").

## Placement rules
1. Bubbles should NOT cover faces. If face_count > 0, keep bubbles away from
   the panel region where faces were detected.
2. Bubbles should be near panel edges, not in the center (x≈0.3-0.7, y≈0.3-0.7).
3. When a panel has characters but face_count=0, bubbles MUST stay in one of
   the four corners: top_left, top_right, bottom_left, bottom_right — to avoid
   obscuring character bodies.
4. Multiple bubbles in the same panel should not overlap.
5. Narration captions are rectangular boxes, typically placed in bottom corners
   (bottom_left first, then bottom_right).
6. Speech/thought bubbles are elliptical and have tails pointing toward the
   speaker (the panel center area).

## Your task
The user will describe what's wrong with the current bubble placements.
You must:
1. Identify WHICH panels need bubble position changes.
2. For each affected panel, compute new coordinates for the problematic bubbles.
3. Provide a concise reason for each change.

## Output format (JSON only, no markdown, no extra text)
{
  "target_panels": [
    {
      "panel_id": "page_001_p03",
      "reason": "Bubble covered the protagonist's face at top-left",
      "revised_bubbles": [
        {
          "dialogue_index": 0,
          "x": 0.15,
          "y": 0.72,
          "w": 0.30,
          "h": 0.08,
          "tail_direction": "up_right",
          "note": "Moved to bottom-left corner to avoid face region"
        }
      ]
    }
  ]
}

## IMPORTANT
- Only change bubbles the user complained about. Leave others alone.
- Keep bubble dimensions (w, h) the same as original unless the user
  specifically asked to resize.
- Do NOT place bubbles at panel center (x=0.3-0.7, y=0.3-0.7) unless the
  panel has NO characters and the user specifically requests it.
- Output ONLY the JSON object. No markdown fences, no extra text.
"""


class BubbleAgent(BaseAgent[BubbleInput, BubbleOutput]):
    """台词气泡放置 Agent — 计算气泡坐标和样式参数，并生成面板预览图。"""

    name = "bubble_agent"
    version = "0.4.0"

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        is_revision = bool(inputs.user_guidance and inputs.previous_output)
        if is_revision:
            return await self._run_revision(inputs, context)
        else:
            return await self._run_first_pass(inputs, context)

    # ------------------------------------------------------------------
    # First pass — original algorithmic pipeline (改进点 1 + 2 已集成)
    # ------------------------------------------------------------------

    async def _run_first_pass(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        """首次运行：YOLO 检测 → 气泡放置 → 预览渲染。"""
        all_placements: dict[str, list[BubblePlacementResult]] = {}
        total_bubbles = 0

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

        # Step 2: YOLO 人脸检测（顺序处理所有有对话的面板）
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

        # Step 4: 在面板图像上实际绘制气泡 + 人脸检测框，生成预览图
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

                bbox = panel_bbox_map.get(pid)
                if bbox is None:
                    continue

                # 收集该面板的气泡
                panel_bubbles_list = [
                    bp for pg_bubbles in page_bubbles.values()
                    for bp in pg_bubbles
                    if bp.panel_id == pid
                ]
                if not panel_bubbles_list:
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
                    logger.warning("面板 %s 渲染失败: %s", pid, exc)

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
                ] + ([f"user_guidance: {inputs.user_guidance[:100]}"] if inputs.user_guidance else []),
            ),
        )

    # ------------------------------------------------------------------
    # Revision mode — LLM 驱动的选择性气泡重放置 (改进点 3)
    # ------------------------------------------------------------------

    async def _run_revision(
        self, inputs: BubbleInput, context: AgentContext
    ) -> BubbleOutput:
        """修订模式：LLM 解释用户反馈 + 选择性重新放置受影响面板的气泡。

        流程：
        1. 从 previous_output 还原上一轮放置结果
        2. 构建反馈上下文 → 调用 LLM 解释
        3. 对 target_panels 应用修订坐标；其余面板复用
        4. 重新渲染受影响面板的预览图
        5. LLM 失败时回退至全量重放置
        """
        prev_placements: dict[str, list[dict]] = inputs.previous_output or {}

        # Step 1: 预计算 bbox（供渲染用）
        panel_bbox_map: dict[str, object] = {}
        for page in inputs.pages:
            bboxes = solve_page_bboxes(
                page,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                gutter_px=inputs.gutter_px,
            )
            for panel, bbox in zip(page.panels, bboxes):
                panel_bbox_map[panel.panel_id] = bbox

        # Step 2: 从上一轮结果推断人脸统计
        face_stats = self._derive_face_stats(prev_placements)

        # Step 3: 调用 LLM 解释反馈
        llm_available = (
            self.config.llm.is_available
            if hasattr(self.config, "llm")
            else False
        )
        if not llm_available:
            logger.warning(
                "LLM 不可用，气泡反馈解释跳过。回退至全量重放置。"
            )
            return await self._run_first_pass(inputs, context)

        try:
            fb_result = await self.interpret_bubble_feedback(
                inputs.user_guidance,
                prev_placements,
                inputs.pages,
                face_stats,
            )
        except Exception as exc:
            logger.warning(
                "LLM 气泡反馈解释失败 (%s)，回退至全量重放置。", exc,
            )
            return await self._run_first_pass(inputs, context)

        # Step 4: 解析 LLM 输出
        target_panels_list = fb_result.get("target_panels", [])
        if not isinstance(target_panels_list, list) or not target_panels_list:
            logger.info("LLM 未识别到需要调整的面板，回退至全量重放置。")
            return await self._run_first_pass(inputs, context)

        target_panel_ids = {tp.get("panel_id", "") for tp in target_panels_list}
        target_panel_ids.discard("")
        logger.info(
            "LLM 反馈解释完成: %d 个面板需要调整 (%s)",
            len(target_panel_ids), ", ".join(sorted(target_panel_ids)),
        )

        # Step 5: 构建修订坐标查找表
        revised_map: dict[str, dict[int, dict]] = {}
        for tp in target_panels_list:
            pid = tp.get("panel_id", "")
            if not pid:
                continue
            revised_map[pid] = {}
            for rb in tp.get("revised_bubbles", []):
                di = rb.get("dialogue_index", 0)
                revised_map[pid][di] = rb

        # Step 6: 按页重建气泡放置
        all_placements: dict[str, list[BubblePlacementResult]] = {}
        total_bubbles = 0
        page_bubbles_render: dict[str, list[BubblePlacement]] = {}

        for page in inputs.pages:
            page_id = page.page_id
            bboxes = solve_page_bboxes(
                page,
                page_width_px=inputs.page_width_px,
                page_height_px=inputs.page_height_px,
                margin_px=inputs.margin_px,
                gutter_px=inputs.gutter_px,
            )

            page_results: list[BubblePlacementResult] = []
            page_bp_list: list[BubblePlacement] = []

            for panel, bbox in zip(page.panels, bboxes):
                pid = panel.panel_id
                prev_panel_bubbles = [
                    d for d in prev_placements.get(page_id, [])
                    if d.get("panel_id") == pid
                ]

                if pid in target_panel_ids:
                    # 受影响面板：应用 LLM 修订坐标
                    revised = revised_map.get(pid, {})
                    for bdict in prev_panel_bubbles:
                        di = bdict.get("dialogue_index", 0)
                        rev = revised.get(di)
                        if rev is not None:
                            # 使用 LLM 修订的坐标，其他字段保留原值
                            new_x = float(rev.get("x", bdict.get("x", 0)))
                            new_y = float(rev.get("y", bdict.get("y", 0)))
                            new_w = float(rev.get("w", bdict.get("w", 0.1)))
                            new_h = float(rev.get("h", bdict.get("h", 0.05)))
                            new_tail = str(rev.get("tail_direction", bdict.get("tail_direction", "auto")))
                            # 钳制在面板内
                            new_x = max(bbox.x + 0.01, min(new_x, bbox.x + bbox.width - new_w - 0.01))
                            new_y = max(bbox.y + 0.01, min(new_y, bbox.y + bbox.height - new_h - 0.01))

                            result = BubblePlacementResult(
                                panel_id=pid,
                                dialogue_index=di,
                                bubble_type=bdict.get("bubble_type", "speech"),
                                speaker=bdict.get("speaker", ""),
                                text=bdict.get("text", ""),
                                x=new_x,
                                y=new_y,
                                w=new_w,
                                h=new_h,
                                font_size_pt=bdict.get("font_size_pt", 14),
                                tail_direction=new_tail,
                                occlusion_score=0.0,
                                face_count=bdict.get("face_count", 0),
                                style=bdict.get("style", {}),
                            )
                        else:
                            # LLM 未修订此对话 → 复用原值
                            result = self._dict_to_placement_result(bdict)
                        page_results.append(result)
                        page_bp_list.append(BubblePlacement(
                            panel_id=result.panel_id,
                            dialogue_index=result.dialogue_index,
                            bubble_type=result.bubble_type,
                            speaker=result.speaker,
                            text=result.text,
                            x=result.x,
                            y=result.y,
                            w=result.w,
                            h=result.h,
                            font_size_pt=result.font_size_pt,
                            tail_direction=result.tail_direction,
                            occlusion_score=result.occlusion_score,
                            style=result.style,
                        ))
                        total_bubbles += 1
                else:
                    # 未受影响面板：复用上一轮结果
                    for bdict in prev_panel_bubbles:
                        result = self._dict_to_placement_result(bdict)
                        page_results.append(result)
                        page_bp_list.append(BubblePlacement(
                            panel_id=result.panel_id,
                            dialogue_index=result.dialogue_index,
                            bubble_type=result.bubble_type,
                            speaker=result.speaker,
                            text=result.text,
                            x=result.x,
                            y=result.y,
                            w=result.w,
                            h=result.h,
                            font_size_pt=result.font_size_pt,
                            tail_direction=result.tail_direction,
                            occlusion_score=result.occlusion_score,
                            style=result.style,
                        ))
                        total_bubbles += 1

            all_placements[page_id] = page_results
            page_bubbles_render[page_id] = page_bp_list

        # Step 7: 仅重新渲染受影响面板的预览图
        bubbled_images = await self._rerender_affected_panels(
            inputs, target_panel_ids, all_placements, panel_bbox_map,
        )

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
                    f"修订模式: LLM 识别了 {len(target_panel_ids)} 个受影响面板",
                    f"已重放置 {total_bubbles} 个气泡",
                    f"user_guidance: {inputs.user_guidance[:100]}",
                ],
            ),
        )

    # ------------------------------------------------------------------
    # LLM feedback interpreter
    # ------------------------------------------------------------------

    async def interpret_bubble_feedback(
        self,
        user_guidance: str,
        previous_placements: dict[str, list[dict]],
        pages: list[PageLayout],
        face_stats: dict[str, int],
    ) -> dict:
        """使用 LLM 解释用户对气泡放置的反馈，返回结构化修订指令。

        Parameters
        ----------
        user_guidance:
            用户自然语言反馈。
        previous_placements:
            上一轮气泡放置结果 {page_id: [serialized BubblePlacementResult]}.
        pages:
            当前所有页面布局。
        face_stats:
            每面板检测到的人脸数 {panel_id: int}.

        Returns
        -------
        dict
            {"target_panels": [{"panel_id": str, "reason": str,
              "revised_bubbles": [{"dialogue_index": int, "x": float, ...}]}]}
        """
        from comicweaver.api import OpenAICompatibleLLMClient

        system_prompt = _BUBBLE_FEEDBACK_SYSTEM_PROMPT
        user_message = self._build_feedback_user_message(
            user_guidance, previous_placements, pages, face_stats,
        )

        client = OpenAICompatibleLLMClient(self.config.llm)
        result = await asyncio.to_thread(
            client.complete_json,
            system_prompt,
            {"user_message": user_message},
        )
        if not isinstance(result, dict):
            logger.warning("LLM 气泡反馈解释返回非 dict 类型: %s", type(result))
            return {"target_panels": []}
        return result

    def _build_feedback_user_message(
        self,
        user_guidance: str,
        previous_placements: dict[str, list[dict]],
        pages: list[PageLayout],
        face_stats: dict[str, int],
    ) -> str:
        """构造 LLM 反馈解释器的 user message。

        包含：
        - 用户原始反馈
        - 每个面板的元数据（角色、景别、人脸数）
        - 每个面板当前的气泡放置详情（坐标、类型、文本）
        """
        panels_info: list[dict] = []
        for page in pages:
            for panel in page.panels:
                pid = panel.panel_id
                page_placements = previous_placements.get(page.page_id, [])
                panel_bubbles = [
                    b for b in page_placements
                    if b.get("panel_id") == pid
                ]

                shot_val = (
                    panel.shot_size.value
                    if hasattr(panel.shot_size, "value")
                    else str(panel.shot_size)
                )
                char_list = getattr(panel, "characters_in_panel", []) or []

                info = {
                    "panel_id": pid,
                    "shot_size": shot_val,
                    "characters": char_list,
                    "faces_detected": face_stats.get(pid, 0),
                    "emotion_intensity": getattr(panel, "emotion_intensity", 0.5) or 0.5,
                    "current_bubbles": [
                        {
                            "dialogue_index": b.get("dialogue_index", 0),
                            "type": b.get("bubble_type", "speech"),
                            "speaker": b.get("speaker", ""),
                            "text": (b.get("text", "") or "")[:80],
                            "x": round(b.get("x", 0), 3),
                            "y": round(b.get("y", 0), 3),
                            "w": round(b.get("w", 0), 3),
                            "h": round(b.get("h", 0), 3),
                            "tail": b.get("tail_direction", "auto"),
                            "font_size_pt": b.get("font_size_pt", 14),
                        }
                        for b in panel_bubbles
                    ],
                }
                panels_info.append(info)

        return (
            f"## User Feedback\n{user_guidance}\n\n"
            f"## Current Bubble Placements (for reference)\n"
            f"```json\n{json.dumps(panels_info, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Please identify which panels need repositioning and provide revised coordinates."
        )

    # ------------------------------------------------------------------
    # Revision helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_face_stats(placements: dict[str, list[dict]]) -> dict[str, int]:
        """从上一轮放置结果推断每面板的人脸检测数。"""
        stats: dict[str, int] = {}
        for page_bubbles in placements.values():
            for bdict in page_bubbles:
                pid = bdict.get("panel_id", "")
                if not pid:
                    continue
                fc = bdict.get("face_count", 0)
                if pid not in stats or fc > stats[pid]:
                    stats[pid] = fc
        return stats

    @staticmethod
    def _dict_to_placement_result(bdict: dict) -> BubblePlacementResult:
        """将序列化的 dict 还原为 BubblePlacementResult。"""
        return BubblePlacementResult(
            panel_id=bdict.get("panel_id", ""),
            dialogue_index=bdict.get("dialogue_index", 0),
            bubble_type=bdict.get("bubble_type", "speech"),
            speaker=bdict.get("speaker", ""),
            text=bdict.get("text", ""),
            x=float(bdict.get("x", 0)),
            y=float(bdict.get("y", 0)),
            w=float(bdict.get("w", 0.1)),
            h=float(bdict.get("h", 0.05)),
            font_size_pt=bdict.get("font_size_pt", 14),
            tail_direction=bdict.get("tail_direction", "auto"),
            occlusion_score=float(bdict.get("occlusion_score", 0)),
            face_count=bdict.get("face_count", 0),
            style=bdict.get("style", {}),
        )

    async def _rerender_affected_panels(
        self,
        inputs: BubbleInput,
        target_panel_ids: set[str],
        all_placements: dict[str, list[BubblePlacementResult]],
        panel_bbox_map: dict[str, object],
    ) -> dict[str, str]:
        """重新渲染受影响面板的气泡预览图。

        对 target_panel_ids 中的面板重新渲染；
        其他面板从磁盘读取已有的 _bubbled.png 路径。
        """
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

                bbox = panel_bbox_map.get(pid)
                if bbox is None:
                    continue

                # 收集该面板的气泡
                panel_bubbles_list: list[BubblePlacement] = []
                for page_bubbles in all_placements.values():
                    for bp in page_bubbles:
                        if bp.panel_id == pid:
                            panel_bubbles_list.append(BubblePlacement(
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
                                style=bp.style,
                            ))

                if pid in target_panel_ids and panel_bubbles_list:
                    # 受影响面板：重新渲染
                    try:
                        img = Image.open(img_path)
                        bubbled = render_bubbles_on_panel_image(
                            img, bbox, panel_bubbles_list,
                        )
                        out_path = Path(img_path).parent / f"{pid}_bubbled.png"
                        bubbled.save(str(out_path))
                        bubbled_images[pid] = str(out_path)
                    except Exception as exc:
                        logger.warning("面板 %s 修订后气泡渲染失败: %s", pid, exc)
                        bubbled_images[pid] = str(
                            Path(img_path).parent / f"{pid}_bubbled.png"
                        )
                else:
                    # 未受影响面板：指向已有文件
                    existing = Path(img_path).parent / f"{pid}_bubbled.png"
                    if existing.exists():
                        bubbled_images[pid] = str(existing)

        return bubbled_images

    # ------------------------------------------------------------------
    # YOLO face detection
    # ------------------------------------------------------------------

    async def _detect_faces(
        self, inputs: BubbleInput
    ) -> dict[str, list[FaceRegion]]:
        """YOLO 人脸检测 — 顺序处理所有有对话的面板。

        只检测有对话的面板（无对话的面板不需要气泡放置）。

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
                "%d/%d 个面板的人脸检测失败，这些面板将使用纯启发式气泡放置",
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

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def astream(
        self, inputs: BubbleInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """流式输出气泡放置进度。"""
        is_revision = bool(inputs.user_guidance and inputs.previous_output)

        if is_revision:
            yield self._make_event(StreamEventType.LOG, "收到用户反馈，开始 LLM 解释与气泡修正...")
            yield self._make_event(StreamEventType.PROGRESS, 0.1)
            yield self._make_event(
                StreamEventType.THINKING,
                f"分析反馈: {inputs.user_guidance[:100]}...",
            )
            yield self._make_event(StreamEventType.PROGRESS, 0.3)
        else:
            yield self._make_event(StreamEventType.LOG, "开始台词气泡放置...")
            yield self._make_event(StreamEventType.PROGRESS, 0.05)

            # 统计面板和对话数
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
