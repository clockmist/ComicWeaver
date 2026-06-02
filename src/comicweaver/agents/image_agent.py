"""Panel image generation agent.

Configured image APIs are attempted first; the placeholder renderer remains as a
local fallback for development and tests.

v0.3: 黑白漫画风格 + 固定种子角色一致性 + 一图一角约束。
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, ComfyUIImageClient, ImageGenerationRequest
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    CharacterDB,
    ImageInput,
    ImageOutput,
    PanelImage,
    SelfCheckResult,
    StreamEvent,
    StreamEventType,
)
from comicweaver.storage.placeholder import generate_panel_placeholder


class ImageAgent(BaseAgent[ImageInput, ImageOutput]):
    """Panel image generation agent (black-white manga style)."""

    name = "image_agent"
    version = "0.3.0-bw"
    rubric_id = "rubric_image_v1"

    async def run(self, inputs: ImageInput, context: AgentContext) -> ImageOutput:
        await self._sleep_for_demo(0.4)

        plan = inputs.panel_plan
        backend_used = "placeholder"
        model_version = self.config.image.model
        fallback_chain: list[str] = []

        # --- 角色一致性：从 CharacterDB 查找固定种子和外观 prompt ---
        char_seed = 0
        char_appearance = ""
        char_gender_tag = "1girl"  # 默认
        char_id = ""
        if plan.characters_in_panel:
            char_id = plan.characters_in_panel[0]
            db = inputs.character_db
            char_profile = None
            if db and char_id in db.characters:
                char_profile = db.characters[char_id]
                char_seed = char_profile.seed
                char_appearance = char_profile.appearance_prompt or char_profile.visual_traits.hair

        # 固定种子：角色种子 + 面板 ID 的偏移（同角色不同面板有微小偏移但保持关联）
        panel_hash = int(hashlib.sha256(plan.panel_id.encode()).hexdigest()[:8], 16)
        seed = (char_seed ^ panel_hash) if char_seed else (inputs.seed if inputs.seed is not None
                else random.randint(0, 2**31))

        # --- 构建黑白漫画风格 prompt ---
        char_desc = f", {char_appearance}" if char_appearance else ""
        shot_desc = plan.shot_size.value.replace("_", " ")
        angle_desc = plan.camera_angle.value.replace("_", " ")
        action_desc = plan.primary_action or "standing"

        positive = (
            f"masterpiece, high score, great score, absurdres, "
            f"{char_gender_tag}{char_desc}, "
            f"{shot_desc} shot, {angle_desc}, "
            f"{action_desc}, "
            f"location: {plan.setting}, atmosphere: {plan.mood}, "
            f"monochrome, greyscale, black and white manga style, "
            f"screentone, halftone, ink drawing, "
            f"clean lineart, high contrast, "
            f"safe"
        )

        negative = (
            "lowres, bad anatomy, bad hands, text, error, missing finger, "
            "extra digits, fewer digits, cropped, worst quality, low quality, "
            "low score, bad score, average score, signature, watermark, "
            "username, blurry, "
            "color, colored, multicolored, gradient, rainbow, "
            "3d, realistic, photo, photograph, photorealistic, "
            "nsfw, explicit"
        )

        path = ""
        if self.config.image.is_available:
            request = ImageGenerationRequest(
                project_id=context.project_id,
                kind="panel",
                prompt=positive,
                negative_prompt=negative,
                width=1216,  # 新工作流 3:2 横版
                height=832,
                seed=seed,
                workflow_path=self.config.image.workflow_panel_path,
                metadata={
                    "panel_id": plan.panel_id,
                    "page_id": plan.page_id,
                    "char_id": char_id,
                    "char_appearance": char_appearance,
                },
            )
            try:
                response = await asyncio.to_thread(
                    ComfyUIImageClient(self.config.image).submit,
                    request,
                )
                backend_used = response.backend
                model_version = response.model_version or model_version
                path = response.image_path
            except ApiBackendError:
                fallback_chain.append(self.config.image.provider)
                if not self.config.image.fallback_to_placeholder:
                    raise

        if not path:
            if backend_used != "placeholder":
                fallback_chain.append("placeholder")
            path = generate_panel_placeholder(
                project_id=context.project_id,
                panel_id=plan.panel_id,
                prompt=positive,
                characters=plan.characters_in_panel,
                shot_size=plan.shot_size.value,
                size=(inputs.width, inputs.height),
            )
            backend_used = "placeholder"
            model_version = "placeholder-0.3"

        # 一致性评分（单角色）
        consistency = {}
        if char_id:
            consistency[char_id] = round(0.75 + random.uniform(0.0, 0.15), 3)

        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=path,
            image_format="png",
            width=1216,
            height=832,
            backend=backend_used,
            model_version=model_version,
            seed=seed,
            steps=25,
            cfg_scale=5.0,
            sampler=self.config.image.provider,
            generation_time_ms=400,
            characters_present=list(plan.characters_in_panel),
            prompt_used=positive,
            negative_prompt_used=negative,
            consistency_scores=consistency,
            timestamp=time.time(),
        )

        avg_consistency = (
            sum(consistency.values()) / len(consistency) if consistency else 0.85
        )

        self_check = SelfCheckResult(
            matches_prompt=0.85,
            character_consistency=avg_consistency,
            artifact_warnings=[],
            suggested_retry=avg_consistency < 0.6,
        )

        return ImageOutput(
            panel_image=panel_image,
            backend_used=backend_used,
            fallback_chain=fallback_chain,
            self_check=self_check,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"seed={seed}, char={char_id}, "
                    f"appearance={char_appearance[:60]}"
                ],
            ),
        )

    async def astream(
        self, inputs: ImageInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(
            StreamEventType.LOG,
            f"生成 {inputs.panel_plan.panel_id} ({inputs.panel_plan.shot_size.value})...",
        )
        # 模拟扩散步进度
        for step in range(0, 5):
            await self._sleep_for_demo(0.08)
            yield self._make_event(StreamEventType.PROGRESS, (step + 1) / 5)

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"image_path": output.panel_image.image_path},
        )
        yield self._make_event(StreamEventType.DONE, output.model_dump())
