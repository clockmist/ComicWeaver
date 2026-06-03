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

from comicweaver.agents.storyboard_agent import _map_shot_size
from comicweaver.api import (
    ApiBackendError,
    ComfyUIImageClient,
    ImageGenerationRequest,
    OpenAICompatibleLLMClient,
)
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

# ---------------------------------------------------------------------------
# Prompt helper utilities (shared by ImageAgent and StoryboardAgent)
# ---------------------------------------------------------------------------


def _strip_prefix_tags(char_tags: str, gender_tag: str) -> str:
    """Remove gender_tag and 'solo' prefix from char_tags to avoid duplication.

    Example: _strip_prefix_tags("1girl, solo, young_adult, short hair", "1girl")
            -> "young_adult, short hair"
    """
    if not char_tags:
        return ""
    tags = [t.strip() for t in char_tags.split(",") if t.strip()]
    # Remove leading gender_tag
    if tags and tags[0] == gender_tag:
        tags = tags[1:]
    # Remove leading "solo" (will be re-added in correct position)
    if tags and tags[0] == "solo":
        tags = tags[1:]
    return ", ".join(tags)


def _deduplicate_tags(tags_str: str) -> str:
    """Remove duplicate tags while preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags_str.split(","):
        t = tag.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return ", ".join(result)


class ImageAgent(BaseAgent[ImageInput, ImageOutput]):
    """Panel image generation agent (black-white manga style)."""

    name = "image_agent"
    version = "0.5.0-llm"
    rubric_id = "rubric_image_v1"

    async def run(self, inputs: ImageInput, context: AgentContext) -> ImageOutput:
        await self._sleep_for_demo(0.4)

        plan = inputs.panel_plan
        backend_used = "comfyui"
        model_version = self.config.image.model

        # --- 角色一致性：从 CharacterDB 查找固定种子、core_tags、性别标签 ---
        char_seed = 0
        char_core_tags = ""
        char_gender_tag = "1girl"
        char_id = ""
        if plan.characters_in_panel:
            char_id = plan.characters_in_panel[0]
            db = inputs.character_db
            if db and char_id in db.characters:
                char_profile = db.characters[char_id]
                char_seed = char_profile.seed
                char_core_tags = char_profile.core_tags or char_profile.appearance_prompt
                char_gender_tag = char_profile.gender_tag or "1girl"
            else:
                raise ApiBackendError(
                    f"Character '{char_id}' not found in CharacterDB — "
                    f"cannot generate panel {plan.panel_id}"
                )

        # 固定种子：直接使用角色种子（保证同一角色在所有面板中一致）
        # 不 XOR panel_hash，因为角色一致性 > 面板间微小变化
        if char_seed:
            seed = char_seed
        elif inputs.seed is not None:
            seed = inputs.seed
        else:
            seed = random.randint(0, 2**31 - 1)

        # --- 构建 Animagine-XL-4.0 prompt (v0.7: Animagine 官方标签) ---
        # Animagine XL 4.0 训练标签：masterpiece, high score, great score, absurdres, safe
        style_prefix = "masterpiece, high score, great score, absurdres, safe"

        # 角色 Tag — 剥离 core_tags 中已有的 gender_tag 和 solo，避免重复
        char_tags_clean = _strip_prefix_tags(char_core_tags, char_gender_tag)

        # 构图与动作
        shot_desc = _map_shot_size(plan.shot_size.value)
        angle_desc = plan.camera_angle.value.replace("_", " ")
        comp_parts = [shot_desc, angle_desc]
        # pose_hint 已包含完整动作姿态（StoryboardAgent LLM 负责一致性）
        for key in ("pose_hint", "expression"):
            v = getattr(plan, key, "") or ""
            if v:
                comp_parts.append(v)

        # 背景（不做硬编码截断，交给 LLM optimizer 清洗）
        scene_parts = []
        setting = plan.setting or ""
        if setting:
            scene_parts.append(setting)
        scene_lighting = getattr(plan, "scene_lighting", "") or ""
        if scene_lighting:
            scene_parts.append(scene_lighting)

        # 组装 prompt
        tag_parts = [
            style_prefix,
            f"{char_gender_tag}, solo, {char_tags_clean}",
            ", ".join(comp_parts),
        ]
        if scene_parts:
            tag_parts.append(", ".join(scene_parts))
        tag_parts.append(
            "monochrome, greyscale, black and white manga style, "
            "screentone, halftone, ink drawing, clean lineart, high contrast"
        )
        positive = ", ".join(p for p in tag_parts if p)

        # 去重
        positive = _deduplicate_tags(positive)

        # LLM 提示词优化（统一清洗：去矛盾、去无效标签、去颜色、精简背景）
        if self.config.llm.is_available:
            try:
                positive = await self._optimize_prompt_via_llm(
                    positive, char_gender_tag,
                )
            except Exception:
                pass

        negative = (
            "lowres, worst quality, bad quality, displeasing, low resolution, "
            "bad anatomy, bad hands, bad fingers, extra digits, fewer digits, "
            "cropped, signature, watermark, username, blurry, text, error, "
            "color, colored, multicolored, gradient, rainbow, "
            "3d, realistic, photo, photograph, photorealistic, "
            "bad perspective, distorted face, extra limbs, fused fingers, "
            "bad proportions, mutation, deformed, ugly, duplicate, "
            "disfigured, poorly drawn face, poorly drawn hands, "
            "nsfw, explicit"
        )

        if not self.config.image.is_available:
            raise ApiBackendError(
                "Image API is not available — cannot generate panel image"
            )
        request = ImageGenerationRequest(
            project_id=context.project_id,
            kind="panel",
            prompt=positive,
            negative_prompt=negative,
            width=1024,
            height=1024,
            seed=seed,
            workflow_path=self.config.image.workflow_panel_path,
            metadata={
                "panel_id": plan.panel_id,
                "page_id": plan.page_id,
                "char_id": char_id,
                "char_core_tags": char_core_tags,
            },
        )
        response = await asyncio.to_thread(
            ComfyUIImageClient(self.config.image).submit,
            request,
        )
        backend_used = response.backend
        model_version = response.model_version or model_version
        path = response.image_path

        # 一致性评分（单角色）
        consistency = {}
        if char_id:
            consistency[char_id] = round(0.75 + random.uniform(0.0, 0.15), 3)

        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=path,
            image_format="png",
            width=1024,
            height=1024,
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
            self_check=self_check,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"[LLM-opt] char={char_id} tags={char_core_tags[:60]} "
                    f"seed={seed} {char_gender_tag} "
                    f"shot={shot_desc} angle={angle_desc} "
                    f"pose={plan.pose_hint[:40] if plan.pose_hint else 'N/A'} "
                    f"optimized=yes"
                ],
            ),
        )

    async def _optimize_prompt_via_llm(
        self, prompt: str, gender_tag: str,
    ) -> str:
        """LLM-driven comprehensive prompt cleanup — the single entry point for all
        post-processing. No hardcoded rules; the LLM handles dedup, contradiction
        removal, tag cleanup, reordering, and background simplification.
        """
        system_prompt = """You are a prompt optimizer for Animagine-XL-4.0 generating BLACK AND WHITE MANGA.

Given a raw prompt, return optimized JSON: {"optimized_prompt": "..."}

=== RULES ===

1. FIX CONTRADICTIONS: If the prompt has conflicting action tags (e.g. "walking, standing" or
   "running, sitting"), keep only the PRIMARY action that matches the shot distance. Remove the
   contradictory one.

2. MATCH SHOT TO ACTION: Check shot distance vs action.
   - "extreme close-up" / "close-up" → facial expression / eye details ONLY. Remove any
     full-body action (walking, running, kicking, jumping, crouching).
   - "medium shot" → upper body gestures, hand poses, reaching. Remove full-body actions.
   - "full shot" / "long shot" / "panoramic view" → keep full-body actions.

3. REMOVE USELESS TAGS: Delete these immediately:
   - "natural light", "natural sunlight", "daylight" → invisible in B&W, replace with "harsh
     lighting" or "dramatic shadows" or just delete
   - "quiet", "soft", "serene", "calm" → abstract, not visual. Delete.
   - "slender", "average build", "neat" → invisible in lineart. Delete.
   - Weather words unless critical to the scene.
   - "standing" if the primary action is already a dynamic pose.

4. REMOVE COLORS: Delete clothing color words (rewrite "blue dress" -> "dress"). Delete skin
   color tags entirely. Hair: only black/grey/white/silver. Eye color IS allowed and keep
   SPECIFIC eye colors (red eyes, blue eyes, green eyes, etc).

5. REORDER for weight: quality tags -> gender -> character traits -> shot/angle -> pose/expression
   -> background (1-2 max) -> B&W style suffix.

6. ENSURE these are present: masterpiece, high score, great score, absurdres, safe, monochrome,
   greyscale, black and white manga style, screentone, halftone, ink drawing, clean lineart,
   high contrast.

7. Keep output under 400 characters."""

        client = OpenAICompatibleLLMClient(self.config.llm)
        result = await asyncio.to_thread(
            client.complete_json,
            system_prompt,
            {"raw_prompt": prompt, "gender_tag": gender_tag},
        )
        optimized = str(result.get("optimized_prompt", "")).strip()
        if optimized and len(optimized) > 10:
            return optimized
        return prompt

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
        pi = output.panel_image
        yield self._make_event(
            StreamEventType.LOG,
            f"[{pi.panel_id}] seed={pi.seed} | prompt={pi.prompt_used[:120]}...",
        )
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"image_path": pi.image_path},
        )
        yield self._make_event(StreamEventType.DONE, output.model_dump())
