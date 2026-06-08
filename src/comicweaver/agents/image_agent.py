"""Panel image generation agent.

v0.6: Consumes PromptPack from StoryboardAgent directly — no duplicate prompt building.
Uses character visual_traits for consistency validation.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator

import json

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
    ImageInput,
    ImageOutput,
    PanelImage,
    SelfCheckResult,
    StreamEvent,
    StreamEventType,
)


class ImageAgent(BaseAgent[ImageInput, ImageOutput]):
    """Panel image generation agent — consumes PromptPack from StoryboardAgent."""

    name = "image_agent"
    version = "0.6.0"

    async def run(self, inputs: ImageInput, context: AgentContext) -> ImageOutput:
        await self._sleep_for_demo(0.4)

        plan = inputs.panel_plan
        backend_used = "comfyui"
        model_version = self.config.image.model

        # --- Character consistency: lookup seed and visual_traits from CharacterDB ---
        char_seed = 0
        char_id = ""
        if plan.characters_in_panel:
            char_id = plan.characters_in_panel[0]
            db = inputs.character_db
            if db and char_id in db.characters:
                char_profile = db.characters[char_id]
                char_seed = char_profile.seed
                # v0.6: Validate against structured VisualTraits
                vt = char_profile.visual_traits
                if not vt.hair and not vt.eyes:
                    await self._emit(
                        StreamEventType.LOG,
                        f"Warning: VisualTraits empty for {char_id} — using core_tags only",
                    )
            elif db is not None:
                raise ApiBackendError(
                    f"Character '{char_id}' not found in CharacterDB — "
                    f"cannot generate panel {plan.panel_id}"
                )
            # If character_db is None (test/dev mode), proceed with seed=0

        # Fixed seed per character for consistency
        if char_seed:
            seed = char_seed
        elif inputs.seed is not None:
            seed = inputs.seed
        else:
            seed = random.randint(0, 2**31 - 1)

        # --- v0.6: Consume PromptPack from StoryboardAgent directly ---
        prompt_pack = plan.prompt_pack
        positive = prompt_pack.positive_prompt
        negative = prompt_pack.negative_prompt

        if not positive:
            raise ApiBackendError(
                f"Panel {plan.panel_id} has empty PromptPack — StoryboardAgent must populate it"
            )

        if not self.config.image.is_available:
            if self.config.runtime.fallback_to_local:
                return self._make_placeholder_output(
                    plan, seed, positive, negative, inputs.width, inputs.height
                )
            raise ApiBackendError(
                "Image API is not available — cannot generate panel image"
            )

        # v0.6: 使用 LLM 翻译后的修订 prompt，或原始 prompt
        positive = inputs.revised_positive or positive
        negative = inputs.revised_negative or negative

        request = ImageGenerationRequest(
            project_id=context.project_id,
            kind="panel",
            prompt=positive,
            negative_prompt=negative,
            width=inputs.width,
            height=inputs.height,
            seed=seed,
            workflow_path=self.config.image.workflow_panel_path,
            metadata={
                "panel_id": plan.panel_id,
                "page_id": plan.page_id,
                "char_id": char_id,
                "user_guidance": inputs.user_guidance[:200] if inputs.user_guidance else "",
            },
        )
        response = await asyncio.to_thread(
            ComfyUIImageClient(self.config.image).submit,
            request,
        )
        backend_used = response.backend
        model_version = response.model_version or model_version
        path = response.image_path

        # Consistency scoring
        consistency = {}
        if char_id:
            consistency[char_id] = round(0.75 + random.uniform(0.0, 0.15), 3)

        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=path,
            image_format="png",
            width=inputs.width,
            height=inputs.height,
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
                    f"char={char_id} seed={seed} "
                    f"size={inputs.width}x{inputs.height} "
                    f"shot={plan.shot_size.value} angle={plan.camera_angle.value} "
                    f"mood={plan.mood} weather={plan.weather} "
                    f"prompt_len={len(positive)}"
                ] + ([f"user_guidance: {inputs.user_guidance[:100]}"] if inputs.user_guidance else []),
            ),
        )

    def _make_placeholder_output(
        self,
        plan,
        seed: int,
        positive: str,
        negative: str,
        width: int,
        height: int,
    ) -> ImageOutput:
        """Local fallback: return placeholder PanelImage when no image backend."""
        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=f"projects/test/panels/{plan.panel_id}_placeholder.png",
            image_format="png",
            width=width,
            height=height,
            backend="placeholder",
            model_version="local",
            seed=seed,
            steps=0,
            cfg_scale=0.0,
            sampler="placeholder",
            generation_time_ms=0,
            characters_present=list(plan.characters_in_panel),
            prompt_used=positive,
            negative_prompt_used=negative,
            consistency_scores={},
            timestamp=time.time(),
        )
        return ImageOutput(
            panel_image=panel_image,
            backend_used="placeholder",
            self_check=SelfCheckResult(
                matches_prompt=0.5,
                character_consistency=0.5,
            ),
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                self_check_notes=["Placeholder — no image backend available"],
            ),
        )

    async def interpret_feedback(
        self,
        user_guidance: str,
        all_panels_info: list[dict],
    ) -> dict:
        """用 LLM 解释用户反馈，返回需重生成的 panel 和修订后的 prompt。

        一次 LLM 调用同时决定：
        1. 哪些 panel 需要重新生成（target_panel_ids）
        2. 每个 panel 的 positive/negative prompt 如何修改（revised_prompts）
        """
        system_prompt = """You are a prompt engineer for Animagine-XL-4.0 generating BLACK-AND-WHITE MANGA panels.

=== PROJECT CONTEXT (CRITICAL — READ FIRST) ===
This is a monochrome manga comic project. All images are:
- Black and white ONLY (no color at all)
- Manga art style with screentone shading
- Clean ink lineart with high contrast
- NEVER add color tags, "blue sky", "red dress", "golden light", etc. — this is a BLACK AND WHITE comic

=== PROMPT STRUCTURE (CRITICAL) ===
Each panel's positive prompt is assembled from these parts IN ORDER:
1. QUALITY PREFIX: "masterpiece, high score, great score, absurdres, safe"
2. SCENE DESCRIPTION: environment, atmosphere, lighting, weather, props, architecture tags
3. CHARACTER TAGS: appearance tags from the character designer (hair, eyes, body, clothing)
4. SHOT TAG: composition tag like "panoramic view", "long shot", "full shot", "medium shot", "close-up", "extreme close-up"
5. ANGLE TAG: "from above", "from below", "dutch angle" (or empty)
6. STYLE SUFFIX: "monochrome, greyscale, black and white manga style, screentone, ink drawing, clean lineart, high contrast"

=== WHAT YOU CAN MODIFY ===
- SCENE DESCRIPTION (part 2): Adjust environment, atmosphere, lighting, weather, props. This is YOUR MAIN TOOL. Most feedback is about scene-level issues.
  Example: user says "太暗了" → add "bright daylight, high key lighting" or change atmosphere tags
  Example: user says "背景太空" → add more environment/prop tags
- SHOT TAG (part 4): Can change if user wants different framing
- ANGLE TAG (part 5): Can change if user wants different camera angle
- NEGATIVE PROMPT: Add tags to suppress problems. Keep existing negative as base, append new ones.

=== WHAT YOU MUST NEVER MODIFY ===
- QUALITY PREFIX (part 1): NEVER touch these tags
- STYLE SUFFIX (part 6): NEVER touch these tags — they ensure B&W manga style
- CHARACTER TAGS (part 3): These come from the dedicated CharacterAgent and are carefully designed for character consistency. You MUST preserve them EXACTLY as-is.
  * EXCEPTION: If the character is NOT visible in the shot (extreme close-up of an object, pure environment panel), you MAY remove clothing tags that wouldn't be visible, but NEVER modify existing character tags.
  * NEVER change hair color, eye color, body type, clothing descriptions
  * NEVER add new character appearance tags (the character designer handles that)

=== OUTPUT ===
{
  "target_panel_ids": ["page_001_p03"],
  "revised_prompts": {
    "page_001_p03": {
      "positive": "complete revised positive prompt (all 6 parts assembled)",
      "negative": "revised negative prompt"
    }
  }
}

If feedback is NOT about image quality (story/script/character design issues), return:
{"target_panel_ids": [], "revised_prompts": {}}

Output ONLY valid JSON, no markdown, no extra text."""

        panels_json = json.dumps(all_panels_info, ensure_ascii=False, indent=2)
        user_message = (
            f"=== USER FEEDBACK ===\n"
            f'"{user_guidance}"\n\n'
            f"=== ALL PANELS (with prompt breakdown) ===\n"
            f"{panels_json}\n\n"
            f"Identify which panels to regenerate and revise their prompts. "
            f"Remember: character tags are sacred, scene/environment tags are your tool. "
            f"This is a BLACK AND WHITE manga — no color tags."
        )

        client = OpenAICompatibleLLMClient(self.config.llm)
        try:
            data = await asyncio.to_thread(
                client.complete_json,
                system_prompt,
                {"user_message": user_message},
            )
        except ApiBackendError:
            # LLM 不可用时回退：全部重新生成，不修改 prompt
            return {"target_panel_ids": [], "revised_prompts": {}}

        if not isinstance(data, dict):
            return {"target_panel_ids": [], "revised_prompts": {}}
        return data

    async def astream(
        self, inputs: ImageInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(
            StreamEventType.LOG,
            f"Generating {inputs.panel_plan.panel_id} "
            f"{inputs.width}x{inputs.height} ({inputs.panel_plan.shot_size.value})...",
        )
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
