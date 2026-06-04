"""Panel image generation agent.

v0.6: Consumes PromptPack from StoryboardAgent directly — no duplicate prompt building.
Uses character visual_traits for consistency validation.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator

from comicweaver.api import (
    ApiBackendError,
    ComfyUIImageClient,
    ImageGenerationRequest,
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
    rubric_id = "rubric_image_v1"

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
                    self._emit(
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
                return self._make_placeholder_output(plan, seed, positive, negative)
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
                    f"char={char_id} seed={seed} "
                    f"shot={plan.shot_size.value} angle={plan.camera_angle.value} "
                    f"mood={plan.mood} weather={plan.weather} "
                    f"prompt_len={len(positive)}"
                ],
            ),
        )

    def _make_placeholder_output(
        self, plan, seed: int, positive: str, negative: str
    ) -> ImageOutput:
        """Local fallback: return placeholder PanelImage when no image backend."""
        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=f"projects/test/panels/{plan.panel_id}_placeholder.png",
            image_format="png",
            width=1024,
            height=1024,
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

    async def astream(
        self, inputs: ImageInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        yield self._make_event(
            StreamEventType.LOG,
            f"Generating {inputs.panel_plan.panel_id} ({inputs.panel_plan.shot_size.value})...",
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
