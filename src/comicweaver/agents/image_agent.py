"""图像生成 Agent - Mock 实现。

真实实现:调用 ComfyUI/diffusers + IP-Adapter + ControlNet。
Mock: 调用 storage.placeholder.generate_panel_placeholder 输出占位图。
"""
from __future__ import annotations

import random
import time
from typing import AsyncIterator

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
from comicweaver.storage.placeholder import generate_panel_placeholder


class ImageAgent(BaseAgent[ImageInput, ImageOutput]):
    """图像生成 Agent (Mock)."""

    name = "image_agent"
    version = "0.1.0-mock"
    rubric_id = "rubric_image_v1"

    async def run(self, inputs: ImageInput, context: AgentContext) -> ImageOutput:
        await self._sleep_for_demo(0.4)

        plan = inputs.panel_plan
        seed = inputs.seed if inputs.seed is not None else random.randint(0, 2 ** 31)

        path = generate_panel_placeholder(
            project_id=context.project_id,
            panel_id=plan.panel_id,
            prompt=plan.prompt_pack.positive_prompt,
            characters=plan.characters_in_panel,
            shot_size=plan.shot_size.value,
            size=(inputs.width, inputs.height),
        )

        # 模拟一致性评分
        consistency = {
            char_id: round(0.7 + random.uniform(0.0, 0.2), 3)
            for char_id in plan.characters_in_panel
        }

        panel_image = PanelImage(
            panel_id=plan.panel_id,
            image_path=path,
            image_format="png",
            width=inputs.width,
            height=inputs.height,
            backend="mock",
            model_version="mock-0.1",
            seed=seed,
            steps=20,
            cfg_scale=7.5,
            sampler="mock",
            generation_time_ms=400,
            characters_present=list(plan.characters_in_panel),
            prompt_used=plan.prompt_pack.positive_prompt,
            negative_prompt_used=plan.prompt_pack.negative_prompt,
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
            backend_used="mock",
            self_check=self_check,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
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
