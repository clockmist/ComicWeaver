"""Character management agent with image API integration hooks."""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, ComfyUIImageClient, ImageGenerationRequest
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    CharacterDB,
    CharacterDraft,
    CharacterInput,
    CharacterOutput,
    CharacterProfile,
    ReferenceImage,
    ReferenceWindow,
    StreamEvent,
    StreamEventType,
    VisualTraits,
    WeightedReference,
)
from comicweaver.storage.placeholder import generate_character_placeholder


class CharacterAgent(BaseAgent[CharacterInput, CharacterOutput]):
    """Character management agent."""

    name = "character_agent"
    version = "0.2.0-api"
    rubric_id = "rubric_character_v1"

    async def run(self, inputs: CharacterInput, context: AgentContext) -> CharacterOutput:
        if inputs.operation == "init":
            return await self._init(inputs, context)
        if inputs.operation == "build_reference_window":
            return await self._build_window(inputs, context)
        return CharacterOutput(operation=inputs.operation)

    async def _init(
        self, inputs: CharacterInput, context: AgentContext
    ) -> CharacterOutput:
        await self._sleep_for_demo(0.3)

        drafts = inputs.character_drafts or []
        profiles: dict[str, CharacterProfile] = {}

        for draft in drafts:
            # 固定种子：从 char_id 的 SHA256 前 16 位导出（保证每个角色种子唯一且可复现）
            seed = int(hashlib.sha256(draft.char_id.encode()).hexdigest()[:16], 16) % (2**63)

            # 提取视觉特征并构建角色外观 prompt
            traits = _extract_traits_from_appearance(draft.appearance)
            appearance_parts = []
            if traits.hair:
                appearance_parts.append(traits.hair)
            if traits.eyes:
                appearance_parts.append(traits.eyes)
            if traits.clothing:
                appearance_parts.append(traits.clothing)
            appearance_prompt = ", ".join(appearance_parts) if appearance_parts else draft.appearance

            ref_path = await self._reference_image_path(
                draft, context, inputs.style_preset, seed, appearance_prompt,
            )
            profiles[draft.char_id] = CharacterProfile(
                char_id=draft.char_id,
                name=draft.name,
                base_reference=ReferenceImage(
                    image_id=f"{draft.char_id}_base",
                    image_path=ref_path,
                    source="generated",
                    generation_prompt=f"character design sheet, {appearance_prompt}",
                    confidence=0.85,
                ),
                visual_traits=traits,
                clip_embedding_id=f"emb_{draft.char_id}",
                style_preset=inputs.style_preset or context.style_preset,
                seed=seed,
                appearance_prompt=appearance_prompt,
            )

        db = CharacterDB(
            project_id=context.project_id,
            characters=profiles,
        )

        return CharacterOutput(
            operation="init",
            character_db=db,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                self_check_notes=[f"已初始化{len(profiles)}个角色档案"],
            ),
        )

    async def _reference_image_path(
        self,
        draft: CharacterDraft,
        context: AgentContext,
        style_preset: str | None,
        seed: int = 0,
        appearance_prompt: str = "",
    ) -> str:
        if self.config.image.is_available:
            # 黑白漫画风格角色参考图 prompt（严格单色 + 线稿风格）
            positive = (
                f"masterpiece, high score, great score, absurdres, "
                f"1girl, {appearance_prompt}, "
                f"upper body, portrait, bust shot, "
                f"looking at viewer, neutral expression, "
                f"monochrome, greyscale, black and white manga style, "
                f"screentone, ink drawing, clean lineart, high contrast, "
                f"white background, safe"
            )
            negative = (
                "lowres, bad anatomy, bad hands, text, error, missing finger, "
                "extra digits, fewer digits, cropped, worst quality, low quality, "
                "low score, bad score, average score, signature, watermark, "
                "username, blurry, color, colored, multicolored, gradient, "
                "rainbow, 3d, realistic, photo, photograph, photorealistic, "
                "nsfw, explicit"
            )
            request = ImageGenerationRequest(
                project_id=context.project_id,
                kind="character_reference",
                prompt=positive,
                negative_prompt=negative,
                width=1024,
                height=1024,
                seed=seed,
                workflow_path=self.config.image.workflow_character_path,
                metadata={"char_id": draft.char_id, "name": draft.name},
            )
            try:
                response = await asyncio.to_thread(
                    ComfyUIImageClient(self.config.image).submit,
                    request,
                )
                if response.image_path:
                    return response.image_path
            except ApiBackendError:
                if not self.config.image.fallback_to_placeholder:
                    raise

        return generate_character_placeholder(
            project_id=context.project_id,
            char_id=draft.char_id,
            name=draft.name,
            appearance=draft.appearance,
        )

    async def _build_window(
        self, inputs: CharacterInput, context: AgentContext
    ) -> CharacterOutput:
        """构造链式参考窗口 - 本地版本只用 base_reference。"""
        await self._sleep_for_demo(0.05)

        windows: dict[str, ReferenceWindow] = {}
        for char_id in (inputs.target_characters or []):
            windows[char_id] = ReferenceWindow(
                panel_id=inputs.panel_id or "unknown",
                char_id=char_id,
                references=[
                    WeightedReference(
                        image_path=f"projects/{context.project_id}/characters/"
                                   f"{char_id}/base.png",
                        weight=1.0,
                        role="base",
                    )
                ],
                window_strategy="base_only",
            )

        return CharacterOutput(
            operation="build_reference_window",
            reference_windows=windows,
        )

    async def astream(
        self, inputs: CharacterInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        if inputs.operation == "init":
            yield self._make_event(StreamEventType.LOG, "开始构建角色档案...")
            yield self._make_event(StreamEventType.PROGRESS, 0.2)

            n = len(inputs.character_drafts or [])
            for i in range(n):
                await self._sleep_for_demo(0.15)
                draft = (inputs.character_drafts or [])[i]
                yield self._make_event(
                    StreamEventType.LOG,
                    f"生成角色「{draft.name}」的参考图...",
                )
                yield self._make_event(
                    StreamEventType.PROGRESS,
                    0.2 + (i + 1) / max(n, 1) * 0.7,
                )

            output = await self.run(inputs, context)
            yield self._make_event(
                StreamEventType.PARTIAL_OUTPUT,
                {"character_count": len(output.character_db.characters)
                                     if output.character_db else 0},
            )
            yield self._make_event(StreamEventType.PROGRESS, 1.0)
            yield self._make_event(StreamEventType.DONE, output.model_dump())
        else:
            output = await self.run(inputs, context)
            yield self._make_event(StreamEventType.DONE, output.model_dump())


def _extract_traits_from_appearance(appearance: str) -> VisualTraits:
    """从外观描述中提取关键视觉特征（中英文双语支持）。"""
    parts = [p.strip() for p in appearance.replace(",", ",").replace("，", ",").split(",") if p.strip()]
    traits = VisualTraits()
    hair_keywords = ("发", "髮", "hair", "haired")
    eye_keywords = ("眼", "瞳", "eye", "eyed")
    cloth_keywords = ("衫", "服", "袍", "裙", "外套", "shirt", "coat", "jacket",
                      "dress", "suit", "uniform", "robe", "cloak", "hoodie", "cape")
    for p in parts:
        p_lower = p.lower()
        if any(k in p or k in p_lower for k in hair_keywords):
            traits.hair = p
        elif any(k in p or k in p_lower for k in eye_keywords):
            traits.eyes = p
        elif any(k in p or k in p_lower for k in cloth_keywords):
            traits.clothing = p
    return traits
