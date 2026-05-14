"""角色管理 Agent - Mock 实现。

真实实现: CLIP特征提取 + IP-Adapter + FAISS索引。
Mock: 用占位图像与确定性权重计算。
"""
from __future__ import annotations

from typing import AsyncIterator

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
    """角色管理 Agent (Mock)."""

    name = "character_agent"
    version = "0.1.0-mock"
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
            ref_path = generate_character_placeholder(
                project_id=context.project_id,
                char_id=draft.char_id,
                name=draft.name,
                appearance=draft.appearance,
            )
            profiles[draft.char_id] = CharacterProfile(
                char_id=draft.char_id,
                name=draft.name,
                base_reference=ReferenceImage(
                    image_id=f"{draft.char_id}_base",
                    image_path=ref_path,
                    source="generated",
                    generation_prompt=draft.appearance,
                    confidence=0.85,
                ),
                visual_traits=_extract_traits_from_appearance(draft.appearance),
                clip_embedding_id=f"emb_{draft.char_id}",
                style_preset=inputs.style_preset or context.style_preset,
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

    async def _build_window(
        self, inputs: CharacterInput, context: AgentContext
    ) -> CharacterOutput:
        """构造链式参考窗口 - Mock 版本只用 base_reference。"""
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
    """从外观描述中粗略提取关键特征。"""
    parts = [p.strip() for p in appearance.replace(",", ",").split(",") if p.strip()]
    traits = VisualTraits()
    for p in parts:
        if any(k in p for k in ("发", "髮")):
            traits.hair = p
        elif any(k in p for k in ("眼", "瞳")):
            traits.eyes = p
        elif any(k in p for k in ("衫", "服", "袍", "裙", "外套")):
            traits.clothing = p
    return traits
