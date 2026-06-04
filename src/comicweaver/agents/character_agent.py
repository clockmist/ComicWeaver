"""Character management agent with LLM-driven tag generation and image API hooks."""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator

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
class CharacterAgent(BaseAgent[CharacterInput, CharacterOutput]):
    """Character management agent."""

    name = "character_agent"
    version = "0.4.0-llm"
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

        # 批量调用 LLM 生成所有角色的 Tag / 性别 / 外观描述（一次 API 调用）
        try:
            llm_data_map = await self._generate_core_tags_batch(drafts)
        except ApiBackendError:
            if self.config.runtime.fallback_to_local:
                llm_data_map = self._generate_core_tags_local(drafts)
            else:
                raise

        for draft in drafts:
            # 随机种子：首次生成角色人设图时使用随机种子，
            # 然后保存到 CharacterProfile，后续所有 panel 生成复用此种子保证一致性
            seed = random.randint(0, 2**31 - 1)

            # 获取 LLM 生成的全部角色数据
            llm_data = llm_data_map.get(draft.char_id, {})
            core_tags = llm_data.get("core_tags", draft.appearance)
            gender_tag = llm_data.get("gender_tag", "1girl")
            appearance_prompt = llm_data.get("appearance_prompt", draft.appearance)

            # v0.4: 从 LLM 响应中提取结构化的 VisualTraits
            vt_raw = llm_data.get("visual_traits", {})
            vt = VisualTraits(
                hair=str(vt_raw.get("hair", "")),
                eyes=str(vt_raw.get("eyes", "")),
                body=str(vt_raw.get("body", "")),
                clothing=str(vt_raw.get("clothing", "")),
                distinctive=list(vt_raw.get("distinctive", [])) if isinstance(vt_raw.get("distinctive"), list) else [],
            )

            ref_path = await self._reference_image_path(
                draft, context, inputs.style_preset, seed, core_tags, gender_tag,
            )
            profiles[draft.char_id] = CharacterProfile(
                char_id=draft.char_id,
                name=draft.name,
                base_reference=ReferenceImage(
                    image_id=f"{draft.char_id}_base",
                    image_path=ref_path,
                    source="generated",
                    generation_prompt=f"character design sheet: {core_tags}",
                    confidence=0.85,
                ),
                visual_traits=vt,
                clip_embedding_id=f"emb_{draft.char_id}",
                style_preset=inputs.style_preset or context.style_preset,
                seed=seed,
                appearance_prompt=appearance_prompt,
                gender_tag=gender_tag,
                core_tags=core_tags,
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
        core_tags: str = "",
        gender_tag: str = "1girl",
    ) -> str:
        if self.config.image.is_available:
            # 使用 LLM 生成的 Danbooru Tag 构建人设图 Prompt
            # 风格前缀 + 角色 Tag + 构图描述（构图部分保持硬编码以确保胸像效果）
            char_tags = core_tags or draft.appearance
            composition = (
                "portrait, head and shoulders, "
                "looking at viewer, entire head visible, full hair in frame, "
                "centered framing, ample headroom, "
                "simple background, white background, "
                "neutral expression, front view"
            )
            # Animagine XL 4.0 质量前缀（官方训练标签）
            positive = (
                f"masterpiece, high score, great score, absurdres, safe, "
                f"{char_tags}, "
                f"{composition}, "
                f"monochrome, greyscale, black and white manga style, "
                f"screentone, ink drawing, clean lineart, high contrast"
            )
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
            response = await asyncio.to_thread(
                ComfyUIImageClient(self.config.image).submit,
                request,
            )
            if response.image_path:
                return response.image_path
            raise ApiBackendError(
                f"ComfyUI returned empty image_path for char_id={draft.char_id}"
            )

        # Local/dev fallback: return placeholder path when image API unavailable
        if self.config.runtime.fallback_to_local:
            return f"projects/{context.project_id}/characters/{draft.char_id}/placeholder.png"

        raise ApiBackendError(
            "Image API is not available — cannot generate character reference"
        )

    async def _generate_core_tags_batch(
        self, drafts: list[CharacterDraft]
    ) -> dict[str, dict]:
        """调用 LLM 批量生成角色的 Danbooru Tag、性别标签和外观描述。

        一次 API 调用处理所有角色，LLM 直接输出 core_tags / gender_tag / appearance_prompt，
        不再依赖硬编码的 _detect_gender / _extract_traits_from_appearance。
        """
        system_prompt = """Convert character appearance from natural language into Danbooru-style tags (English only) for BLACK AND WHITE MANGA (monochrome, screentone, ink drawing).

CRITICAL RULES:
1. Determine gender_tag from the appearance text: "1girl", "1boy", or "1other". Infer from explicit gender words (male/female/man/woman/boy/girl) or from name/pronouns in the description.
2. core_tags: Start with gender_tag, then include ONLY these categories (comma-separated Danbooru tags):
   - age (child/teenager/young_adult/adult/elderly)
   - hair: LENGTH (short/medium/long/very_long) and STYLE (ponytail/braid/bob/messy/spiky/straight/wavy/curly/bangs/undercut/bun/twin_tails) — NOT color
   - hair greyscale: ONLY "black hair" / "grey hair" / "white hair" / "silver hair" / "dark hair" / "light hair" — no other hair colors
   - eyes: shape (sharp/round/narrow/large/deep-set) AND SPECIFIC eye color (red/blue/green/grey/golden/purple/brown/heterochromia). ALWAYS include a concrete eye color — "dark eyes" or "round eyes" without color is FORBIDDEN. Use "red eyes", "blue eyes", "green eyes", "grey eyes" etc.
   - clothing: describe by STYLE, FIT, TEXTURE, TYPE (hoodie, jacket, shirt, dress, suit, armor, robe, uniform, tight, loose, layered, ripped, oversized) — NEVER use color for clothing
   - build: body type (slim/muscular/petite/tall/short/athletic/lean/broad)
   - accessories: eyewear, jewelry, weapons, scarves, hats (describe by SHAPE and type, not color)
3. appearance_prompt: A short natural-language visual description focusing on hair style/length, eye shape, clothing style/fit/texture, build. Used as fallback. Keep under 80 words. Describe TEXTURE and SHAPE, not color.
4. visual_traits: A structured breakdown of the character's appearance with these EXACT fields:
   - hair: brief description of hair style, length, texture (e.g. "short spiky", "long wavy")
   - eyes: brief description of eye shape and specific color (e.g. "narrow sharp grey", "large round blue") — MUST include a specific eye color
   - body: brief description of build and height (e.g. "slim athletic tall", "petite slender")
   - clothing: brief description of clothing style, fit, layers, textures (e.g. "loose hoodie, layered collar, slim-fit pants")
   - distinctive: list of 0-3 unique visual features (e.g. ["scar on left cheek", "always wears pendant"])
5. STRICTLY FORBIDDEN (will degrade black and white manga quality):
   - NO clothing color tags (no "red dress", "blue jacket", "golden necklace", etc.)
   - NO skin color tags (no "dark skin", "pale skin", "brown skin", "black skin", "blue skin", etc.)
   - NO hair colors except black/grey/white/silver
   - NO colored accessory descriptions
6. NEVER include style tags (anime, realistic, masterpiece, quality, detailed, beautiful, etc.)
7. NEVER include background tags (simple background, white background, etc.)
8. Always include "solo" in core_tags (this is for single-character reference images).
9. Output valid JSON: {"characters": [{"char_id": "...", "core_tags": "1girl, solo, ...", "gender_tag": "1girl", "appearance_prompt": "...", "visual_traits": {"hair": "...", "eyes": "...", "body": "...", "clothing": "...", "distinctive": [...]}}]}"""

        char_list = [
            {
                "char_id": d.char_id,
                "name": d.name,
                "age_range": d.age_hint or "young_adult",
                "appearance": d.appearance,
            }
            for d in drafts
        ]

        user_prompt = f"Characters to convert:\n{json.dumps(char_list, ensure_ascii=False, indent=2)}"

        client = OpenAICompatibleLLMClient(self.config.llm)
        data = await asyncio.to_thread(
            client.complete_json,
            system_prompt,
            {"characters": char_list},
        )

        draft_appearances = {d.char_id: d.appearance for d in drafts}
        result: dict[str, dict] = {}
        for item in data.get("characters", []):
            cid = item.get("char_id", "")
            tags = str(item.get("core_tags", "")).strip()
            gender = str(item.get("gender_tag", "")).strip()
            appearance = str(item.get("appearance_prompt", "")).strip()
            # 清理可能的 markdown 残留
            tags = re.sub(r"^[`'\"\[\]]+|[`'\"\[\]]+$", "", tags).strip()
            gender = re.sub(r"^[`'\"\[\]]+|[`'\"\[\]]+$", "", gender).strip()
            if not tags:
                raise ApiBackendError(
                    f"LLM returned empty core_tags for char_id={cid}"
                )
            # 校验 gender_tag 必须是合法的 Animagine 标签
            if gender not in ("1girl", "1boy", "1other"):
                gender = "1girl"  # 安全回退（理论上 LLM 不应该输出非法值）
            result[cid] = {
                "core_tags": tags,
                "gender_tag": gender,
                "appearance_prompt": appearance or draft_appearances.get(cid, ""),
            }

        # 确保所有角色都有结果
        for d in drafts:
            if d.char_id not in result:
                raise ApiBackendError(
                    f"LLM did not return core_tags for char_id={d.char_id}"
                )

        return result

    def _generate_core_tags_local(
        self, drafts: list[CharacterDraft]
    ) -> dict[str, dict]:
        """Local fallback: generate mock core_tags and visual_traits (v0.4)."""
        result: dict[str, dict] = {}
        for d in drafts:
            # Simple gender detection from appearance text
            appearance_lower = d.appearance.lower()
            if "female" in appearance_lower or "女" in d.appearance or "girl" in appearance_lower:
                gender_tag = "1girl"
            elif "male" in appearance_lower or "男" in d.appearance or "boy" in appearance_lower:
                gender_tag = "1boy"
            else:
                gender_tag = "1girl"

            result[d.char_id] = {
                "core_tags": f"{gender_tag}, solo, young_adult, short hair, black hair, round eyes, casual clothes, slim",
                "gender_tag": gender_tag,
                "appearance_prompt": d.appearance,
                "visual_traits": {
                    "hair": "short straight hair",
                    "eyes": "round dark eyes",
                    "body": "slim build",
                    "clothing": "casual clothes",
                    "distinctive": [],
                },
            }
        return result

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
