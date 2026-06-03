"""Script understanding agent.

The agent prefers a configured JSON LLM API and falls back to a deterministic
local generator when no backend is configured.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from comicweaver.api import ApiBackendError, OpenAICompatibleLLMClient
from comicweaver.core import (
    AgentContext,
    AgentOutputMeta,
    BaseAgent,
    CharacterDraft,
    CreationMode,
    Dialogue,
    NarrativeStructure,
    Scene,
    ScriptInput,
    ScriptOutput,
    StreamEvent,
    StreamEventType,
)

_MOCK_CHARACTERS = [
    ("林染", "protagonist", "男性,短发凌乱,丹凤眼,宽松连帽衫配修身长裤,瘦高身材"),
    ("陈夜", "antagonist", "男性,灰色长直发,细长眼,修身长袍,高瘦身形"),
    ("苏白", "supporting", "女性,波浪短发,大圆眼,荷叶边衬衫配百褶裙,娇小身材"),
]

_MOCK_LOCATIONS = ["雨夜城市街道", "废弃工厂", "霓虹咖啡馆", "天台",
                    "地铁站", "图书馆", "山顶观景台"]
_MOCK_ATMOSPHERES = ["阴郁压抑", "紧张悬疑", "温暖怀旧", "明亮欢快", "孤独沉静"]


class ScriptAgent(BaseAgent[ScriptInput, ScriptOutput]):
    """Script understanding agent."""

    name = "script_agent"
    version = "0.2.0-api"
    rubric_id = "rubric_script_v1"

    async def run(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        if self.config.llm.is_available:
            try:
                return await self._run_api(inputs, context)
            except ApiBackendError:
                if not self.config.runtime.fallback_to_local:
                    raise
        return await self._run_local(inputs, context)

    async def _run_api(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        from pydantic import ValidationError

        client = OpenAICompatibleLLMClient(self.config.llm)

        system_prompt = (
            "You are a professional comic script writer for ComicWeaver, a SINGLE-CHARACTER-PER-PANEL "
            "black-and-white manga system. Every scene features EXACTLY ONE character. Your JSON "
            "output will be REJECTED if any scene contains more than 1 character in characters_present.\n\n"

            "=== RULE #1 (MOST IMPORTANT — VIOLATION = REJECTION) ===\n"
            'Every "characters_present" array MUST contain EXACTLY ONE character ID. '
            "ONE and ONLY ONE. Never two. Never three. ONE.\n"
            '  CORRECT: "characters_present": ["char_000"]\n'
            '  CORRECT: "characters_present": ["char_001"]\n'
            '  WRONG:   "characters_present": ["char_000", "char_001"]  <-- WILL BE REJECTED\n'
            '  WRONG:   "characters_present": []                        <-- WILL BE REJECTED\n'
            "If two characters interact, put them in SEPARATE scenes. "
            "Dialogue/text can reference other characters even when only one is visible.\n\n"

            "=== OTHER RULES ===\n"
            "2. Return a single JSON object matching this EXACT structure:\n"
            '{\n'
            '  "title": "A compelling title for the comic",\n'
            '  "summary": "2-3 sentence summary of the story",\n'
            '  "genre": ["action", "mystery"],\n'
            '  "characters": [\n'
            '    {\n'
            '      "char_id": "char_000",\n'
            '      "name": "Character Name",\n'
            '      "role": "protagonist",\n'
            '      "appearance": "Start with gender (male/female), then describe hair '
            '(length, style, texture — NOT color except black/white/grey for hair), '
            'eyes (shape and color allowed), clothing (style, fit, layers, texture — '
            'NOT color), build. Describe TEXTURE, SHAPE, PATTERN, STYLE, LENGTH, FIT — '
            'NOT colors. Example: Male, short spiky hair, narrow sharp grey eyes, '
            'loose hoodie with layered collar, slim athletic build.",\n'
            '      "first_appearance_scene": 0\n'
            '    }\n'
            '  ],\n'
            '  "scenes": [\n'
            '    {\n'
            '      "scene_id": "scene_000",\n'
            '      "order": 0,\n'
            '      "location": "Specific location name",\n'
            '      "time_of_day": "morning/afternoon/evening/night",\n'
            '      "atmosphere": "Mood (e.g., tense, peaceful)",\n'
            '      "characters_present": ["char_000"],\n'
            '      "actions": [{"actor": "char_000", "description": "Action description"}],\n'
            '      "dialogues": [{"speaker": "char_000", "text": "Dialogue", "tone": "neutral"}],\n'
            '      "narration": null,\n'
            '      "emotion_intensity": 0.5,\n'
            '      "panel_hint": 2\n'
            '    }\n'
            '  ],\n'
            '  "emotion_curve": [0.3, 0.5, 0.8, 0.4],\n'
            '  "narrative_structure": {\n'
            '    "setup_scenes": [0],\n'
            '    "rising_scenes": [1],\n'
            '    "climax_scenes": [2],\n'
            '    "resolution_scenes": [3],\n'
            '    "pacing": "varied"\n'
            '  }\n'
            '}\n\n'
            "3. char_id format: char_000, char_001, char_002. "
            "scene_id format: scene_000, scene_001, etc.\n"
            "4. role must be: protagonist, antagonist, supporting, extra.\n"
            "5. emotion_intensity: 0.0 (calm) to 1.0 (climax).\n"
            "6. emotion_curve: ONE float per scene, in scene order.\n"
            "7. narrative_structure: arrays contain scene INDICES (order field values).\n"
            "8. Each scene: at least 1 action + 1 dialogue.\n"
            "9. Title must be creative, never 'Untitled'.\n"
            "10. Output ONLY the JSON object, no markdown, no extra text.\n\n"

            "=== REMINDER: characters_present ===\n"
            "I repeat Rule #1 because it is the most common cause of rejection:\n"
            'characters_present = ["ONE_CHAR_ID"] — exactly 1 element, always.\n'
            'Never ["char_000", "char_001"]. Never []. Never more than 1.\n'
        )

        target_pages = inputs.target_pages
        # 根据页数推断场景数：每页约 2-4 个面板 → 2-4 个场景，最少 4 个场景
        target_scenes = max(4, target_pages * 2)

        user_message = (
            f"Create a comic script based on this story idea:\n\n"
            f'"{inputs.raw_text}"\n\n'
            f"Parameters:\n"
            f"- Target pages: {target_pages}\n"
            f"- Target scenes: {target_scenes}\n"
            f"- Style: {inputs.style_hint or 'manga'}\n"
            f"- Language: {inputs.language}\n"
            f"- Character count: 2-3\n\n"
            f"CRITICAL CONSTRAINT (your output will be REJECTED if violated):\n"
            f"  Every scene MUST have EXACTLY ONE character in characters_present.\n"
            f"  This is a single-character-per-panel system.\n"
            f"  If a scene involves two characters meeting or interacting,\n"
            f"  split them into SEPARATE SCENES — each scene shows only one of them.\n\n"
            f"Create a complete story arc with clear setup, rising action, climax, and resolution."
        )

        payload = {"user_message": user_message}

        # 尝试 LLM 调用，失败时回退到本地
        try:
            data = await asyncio.to_thread(
                client.complete_json,
                system_prompt,
                payload,
            )
        except ApiBackendError:
            # API 连接/响应错误 → 回退本地
            if not self.config.runtime.fallback_to_local:
                raise
            return await self._run_local(inputs, context)

        # 尝试 Pydantic 验证，失败时重试一次或回退
        try:
            output = ScriptOutput.model_validate(data)
        except ValidationError as ve:
            # 可能是 LLM 返回了嵌套结构（如 {"script": {...}}）
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                try:
                    output = ScriptOutput.model_validate(list(data.values())[0])
                except ValidationError:
                    if self.config.runtime.fallback_to_local:
                        return await self._run_local(inputs, context)
                    raise ApiBackendError(
                        f"LLM response validation failed after unwrap: {ve}"
                    ) from ve
            elif self.config.runtime.fallback_to_local:
                return await self._run_local(inputs, context)
            else:
                raise ApiBackendError(
                    f"LLM response validation failed: {ve}"
                ) from ve

        # 质量检查：如果标题为空/Untitled，补充处理
        title = output.title
        if not title or title.strip().lower() in ("untitled", "无题", ""):
            title = _make_title(inputs.raw_text)
            output = output.model_copy(update={"title": title})

        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"LLM API provider: {self.config.llm.provider}",
                    f"Generated {len(output.scenes)} scenes, {len(output.characters)} characters",
                ],
            )
        })

    async def _run_local(self, inputs: ScriptInput, context: AgentContext) -> ScriptOutput:
        """根据用户输入构造一个可信的虚假剧本。"""
        await self._sleep_for_demo(0.3)

        # 选取角色数量(2-3)
        n_chars = 2 if inputs.creation_mode == CreationMode.SIMPLE else 3
        characters = [
            CharacterDraft(
                char_id=f"char_{i:03d}",
                name=name,
                role=role,  # type: ignore[arg-type]
                appearance=appearance,
                first_appearance_scene=0 if i == 0 else i,
            )
            for i, (name, role, appearance) in enumerate(
                _MOCK_CHARACTERS[:n_chars]
            )
        ]

        # 构造场景
        n_scenes = max(4, inputs.target_pages * 2)
        scenes: list[Scene] = []
        for i in range(n_scenes):
            # 每个场景仅 1 个角色（黑白漫画单角色分镜系统）
            present = [characters[i % len(characters)].char_id]
            scenes.append(
                Scene(
                    scene_id=f"scene_{i:03d}",
                    order=i,
                    location=_MOCK_LOCATIONS[i % len(_MOCK_LOCATIONS)],
                    time_of_day="夜晚" if i % 2 == 0 else "白天",
                    atmosphere=_MOCK_ATMOSPHERES[i % len(_MOCK_ATMOSPHERES)],
                    characters_present=present,
                    dialogues=[
                        Dialogue(
                            speaker=present[0],
                            text=f"这是第{i + 1}场的开场对话。",
                            tone="neutral",
                        ),
                    ],
                    emotion_intensity=_emotion_for_position(i, n_scenes),
                    panel_hint=2 if i in (n_scenes // 2, n_scenes - 1) else 1,
                )
            )

        emotion_curve = [s.emotion_intensity for s in scenes]

        # 叙事结构: 起承转合
        quarter = max(1, n_scenes // 4)
        structure = NarrativeStructure(
            setup_scenes=list(range(0, quarter)),
            rising_scenes=list(range(quarter, 2 * quarter)),
            climax_scenes=list(range(2 * quarter, 3 * quarter)),
            resolution_scenes=list(range(3 * quarter, n_scenes)),
            pacing="varied",
        )

        title = _make_title(inputs.raw_text)

        return ScriptOutput(
            title=title,
            summary=f"基于「{inputs.raw_text[:30]}」改编的{n_scenes}场漫画故事。",
            genre=["剧情", inputs.style_hint or "原创"],
            characters=characters,
            scenes=scenes,
            emotion_curve=emotion_curve,
            narrative_structure=structure,
            meta=AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[
                    f"生成{n_scenes}场景,{len(characters)}角色",
                    "本地规则后端,可通过配置切换到 LLM API",
                ],
            ),
        )

    async def astream(
        self, inputs: ScriptInput, context: AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """带流式事件的执行。"""
        yield self._make_event(StreamEventType.LOG, "开始解析剧本...")
        yield self._make_event(StreamEventType.PROGRESS, 0.1)

        await self._sleep_for_demo(0.2)
        yield self._make_event(StreamEventType.THINKING, "扩展主题为完整大纲...")
        yield self._make_event(StreamEventType.PROGRESS, 0.3)

        await self._sleep_for_demo(0.2)
        yield self._make_event(StreamEventType.THINKING, "拆分场景与分配情感强度...")
        yield self._make_event(StreamEventType.PROGRESS, 0.6)

        await self._sleep_for_demo(0.2)
        yield self._make_event(StreamEventType.THINKING, "提取角色与对话...")
        yield self._make_event(StreamEventType.PROGRESS, 0.85)

        output = await self.run(inputs, context)
        yield self._make_event(
            StreamEventType.PARTIAL_OUTPUT,
            {"title": output.title, "scene_count": len(output.scenes)},
        )
        yield self._make_event(StreamEventType.PROGRESS, 1.0)
        yield self._make_event(StreamEventType.DONE, output.model_dump())


def _make_title(raw: str) -> str:
    cleaned = raw.strip().replace("\n", " ")
    if not cleaned:
        return "无题"
    return cleaned[:12] + ("..." if len(cleaned) > 12 else "")


def _emotion_for_position(i: int, total: int) -> float:
    """钟形情感曲线: 起→承→转(高潮)→合。"""
    if total <= 1:
        return 0.5
    progress = i / (total - 1)
    if progress < 0.25:
        return 0.3 + progress * 0.4
    if progress < 0.5:
        return 0.4 + progress * 0.4
    if progress < 0.75:
        return 0.7 + (progress - 0.5) * 1.2  # 高潮
    return 0.4 + random.uniform(-0.1, 0.1)
