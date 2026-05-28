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
    ("林染", "protagonist", "黑色短发,蓝色眼睛,白衬衫黑外套", "好奇心强,坚定"),
    ("陈夜", "antagonist", "灰色长发,黄色瞳孔,黑色长袍", "神秘,深谋远虑"),
    ("苏白", "supporting", "金色卷发,绿眼,粉色连衣裙", "活泼,善良"),
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
        client = OpenAICompatibleLLMClient(self.config.llm)
        payload = {
            "task": "generate_structured_comic_script",
            "input_schema": ScriptInput.model_json_schema(),
            "output_schema": ScriptOutput.model_json_schema(),
            "input": inputs.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "requirements": [
                "Return JSON only.",
                "Use stable char_id and scene_id values.",
                "Create an emotion_curve with one value per scene.",
                "Keep scenes visually actionable for comic panels.",
            ],
        }
        data = await asyncio.to_thread(
            client.complete_json,
            "You generate structured comic scripts for ComicWeaver.",
            payload,
        )
        output = ScriptOutput.model_validate(data)
        return output.model_copy(update={
            "meta": AgentOutputMeta(
                agent=self.name,
                version=self.version,
                inputs_hash=self._inputs_hash(inputs),
                retry_count=context.retry_count,
                self_check_notes=[f"LLM API provider: {self.config.llm.provider}"],
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
                personality=personality,
                first_appearance_scene=0 if i == 0 else i,
            )
            for i, (name, role, appearance, personality) in enumerate(
                _MOCK_CHARACTERS[:n_chars]
            )
        ]

        # 构造场景
        n_scenes = max(4, inputs.target_pages * 2)
        scenes: list[Scene] = []
        for i in range(n_scenes):
            present = [c.char_id for c in characters[: 1 + (i % len(characters))]]
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
