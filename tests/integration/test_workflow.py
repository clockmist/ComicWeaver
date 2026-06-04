"""端到端集成测试 — v1.0 流水线 (storyboard → image → layout)。

StoryAgent、CharacterAgent、ScriptAgent 需要真实 LLM/ComfyUI API，
不在集成测试范围内。
"""
import pytest

from comicweaver.core import make_initial_state
from comicweaver.orchestrator import ComicWorkflow, WorkflowEvent


@pytest.mark.asyncio
async def test_full_workflow_full_auto():
    """从 storyboard 阶段开始测试 storyboard→image→layout 链路。

    预填充 developed_story、character_db、structured_script（模拟前三个阶段已完成）。
    """
    state = make_initial_state(
        user_input="一个少年在雨夜的城市中寻找失踪的朋友",
        creation_mode="simple",
        style_preset="manga",
        interaction_mode="full_auto",
        target_pages=2,
    )

    # 预填充 developed_story
    state["developed_story"] = {
        "title": "雨夜寻踪",
        "author_note": "一个关于寻找与发现的故事",
        "tone": "mysterious noir",
        "genre": ["mystery", "drama"],
        "story_text": "在一个雨夜的城市中，一个少年踏上了寻找失踪朋友的旅程...",
        "characters": [
            {"name": "林染", "role": "protagonist", "brief_description": "勇敢的年轻人"},
        ],
        "core_conflict": "少年必须在时间耗尽前找到朋友",
        "setting": "雨夜的城市街道",
        "target_pages": 2,
    }

    # 预填充 character_db（模拟 CharacterAgent）
    from comicweaver.core import CharacterProfile, ReferenceImage, VisualTraits
    state["character_db"] = {
        "project_id": state["project_id"],
        "characters": {
            "char_000": CharacterProfile(
                char_id="char_000",
                name="林染",
                base_reference=ReferenceImage(
                    image_id="char_000_base",
                    image_path="projects/test/characters/char_000/placeholder.png",
                    source="generated",
                ),
                visual_traits=VisualTraits(
                    hair="short spiky", eyes="narrow sharp grey",
                    body="slim athletic", clothing="loose hoodie",
                ),
                seed=42,
                appearance_prompt="short spiky hair, narrow sharp grey eyes",
                gender_tag="1boy",
                core_tags="1boy, solo, young_adult, short hair, spiky, narrow eyes, grey eyes, hoodie, slim",
            ).model_dump(),
        },
        "version": 1,
    }

    # 预填充 structured_script — v1.0 pages 格式（模拟 ScriptAgent）
    state["structured_script"] = {
        "title": "雨夜寻踪",
        "summary": "一个少年在雨夜寻找朋友的故事",
        "genre": ["mystery"],
        "pages": [
            {
                "page_number": 1,
                "page_note": "建立世界观与主角登场",
                "panels": [
                    {
                        "panel_id": "page_001_p01", "page_number": 1, "order_in_page": 1,
                        "narrative_purpose": "建立场景：雨夜的城市街道",
                        "character_id": "",
                        "character_action": "",
                        "dialogue_text": "", "dialogue_tone": "", "is_thought": False,
                        "narration": "一个雨夜...",
                        "emotion": "tension", "emotion_intensity": 0.3,
                        "is_key_panel": False,
                        "location": "城市街道", "time_of_day": "夜晚", "atmosphere": "阴郁",
                    },
                    {
                        "panel_id": "page_001_p02", "page_number": 1, "order_in_page": 2,
                        "narrative_purpose": "介绍主角登场——展示他的急迫",
                        "character_id": "char_000",
                        "character_action": "在雨中奔跑，焦急地四处张望",
                        "dialogue_text": "到底在哪...", "dialogue_tone": "desperate",
                        "is_thought": False,
                        "narration": "",
                        "emotion": "tension", "emotion_intensity": 0.5,
                        "is_key_panel": True,
                        "location": "城市街道", "time_of_day": "夜晚", "atmosphere": "阴郁",
                    },
                ],
            },
            {
                "page_number": 2,
                "page_note": "关键线索与剧情转折",
                "panels": [
                    {
                        "panel_id": "page_002_p01", "page_number": 2, "order_in_page": 1,
                        "narrative_purpose": "发现关键线索——地上的照片",
                        "character_id": "char_000",
                        "character_action": "发现地上的照片，弯腰拾起",
                        "dialogue_text": "这是...", "dialogue_tone": "surprised",
                        "is_thought": False,
                        "narration": "",
                        "emotion": "surprise", "emotion_intensity": 0.7,
                        "is_key_panel": True,
                        "location": "小巷", "time_of_day": "夜晚", "atmosphere": "紧张",
                    },
                    {
                        "panel_id": "page_002_p02", "page_number": 2, "order_in_page": 2,
                        "narrative_purpose": "主角的决心——看向远方",
                        "character_id": "char_000",
                        "character_action": "紧握照片，坚定地看向街道尽头",
                        "dialogue_text": "", "dialogue_tone": "", "is_thought": False,
                        "narration": "",
                        "emotion": "determination", "emotion_intensity": 0.8,
                        "is_key_panel": False,
                        "location": "小巷出口", "time_of_day": "夜晚", "atmosphere": "充满希望",
                    },
                ],
            },
        ],
    }

    # 从 storyboard 阶段开始（前三个阶段需要 LLM/ComfyUI）
    wf = ComicWorkflow(resume_phase="storyboard")
    events = []
    async for msg in wf.arun(state):
        events.append(msg)

    assert len(events) > 0, "Should see workflow events"
    assert state.get("developed_story"), "Pre-populated"
    assert state.get("character_db"), "Pre-populated"
    assert state.get("structured_script"), "Pre-populated"
    assert state.get("storyboard_plan"), "StoryboardAgent should populate storyboard_plan"
    assert state.get("panel_images"), "ImageAgent should populate panel_images"
    assert state.get("final_pages"), "LayoutAgent should populate final_pages"

    done_msgs = [
        m for m in events
        if hasattr(m, "event") and m.event == WorkflowEvent.WORKFLOW_DONE
    ]
    assert len(done_msgs) == 1
