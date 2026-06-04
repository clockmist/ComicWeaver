"""端到端集成测试 - 完整工作流（从 script 阶段开始，跳过需要 LLM 的 StoryAgent）。"""
import pytest

from comicweaver.core import make_initial_state
from comicweaver.orchestrator import ComicWorkflow, WorkflowEvent


@pytest.mark.asyncio
async def test_full_workflow_full_auto():
    """全自动模式下端到端跑通（script → character → storyboard → image → layout）。

    StoryAgent 需要真实 LLM API，不在集成测试范围内；此处预填充
    developed_story 并直接从 script 阶段启动。
    """
    state = make_initial_state(
        user_input="一个少年在雨夜的城市中寻找失踪的朋友",
        creation_mode="simple",
        style_preset="manga",
        interaction_mode="full_auto",
        target_pages=2,
    )

    # 预填充 developed_story（模拟 StoryAgent 已完成），直接从 script 开始
    state["developed_story"] = {
        "title": "测试故事",
        "author_note": "一个关于寻找与发现的故事",
        "tone": "mysterious",
        "genre": ["mystery", "drama"],
        "story_text": "在一个雨夜的城市中，一个少年踏上了寻找失踪朋友的旅程。",
        "characters": [
            {"name": "少年", "role": "protagonist", "brief_description": "勇敢而执着的年轻人"},
            {"name": "朋友", "role": "supporting", "brief_description": "失踪的挚友"},
        ],
        "core_conflict": "少年必须在时间耗尽前找到朋友",
        "setting": "雨夜的城市街道",
        "target_pages": 2,
    }

    # 从 script 阶段开始（跳过需要 LLM 的 StoryAgent）
    wf = ComicWorkflow(resume_phase="script")
    events = []
    async for msg in wf.arun(state):
        events.append(msg)

    # 至少应该看到开始与结束
    assert len(events) > 0

    # 验证所有阶段 state 被填充（5 agent 流水线，story 预填充）
    assert state.get("developed_story"), "developed_story should be pre-populated"
    assert state.get("structured_script"), "ScriptAgent should populate structured_script"
    assert state.get("character_db"), "CharacterAgent should populate character_db"
    assert state.get("storyboard_plan"), "StoryboardAgent should populate storyboard_plan"
    assert state.get("panel_images"), "ImageAgent should populate panel_images"
    assert state.get("final_pages"), "LayoutAgent should populate final_pages"

    # 验证有完成事件
    done_msgs = [
        m for m in events
        if hasattr(m, "event") and m.event == WorkflowEvent.WORKFLOW_DONE
    ]
    assert len(done_msgs) == 1
