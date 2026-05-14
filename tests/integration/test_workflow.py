"""端到端集成测试 - 完整工作流。"""
import pytest

from comicweaver.core import make_initial_state
from comicweaver.orchestrator import ComicWorkflow, WorkflowEvent


@pytest.mark.asyncio
async def test_full_workflow_full_auto():
    """全自动模式下端到端跑通。"""
    state = make_initial_state(
        user_input="一个少年在雨夜的城市中寻找失踪的朋友",
        creation_mode="simple",
        style_preset="manga",
        interaction_mode="full_auto",
        target_pages=2,
    )

    wf = ComicWorkflow()
    events = []
    async for msg in wf.arun(state):
        events.append(msg)

    # 至少应该看到开始与结束
    assert len(events) > 0

    # 验证最终state被填充
    assert state.get("structured_script")
    assert state.get("character_db")
    assert state.get("storyboard_plan")
    assert state.get("panel_images")
    assert state.get("final_pages")

    # 验证审查记录被填充
    assert len(state.get("review_results", [])) >= 4

    # 验证有完成事件
    done_msgs = [
        m for m in events
        if hasattr(m, "event") and m.event == WorkflowEvent.WORKFLOW_DONE
    ]
    assert len(done_msgs) == 1
