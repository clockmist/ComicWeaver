"""测试核心抽象层 - Schema 与 BaseAgent。"""
import pytest

from comicweaver.core import (
    AgentContext,
    CreationMode,
    InteractionMode,
    ScriptInput,
)


def test_script_input_validation():
    """测试 ScriptInput 的 Pydantic 校验。"""
    inp = ScriptInput(
        creation_mode=CreationMode.SIMPLE,
        raw_text="一个少年在雨夜寻找朋友",
        target_pages=4,
    )
    assert inp.creation_mode == CreationMode.SIMPLE
    assert inp.target_pages == 4


def test_agent_context():
    """测试 AgentContext 构造。"""
    ctx = AgentContext(
        project_id="test_proj",
        interaction_mode=InteractionMode.SEMI_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    assert ctx.project_id == "test_proj"
    assert ctx.retry_count == 0


@pytest.mark.asyncio
async def test_base_agent_run():
    """测试 BaseAgent 的默认 astream 实现。"""
    from comicweaver.agents import LayoutAgent
    from comicweaver.core import BoundingBox, CameraAngle, LayoutInput, PageLayout, PromptPack, ShotSize
    from comicweaver.core import PanelPlan as PP

    agent = LayoutAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    plan = PP(
        panel_id="p001", page_id="page_001", order_in_page=1,
        bbox=BoundingBox(x=0, y=0, width=1, height=1),
        shot_size=ShotSize.MEDIUM, camera_angle=CameraAngle.EYE_LEVEL,
        prompt_pack=PromptPack(positive_prompt="test"),
    )
    page = PageLayout(page_id="page_001", page_number=1, panels=[plan])
    from comicweaver.core import PanelImage
    panel_img = PanelImage(panel_id="p001", image_path="fake.png", width=768, height=1024)
    inp = LayoutInput(pages=[page], panel_images={"p001": panel_img})

    events = []
    async for evt in agent.astream(inp, ctx):
        events.append(evt)

    assert len(events) > 0
    # 最后一个事件应该是 DONE
    assert events[-1].type.value == "done"
    assert "final_pages" in events[-1].content
