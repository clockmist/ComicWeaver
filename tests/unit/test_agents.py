"""测试 5 个生产 Agent 的默认本地 fallback。"""
import pytest

from comicweaver.agents import (
    CharacterAgent,
    ImageAgent,
    LayoutAgent,
    ScriptAgent,
    StoryboardAgent,
)
from comicweaver.core import (
    AgentContext,
    CharacterDraft,
    CharacterInput,
    CreationMode,
    ImageInput,
    InteractionMode,
    LayoutInput,
    PageLayout,
    PanelImage,
    PanelPlan,
    Scene,
    ScriptInput,
    StoryboardInput,
)


@pytest.mark.asyncio
async def test_script_agent():
    """ScriptAgent v1.0 — requires LLM. Test output schema with pre-populated story."""
    agent = ScriptAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    # ScriptAgent requires LLM — skip if not available
    if not agent.config.llm.is_available:
        pytest.skip("LLM API not configured — ScriptAgent requires LLM")

    from comicweaver.core import StoryOutput
    story = StoryOutput(
        title="测试",
        story_text="一个少年在雨夜寻找失踪的朋友。",
        characters=[{"name": "少年", "role": "protagonist", "brief_description": "勇敢的年轻人"}],
    )
    inp = ScriptInput(
        creation_mode=CreationMode.SIMPLE,
        story=story,
        target_pages=2,
        target_panels_per_page=3,
    )
    out = await agent.run(inp, ctx)
    assert out.title
    assert len(out.pages) >= 1
    total_panels = sum(len(p.panels) for p in out.pages)
    assert total_panels >= 2


@pytest.mark.asyncio
async def test_character_agent():
    agent = CharacterAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    drafts = [
        CharacterDraft(
            char_id="char_001",
            name="林染",
            role="protagonist",
            appearance="黑色短发,蓝色眼睛",
            personality="好奇",
            first_appearance_scene=0,
        )
    ]
    inp = CharacterInput(
        operation="init",
        character_drafts=drafts,
        style_preset="manga",
    )
    out = await agent.run(inp, ctx)
    assert out.character_db is not None
    assert "char_001" in out.character_db.characters


@pytest.mark.asyncio
async def test_storyboard_agent():
    agent = StoryboardAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    scenes = [
        Scene(
            scene_id="scene_001",
            order=0,
            location="街道",
            time_of_day="夜晚",
            atmosphere="阴郁",
            characters_present=["char_001"],
            dialogues=[],
            emotion_intensity=0.5,
            panel_hint=2,
        )
    ]
    inp = StoryboardInput(
        scenes=scenes,
        emotion_curve=[0.5],
        target_pages=1,
        style_preset="manga",
    )
    out = await agent.run(inp, ctx)
    assert len(out.pages) >= 1
    assert out.total_panels >= 1


@pytest.mark.asyncio
async def test_image_agent():
    agent = ImageAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    from comicweaver.core import (
        BoundingBox,
        CameraAngle,
        PromptPack,
        ShotSize,
    )

    plan = PanelPlan(
        panel_id="p001",
        page_id="page_001",
        order_in_page=1,
        source_scene_id="scene_001",
        bbox=BoundingBox(x=0, y=0, width=1, height=1),
        shot_size=ShotSize.MEDIUM,
        camera_angle=CameraAngle.EYE_LEVEL,
        characters_in_panel=["char_001"],
        prompt_pack=PromptPack(
            positive_prompt="test prompt",
            negative_prompt="",
        ),
    )
    inp = ImageInput(panel_plan=plan, style_preset="manga", width=640, height=896)
    out = await agent.run(inp, ctx)
    assert out.panel_image.panel_id == "p001"
    assert out.panel_image.width == 640
    assert out.panel_image.height == 896
    assert out.panel_image.image_path


@pytest.mark.asyncio
async def test_layout_agent():
    agent = LayoutAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    from comicweaver.core import BoundingBox, CameraAngle, PromptPack, ShotSize

    plan = PanelPlan(
        panel_id="p001",
        page_id="page_001",
        order_in_page=1,
        source_scene_id="scene_001",
        bbox=BoundingBox(x=0, y=0, width=1, height=1),
        shot_size=ShotSize.MEDIUM,
        camera_angle=CameraAngle.EYE_LEVEL,
        characters_in_panel=[],
        prompt_pack=PromptPack(positive_prompt="test", negative_prompt=""),
    )
    page = PageLayout(
        page_id="page_001",
        page_number=1,
        layout_template="full_bleed",
        panels=[plan],
    )
    panel_img = PanelImage(
        panel_id="p001",
        image_path="fake.png",
        width=768,
        height=1024,
    )
    inp = LayoutInput(
        pages=[page],
        panel_images={"p001": panel_img},
        style_preset="manga",
    )
    out = await agent.run(inp, ctx)
    assert len(out.final_pages) == 1
    assert out.overall_metrics.total_pages == 1
