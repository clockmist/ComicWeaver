"""测试核心抽象层 - Schema 与 BaseAgent。"""
import pytest

from comicweaver.core import (
    AgentContext,
    BaseAgent,
    CreationMode,
    InteractionMode,
    ScriptInput,
    ScriptOutput,
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
    from comicweaver.agents import ScriptAgent

    agent = ScriptAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )
    inp = ScriptInput(
        creation_mode=CreationMode.SIMPLE,
        raw_text="测试剧本",
        target_pages=2,
    )

    events = []
    async for evt in agent.astream(inp, ctx):
        events.append(evt)

    assert len(events) > 0
    # 最后一个事件应该是 DONE
    assert events[-1].type.value == "done"
    assert "title" in events[-1].content
