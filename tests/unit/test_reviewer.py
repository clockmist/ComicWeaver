"""测试审查 Agent 与 Rubric 机制。"""
import pytest

from comicweaver.agents import ReviewerAgent
from comicweaver.agents.reviewer_agent import _rubric_payload
from comicweaver.core import AgentContext, CreationMode, InteractionMode, ReviewInput
from comicweaver.review import decide, get_rubric


def test_rubric_registry():
    """测试 Rubric 注册表。"""
    rubric = get_rubric("rubric_script_v1")
    assert rubric is not None
    assert rubric.target_agent == "script_agent"
    assert len(rubric.dimensions) > 0


def test_rubric_payload_serializes_dataclass():
    """API reviewer payload can serialize dataclass rubrics."""
    payload = _rubric_payload(get_rubric("rubric_script_v1"))

    assert payload is not None
    assert payload["rubric_id"] == "rubric_script_v1"
    assert payload["dimensions"][0]["dim_id"] == "completeness"


def test_decision_logic():
    """测试决策器逻辑。"""
    from comicweaver.core import ReviewDecision

    # 高分 -> PASS
    assert decide(8.0, 0, 3) == ReviewDecision.PASS
    # 低分 -> ESCALATE
    assert decide(3.0, 0, 3) == ReviewDecision.ESCALATE
    # 中分 -> REVISE
    assert decide(5.5, 0, 3) == ReviewDecision.REVISE
    # 超过重试次数 -> ESCALATE
    assert decide(5.5, 3, 3) == ReviewDecision.ESCALATE


@pytest.mark.asyncio
async def test_reviewer_agent():
    """测试审查 Agent 的默认本地评分。"""
    agent = ReviewerAgent()
    ctx = AgentContext(
        project_id="test",
        interaction_mode=InteractionMode.FULL_AUTO,
        style_preset="manga",
        creation_mode=CreationMode.SIMPLE,
    )

    # 构造一个假的 script_agent 输出
    target_output = {
        "title": "测试剧本",
        "scenes": [{"scene_id": "s1"}] * 5,
        "characters": [{"char_id": "c1"}, {"char_id": "c2"}],
        "emotion_curve": [0.5] * 5,
    }

    inp = ReviewInput(
        target_agent="script_agent",
        target_rubric_id="rubric_script_v1",
        target_output=target_output,
        retry_count=0,
        max_retries=3,
    )

    out = await agent.run(inp, ctx)
    assert out.feedback is not None
    assert out.schema_check.passed
    assert out.feedback.overall_score > 0
    assert len(out.feedback.dimension_scores) > 0
