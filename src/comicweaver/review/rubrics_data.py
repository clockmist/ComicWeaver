"""审查Rubric数据 - 基于 docs/agents/06-reviewer-agent.md。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RubricDimension:
    dim_id: str
    name: str
    description: str
    weight: float


@dataclass
class Rubric:
    rubric_id: str
    target_agent: str
    version: str
    dimensions: list[RubricDimension]
    pass_threshold: float = 7.0
    escalate_threshold: float = 4.0


SCRIPT_RUBRIC = Rubric(
    rubric_id="rubric_script_v1",
    target_agent="script_agent",
    version="1.0",
    dimensions=[
        RubricDimension("completeness", "完整性",
                        "场景信息是否齐全(角色/地点/对话/动作)", 0.25),
        RubricDimension("coherence", "逻辑连贯性",
                        "前后场景因果是否合理", 0.25),
        RubricDimension("emotion_arc", "情感曲线合理性",
                        "情感起伏与高潮是否合理", 0.20),
        RubricDimension("character_clarity", "角色身份明确性",
                        "角色描述是否具象无歧义", 0.15),
        RubricDimension("visualizability", "可视化程度",
                        "描述是否可被图像生成模型理解", 0.15),
    ],
)

CHARACTER_RUBRIC = Rubric(
    rubric_id="rubric_character_v1",
    target_agent="character_agent",
    version="1.0",
    dimensions=[
        RubricDimension("specificity", "描述具象度",
                        "外观细节是否充分", 0.30),
        RubricDimension("trait_clarity", "视觉特征明确度",
                        "关键特征是否突出", 0.25),
        RubricDimension("style_consistency", "风格一致性",
                        "是否对齐style_preset", 0.20),
        RubricDimension("recognizability", "辨识度",
                        "是否有显著区分度", 0.25),
    ],
)

STORYBOARD_RUBRIC = Rubric(
    rubric_id="rubric_storyboard_v1",
    target_agent="storyboard_agent",
    version="1.0",
    dimensions=[
        RubricDimension("shot_diversity", "景别多样性",
                        "是否使用多种shot_size", 0.20),
        RubricDimension("reading_flow", "阅读流",
                        "阅读顺序是否顺畅", 0.25),
        RubricDimension("emotion_mapping", "情感映射",
                        "布局是否反映情感强度", 0.25),
        RubricDimension("prompt_quality", "提示词质量",
                        "提示词是否具体可执行", 0.30),
    ],
)

IMAGE_RUBRIC = Rubric(
    rubric_id="rubric_image_v1",
    target_agent="image_agent",
    version="1.0",
    dimensions=[
        RubricDimension("character_consistency", "角色一致性",
                        "与base_reference的相似度", 0.35),
        RubricDimension("prompt_match", "提示词匹配",
                        "图像与prompt一致程度", 0.25),
        RubricDimension("artifact_free", "无明显瑕疵",
                        "无畸变/串脸/纯色等问题", 0.20),
        RubricDimension("composition", "构图质量",
                        "视觉重心与平衡", 0.20),
    ],
)

LAYOUT_RUBRIC = Rubric(
    rubric_id="rubric_layout_v1",
    target_agent="layout_agent",
    version="1.0",
    dimensions=[
        RubricDimension("bubble_avoidance", "气泡避让",
                        "是否遮挡关键画面", 0.30),
        RubricDimension("flow", "阅读流",
                        "气泡顺序与画面顺序对齐", 0.25),
        RubricDimension("text_fit", "文字适配",
                        "无溢出/截断", 0.20),
        RubricDimension("aesthetics", "整体美观",
                        "页面整体观感", 0.25),
    ],
)


ALL_RUBRICS: dict[str, Rubric] = {
    r.rubric_id: r
    for r in (
        SCRIPT_RUBRIC,
        CHARACTER_RUBRIC,
        STORYBOARD_RUBRIC,
        IMAGE_RUBRIC,
        LAYOUT_RUBRIC,
    )
}


def get_rubric(rubric_id: str) -> Rubric | None:
    return ALL_RUBRICS.get(rubric_id)
