"""共享数据模型 - 对应 docs/agents/00-base.md 与各Agent文档中的 Pydantic 模型。

本模块定义了所有Agent间流转的标准化数据结构。
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ===========================================================================
# 枚举类型
# ===========================================================================

class InteractionMode(str, Enum):
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"
    FULL_INTERACTIVE = "full_interactive"


class CreationMode(str, Enum):
    SIMPLE = "simple"
    DETAILED = "detailed"
    ADAPTATION = "adaptation"


class StreamEventType(str, Enum):
    THINKING = "thinking"
    PROGRESS = "progress"
    PARTIAL_OUTPUT = "partial"
    LOG = "log"
    ERROR = "error"
    DONE = "done"


class ReviewDecision(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    ESCALATE = "escalate"


class ShotSize(str, Enum):
    EXTREME_LONG = "extreme_long"
    LONG = "long"
    FULL = "full"
    MEDIUM = "medium"
    CLOSE = "close"
    EXTREME_CLOSE = "extreme_close"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    HIGH = "high_angle"
    LOW = "low_angle"
    DUTCH = "dutch_angle"


class PanelShape(str, Enum):
    RECTANGLE = "rectangle"
    DIAGONAL = "diagonal"
    BLEED = "bleed"
    SPLASH = "splash"
    CIRCULAR = "circular"
    JAGGED = "jagged"


# ===========================================================================
# 通用元信息与流式事件
# ===========================================================================

class AgentOutputMeta(BaseModel):
    agent: str
    version: str
    generated_at: float = Field(default_factory=time.time)
    inputs_hash: str = ""
    retry_count: int = 0
    self_check_notes: list[str] = Field(default_factory=list)


class StreamEvent(BaseModel):
    agent: str
    type: StreamEventType
    content: Any = None
    timestamp: float = Field(default_factory=time.time)
    metadata: dict = Field(default_factory=dict)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


# ===========================================================================
# 审查反馈
# ===========================================================================

class ReviewFeedback(BaseModel):
    decision: ReviewDecision
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    revise_prompt: str = ""
    rubric_id: str = ""
    reviewer_version: str = "0.1.0"


# ===========================================================================
# 剧本理解 Agent 模型
# ===========================================================================

class CharacterDraft(BaseModel):
    char_id: str
    name: str
    role: Literal["protagonist", "antagonist", "supporting", "extra"] = "supporting"
    appearance: str = ""
    personality: str = ""
    age_hint: str | None = None
    first_appearance_scene: int = 0


class Action(BaseModel):
    actor: str
    description: str
    target: str | None = None


class Dialogue(BaseModel):
    speaker: str
    text: str
    tone: str = "neutral"
    is_thought: bool = False


class Scene(BaseModel):
    """[DEPRECATED v1.0] 保留用于 StoryboardAgent 过渡期，将被 PanelTask 替代。"""
    scene_id: str
    order: int
    location: str = ""
    time_of_day: str = ""
    atmosphere: str = ""
    characters_present: list[str] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    dialogues: list[Dialogue] = Field(default_factory=list)
    narration: str | None = None
    emotion_intensity: float = 0.5
    panel_hint: int = 1
    visual_hook: str = ""
    shot_sequence_hint: str = ""


class NarrativeStructure(BaseModel):
    setup_scenes: list[int] = Field(default_factory=list)
    rising_scenes: list[int] = Field(default_factory=list)
    climax_scenes: list[int] = Field(default_factory=list)
    resolution_scenes: list[int] = Field(default_factory=list)
    pacing: Literal["slow", "medium", "fast", "varied"] = "medium"


# ===========================================================================
# 面板级剧本模型 (v1.0)
# ===========================================================================

class PanelTask(BaseModel):
    """单格面板的叙事任务 — ScriptAgent v1.0 的核心输出单元。

    定义这一格需要完成的叙事功能，不涉及视觉/镜头设计。
    """
    panel_id: str = ""                    # "page_001_p01"
    page_number: int = 1
    order_in_page: int = 1

    # 叙事功能（自由文本）
    narrative_purpose: str = ""           # "建立场景：雨夜的城市街道"

    # 谁、做什么、说什么
    character_id: str = ""                # 空 = 纯场景/转场格，无角色登场
    character_action: str = ""            # 角色在做什么（叙事动作，非镜头描述）
    dialogue_text: str = ""               # 这一格的对白
    dialogue_tone: str = ""               # 对白情感语气
    is_thought: bool = False              # 内心独白
    narration: str = ""                   # 旁白文字

    # 情感与节奏
    emotion: str = ""                     # tension / sorrow / joy / fear / determination / ...
    emotion_intensity: float = 0.5        # 0.0 ~ 1.0
    is_key_panel: bool = False            # 本页焦点格

    # 场景上下文（叙事层面，非视觉层面）
    location: str = ""                    # 发生在哪
    time_of_day: str = ""                 # 时间段
    atmosphere: str = ""                  # 场景氛围


class ScriptPage(BaseModel):
    """一页漫画的面板列表。"""
    page_number: int
    panels: list[PanelTask] = Field(default_factory=list)
    page_note: str = ""                   # 页面级叙事说明


# ===========================================================================
# 故事开发 Agent 模型 (v0.4)
# ===========================================================================

class StoryOutput(BaseModel):
    """StoryAgent v0.2 输出：完整的叙事故事（散文体，非结构化脚本）。"""
    title: str = ""
    author_note: str = ""           # 1-2 句高概念 / logline
    tone: str = ""                  # 叙事基调 (如 "dark noir mystery")
    genre: list[str] = Field(default_factory=list)
    story_text: str = ""            # ★ 核心输出：完整叙事故事，可读的散文体
    characters: list[dict] = Field(default_factory=list)
    # [{name, role, brief_description}] — 轻量角色，详细设计留给 CharacterAgent
    core_conflict: str = ""         # 核心戏剧冲突
    setting: str = ""               # 时间/地点/世界观
    target_pages: int = 4
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="story_agent", version="0.2.0"
    ))


class StoryInput(BaseModel):
    """StoryAgent 输入：用户的原始故事创意。"""
    raw_text: str
    creation_mode: CreationMode = CreationMode.SIMPLE
    target_pages: int = 4
    genre_hint: str | None = None
    style_hint: str | None = None
    language: str = "zh"
    feedback: ReviewFeedback | None = None


class ScriptInput(BaseModel):
    """ScriptAgent v1.0 输入：故事 + 角色设定 → 面板级剧本。"""
    creation_mode: CreationMode = CreationMode.SIMPLE
    raw_text: str = ""                                          # 兼容旧路径
    story: StoryOutput | None = None                            # 故事上下文
    character_db: "CharacterDB | None" = None                   # v1.0: 角色设定
    target_pages: int = 4
    target_panels_per_page: int = 4
    style_hint: str | None = None
    language: str = "zh"
    feedback: ReviewFeedback | None = None


class ScriptOutput(BaseModel):
    """ScriptAgent v1.0 输出：按页组织的面板级叙事任务。"""
    title: str = "Untitled"
    summary: str = ""
    genre: list[str] = Field(default_factory=list)
    pages: list[ScriptPage] = Field(default_factory=list)       # v1.0: 替代 scenes
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="script_agent", version="1.0.0"
    ))
    # [DEPRECATED v0.x] 以下字段保留向后兼容，新代码不应使用:
    characters: list[CharacterDraft] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    emotion_curve: list[float] = Field(default_factory=list)
    narrative_structure: NarrativeStructure = Field(default_factory=NarrativeStructure)


# ===========================================================================
# 角色管理 Agent 模型
# ===========================================================================

class ReferenceImage(BaseModel):
    image_id: str
    image_path: str
    source: Literal["generated", "uploaded", "panel"] = "generated"
    generation_prompt: str | None = None
    confidence: float = 0.8
    created_at: float = Field(default_factory=time.time)


class VisualTraits(BaseModel):
    hair: str = ""
    eyes: str = ""
    body: str = ""
    clothing: str = ""
    distinctive: list[str] = Field(default_factory=list)


class ArchiveEntry(BaseModel):
    entry_id: str
    panel_id: str
    image_path: str
    pose: str = ""
    expression: str = ""
    quality_score: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class CharacterProfile(BaseModel):
    char_id: str
    name: str
    base_reference: ReferenceImage
    archive: list[ArchiveEntry] = Field(default_factory=list)
    visual_traits: VisualTraits = Field(default_factory=VisualTraits)
    clip_embedding_id: str = ""
    style_preset: str = ""
    locked: bool = False
    # 角色一致性：固定种子 + 特征 prompt（每次生成相同角色时复用）
    seed: int = 0
    appearance_prompt: str = ""
    # Animagine-XL-4.0 角色数量标签（1girl / 1boy / 1other）
    gender_tag: str = "1girl"
    # LLM 生成的 Danbooru 风格角色 Tag（不含画风/质量 Tag）
    core_tags: str = ""


class CharacterDB(BaseModel):
    project_id: str
    characters: dict[str, CharacterProfile] = Field(default_factory=dict)
    version: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class WeightedReference(BaseModel):
    image_path: str
    weight: float
    role: str = "base"
    embedding_id: str | None = None


class ReferenceWindow(BaseModel):
    panel_id: str
    char_id: str
    references: list[WeightedReference] = Field(default_factory=list)
    window_strategy: Literal["chain", "base_only", "fallback"] = "chain"


class CharacterInput(BaseModel):
    operation: Literal["init", "build_reference_window", "update_archive"] = "init"
    character_drafts: list[CharacterDraft] | None = None
    style_preset: str | None = None
    panel_id: str | None = None
    target_characters: list[str] | None = None
    feedback: ReviewFeedback | None = None


class CharacterOutput(BaseModel):
    operation: str
    character_db: CharacterDB | None = None
    reference_windows: dict[str, ReferenceWindow] = Field(default_factory=dict)
    archive_updated: bool = False
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="character_agent", version="0.1.0"
    ))


# ===========================================================================
# 分镜设计 Agent 模型
# ===========================================================================

class PromptPack(BaseModel):
    positive_prompt: str = ""
    negative_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    composition_tags: list[str] = Field(default_factory=list)
    quality_tags: list[str] = Field(default_factory=list)
    seed_hint: int | None = None


class BubbleHint(BaseModel):
    dialogue_index: int
    suggested_position: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right", "center", "auto"
    ] = "auto"
    bubble_type: Literal["speech", "thought", "shout", "whisper", "narration"] = "speech"
    avoid_zones: list[BoundingBox] = Field(default_factory=list)


class PreviousPanelContext(BaseModel):
    """前一面板的上下文，用于保持视觉连续性 (v0.4, v0.6: +narrative_purpose)。"""
    panel_id: str = ""
    shot_size: str = ""             # ShotSize value
    camera_angle: str = ""          # CameraAngle value
    characters_in_panel: list[str] = Field(default_factory=list)
    pose_hint: str = ""             # 前一格的动作姿态
    expression: str = ""            # 前一格的表情
    setting_summary: str = ""       # 前一格的场景/背景
    emotion: str = ""               # 前一格的情绪 (mood)
    emotion_intensity: float = 0.5
    narrative_purpose: str = ""     # v0.6: 前一格的叙事目的 — 面板间叙事连续性


class PanelPlan(BaseModel):
    panel_id: str
    page_id: str
    order_in_page: int
    source_scene_id: str = ""
    source_dialogue_indices: list[int] = Field(default_factory=list)
    bbox: BoundingBox = Field(default_factory=lambda: BoundingBox(x=0, y=0, width=1.0, height=1.0))
    shape: PanelShape = PanelShape.RECTANGLE
    size_ratio: float = 0.25
    shot_size: ShotSize = ShotSize.MEDIUM
    camera_angle: CameraAngle = CameraAngle.EYE_LEVEL
    characters_in_panel: list[str] = Field(default_factory=list)
    primary_action: str = ""
    setting: str = ""
    mood: str = ""
    emotion_intensity: float = 0.5
    prompt_pack: PromptPack = Field(default_factory=PromptPack)
    dialogues_in_panel: list[Dialogue] = Field(default_factory=list)
    speech_bubble_hints: list[BubbleHint] = Field(default_factory=list)
    # LLM 生成的分镜细节（StoryboardAgent 通过 LLM 设计）
    pose_hint: str = ""          # 角色身体姿势描述
    expression: str = ""         # 角色面部表情
    scene_lighting: str = ""     # 场景光照描述
    weather: str = ""            # 天气
    time_of_day: str = ""        # 时间段
    # v0.4: 前格上下文 — 保持面板间视觉连续性
    previous_panel_context: PreviousPanelContext | None = None


class PageLayout(BaseModel):
    page_id: str
    page_number: int
    layout_template: str = "grid_2x2"
    layout_hint: str = "standard"  # "standard" | "climax" | "action" | "dialogue" | "establishing"
    panels: list[PanelPlan] = Field(default_factory=list)
    page_emotion_avg: float = 0.5
    is_climax_page: bool = False


class StoryboardInput(BaseModel):
    scenes: list[Scene] = Field(default_factory=list)           # v0.6: 已弃用，保留兼容
    pages: list[ScriptPage] = Field(default_factory=list)       # v0.6: 直接消费 PanelTask（替代 scenes）
    emotion_curve: list[float] = Field(default_factory=list)
    character_db: CharacterDB | None = None
    style_preset: str = "manga"
    target_pages: int = 4
    target_panels_per_page: int = 4
    feedback: ReviewFeedback | None = None
    # v0.4: 故事上下文 — 让 StoryboardAgent 知道叙事阶段
    story_summary: str = ""                                     # 从 StoryOutput.summary
    narrative_structure: NarrativeStructure | None = None       # 当前叙事阶段


class StoryboardOutput(BaseModel):
    pages: list[PageLayout] = Field(default_factory=list)
    total_panels: int = 0
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="storyboard_agent", version="0.1.0"
    ))


# ===========================================================================
# 图像生成 Agent 模型
# ===========================================================================

class PanelImage(BaseModel):
    panel_id: str
    image_path: str
    image_format: Literal["png", "webp"] = "png"
    width: int = 768
    height: int = 1024
    backend: str = "placeholder"
    model_version: str = "0.0"
    seed: int = 0
    steps: int = 0
    cfg_scale: float = 0.0
    sampler: str = "api"
    generation_time_ms: int = 0
    characters_present: list[str] = Field(default_factory=list)
    prompt_used: str = ""
    negative_prompt_used: str = ""
    consistency_scores: dict[str, float] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class SelfCheckResult(BaseModel):
    matches_prompt: float = 0.0
    character_consistency: float = 0.0
    artifact_warnings: list[str] = Field(default_factory=list)
    suggested_retry: bool = False


class ImageInput(BaseModel):
    panel_plan: PanelPlan
    reference_windows: dict[str, ReferenceWindow] = Field(default_factory=dict)
    character_db: CharacterDB | None = None  # 角色数据库（用于读取固定种子和外观 prompt）
    style_preset: str = "manga"
    backend_preference: str = "auto"
    quality_target: Literal["draft", "standard", "high"] = "standard"
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    feedback: ReviewFeedback | None = None


class ImageOutput(BaseModel):
    panel_image: PanelImage
    backend_used: str = "placeholder"
    fallback_chain: list[str] = Field(default_factory=list)
    self_check: SelfCheckResult = Field(default_factory=SelfCheckResult)
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="image_agent", version="0.1.0"
    ))


# ===========================================================================
# 排版合成 Agent 模型
# ===========================================================================

class FontConfig(BaseModel):
    speech_font: str = "default"
    thought_font: str = "default"
    narration_font: str = "default"
    base_size_pt: int = 12
    min_size_pt: int = 8
    max_size_pt: int = 20


class PlacedBubble(BaseModel):
    bubble_id: str
    panel_id: str
    dialogue_index: int = 0
    bubble_type: str = "speech"
    bbox: BoundingBox
    text: str = ""
    font_size_pt: int = 12
    occlusion_score: float = 0.0


class FinalPage(BaseModel):
    page_id: str
    page_number: int
    image_path: str
    width_px: int = 2480  # A4@300dpi
    height_px: int = 3508
    bubbles: list[PlacedBubble] = Field(default_factory=list)
    layout_warnings: list[str] = Field(default_factory=list)


class ExportArtifact(BaseModel):
    format: str
    file_path: str
    file_size_bytes: int = 0
    page_count: int = 0


class BubblePlacementResult(BaseModel):
    """单个气泡放置结果 — BubbleAgent 输出的气泡坐标和样式参数。"""
    panel_id: str
    dialogue_index: int
    bubble_type: str = "speech"
    speaker: str = ""
    text: str = ""
    x: float = 0.0              # 归一化坐标 (0-1), 相对整页
    y: float = 0.0
    w: float = 0.1
    h: float = 0.05
    font_size_pt: int = 14
    tail_direction: str = "auto"
    occlusion_score: float = 0.0
    face_count: int = 0         # 该面板检测到的人脸数（日志用）


class BubbleInput(BaseModel):
    """BubbleAgent 输入"""
    pages: list[PageLayout]
    panel_images: dict[str, PanelImage] = Field(default_factory=dict)
    font_config: FontConfig = Field(default_factory=FontConfig)
    page_width_px: int = 1240
    page_height_px: int = 1754
    margin_px: int = 40
    gutter_px: int = 10
    reading_direction: Literal["ltr", "rtl"] = "ltr"
    feedback: ReviewFeedback | None = None


class BubbleOutput(BaseModel):
    """BubbleAgent 输出"""
    bubble_placements: dict[str, list[BubblePlacementResult]] = Field(default_factory=dict)
    total_bubbles: int = 0
    face_detection_stats: dict[str, int] = Field(default_factory=dict)
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="bubble_agent", version="0.1.0"
    ))


class LayoutInput(BaseModel):
    pages: list[PageLayout]
    panel_images: dict[str, PanelImage] = Field(default_factory=dict)
    bubble_placements: dict[str, list[BubblePlacementResult]] = Field(default_factory=dict)
    style_preset: str = "manga"
    font_config: FontConfig = Field(default_factory=FontConfig)
    export_formats: list[str] = Field(default_factory=lambda: ["png"])
    feedback: ReviewFeedback | None = None
    # Page geometry
    page_width_px: int = 1240       # A4 @150dpi
    page_height_px: int = 1754
    margin_px: int = 40
    gutter_px: int = 10
    reading_direction: Literal["ltr", "rtl"] = "ltr"


class LayoutMetrics(BaseModel):
    total_pages: int = 0
    total_bubbles: int = 0
    avg_occlusion_score: float = 0.0
    bubbles_overflow_count: int = 0


class LayoutOutput(BaseModel):
    final_pages: list[FinalPage] = Field(default_factory=list)
    exports: list[ExportArtifact] = Field(default_factory=list)
    overall_metrics: LayoutMetrics = Field(default_factory=LayoutMetrics)
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="layout_agent", version="0.1.0"
    ))


# ===========================================================================
# 审查 Agent 模型
# ===========================================================================

class SchemaCheckResult(BaseModel):
    passed: bool = True
    errors: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)


class QualityCheckResult(BaseModel):
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    rubric_id: str = ""
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class UserOption(BaseModel):
    option_id: str
    label: str
    description: str = ""
    requires_input: bool = False


class EscalationSummary(BaseModel):
    headline: str
    user_options: list[UserOption] = Field(default_factory=list)
    technical_details: str = ""
    suggested_action: str = ""


class ReviewInput(BaseModel):
    target_agent: str
    target_rubric_id: str
    target_output: dict
    target_inputs: dict = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    escalate_threshold: float = 4.0
    pass_threshold: float = 7.0


class ReviewOutput(BaseModel):
    feedback: ReviewFeedback
    schema_check: SchemaCheckResult = Field(default_factory=SchemaCheckResult)
    quality_check: QualityCheckResult = Field(default_factory=QualityCheckResult)
    escalation_summary: EscalationSummary | None = None
    meta: AgentOutputMeta = Field(default_factory=lambda: AgentOutputMeta(
        agent="reviewer_agent", version="0.1.0"
    ))


# ===========================================================================
# Agent 上下文与回调
# ===========================================================================

StreamCallback = Callable[[StreamEvent], Awaitable[None]]


class AgentContext(BaseModel):
    """跨Agent共享的运行时上下文。"""
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    interaction_mode: InteractionMode = InteractionMode.SEMI_AUTO
    style_preset: str = "manga"
    creation_mode: CreationMode = CreationMode.SIMPLE
    retry_count: int = 0

    model_config = {"arbitrary_types_allowed": True}
