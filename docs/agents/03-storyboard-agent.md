# 分镜设计 Agent (StoryboardAgent)

> 系统的"视觉导演"。将结构化剧本转化为可执行的分镜规划,决定每一格的景别、角度、布局与提示词。

## 1. 职责边界

**做什么**:
- 为每个场景规划分镜数量与切分
- 应用视觉语法规则(景别/角度/构图)
- 基于情感曲线的动态布局生成
- 为每个分镜生成图像生成的核心提示词
- 优化阅读流(视线引导)

**不做什么**:
- 不调用图像生成模型(交给图像Agent)
- 不处理对话框排版(交给排版Agent)
- 不修改剧本内容(只读)

## 2. 接口定义

```python
class StoryboardAgent(BaseAgent[StoryboardInput, StoryboardOutput]):
    name = "storyboard_agent"
    version = "1.0.0"
    rubric_id = "rubric_storyboard_v1"
```

## 3. 输入模型

```python
class StoryboardInput(BaseModel):
    scenes: list[Scene]              # 来自ScriptOutput
    emotion_curve: list[float]
    character_db: CharacterDB        # 来自CharacterOutput
    style_preset: str
    target_pages: int
    target_panels_per_page: int      # 默认值,可被情感曲线覆盖
    page_size: PageSize = PageSize.A4

    feedback: ReviewFeedback | None = None
```

## 4. 输出模型

```python
class StoryboardOutput(BaseModel):
    pages: list[PageLayout]          # 每页分镜规划
    total_panels: int
    reading_flow: ReadingFlow

    meta: AgentOutputMeta


class PageLayout(BaseModel):
    page_id: str
    page_number: int
    layout_template: LayoutTemplate  # 布局模板ID
    panels: list[PanelPlan]
    page_emotion_avg: float          # 该页平均情感强度
    is_climax_page: bool             # 是否为高潮页


class PanelPlan(BaseModel):
    panel_id: str                    # "p001_01" (page1的第1格)
    page_id: str
    order_in_page: int               # 阅读顺序(1开始)

    # 来源映射
    source_scene_id: str
    source_dialogue_indices: list[int]  # 该格涵盖的对话索引

    # 布局
    bbox: BoundingBox                # 在页面中的位置
    shape: PanelShape                # 矩形/斜切/异形
    size_ratio: float                # 占页面面积比例

    # 视觉语法
    shot_size: ShotSize              # 远景/全景/中景/近景/特写
    camera_angle: CameraAngle        # 平视/俯视/仰视/倾斜

    # 内容
    characters_in_panel: list[str]   # char_id列表
    primary_action: str              # 主要动作描述
    setting: str                     # 环境描述
    mood: str                        # 氛围
    emotion_intensity: float

    # 给图像Agent的提示词包
    prompt_pack: PromptPack

    # 给排版Agent的对话信息
    dialogues_in_panel: list[Dialogue]
    speech_bubble_hints: list[BubbleHint]


class BoundingBox(BaseModel):
    x: float                         # 0.0-1.0,相对页面
    y: float
    width: float
    height: float


class PanelShape(str, Enum):
    RECTANGLE = "rectangle"
    DIAGONAL = "diagonal"            # 斜切
    BLEED = "bleed"                  # 出血
    SPLASH = "splash"                # 跨页/全页
    CIRCULAR = "circular"            # 圆形(回忆)
    JAGGED = "jagged"                # 锯齿(冲击/对话)


class ShotSize(str, Enum):
    EXTREME_LONG = "extreme_long"    # 大远景
    LONG = "long"                    # 远景
    FULL = "full"                    # 全景
    MEDIUM = "medium"                # 中景
    CLOSE = "close"                  # 近景
    EXTREME_CLOSE = "extreme_close"  # 特写


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    HIGH = "high_angle"              # 俯视
    LOW = "low_angle"                # 仰视
    DUTCH = "dutch_angle"            # 倾斜


class PromptPack(BaseModel):
    """传给图像Agent的完整提示词包。"""
    positive_prompt: str             # 正向提示词
    negative_prompt: str             # 负向提示词
    style_tags: list[str]            # 风格标签
    composition_tags: list[str]      # 构图标签
    quality_tags: list[str]          # 质量标签
    seed_hint: int | None = None     # 建议seed(用于可复现性)


class BubbleHint(BaseModel):
    """对话气泡的位置建议。"""
    dialogue_index: int
    suggested_position: Literal["top_left", "top_right", "bottom_left",
                                  "bottom_right", "center", "auto"]
    bubble_type: Literal["speech", "thought", "shout", "whisper", "narration"]
    avoid_zones: list[BoundingBox]   # 不可遮挡的区域(角色脸/关键物)


class LayoutTemplate(str, Enum):
    GRID_2X2 = "grid_2x2"            # 2行2列规整
    GRID_3X2 = "grid_3x2"
    GRID_3X3 = "grid_3x3"
    SPLASH_TOP = "splash_top"        # 顶部大格
    SPLASH_BOTTOM = "splash_bottom"
    DIAGONAL = "diagonal"            # 斜切动感
    FULL_BLEED = "full_bleed"        # 整页出血
    CHAOTIC = "chaotic"              # 混乱布局(高潮)


class ReadingFlow(BaseModel):
    """阅读流分析结果。"""
    direction: Literal["LTR", "RTL", "TTB"]  # 默认LTR
    flow_score: float                # 顺畅度0-1
    ambiguity_warnings: list[str]    # 阅读顺序歧义警告
```

## 5. 处理流程

```
输入: 场景列表 + 情感曲线
   ↓
[Stage 1] 场景→分镜数映射
   → 基于emotion_intensity分配panel_hint
   → 高潮场景多分镜(4-6格),低谷少分镜(1-2格)
   ↓
[Stage 2] 分页规划
   → 总分镜数 / target_panels_per_page → 页数
   → 高潮分镜尽量集中在同一页
   ↓
[Stage 3] 每页布局选择
   → 情感曲线区间 → LayoutTemplate
   → 高潮页用SPLASH/DIAGONAL,普通页用GRID
   ↓
[Stage 4] 视觉语法应用
   → 为每个panel选择shot_size/camera_angle
   → 应用规则: 对话→中近景, 动作→中景, 环境→远景
   ↓
[Stage 5] 提示词生成 (LLM)
   → 拼装PromptPack
   ↓
[Stage 6] 阅读流优化
   → 检测视线跳跃距离
   → 标注歧义警告
   ↓
[Schema校验] → StoryboardOutput
```

## 6. 核心算法

### 6.1 情感→布局映射

```python
def select_layout(page_emotion_avg, max_panel_emotion):
    if max_panel_emotion >= 0.85:
        return LayoutTemplate.SPLASH_TOP if page_emotion_avg > 0.6 \
               else LayoutTemplate.FULL_BLEED
    if max_panel_emotion >= 0.6:
        return LayoutTemplate.DIAGONAL
    if page_emotion_avg < 0.3:
        return LayoutTemplate.GRID_3X3   # 低谷:规整密集
    return LayoutTemplate.GRID_2X2       # 标准
```

### 6.2 景别选择规则

```python
SHOT_RULES = {
    # (场景类型, 角色数, 情感强度) → ShotSize
    ("dialogue", 2, "low"): ShotSize.MEDIUM,
    ("dialogue", 2, "high"): ShotSize.CLOSE,
    ("action", "many", "high"): ShotSize.LONG,
    ("emotion", 1, "high"): ShotSize.EXTREME_CLOSE,
    ("setting", 0, "any"): ShotSize.LONG,
    ("transition", "any", "any"): ShotSize.EXTREME_LONG,
}
```

### 6.3 视觉重量平衡

页面内分镜大小分配遵循:
- 高潮分镜面积占比 30-50%
- 标准分镜面积占比 15-25%
- 过渡分镜面积占比 10-15%
- 单页所有分镜面积总和 = 100% ± 5%(允许间隙)

### 6.4 阅读流验证

```python
def validate_flow(panels):
    warnings = []
    for i in range(len(panels) - 1):
        curr_center = panels[i].bbox.center()
        next_center = panels[i+1].bbox.center()
        # 阅读方向LTR: 期望next在curr右下方
        if next_center.x < curr_center.x and next_center.y < curr_center.y:
            warnings.append(f"Panel {i}→{i+1} 阅读顺序可能产生歧义")
    return warnings
```

## 7. 提示词生成模板

```
[正向]: {style_preset_tags},
       {shot_size_tag} of {characters_desc} in {setting},
       {primary_action}, {mood} mood,
       {camera_angle_tag},
       {composition_tags},
       masterpiece, best quality

[负向]: {style_preset_negative},
       low quality, blurry, distorted,
       multiple panels, comic page,  # 避免生成漫画页面而非单格
       text, watermark
```

## 8. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| Stage 1 | PROGRESS | 0.0→0.15 |
| Stage 2 | PROGRESS | →0.30 |
| Stage 3 | PARTIAL_OUTPUT | 布局缩略图(纯框架) |
| Stage 4 | THINKING | 视觉语法决策思路 |
| Stage 5 | PARTIAL_OUTPUT | 提示词预览 |
| Stage 6 | LOG | 阅读流警告(如有) |
| 完成 | DONE | StoryboardOutput |

## 9. 返工策略

| 审查问题 | 返工动作 |
|---------|---------|
| 景别多样性不足 | 强制要求至少使用4种shot_size |
| 阅读流不顺 | 切换到规整GRID模板 |
| 情感曲线未体现 | 重跑Stage 3,加大模板差异 |
| 提示词模糊 | 重跑Stage 5,加入具体细节约束 |
| 单页分镜过多 | 调整target_panels_per_page,重新分页 |

## 10. 性能指标

| 指标 | 目标 |
|------|------|
| 8页 P50 | ≤ 30s |
| 提示词Token数 | 每panel 80-200 tokens |
| 阅读流警告率 | ≤ 10% |
| 景别多样性熵 | ≥ 1.5 (5类近均匀) |

## 11. 测试要点

- 极端情感曲线(全平/全高潮)的布局退化
- 单角色 vs 多角色场景的景别选择
- 跨页布局(splash)的页码分配
- 阅读流警告的误报率
- 返工时保留已确认panel的能力

---

**文档版本**: v1.0
**负责成员**: C
**依赖**: BaseAgent, ScriptOutput, CharacterDB
**下游**: 图像生成Agent(消费PromptPack), 排版合成Agent(消费BubbleHint+bbox)
