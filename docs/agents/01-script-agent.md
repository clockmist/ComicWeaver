# 剧本理解 Agent (ScriptAgent)

> 系统的"入口网关"。将用户输入(简单Prompt/详细剧本/小说原文)转换为结构化叙事表示。

## 1. 职责边界

**做什么**:
- 解析三种输入模式:简单主题、详细剧本、长文本小说
- 输出结构化的场景/角色/对话/情感曲线
- 进行叙事结构分析(起承转合)
- 提取角色清单与角色画像草稿

**不做什么**:
- 不生成视觉描述(交给分镜Agent)
- 不分配角色参考图(交给角色Agent)
- 不做内容审核(交给审查Agent)

## 2. 接口定义

```python
class ScriptAgent(BaseAgent[ScriptInput, ScriptOutput]):
    name = "script_agent"
    version = "1.0.0"
    rubric_id = "rubric_script_v1"
```

## 3. 输入模型

```python
class ScriptInput(BaseModel):
    creation_mode: Literal["simple", "detailed", "adaptation"]
    raw_text: str                   # 用户原始输入
    target_pages: int = 8           # 期望漫画页数
    target_panels_per_page: int = 4 # 每页期望分镜数
    style_hint: str | None = None   # 风格提示(影响叙事节奏)
    language: str = "zh"            # 输出语言

    # 改编模式专用
    source_format: Literal["novel", "manga"] | None = None
    preserve_dialogue: bool = True  # 改编时是否保留原对话

    # 返工时携带的反馈
    feedback: ReviewFeedback | None = None
```

**输入验证规则**:
- `simple`模式: `raw_text` 长度 ≤ 200字
- `detailed`模式: `raw_text` 长度 200-5000字
- `adaptation`模式: `raw_text` 长度 ≤ 50000字,需指定 `source_format`
- `target_pages` ∈ [1, 50]

## 4. 输出模型

```python
class ScriptOutput(BaseModel):
    # 元信息
    title: str                      # 自动生成的作品标题
    summary: str                    # 一句话简介
    genre: list[str]                # 题材标签

    # 角色清单
    characters: list[CharacterDraft]

    # 场景列表(按叙事顺序)
    scenes: list[Scene]

    # 情感曲线(每个场景一个值)
    emotion_curve: list[float]      # 0.0-1.0,长度==len(scenes)

    # 叙事结构
    narrative_structure: NarrativeStructure

    meta: AgentOutputMeta


class CharacterDraft(BaseModel):
    char_id: str                    # "char_001"
    name: str
    role: Literal["protagonist", "antagonist", "supporting", "extra"]
    appearance: str                 # 外貌描述(供角色Agent生成参考图)
    personality: str
    age_hint: str | None = None
    first_appearance_scene: int     # 首次出场场景索引


class Scene(BaseModel):
    scene_id: str                   # "scene_001"
    order: int                      # 叙事顺序
    location: str                   # 地点描述
    time_of_day: str                # 白天/夜晚/黄昏等
    atmosphere: str                 # 氛围描述
    characters_present: list[str]   # 出场角色char_id列表
    actions: list[Action]           # 角色动作序列
    dialogues: list[Dialogue]       # 对话序列
    narration: str | None = None    # 旁白
    emotion_intensity: float        # 0.0-1.0
    panel_hint: int = 1             # 建议分镜数(供分镜Agent参考)


class Action(BaseModel):
    actor: str                      # char_id或"narrator"
    description: str                # 动作描述
    target: str | None = None       # 动作对象


class Dialogue(BaseModel):
    speaker: str                    # char_id
    text: str
    tone: str                       # 平静/愤怒/喜悦/悲伤等
    is_thought: bool = False        # 是否为内心独白


class NarrativeStructure(BaseModel):
    setup_scenes: list[int]         # 起 - 场景索引
    rising_scenes: list[int]        # 承
    climax_scenes: list[int]        # 转
    resolution_scenes: list[int]    # 合
    pacing: Literal["slow", "medium", "fast", "varied"]
```

## 5. 处理流程

### 5.1 简单模式 (simple)

```
用户输入: "赛博朋克侦探,蒸汽朋克风"
   ↓
[LLM Stage 1] 主题扩展
   → 生成完整大纲(背景/角色/冲突/结局)
   ↓
[LLM Stage 2] 场景拆分
   → 按目标页数拆分为N个场景
   ↓
[LLM Stage 3] 细化填充
   → 为每场景填充角色/动作/对话
   ↓
[规则计算] 情感曲线
   → 基于场景类型分配情感强度
   ↓
[Schema校验] → ScriptOutput
```

### 5.2 详细模式 (detailed)

```
用户输入: 已结构化的剧本文本
   ↓
[LLM] 结构识别
   → 抽取角色/场景/对话
   ↓
[LLM] 缺失补全
   → 推断未明确的氛围/情感
   ↓
[Schema校验] → ScriptOutput
```

### 5.3 改编模式 (adaptation)

```
用户输入: 小说全文(≤50K字)
   ↓
[预处理] 分段
   → 按章节/段落切块
   ↓
[LLM Map] 章节摘要 + 场景候选
   → 每段独立处理(可并行)
   ↓
[LLM Reduce] 全局整合
   → 合并角色清单、过滤冗余场景
   ↓
[LLM] 浓缩到目标页数
   → 选取关键场景,保留核心对话
   ↓
[Schema校验] → ScriptOutput
```

## 6. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| 启动 | LOG | "开始解析剧本..." |
| Stage 1-3 | THINKING | LLM token流 |
| 大纲完成 | PARTIAL_OUTPUT | 大纲文本(供前端预览) |
| 场景拆分完成 | PARTIAL_OUTPUT | 场景标题列表 |
| 角色识别 | PARTIAL_OUTPUT | 角色卡片 |
| 情感曲线计算 | PARTIAL_OUTPUT | 曲线数据 |
| 完成 | DONE | ScriptOutput |
| 各阶段 | PROGRESS | 0.0→1.0 |

## 7. 返工策略

收到 `ReviewFeedback` 时:

| 问题维度 | 返工动作 |
|---------|---------|
| 完整性不足 | 重跑Stage 3,提高细化深度 |
| 逻辑不连贯 | 重跑Stage 2,加入连贯性约束prompt |
| 情感曲线不合理 | 仅重算emotion_curve,保留场景结构 |
| 角色身份不明 | 仅重生CharacterDraft,保留场景 |
| 描述不可视化 | 重跑Stage 3,加入"可视化"约束prompt |

返工时温度参数递增:`0.7 → 0.85 → 1.0`

## 8. LLM Prompt 模板

### 8.1 Stage 1 (主题扩展)

```
你是一名专业漫画编剧。基于以下主题创作完整故事大纲:

主题: {raw_text}
风格: {style_hint}
目标页数: {target_pages}页

要求:
1. 起承转合结构清晰
2. 角色数量 2-5个
3. 包含至少一个情感高潮
4. 输出JSON,字段: title, summary, characters, plot_outline

{feedback_section}
```

### 8.2 反馈注入片段

```python
def build_feedback_section(fb: ReviewFeedback | None) -> str:
    if not fb:
        return ""
    return f"""
你之前的输出存在以下问题需要修正:
{chr(10).join(f"- {issue}" for issue in fb.issues)}

具体改进建议:
{fb.revise_prompt}
"""
```

## 9. 性能指标

| 指标 | 目标 |
|------|------|
| 简单模式 P50 | ≤ 15s |
| 详细模式 P50 | ≤ 25s |
| 改编模式 P50 | ≤ 90s (按10K字计) |
| Schema通过率 | ≥ 95% |
| 审查PASS率(首次) | ≥ 70% |

## 10. 测试要点

- 三种模式的端到端冒烟测试
- 边界场景: 空输入、超长输入、混合中英文
- Schema失败注入测试(LLM返回非法JSON)
- 返工循环测试(模拟REVISE反馈)
- 流式事件顺序与完整性测试

---

**文档版本**: v1.0
**负责成员**: A
**依赖**: BaseAgent (00-base.md)
**下游**: 角色管理Agent、分镜设计Agent
