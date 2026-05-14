# 审查 Agent (ReviewerAgent)

> 系统的"质量守门人"。横向审查所有生产Agent的输出,执行格式校验、质量评分与返工/升级决策。

## 1. 职责边界

**做什么**:
- 对每个生产Agent的输出执行双层审查(格式 + 质量)
- 按Rubric维度评分并生成改进建议
- 输出PASS/REVISE/ESCALATE决策
- 维护各Agent的重试计数与升级阈值
- 生成用户友好的问题摘要(ESCALATE时)

**不做什么**:
- 不修改生产Agent的输出(只读评分)
- 不直接调用生产Agent(由编排层负责)
- 不参与最终内容生成

## 2. 接口定义

```python
class ReviewerAgent(BaseAgent[ReviewInput, ReviewOutput]):
    name = "reviewer_agent"
    version = "1.0.0"
    rubric_id = "rubric_reviewer_meta"  # 自身不被审查,但记录元信息
```

**特性**: 与其他Agent不同,审查Agent的`run()`即输出`ReviewFeedback`(格式上是`ReviewOutput.feedback`)。

## 3. 输入模型

```python
class ReviewInput(BaseModel):
    target_agent: str                # 被审查的Agent name
    target_rubric_id: str            # 使用的Rubric ID
    target_output: dict              # 被审查的输出(序列化)
    target_inputs: dict              # 被审查Agent的原始输入

    retry_count: int = 0             # 当前重试次数
    max_retries: int = 3
    escalate_threshold: float = 4.0  # 评分低于此值直接ESCALATE
    pass_threshold: float = 7.0

    # 上下文(可选)
    project_context: dict | None = None
```

## 4. 输出模型

```python
class ReviewOutput(BaseModel):
    feedback: ReviewFeedback         # 见00-base.md
    schema_check: SchemaCheckResult
    quality_check: QualityCheckResult
    escalation_summary: EscalationSummary | None  # 仅在ESCALATE时

    meta: AgentOutputMeta


class SchemaCheckResult(BaseModel):
    passed: bool
    errors: list[str]                # Pydantic校验错误列表
    missing_fields: list[str]
    invalid_fields: list[str]


class QualityCheckResult(BaseModel):
    overall_score: float             # 0-10
    dimension_scores: dict[str, float]
    rubric_id: str
    strengths: list[str]             # 优点(供用户参考)
    issues: list[str]
    suggestions: list[str]
    confidence: float                # LLM对自身评分的信心


class EscalationSummary(BaseModel):
    """升级到用户时的友好摘要。"""
    headline: str                    # 一句话说明
    user_options: list[UserOption]   # 用户可选操作
    technical_details: str           # 技术细节(折叠展示)
    suggested_action: str            # 推荐操作


class UserOption(BaseModel):
    option_id: str                   # "force_accept" | "manual_edit" | "retry_with_hint"
    label: str                       # 用户看到的按钮文案
    description: str
    requires_input: bool             # 是否需要用户填写额外内容
```

## 5. 双层审查流程

```
输入: target_output + target_rubric_id
   ↓
[Layer 1] 格式审查(硬性)
   → 加载target Agent的Pydantic模型
   → 校验target_output
   → 失败 → 返回 REVISE(不计入retry_count)
   ↓
[Layer 2] 质量审查(软性)
   → 加载Rubric定义
   → 构造审查Prompt
   → LLM按维度评分
   → 解析评分结果
   ↓
[决策]
   → score ≥ pass_threshold:    PASS
   → score ∈ [escalate, pass):  REVISE(若retry_count < max,否则ESCALATE)
   → score < escalate_threshold: ESCALATE
   ↓
[ESCALATE分支]
   → 生成EscalationSummary
   → 准备UserOption
   ↓
输出: ReviewOutput
```

## 6. Rubric 定义

### 6.1 Rubric 数据结构

```python
class Rubric(BaseModel):
    rubric_id: str
    target_agent: str
    version: str
    dimensions: list[RubricDimension]
    pass_threshold: float = 7.0
    escalate_threshold: float = 4.0


class RubricDimension(BaseModel):
    dim_id: str                      # "completeness"
    name: str                        # "完整性"
    description: str                 # 评分准则
    weight: float                    # 权重(总和=1.0)
    examples: list[ScoringExample]   # Few-shot示例


class ScoringExample(BaseModel):
    score: float
    justification: str
    sample_output_excerpt: str
```

### 6.2 五大Rubric概览

| Rubric ID | 目标Agent | 关键维度 |
|-----------|----------|---------|
| rubric_script_v1 | ScriptAgent | 完整性/连贯性/情感曲线/角色明确性/可视化 |
| rubric_character_v1 | CharacterAgent | 描述具象度/视觉特征明确度/风格一致/辨识度 |
| rubric_storyboard_v1 | StoryboardAgent | 景别多样性/阅读流/情感映射/提示词质量 |
| rubric_image_v1 | ImageAgent | 角色一致性/提示词匹配/无瑕疵/构图质量 |
| rubric_layout_v1 | LayoutAgent | 气泡避让/阅读流/文字适配/整体美观 |

### 6.3 示例: rubric_script_v1

```yaml
rubric_id: rubric_script_v1
target_agent: script_agent
version: "1.0"
pass_threshold: 7.0
escalate_threshold: 4.0
dimensions:
  - dim_id: completeness
    name: 完整性
    weight: 0.25
    description: |
      场景信息是否完整:角色出场、地点、时间、动作、对话齐全
      10分: 所有场景元素饱满,无缺失
      7分: 主要元素完整,个别细节缺失
      4分: 多个场景缺失关键元素
      1分: 大量场景仅有骨架
  - dim_id: coherence
    name: 逻辑连贯性
    weight: 0.25
    description: |
      场景间因果是否合理,角色行为是否一致
  - dim_id: emotion_arc
    name: 情感曲线合理性
    weight: 0.20
    description: |
      情感强度变化是否有起伏,高潮位置是否合理
  - dim_id: character_clarity
    name: 角色身份明确性
    weight: 0.15
    description: |
      每个角色描述是否具象,无歧义
  - dim_id: visualizability
    name: 可视化程度
    weight: 0.15
    description: |
      场景描述是否能被图像生成模型理解
```

## 7. 审查 Prompt 模板

### 7.1 通用模板

```
你是一名严格的{target_agent_role}质量审查员。

【审查任务】
评估以下输出是否符合{target_agent_name}的质量要求。

【原始输入】
{target_inputs_summary}

【待审查输出】
{target_output_summary}

【评分维度】
{rubric_dimensions_yaml}

【输出格式】
请严格输出JSON:
{
  "dimension_scores": {
    "<dim_id>": {"score": <0-10>, "justification": "<理由>"}
  },
  "issues": ["<具体问题>", ...],
  "suggestions": ["<改进建议>", ...],
  "strengths": ["<优点>", ...],
  "confidence": <0-1>
}

注意:
- 分数与justification必须对应
- issues要具体,可被生产Agent理解
- suggestions要可操作
- 不要过分宽松,也不要鸡蛋里挑骨头
```

### 7.2 角色化(每个Rubric对应不同角色)

| Rubric | LLM角色设定 |
|--------|------------|
| script | 资深漫画编剧 |
| character | 角色设计师 |
| storyboard | 资深分镜师 |
| image | 漫画美术指导 |
| layout | 排版编辑 |

## 8. 决策逻辑

```python
def decide(quality: QualityCheckResult, retry_count: int,
           max_retries: int) -> ReviewDecision:
    score = quality.overall_score

    if score >= 7.0:
        return ReviewDecision.PASS

    if score < 4.0:
        return ReviewDecision.ESCALATE

    # 4.0-7.0 之间
    if retry_count >= max_retries:
        return ReviewDecision.ESCALATE
    return ReviewDecision.REVISE
```

## 9. 重试温度递增

为避免重复犯同样错误,审查Agent向编排层建议返工时的参数:

```python
RETRY_TEMPERATURE = {
    0: 0.7,   # 第一次返工
    1: 0.85,  # 第二次返工
    2: 1.0,   # 第三次返工(最后一次)
}
```

## 10. ESCALATE 用户选项

```python
def build_user_options(score, issues):
    options = []

    # 总能选择强制接受
    options.append(UserOption(
        option_id="force_accept",
        label="接受当前结果",
        description=f"质量评分仅{score:.1f},确认接受?",
        requires_input=False,
    ))

    # 总能手动编辑
    options.append(UserOption(
        option_id="manual_edit",
        label="手动编辑",
        description="打开编辑器直接修改输出",
        requires_input=True,
    ))

    # 重试 + 用户提示
    options.append(UserOption(
        option_id="retry_with_hint",
        label="补充提示后重试",
        description="提供额外指引让Agent重新尝试",
        requires_input=True,
    ))

    # 跳过该步骤(若可能)
    if can_skip(target_agent):
        options.append(UserOption(
            option_id="skip",
            label="跳过此步",
            description="使用降级方案继续",
            requires_input=False,
        ))

    return options
```

## 11. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| 启动 | LOG | "审查 {target_agent} 输出..." |
| 格式校验 | LOG | "格式校验: 通过/失败" |
| 质量评分中 | THINKING | LLM token流 |
| 维度评分 | PARTIAL_OUTPUT | 实时显示各维度评分 |
| 决策 | LOG | "决策: PASS/REVISE/ESCALATE" |
| 完成 | DONE | ReviewOutput |

## 12. 性能指标

| 指标 | 目标 |
|------|------|
| 审查 P50 | ≤ 8s |
| LLM token消耗 | ≤ 1500/次 |
| 与人工评分一致率 | ≥ 75% |
| 误判率(False PASS) | ≤ 5% |
| 误判率(False REVISE) | ≤ 15% |

## 13. 防御性设计

### 13.1 LLM输出非法

```python
try:
    quality = parse_quality_json(llm_output)
except (JSONDecodeError, ValidationError):
    # 降级: 使用启发式规则评分
    quality = heuristic_score(target_output)
    quality.confidence = 0.3  # 标记低信心
```

### 13.2 评分极端值检测

若所有维度都是10分或都是0分,触发可疑标记:
- 加大温度重审一次
- 仍异常 → 标记为低信心,信任度归0.5

### 13.3 防止无限循环

编排层硬性约束: 同一Agent单次工作流中最多调用审查Agent `max_retries + 1` 次,超出强制ESCALATE。

## 14. 测试要点

- 每个Rubric的5个分数档位测试样本
- 故意构造低质量输出验证REVISE触发
- 故意构造Schema失败验证Layer 1
- 边界值测试(score=4.0/7.0/10.0)
- LLM输出非法时的降级
- 多次重试后强制ESCALATE
- 与人工评分的一致性测试集

## 15. 数据持久化

```
projects/{project_id}/reviews/
├── {timestamp}_{target_agent}_{retry_count}.json
└── ...
```

每次审查结果都落盘,支持审查历史回溯与质量分析。

---

**文档版本**: v1.0
**负责成员**: A
**依赖**: BaseAgent, 各生产Agent的Pydantic模型
**特殊性**: 横向贯穿,被所有生产Agent的输出触发
