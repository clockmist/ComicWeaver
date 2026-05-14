# BaseAgent 与共享类型规范

> 所有Agent的公共契约。所有生产Agent与审查Agent都必须遵循本文档定义的基类与类型。

## 1. 设计原则

- **统一接口**: 所有Agent暴露相同的 `run()` / `astream()` 方法
- **结构化I/O**: 输入输出必须为Pydantic模型,禁止裸dict
- **流式优先**: 默认支持异步流式输出,适配交互反馈层
- **可审查**: 输出必须可被审查Agent按Rubric评分
- **可重试**: 支持携带审查反馈的返工调用

## 2. BaseAgent 抽象类

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, TypeVar
from pydantic import BaseModel

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)

class BaseAgent(ABC, Generic[TIn, TOut]):
    """所有Agent的基类。

    类型参数:
        TIn: 输入模型(Pydantic)
        TOut: 输出模型(Pydantic)
    """

    name: str               # Agent唯一标识,如 "script_agent"
    version: str            # 语义化版本
    rubric_id: str          # 对应的审查Rubric ID

    @abstractmethod
    async def run(
        self,
        inputs: TIn,
        context: AgentContext,
    ) -> TOut:
        """同步执行,一次返回完整结果。"""

    @abstractmethod
    async def astream(
        self,
        inputs: TIn,
        context: AgentContext,
    ) -> AsyncIterator[StreamEvent]:
        """流式执行,逐步产出中间事件。"""

    async def revise(
        self,
        inputs: TIn,
        previous_output: TOut,
        review_feedback: ReviewFeedback,
        context: AgentContext,
    ) -> TOut:
        """携带审查反馈的返工。默认实现:重新run并注入feedback。"""
        return await self.run(
            inputs.with_feedback(review_feedback),
            context,
        )
```

## 3. 共享类型

### 3.1 AgentContext

```python
class AgentContext(BaseModel):
    """跨Agent共享的运行时上下文。"""
    project_id: str                  # 项目唯一ID
    session_id: str                  # 当前会话ID
    interaction_mode: InteractionMode  # semi_auto | full_auto | full_interactive
    style_preset: str                # 风格预设
    creation_mode: str               # simple | detailed | adaptation
    retry_count: int = 0             # 当前Agent返工次数
    state_ref: ComicStateRef         # 全局状态引用(只读快照)
    stream_callback: StreamCallback | None = None  # 流式回调
```

### 3.2 StreamEvent

```python
class StreamEventType(str, Enum):
    THINKING = "thinking"           # LLM token流
    PROGRESS = "progress"           # 进度更新(0.0-1.0)
    PARTIAL_OUTPUT = "partial"      # 中间产物预览
    LOG = "log"                     # 日志消息
    ERROR = "error"                 # 错误信息
    DONE = "done"                   # 完成事件

class StreamEvent(BaseModel):
    agent: str                      # 来源Agent名
    type: StreamEventType
    content: Any                    # 事件载荷
    timestamp: float
    metadata: dict = {}
```

### 3.3 ReviewFeedback

```python
class ReviewDecision(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    ESCALATE = "escalate"

class ReviewFeedback(BaseModel):
    """审查Agent返回的反馈,用于驱动返工。"""
    decision: ReviewDecision
    overall_score: float            # 0.0-10.0
    dimension_scores: dict[str, float]  # 各维度评分
    issues: list[str]               # 发现的问题
    suggestions: list[str]          # 改进建议
    revise_prompt: str              # 注入返工的提示词片段
    rubric_id: str                  # 使用的Rubric ID
    reviewer_version: str
```

### 3.4 InteractionMode

```python
class InteractionMode(str, Enum):
    SEMI_AUTO = "semi_auto"         # 关键节点暂停(默认)
    FULL_AUTO = "full_auto"         # 全程不暂停
    FULL_INTERACTIVE = "full_interactive"  # 每个Agent都暂停
```

## 4. 错误处理契约

### 4.1 异常分类

```python
class AgentError(Exception):
    """所有Agent异常的基类。"""
    recoverable: bool = True        # 是否可被审查Agent返工修复

class SchemaValidationError(AgentError):
    """输出未通过Pydantic校验。recoverable=True"""

class ResourceError(AgentError):
    """资源不可用(模型/GPU/API)。recoverable=False,触发降级"""

class UserAbortError(AgentError):
    """用户主动中止。recoverable=False"""
```

### 4.2 异常处理流程

| 异常类型 | 编排层处理 |
|---------|-----------|
| SchemaValidationError | 自动返工(不计入重试上限) |
| ResourceError | 触发降级路径(切换后端/降级方案) |
| UserAbortError | 终止工作流,保存现场 |
| 其他Exception | 记入errors,升级到用户决策 |

## 5. 流式回调约定

```python
StreamCallback = Callable[[StreamEvent], Awaitable[None]]

# 编排层注入回调示例
async def on_event(evt: StreamEvent):
    state["stream_messages"].append(evt.dict())
    await ui_queue.put(evt)  # 推送到Gradio前端

context = AgentContext(..., stream_callback=on_event)
```

**事件发送频率指引**:
- THINKING: 每token或每50ms,取较慢
- PROGRESS: 每完成一个子步骤
- PARTIAL_OUTPUT: 关键中间产物完成时
- LOG: 仅重要状态变更

## 6. 审查钩子约定

每个生产Agent必须:
1. 声明 `rubric_id`,对应审查Agent的Rubric表
2. 输出模型必须包含 `meta: AgentOutputMeta` 字段
3. `revise()` 方法必须能解析 `ReviewFeedback.revise_prompt`

```python
class AgentOutputMeta(BaseModel):
    agent: str
    version: str
    generated_at: float
    inputs_hash: str                # 输入快照哈希,便于缓存
    retry_count: int
    self_check_notes: list[str] = []  # Agent自评(可选)
```

## 7. 版本管理

- 每个Agent接口变更须升级 `version` 字段
- 主版本变更需同步更新对应Rubric
- 接口契约变更需在本文档CHANGELOG记录

---

**文档版本**: v1.0
**适用Agent**: 所有(剧本/角色/分镜/图像/排版/审查)
